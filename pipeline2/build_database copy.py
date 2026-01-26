import os
import sys
import glob
import json
import argparse
from tqdm import tqdm

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
# NOTE: Remember we're in pipeline2, adjust paths accordingly for future use

# Path Configuration
SOURCE_DATA_DIR = "data/perva-data"        # Source directory containing images
OUTPUT_JSON_PATH = "database/database_perva.json"  # Output database JSON path
DEVICE = "cuda"                            # Device for model inference (cuda/cpu)
MODEL_PATH = "openbmb/MiniCPM-o-2_6"      # MiniCPM model path

# Run Settings
DEBUG_MODE = False         # Set to False to process the entire dataset
DEBUG_LIMIT = 5            # Number of images to process in debug mode
USE_CLIP_CATEGORY = True   # True = Let R2P detect category via CLIP; False = Pass None/Generic

# SDXL Prompt Templates
# We ask the LLM ONLY for physical description. Style is added via code.
SYSTEM_PROMPT_TRANSLATOR = """
You are an expert Visual Describer.
You will receive a technical sheet of a product.
Convert it into a single, comma-separated description of the physical object.

RULES:
1. Start with the Subject (Category).
2. Describe visuals: material, color, shape.
3. Include specific details: visible text, patterns, flaws.
4. DO NOT add style tags (like '8k', 'photo'). Just describe the object.
5. Output ONLY the string.
"""

HARDCODED_STYLE = " Studio lighting, 8k, sharp focus, hyperrealistic, texture details, highly detailed"


# ============================================================================
# 2. SETUP & IMPORTS
# ============================================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(current_dir, "r2p_core")
utils_dir = os.path.join(core_dir, "utils")
models_dir = os.path.join(core_dir, "models")
database_dir = os.path.join(core_dir, "database")
eval_dir = os.path.join(core_dir, "evaluators")

for path in [core_dir, utils_dir, models_dir, database_dir, eval_dir]:
    if path not in sys.path:
        sys.path.append(path)

try:
    from database.mini_cpm_info import MiniCPMDescription
    
    class MockArgs:
        """Mock arguments for R2P model interface."""
        def __init__(self):
            self.user_defined = False
            self.template_based = True
            
except ImportError as e:
    print(f"❌ Error importing R2P modules: {e}")
    print("Ensure 'r2p_core' contains the 'src' files from the original repo.")
    sys.exit(1)


# ============================================================================
# 3. DATABASE BUILDER CLASS
# ============================================================================

class DatabaseBuilder:
    """
    Builds a database of image fingerprints and SDXL prompts.
    
    This class handles the complete pipeline for processing a directory of images:
    1. Loads the MiniCPM vision model
    2. Scans directories for images
    3. Extracts visual features (fingerprints) from each image
    4. Generates SDXL-compatible text prompts
    5. Saves everything to a structured JSON database
    
    Attributes:
        source_dir (str): Directory containing images to process
        output_path (str): Path where the JSON database will be saved
        device (str): Compute device ('cuda' or 'cpu')
        model_path (str): Path or name of the MiniCPM model
        debug_mode (bool): If True, processes only a limited subset
        debug_limit (int): Number of images to process in debug mode
        use_clip_category (bool): If True, uses CLIP to detect categories
        
    Class Constants:
        VALID_EXTENSIONS: Set of supported image file extensions
        FIELD_MAP: Mapping of fingerprint fields to human-readable labels
        NEGATIVE_TRIGGERS: Phrases that indicate missing/invalid data
    """
    
    # Valid image file extensions
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    
    # Mapping: R2P fingerprint fields -> Human-readable labels for LLM
    FIELD_MAP = [
        ('category', 'Category'),
        ('general', 'Base Description'),
        ('shape', 'Shape'),
        ('material', 'Material'),
        ('color', 'Color'),
        ('pattern', 'Pattern'),
        ('brand/text', 'Visible Text'),
        ('distinct features', 'Distinctive Flaws/Details')
    ]
    
    # Phrases indicating invalid/missing data that should be filtered out
    NEGATIVE_TRIGGERS = [
        "no visible", "none", "n/a", "not readable", 
        "unknown", "no brand", "no text"
    ]
    
    def __init__(self, source_dir, output_path, device="cuda", 
                 model_path="openbmb/MiniCPM-o-2_6",
                 debug_mode=False, debug_limit=5, use_clip_category=True,
                 system_prompt=None, hardcoded_style=None):
        """
        Initialize the DatabaseBuilder.
        
        Args:
            source_dir: Directory containing images to process
            output_path: Where to save the resulting JSON database
            device: Device for model inference ('cuda' or 'cpu')
            model_path: Path or identifier for the MiniCPM model
            debug_mode: If True, process only a subset of images
            debug_limit: Number of images to process in debug mode
            use_clip_category: If True, detect category via CLIP
            system_prompt: Custom prompt for LLM (uses global default if None)
            hardcoded_style: Custom style suffix (uses global default if None)
        """
        self.source_dir = source_dir
        self.output_path = output_path
        self.device = device
        self.model_path = model_path
        self.debug_mode = debug_mode
        self.debug_limit = debug_limit
        self.use_clip_category = use_clip_category
        
        # Use global constants if not provided
        self.system_prompt = system_prompt or SYSTEM_PROMPT_TRANSLATOR
        self.hardcoded_style = hardcoded_style or HARDCODED_STYLE
        
        # Will be initialized during build()
        self.extractor = None
        self.mock_args = None
        self.database_data = {
            "concept_dict": {},
            "path_to_concept": {}
        }
    
    def _get_image_files(self):
        """
        Recursively find all valid image files in the source directory.
        
        Automatically corrects common path errors (underscores vs hyphens).
        
        Returns:
            list: Sorted list of absolute paths to image files
        """
        abs_path = os.path.abspath(self.source_dir)
        print(f"   🔍 Looking in: {abs_path}")
        
        target_dir = self.source_dir
        
        # Automatic path correction for common naming issues
        if not os.path.exists(self.source_dir):
            print(f"   ⚠️ Path '{self.source_dir}' not found.")
            if "_" in self.source_dir and os.path.exists(self.source_dir.replace("_", "-")):
                target_dir = self.source_dir.replace("_", "-")
                print(f"   ✅ Found '{target_dir}' instead. Switching.")
            elif "-" in self.source_dir and os.path.exists(self.source_dir.replace("-", "_")):
                target_dir = self.source_dir.replace("-", "_")
                print(f"   ✅ Found '{target_dir}' instead. Switching.")
            elif os.path.exists(os.path.join("..", self.source_dir)):
                target_dir = os.path.join("..", self.source_dir)
                print(f"   ✅ Found in parent directory. Switching.")
            else:
                return []
        
        files = []
        for root, _, filenames in os.walk(target_dir):
            for filename in filenames:
                ext = os.path.splitext(filename)[1]
                if ext in self.VALID_EXTENSIONS:
                    files.append(os.path.join(root, filename))
                    
        return sorted(files)
    
    def _collect_features_safely(self, info_dict):
        """
        Aggregate fingerprint fields into a human-readable technical sheet.
        
        Filters out empty, invalid, or negative-indicator fields.
        
        Args:
            info_dict: Dictionary containing extracted fingerprints
            
        Returns:
            str: Formatted technical sheet (one field per line)
        """
        lines = []
        
        for key, label in self.FIELD_MAP:
            val = info_dict.get(key)
            
            if val and isinstance(val, str):
                clean_val = val.strip()
                val_lower = clean_val.lower()
                
                # Skip if too short
                if len(clean_val) < 2:
                    continue
                
                # Skip if contains negative indicators
                if any(trigger in val_lower for trigger in self.NEGATIVE_TRIGGERS):
                    continue
                
                lines.append(f"{label}: {clean_val}")
        
        return "\n".join(lines)
    
    def _generate_sdxl_prompt(self, full_info_dict):
        """
        Generate SDXL-compatible prompt from fingerprint data.
        
        Uses LLM to translate technical sheet into natural description,
        then appends hardcoded style suffix.
        
        Args:
            full_info_dict: Dictionary with extracted fingerprints
            
        Returns:
            str: Complete SDXL prompt ready for image generation
        """
        features_text = self._collect_features_safely(full_info_dict)
        
        # Fallback if no valid features found
        if not features_text:
            return f"A photorealistic image of a {full_info_dict.get('category', 'object')}, studio lighting."
        
        query = f"{self.system_prompt}\n\n--- INPUT TECHNICAL SHEET ---\n{features_text}"
        
        try:
            msgs = [{"role": "user", "content": query}]
            res = self.extractor.model.chat(msgs=msgs, tokenizer=self.extractor.tokenizer)
            
            # Handle tuple/list response
            if isinstance(res, (tuple, list)):
                description = res[0]
            else:
                description = res
                
            # Clean and append style
            clean_desc = description.strip().strip('"').strip("'")
            final_prompt = f"{clean_desc}{self.hardcoded_style}"
            
            return final_prompt
                       
        except Exception as e:
            print(f"⚠️ Warning: Failed to generate SDXL prompt: {e}")
            return features_text
    
    def _select_target_images(self, all_images):
        """
        Select which images to process based on debug mode.
        
        In debug mode, selects one image per folder up to debug_limit.
        In normal mode, returns all images.
        
        Args:
            all_images: Complete list of discovered image paths
            
        Returns:
            list: Filtered list of images to process
        """
        if not self.debug_mode:
            return all_images
        
        print(f"⚠️ DEBUG MODE ACTIVE: Selecting diverse images...")
        images_by_folder = {}
        
        for img_path in all_images:
            folder = os.path.dirname(img_path)
            if folder not in images_by_folder:
                images_by_folder[folder] = []
            images_by_folder[folder].append(img_path)
        
        folders = sorted(list(images_by_folder.keys()))
        target_images = []
        
        for folder in folders:
            if len(target_images) >= self.debug_limit:
                break
            target_images.append(images_by_folder[folder][0])
            
        print(f"   Selected {len(target_images)} images from {len(target_images)} different categories.")
        return target_images
    
    def _process_image(self, img_path):
        """
        Process a single image: extract fingerprints and generate SDXL prompt.
        
        Args:
            img_path: Absolute path to the image file
            
        Returns:
            tuple: (concept_id, info_dict) where info_dict includes 'sdxl_prompt'
        """
        filename = os.path.basename(img_path)
        file_stem = os.path.splitext(filename)[0]
        parent_folder = os.path.basename(os.path.dirname(img_path))
        concept_id = f"{parent_folder}_{file_stem}"
        
        cat_arg = None if self.use_clip_category else "object"
        
        # Extract fingerprints using MiniCPM
        json_str = self.extractor.generate_caption(
            image_file=img_path,
            cat=cat_arg,
            concept_identifier=concept_id, 
            args=self.mock_args
        )
        
        # Clean and parse JSON response
        json_str = json_str.replace('```json', '').replace('```', '').strip()
        item_info = json.loads(json_str)
        
        # Generate SDXL prompt and add to info
        sdxl_prompt = self._generate_sdxl_prompt(item_info)
        item_info["sdxl_prompt"] = sdxl_prompt
        
        return concept_id, item_info
    
    def _initialize_models(self):
        """Load the MiniCPM model and initialize mock arguments."""
        print("\n[1/4] Loading MiniCPM Model...")
        self.extractor = MiniCPMDescription(model_path=self.model_path, device=self.device)
        self.mock_args = MockArgs()
    
    def _save_database(self):
        """Save the database to JSON file."""
        print(f"\n[4/4] Saving database to {self.output_path}...")
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        with open(self.output_path, "w") as f:
            json.dump(self.database_data, f, indent=4)
    
    def build_database(self):
        """
        Execute the complete database building pipeline.
        
        This is the main method that orchestrates the entire process:
        1. Initialize models
        2. Scan for images
        3. Process each image (extract + generate prompt)
        4. Save database to JSON
        
        Returns:
            dict: Statistics with keys:
                - success_count: Number of successfully processed images
                - total_images: Total number of images attempted
                - database_path: Absolute path to the saved database file
        """
        print(f"🚀 Starting Build Database")
        print(f"📂 Source: {self.source_dir}")
        print(f"💾 Output: {self.output_path}")
        print(f"🔧 Device: {self.device}")
        print(f"🧪 Debug Mode: {self.debug_mode}")
        
        # Step 1: Initialize models
        self._initialize_models()
        
        # Step 2: Find images
        print("\n[2/4] Scanning for images...")
        all_images = self._get_image_files()
        
        if not all_images:
            print(f"❌ No images found in {self.source_dir}")
            return {"success_count": 0, "total_images": 0}
        
        print(f"Found {len(all_images)} total images.")
        
        # Step 3: Select target images (debug mode filter)
        target_images = self._select_target_images(all_images)
        
        # Step 4: Process images
        print("\n[3/4] Extracting Fingerprints & Generating Prompts...")
        
        success_count = 0
        
        for img_path in tqdm(target_images):
            try:
                concept_id, item_info = self._process_image(img_path)
                
                key = f"<{concept_id}>"
                
                self.database_data["concept_dict"][key] = {
                    "name": concept_id,
                    "image": [img_path],
                    "info": item_info,
                    "category": item_info.get("category", "unknown")
                }
                
                self.database_data["path_to_concept"][img_path] = key
                success_count += 1
                
            except Exception as e:
                print(f"\n❌ Error processing {os.path.basename(img_path)}: {e}")
                continue
        
        # Step 5: Save database
        self._save_database()
        
        print(f"✅ Done! Processed {success_count}/{len(target_images)} images.")
        print(f"   Database saved at: {os.path.abspath(self.output_path)}")
        
        return {
            "success_count": success_count,
            "total_images": len(target_images),
            "database_path": os.path.abspath(self.output_path)
        }


# --- 5. MAIN ENTRY POINT ---

def main():
    """
    Script entry point.
    
    Creates a DatabaseBuilder instance with global configuration
    and executes the build process.
    """
    builder = DatabaseBuilder(
        source_dir=SOURCE_DATA_DIR,
        output_path=OUTPUT_JSON_PATH,
        device=DEVICE,
        model_path=MODEL_PATH,
        debug_mode=DEBUG_MODE,
        debug_limit=DEBUG_LIMIT,
        use_clip_category=USE_CLIP_CATEGORY
    )
    
    stats = builder.build_database()
    return stats


if __name__ == "__main__":
    main()