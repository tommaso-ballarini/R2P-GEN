import os
import sys
import glob
import json
import argparse
import numpy as np
from tqdm import tqdm

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
# All configuration has been moved to config.py for centralized management.
# Import Config to access settings via Config.BuildDatabase.*, Config.Models.*, etc.


# ============================================================================
# 2. SETUP & IMPORTS
# ============================================================================

# Path Configuration: We're in pipeline/, r2p_core is at the project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up one level from pipeline/
core_dir = os.path.join(project_root, "r2p_core")
utils_dir = os.path.join(core_dir, "utils")
models_dir = os.path.join(core_dir, "models")
database_dir = os.path.join(project_root, "database")
eval_dir = os.path.join(core_dir, "evaluators")

for path in [project_root, core_dir, utils_dir, models_dir, database_dir, eval_dir]:
    if path not in sys.path:
        sys.path.append(path)

# Import SDXL prompt templates from prompts module
from pipeline.prompts import (
    SYSTEM_PROMPT_SIMPLE,
    SYSTEM_PROMPT_GEMINI,
    SYSTEM_PROMPT_OPTIMIZED,
    HARDCODED_STYLE
)

# Import centralized config
from config import Config

try:
    from database.mini_cpm_info import MiniCPMDescription
    from database.create_train_test_perva_split import CLIPImageProcessor
    
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
    Builds a database of image fingerprints and SDXL prompts following R2P workflow.

    This class processes images at the CONCEPT level (not individual images):
    - Each concept = a unique physical object with multiple views
    - Selects ONE representative image per concept
    - Extracts fingerprints from that single image
    - Stores the selected image in the database

    R2P Dataset Structure:
        data/perva-data/
        ├── train/
        │   ├── bag/
        │   │   ├── alx/  ← concept (unique object)
        │   │   │   ├── 1.jpg, 2.jpg, ..., 8.jpg  ← multi-view
        │   │   │   └── laion/ (ignored - training data)
        │   │   ├── ash/
        │   │   └── ...
        │   └── ...
        └── test/ (same structure)

    Attributes:
        source_dir (str): Base directory containing train/test splits
        output_path (str): Where to save the JSON database
        device (str): Compute device ('cuda' or 'cpu')
        model_path (str): MiniCPM model identifier
        dataset_split (str): Which split to process ('train', 'test', 'all')
        debug_mode (bool): If True, process only debug_limit concepts
        debug_limit (int): Number of concepts to process in debug mode
        use_clip_category (bool): If True, auto-detect category with CLIP
        use_clip_selection (bool): If True, select most representative image via CLIP
        ignore_laion (bool): If True, skip 'laion' subdirectories
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
                 dataset_split="train",
                 debug_mode=False, debug_limit=5, use_clip_category=True,
                 use_clip_selection=True, ignore_laion=True, seed=42,
                 prompt_strategy='gemini', system_prompt=None, hardcoded_style=None):
        """
        Initialize the DatabaseBuilder following R2P workflow.

        Note: images_per_concept has been removed. The script now always uses
        1 image per concept (R2P standard). All images are still stored in the
        database for backward compatibility.

        Args:
            source_dir: Directory containing images to process
            output_path: Where to save the resulting JSON database
            device: Device for model inference ('cuda' or 'cpu')
            model_path: Path or identifier for the MiniCPM model
            dataset_split: Which split to process ('train', 'test', 'all')
            debug_mode: If True, process only a subset of concepts
            debug_limit: Number of concepts to process in debug mode
            use_clip_category: If True, detect category via CLIP
            use_clip_selection: If True, select images closest to CLIP centroid
            ignore_laion: If True, ignore 'laion' subdirectories
            seed: Random seed for reproducible image selection (R2P compatibility)
            prompt_strategy: Which prompt template to use ('simple'/'optimized'/'gemini')
                            'simple' = Natural description + hardcoded style (baseline)
                            'optimized' = Hierarchical tags with weights (R2P enhanced)
                            'gemini' = Brand-first ultra-concise (SOTA personalization)
            system_prompt: Custom prompt for LLM (uses global default based on strategy)
            hardcoded_style: Custom style suffix (uses global default if None)
        """
        self.source_dir = source_dir
        self.output_path = output_path
        self.device = device
        self.model_path = model_path
        self.dataset_split = dataset_split
        self.debug_mode = debug_mode
        self.debug_limit = debug_limit
        self.use_clip_category = use_clip_category
        self.use_clip_selection = use_clip_selection
        self.ignore_laion = ignore_laion
        self.seed = seed
        self.prompt_strategy = prompt_strategy.lower()
        
        # Select appropriate prompt template if not provided
        if system_prompt is None:
            if self.prompt_strategy == 'gemini':
                self.system_prompt = SYSTEM_PROMPT_GEMINI
            elif self.prompt_strategy == 'optimized':
                self.system_prompt = SYSTEM_PROMPT_OPTIMIZED
            else:  # 'simple'
                self.system_prompt = SYSTEM_PROMPT_SIMPLE
        else:
            self.system_prompt = system_prompt
            
        self.hardcoded_style = hardcoded_style or HARDCODED_STYLE
        
        # Will be initialized during build()
        self.extractor = None
        self.mock_args = None
        self.clip_processor = None
        self.database_data = {
            "concept_dict": {},
            "path_to_concept": {}
        }
    
    def _get_concepts(self):
        """
        Discover all concepts in the dataset following R2P structure.
        
        Scans the dataset and identifies:
        - Categories (bag, pillow, plant, etc.)
        - Concepts within categories (alx, ash, bkq, etc.)
        - Images for each concept
        
        Returns:
            list: List of dicts, each with:
                - 'category': Category name
                - 'concept_id': Unique concept identifier
                - 'concept_path': Full path to concept directory
                - 'images': Sorted list of image paths
                - 'split': Which split this concept belongs to
        """
        abs_path = os.path.abspath(self.source_dir)
        print(f"   🔍 Looking in: {abs_path}")
        print(f"   📊 Split: {self.dataset_split}")
        
        # Automatic path correction
        target_dir = self.source_dir
        if not os.path.exists(self.source_dir):
            print(f"   ⚠️ Path '{self.source_dir}' not found.")
            if "_" in self.source_dir and os.path.exists(self.source_dir.replace("_", "-")):
                target_dir = self.source_dir.replace("_", "-")
                print(f"   ✅ Found '{target_dir}' instead. Switching.")
            elif "-" in self.source_dir and os.path.exists(self.source_dir.replace("-", "_")):
                target_dir = self.source_dir.replace("-", "_")
                print(f"   ✅ Found '{target_dir}' instead. Switching.")
            else:
                return []
        
        concepts = []
        
        # Determine which splits to process
        splits_to_process = []
        if self.dataset_split == "all":
            splits_to_process = ["train", "test"]
        else:
            splits_to_process = [self.dataset_split]
        
        # Scan for concepts
        for split in splits_to_process:
            split_dir = os.path.join(target_dir, split)
            if not os.path.exists(split_dir):
                print(f"   ⚠️ Split directory '{split}' not found, skipping...")
                continue
            
            # Iterate over categories (bag, pillow, etc.)
            for category in sorted(os.listdir(split_dir)):
                category_path = os.path.join(split_dir, category)
                if not os.path.isdir(category_path):
                    continue
                
                # Iterate over concepts (alx, ash, etc.)
                for concept_id in sorted(os.listdir(category_path)):
                    concept_path = os.path.join(category_path, concept_id)
                    if not os.path.isdir(concept_path):
                        continue
                    
                    # Skip 'laion' directories if configured
                    if self.ignore_laion and concept_id.lower() == "laion":
                        continue
                    
                    # Find all images in this concept (skip subdirectories)
                    images = []
                    for item in sorted(os.listdir(concept_path)):
                        item_path = os.path.join(concept_path, item)
                        
                        # Skip subdirectories (like laion/)
                        if os.path.isdir(item_path):
                            continue
                        
                        ext = os.path.splitext(item)[1]
                        if ext in self.VALID_EXTENSIONS:
                            images.append(item_path)
                    
                    if images:
                        concepts.append({
                            'category': category,
                            'concept_id': concept_id,
                            'concept_path': concept_path,
                            'images': images,  # Already sorted
                            'split': split
                        })
        
        return concepts
    
    def _select_images_for_concept(self, images):
        """
        Select ONE representative image from a concept's image list for fingerprinting.

        R2P Standard Approach:
        - Always selects EXACTLY ONE image per concept
        - This image is used for both fingerprinting and stored in the database

        Two selection strategies:
        1. CLIP-based (R2P enhanced):
           - Select the image closest to CLIP feature centroid
           - This is the most representative image of the concept
        2. Simple: Use first image (sorted numerically)

        Args:
            images: List of image paths for a concept

        Returns:
            list: Single-element list containing the selected image path
        """
        if len(images) == 0:
            return []

        # Strategy 1: CLIP-based selection - find most representative image
        if self.use_clip_selection and self.clip_processor is not None:
            try:
                # Extract CLIP features for all images
                clip_features = self.clip_processor.extract_clip_features(images)

                # Get image closest to centroid (the most representative one)
                _, closest_indices = self.clip_processor.get_closest_to_mean_features(
                    clip_features, top_n=1
                )

                # Return only the most representative image
                representative_idx = closest_indices[0]
                return [images[representative_idx]]

            except Exception as e:
                print(f"⚠️ CLIP selection failed: {e}, falling back to simple selection")

        # Strategy 2: Simple selection - use first image (sorted numerically)
        return [images[0]]
    
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
        
        Post-processing applies:
        - Background specification from Config.SDXL_BACKGROUND_STYLE
        - Quality suffix from Config.SDXL_QUALITY_SUFFIX
        - Optional subject weight if Config.SDXL_USE_PROMPT_WEIGHTS=True
        
        Three strategies available (controlled by SDXL_PROMPT_STRATEGY):
        
        1. SIMPLE (baseline): Clean comma-separated tags
           - No weights, refinement-friendly
        
        2. OPTIMIZED (R2P enhanced): Hierarchical tags
           - No weights, discriminative traits first
        
        3. GEMINI (SOTA personalization): Brand-first ultra-concise
           - No weights, maximum identity preservation
        
        Args:
            full_info_dict: Dictionary with extracted fingerprints
            
        Returns:
            str: Complete SDXL prompt ready for image generation
        """
        import re
        
        features_text = self._collect_features_safely(full_info_dict)
        concept_name = full_info_dict.get('category', 'object')

        # Get background and quality suffix from config
        background_template = Config.get_background_template()
        quality_suffix = Config.Generate.SDXL_QUALITY_SUFFIX
        
        # Build suffix (background + quality)
        suffix_parts = []
        if background_template:
            suffix_parts.append(background_template)
        if quality_suffix:
            suffix_parts.append(quality_suffix)
        full_suffix = ", ".join(suffix_parts)
        
        # Fallback if no valid features found
        if not features_text:
            base_prompt = f"a {concept_name}"
            if Config.Generate.SDXL_USE_PROMPT_WEIGHTS:
                base_prompt = f"({concept_name}:{Config.Generate.SDXL_SUBJECT_WEIGHT})"
            return f"{base_prompt}, {full_suffix}"
        
        # Build query with appropriate prompt template
        query = f"{self.system_prompt}\n{features_text}"
        
        try:
            msgs = [{"role": "user", "content": query}]
            res = self.extractor.model.chat(msgs=msgs, tokenizer=self.extractor.tokenizer)
            
            # Handle tuple/list response
            if isinstance(res, (tuple, list)):
                description = res[0]
            else:
                description = res
                
            # Clean output
            clean_desc = description.strip().strip('"').strip("'")

            # Remove any weights the LLM might have added (enforce clean prompts)
            if not Config.Generate.SDXL_USE_PROMPT_WEIGHTS:
                # Remove patterns like (word:1.3) -> word
                clean_desc = re.sub(r'\(([^:]+):\d+\.?\d*\)', r'\1', clean_desc)
            
            # Remove any background/quality tags LLM might have added
            # (we'll add them ourselves for consistency)
            remove_patterns = [
                r',?\s*studio lighting',
                r',?\s*professional product photography',
                r',?\s*8k(\s+resolution)?',
                r',?\s*sharp focus',
                r',?\s*hyperrealistic(\s+photograph)?',
                r',?\s*highly detailed(\s+textures)?',
                r',?\s*white background',
                r',?\s*clean background',
                r',?\s*neutral background',
                r',?\s*seamless\s+\w+\s+background',
                r',?\s*on white surface',
                r',?\s*placed on.*?(?=,|$)',
                r',?\s*soft studio lighting',
                r',?\s*isolated on.*?(?=,|$)',
            ]
            for pattern in remove_patterns:
                clean_desc = re.sub(pattern, '', clean_desc, flags=re.IGNORECASE)
            
            # Clean up multiple commas and trailing commas
            clean_desc = re.sub(r',\s*,', ',', clean_desc)
            clean_desc = re.sub(r',\s*$', '', clean_desc)
            clean_desc = clean_desc.strip()
            
            # Optionally add weight to main subject (if enabled in config)
            if Config.Generate.SDXL_USE_PROMPT_WEIGHTS:
                # Check if category is at the start and wrap it
                if concept_name.lower() in clean_desc.lower()[:50]:
                    # Find and wrap the category mention
                    pattern = rf'\b({re.escape(concept_name)})\b'
                    clean_desc = re.sub(
                        pattern,
                        f'({concept_name}:{Config.Generate.SDXL_SUBJECT_WEIGHT})',
                        clean_desc,
                        count=1,
                        flags=re.IGNORECASE
                    )
            
            # Combine: description + background + quality
            final_prompt = f"{clean_desc}, {full_suffix}"
            
            return final_prompt
                       
        except Exception as e:
            print(f"⚠️ Warning: Failed to generate SDXL prompt: {e}")
            # Fallback
            base_prompt = f"{features_text}"
            return f"{base_prompt}, {full_suffix}"
    
    def _process_concept(self, concept_data):
        """
        Process a concept following R2P workflow:
        1. Select ONE representative image (CLIP centroid or first image)
        2. Extract fingerprints from that image
        3. Generate SDXL prompt from fingerprints
        4. Store the image in the database

        Args:
            concept_data: Dict with 'category', 'concept_id', 'images', etc.

        Returns:
            tuple: (concept_key, entry_dict) for database
        """
        category = concept_data['category']
        concept_id = concept_data['concept_id']
        all_images = concept_data['images']

        # Select ONE representative image for this concept
        selected_images = self._select_images_for_concept(all_images)

        if not selected_images:
            raise ValueError(f"No images selected for concept {concept_id}")

        # Extract fingerprints from the selected image
        representative_image = selected_images[0]

        cat_arg = None if self.use_clip_category else category

        # Extract fingerprints using MiniCPM
        json_str = self.extractor.generate_caption(
            image_file=representative_image,
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

        # Build concept key (R2P format)
        concept_key = f"<{concept_id}>"

        # Build database entry
        entry = {
            "name": concept_id,
            "image": selected_images,  # Single-element list with the representative image
            "info": item_info,
            "category": category
        }

        return concept_key, entry
    
    def _select_target_concepts(self, all_concepts):
        """
        Select which concepts to process based on debug mode.
        
        Args:
            all_concepts: Complete list of discovered concepts
            
        Returns:
            list: Filtered list of concepts to process
        """
        if not self.debug_mode:
            return all_concepts
        
        print(f"⚠️ DEBUG MODE ACTIVE: Selecting {self.debug_limit} concepts...")
        return all_concepts[:self.debug_limit]
    
    def _initialize_models(self):
        """Load the MiniCPM model and CLIP processor, initialize mock arguments."""
        print("\n[1/5] Loading Models...")
        
        # Check available RAM before loading
        try:
            import psutil
            ram_gb = psutil.virtual_memory().available / (1024**3)
            print(f"   💾 Available RAM: {ram_gb:.1f} GB")
            if ram_gb < 20:
                print(f"   ⚠️  WARNING: Low RAM! MiniCPM-o-2.6 needs ~30GB during loading")
        except ImportError:
            pass
        
        print("   - MiniCPM for fingerprint extraction")
        self.extractor = MiniCPMDescription(model_path=self.model_path, device=self.device)
        self.mock_args = MockArgs()
        
        # Load CLIP for image selection if enabled
        if self.use_clip_selection:
            print("   - CLIP for image selection (closest to centroid)")
            self.clip_processor = CLIPImageProcessor(device=self.device)
        else:
            print("   - Using simple selection (first N images)")
            self.clip_processor = None
    
    def _save_database(self):
        """Save the database to JSON file."""
        print(f"\n[4/5] Saving database to {self.output_path}...")
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        with open(self.output_path, "w") as f:
            json.dump(self.database_data, f, indent=4)
    
    def extract_single_image(self, image_path: str, category: str = None) -> dict:
        """
        Extract fingerprints from a single image.
        
        This is a convenience method for processing individual images
        without going through the full database pipeline.
        
        Args:
            image_path: Path to the image file
            category: Optional category hint (e.g., 'bag', 'shoe')
            
        Returns:
            dict: Fingerprints including sdxl_prompt, or None if failed
        """
        # Initialize extractor if not already done
        if self.extractor is None:
            print("   📦 Loading MiniCPM for extraction...")
            from r2p_core.database.mini_cpm_info import MiniCPMDescription
            self.extractor = MiniCPMDescription(model_path=self.model_path, device=self.device)
            self.mock_args = MockArgs()
        
        try:
            # Extract fingerprints
            cat_arg = None if self.use_clip_category else category
            concept_id = os.path.splitext(os.path.basename(image_path))[0]
            
            json_str = self.extractor.generate_caption(
                image_file=image_path,
                cat=cat_arg,
                concept_identifier=concept_id,
                args=self.mock_args
            )
            
            # Clean and parse JSON
            json_str = json_str.replace('```json', '').replace('```', '').strip()
            fingerprints = json.loads(json_str)
            
            # Generate SDXL prompt
            sdxl_prompt = self._generate_sdxl_prompt(fingerprints)
            fingerprints["sdxl_prompt"] = sdxl_prompt
            
            return fingerprints
            
        except Exception as e:
            print(f"   ❌ Extraction failed: {e}")
            return None
    
    def build_database(self):
        """
        Execute the complete database building pipeline following R2P workflow.
        
        Workflow:
        1. Initialize models (MiniCPM)
        2. Discover concepts (not individual images)
        3. For each concept:
           - Select N images
           - Extract fingerprints from first image
           - Generate SDXL prompt
           - Store all selected images
        4. Save database to JSON
        
        Returns:
            dict: Statistics with keys:
                - success_count: Number of successfully processed concepts
                - total_concepts: Total number of concepts attempted
                - total_images: Total number of images stored
                - database_path: Absolute path to the saved database file
        """
        print(f"🚀 Starting Build Database (R2P Workflow)")
        print(f"📂 Source: {self.source_dir}")
        print(f"💾 Output: {self.output_path}")
        print(f"🔧 Device: {self.device}")
        print(f"📊 Split: {self.dataset_split}")
        print(f"🎯 CLIP Selection: {self.use_clip_selection}")
        print(f"🌱 Seed: {self.seed}")
        print(f"✨ SDXL Prompt: {self.prompt_strategy.upper()}")
        print(f"🧪 Debug Mode: {self.debug_mode}")
        
        # Step 1: Initialize models
        self._initialize_models()
        
        # Step 2: Discover concepts
        print("\n[2/5] Discovering concepts...")
        all_concepts = self._get_concepts()
        
        if not all_concepts:
            print(f"❌ No concepts found in {self.source_dir}")
            return {"success_count": 0, "total_concepts": 0, "total_images": 0}
        
        print(f"Found {len(all_concepts)} concepts across categories.")
        
        # Step 3: Select target concepts (debug mode filter)
        target_concepts = self._select_target_concepts(all_concepts)
        print(f"Processing {len(target_concepts)} concepts...")
        
        # Step 4: Process concepts
        print("\n[3/5] Extracting Fingerprints & Generating Prompts...")
        
        success_count = 0
        total_images_stored = 0
        
        for concept_data in tqdm(target_concepts, desc="Processing concepts"):
            try:
                concept_key, entry = self._process_concept(concept_data)
                
                # Add to database
                self.database_data["concept_dict"][concept_key] = entry
                
                # Map each image path to its concept
                for img_path in entry["image"]:
                    self.database_data["path_to_concept"][img_path] = concept_key
                
                success_count += 1
                total_images_stored += len(entry["image"])
                
            except Exception as e:
                concept_id = concept_data.get('concept_id', 'unknown')
                print(f"\n❌ Error processing concept '{concept_id}': {e}")
                continue
        
        # Step 5: Save database
        self._save_database()
        
        print(f"\n✅ Done! Processed {success_count}/{len(target_concepts)} concepts.")
        print(f"   Total images stored: {total_images_stored}")
        print(f"   Database saved at: {os.path.abspath(self.output_path)}")
        
        return {
            "success_count": success_count,
            "total_concepts": len(target_concepts),
            "total_images": total_images_stored,
            "database_path": os.path.abspath(self.output_path)
        }


# ============================================================================
# 4. MAIN ENTRY POINT
# ============================================================================

def main():
    """
    Script entry point.

    Creates a DatabaseBuilder instance with global configuration
    and executes the build process following R2P workflow.

    Output filename format (when Config.Database.CANONICAL_NAME=False):
        database_perva_{split}_{selection}.json
    Examples:
        - database_perva_train_clip.json      (CLIP-based selection)
        - database_perva_train_simple.json    (Simple first-image selection)
        - database_perva_test_clip.json       (Test split, CLIP selection)
        - database_perva_all_simple.json      (Both splits, simple selection)

    When Config.Database.CANONICAL_NAME=True:
        - database.json (canonical name for production/main branch)
    """
    # Calculate absolute paths based on script location
    # This works whether called from pipeline/ or root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_main = os.path.dirname(script_dir)  # Go up from pipeline/ to root

    # Change working directory to project root
    # This ensures all relative paths (e.g., example_database/) work correctly
    os.chdir(project_root_main)

    # Source data is always at project_root/data/perva-data
    source_data_absolute = os.path.join(project_root_main, "data", "perva-data")

    # Generate output filename: canonical "database.json" for production/main,
    # or descriptive name for testing branches (controlled by Config.Database.CANONICAL_NAME)
    if Config.Database.CANONICAL_NAME:
        output_filename = "database.json"
    else:
        selection_method = "clip" if Config.BuildDatabase.USE_CLIP_SELECTION else "simple"
        output_filename = f"database_perva_{Config.BuildDatabase.DATASET_SPLIT}_{selection_method}.json"
    output_path = os.path.join(project_root_main, "database", output_filename)

    print(f"\n" + "="*70)
    print(f"DATABASE BUILDER - R2P Workflow")
    print(f"="*70)
    print(f"Configuration:")
    print(f"  - Dataset Split: {Config.BuildDatabase.DATASET_SPLIT}")
    print(f"  - CLIP Selection: {Config.BuildDatabase.USE_CLIP_SELECTION}")
    print(f"  - Seed: {Config.BuildDatabase.SEED}")
    strategy_names = {'simple': 'SIMPLE (baseline)', 'optimized': 'OPTIMIZED (R2P)', 'gemini': 'GEMINI (SOTA)'}
    print(f"  - SDXL Prompt: {strategy_names.get(Config.BuildDatabase.SDXL_PROMPT_STRATEGY, Config.BuildDatabase.SDXL_PROMPT_STRATEGY)}")
    print(f"  - Ignore Laion: {Config.BuildDatabase.IGNORE_LAION}")
    print(f"  - Output: {output_path}")
    print(f"="*70 + "\n")

    builder = DatabaseBuilder(
        source_dir=source_data_absolute,
        output_path=output_path,
        device=Config.GPU.DEVICE,
        model_path=Config.Models.VLM_MODEL,
        dataset_split=Config.BuildDatabase.DATASET_SPLIT,
        debug_mode=Config.BuildDatabase.DEBUG_MODE,
        debug_limit=Config.BuildDatabase.DEBUG_LIMIT,
        use_clip_category=Config.BuildDatabase.USE_CLIP_CATEGORY,
        use_clip_selection=Config.BuildDatabase.USE_CLIP_SELECTION,
        ignore_laion=Config.BuildDatabase.IGNORE_LAION,
        seed=Config.BuildDatabase.SEED,
        prompt_strategy=Config.BuildDatabase.SDXL_PROMPT_STRATEGY
    )

    stats = builder.build_database()
    return stats


if __name__ == "__main__":
    main()
