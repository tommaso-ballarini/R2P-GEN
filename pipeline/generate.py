"""
R2P-GEN Image Generator using SDXL + IP-Adapter.

This module handles image generation with identity preservation using
IP-Adapter with layer-wise scaling to prevent background contamination.

Usage:
    Standalone: python pipeline/generate.py
    Import: from pipeline.generate import Generator
"""

import json
import os
import sys
import torch
import gc
from PIL import Image
from tqdm import tqdm


# ============================================================================
# PATH CONFIGURATION
# ============================================================================
# We're in pipeline/, config.py and other modules are at project root

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Go up from pipeline/ to root

# Add project root to sys.path BEFORE importing from it
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Now we can safely import from project root
from config import Config
from diffusers import StableDiffusionXLPipeline


# ============================================================================
# GENERATOR CLASS
# ============================================================================

class Generator:
    """
    SDXL + IP-Adapter image generator following R2P workflow.
    
    This class processes a database of fingerprints and generates images
    for each concept using IP-Adapter with configurable scaling strategies.
    
    Attributes:
        database_path (str): Path to the fingerprint database JSON
        output_dir (str): Directory to save generated images
        device (str): Compute device ('cuda' or 'cpu')
        use_layerwise_scaling (bool): If True, use layer-wise IP-Adapter scaling
        
    Example:
        # Standalone usage
        generator = Generator(
            database_path="database/database_perva_train_1_clip.json",
            output_dir="output/generated_images"
        )
        stats = generator.generate_all()
        
        # Import usage in full_loop.py
        from pipeline.generate import Generator
        generator = Generator(database_path=db_path, output_dir=out_dir)
        generator.generate_all()
    """
    
    def __init__(
        self,
        database_path,
        output_dir,
        device=None,
        use_layerwise_scaling=None,
        ip_adapter_scale_global=None,
        ip_adapter_layer_weights=None,
        num_inference_steps=None,
        guidance_scale=None,
        negative_prompt=None,
        reference_image_size=None,
        output_image_size=None,
        sdxl_model=None,
        ip_adapter_repo=None,
        ip_adapter_subfolder=None,
        ip_adapter_weight_name=None,
        seed=None
    ):
        """
        Initialize the Generator with configuration parameters.
        
        All parameters default to values from Config if not specified,
        allowing flexible override for experiments.
        
        Args:
            database_path: Path to the fingerprint database JSON (required)
            output_dir: Directory to save generated images (required)
            device: Compute device ('cuda' or 'cpu')
            use_layerwise_scaling: If True, use layer-wise IP-Adapter scaling
            ip_adapter_scale_global: Global IP-Adapter scale (if not layer-wise)
            ip_adapter_layer_weights: Dict of layer-wise weights
            num_inference_steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale
            negative_prompt: Negative prompt for generation
            reference_image_size: Size to resize reference images
            output_image_size: Size of generated images
            sdxl_model: SDXL model identifier
            ip_adapter_repo: IP-Adapter repository
            ip_adapter_subfolder: IP-Adapter subfolder in repo
            ip_adapter_weight_name: IP-Adapter weight filename
        """
        # Required parameters
        self.database_path = database_path
        self.output_dir = output_dir
        
        # Optional parameters with Config defaults
        self.device = device or Config.GPU.DEVICE
        self.use_layerwise_scaling = use_layerwise_scaling if use_layerwise_scaling is not None else Config.Generate.USE_LAYERWISE_SCALING
        self.ip_adapter_scale_global = ip_adapter_scale_global or Config.Generate.IP_ADAPTER_SCALE_GLOBAL
        self.ip_adapter_layer_weights = ip_adapter_layer_weights or Config.Generate.IP_ADAPTER_LAYER_WEIGHTS
        self.num_inference_steps = num_inference_steps or Config.Generate.NUM_INFERENCE_STEPS
        self.guidance_scale = guidance_scale or Config.Generate.GUIDANCE_SCALE
        self.negative_prompt = negative_prompt or Config.Generate.NEGATIVE_PROMPT
        self.reference_image_size = reference_image_size or Config.Images.REFERENCE_IMAGE_SIZE
        self.output_image_size = output_image_size or Config.Images.OUTPUT_IMAGE_SIZE
        self.sdxl_model = sdxl_model or Config.Models.SDXL_MODEL
        self.ip_adapter_repo = ip_adapter_repo or Config.Models.IP_ADAPTER_REPO
        self.ip_adapter_subfolder = ip_adapter_subfolder or Config.Models.IP_ADAPTER_SUBFOLDER
        self.ip_adapter_weight_name = ip_adapter_weight_name or Config.Models.IP_ADAPTER_WEIGHT_NAME
        self.seed = seed if seed is not None else Config.Generate.SEED

        # Will be initialized during generate_all()
        self.pipe = None
        self.database = None
        self.concept_dict = None

        # Internal counters
        self._iter_count = 0  # Number of generate_single calls; used for lazy cache clearing and per-concept seed
        
        # Statistics
        self.stats = {
            "total_concepts": 0,
            "successful": 0,
            "skipped": 0,
            "failed": 0
        }
    
    def _load_database(self):
        """
        Load the fingerprint database from JSON file.
        
        Returns:
            bool: True if database loaded successfully, False otherwise
        """
        print(f"📂 Loading database from {self.database_path}...")
        
        if not os.path.exists(self.database_path):
            print(f"❌ Database not found at: {self.database_path}")
            print("   Run build_database.py first to create the fingerprint database.")
            return False
        
        with open(self.database_path, 'r', encoding='utf-8') as f:
            self.database = json.load(f)
        
        self.concept_dict = self.database.get("concept_dict", {})
        
        if not self.concept_dict:
            print("❌ Concept dictionary is empty.")
            return False
        
        self.stats["total_concepts"] = len(self.concept_dict)
        print(f"✅ Found {self.stats['total_concepts']} concepts.")
        return True
    
    def _initialize_pipeline(self):
        """
        Load and configure SDXL + IP-Adapter pipeline.
        
        Returns:
            bool: True if pipeline initialized successfully, False otherwise
        """
        print(f"🔌 Loading SDXL with IP-Adapter...")
        
        # Log scaling strategy
        if self.use_layerwise_scaling:
            print(f"   🎨 Strategy: Layer-Wise Scaling (R2P Optimized)")
            print(f"      → Down Blocks: 0.0-0.7 (zero background, shape preservation)")
            print(f"      → Mid Block: 0.9 (very high semantic identity)")
            print(f"      → Up Blocks: 0.6-0.95 (maximum texture/material fidelity)")
        else:
            print(f"   🎨 Strategy: Global Scale {self.ip_adapter_scale_global}")
        
        try:
            # Load SDXL pipeline
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                self.sdxl_model,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )
            
            # Load IP-Adapter
            self.pipe.load_ip_adapter(
                self.ip_adapter_repo,
                subfolder=self.ip_adapter_subfolder,
                weight_name=self.ip_adapter_weight_name
            )
            
            # Apply scaling strategy
            if self.use_layerwise_scaling:
                self.pipe.set_ip_adapter_scale(self.ip_adapter_layer_weights)
                print("   ✅ Layer-wise weights applied successfully")
            else:
                self.pipe.set_ip_adapter_scale(self.ip_adapter_scale_global)
                print(f"   ✅ Global scale {self.ip_adapter_scale_global} applied")
            
            # Memory optimizations
            print("   -> Enabling Model CPU Offload (Fix for Bus Error)...")
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_vae_slicing()
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def _generate_single(self, concept_id, content):
        """
        Generate image for a single concept.
        
        Args:
            concept_id: Unique identifier for the concept
            content: Dict containing name, info, image paths
            
        Returns:
            str or None: Path to saved image, or None if failed
        """
        name = content.get('name', 'unknown')
        info = content.get('info', {})
        prompt = info.get('sdxl_prompt', '')
        
        # Get reference image path
        img_list = content.get('image', [])
        if not img_list:
            self.stats["skipped"] += 1
            return None
        
        ref_img_path = img_list[0]
        
        # Validate inputs
        if not os.path.exists(ref_img_path):
            print(f"⚠️ Skipping {name}: Reference image not found.")
            self.stats["skipped"] += 1
            return None
        
        if not prompt:
            print(f"⚠️ Skipping {name}: No prompt found.")
            self.stats["skipped"] += 1
            return None
        
        # Load and resize reference image
        try:
            ref_image = Image.open(ref_img_path).convert("RGB")
            ref_image = ref_image.resize((self.reference_image_size, self.reference_image_size))
        except Exception as e:
            print(f"Error loading image {ref_img_path}: {e}")
            self.stats["failed"] += 1
            return None
        
        print(f"\n🔹 Generating: {name}")
        print(f"   Prompt: {prompt[:100]}...")

        # Increment counter (used for per-concept seed and lazy cache clearing)
        self._iter_count += 1

        # Generate image
        try:
            gen = torch.Generator(device=self.device).manual_seed(self.seed + self._iter_count)
            with torch.inference_mode():
                generated_image = self.pipe(
                    prompt=prompt,
                    negative_prompt=self.negative_prompt,
                    ip_adapter_image=ref_image,
                    height=self.output_image_size,
                    width=self.output_image_size,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    generator=gen
                ).images[0]

            # Save image
            method_suffix = "layerwise" if self.use_layerwise_scaling else "global"
            save_path = os.path.join(self.output_dir, f"{name}_ipa_{method_suffix}.png")
            generated_image.save(save_path)
            print(f"   💾 Saved to: {save_path}")

            self.stats["successful"] += 1

            # Lazy cache clearing: only every CLEAR_CACHE_EVERY iterations
            if Config.GPU.CLEAR_CACHE_EVERY > 0 and self._iter_count % Config.GPU.CLEAR_CACHE_EVERY == 0:
                torch.cuda.empty_cache()

            return save_path

        except Exception as e:
            print(f"❌ Error generating {name}: {e}")
            self.stats["failed"] += 1
            if Config.GPU.CLEAR_CACHE_EVERY > 0 and self._iter_count % Config.GPU.CLEAR_CACHE_EVERY == 0:
                torch.cuda.empty_cache()
            return None
    
    def generate_all(self):
        """
        Generate images for ALL concepts in the database.
        
        This is the main entry point for batch generation.
        
        Returns:
            dict: Statistics containing:
                - total_concepts: Number of concepts in database
                - successful: Number of successfully generated images
                - skipped: Number of skipped concepts (missing data)
                - failed: Number of failed generations
                - output_dir: Path to output directory
        """
        # Preventive memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
        # Load database
        if not self._load_database():
            return self.stats
        
        # Initialize pipeline
        if not self._initialize_pipeline():
            return self.stats
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate all images
        print(f"\n🎨 Starting Generation with IP-Adapter...")
        
        for concept_id, content in tqdm(self.concept_dict.items()):
            self._generate_single(concept_id, content)
        
        # Final report
        print(f"\n{'='*60}")
        print("📊 GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"   Total concepts: {self.stats['total_concepts']}")
        print(f"   ✅ Successful: {self.stats['successful']}")
        print(f"   ⚠️  Skipped: {self.stats['skipped']}")
        print(f"   ❌ Failed: {self.stats['failed']}")
        print(f"   📁 Output: {self.output_dir}")
        print(f"{'='*60}")
        
        self.stats["output_dir"] = self.output_dir
        return self.stats
    
    def cleanup(self):
        """Release GPU memory and cleanup resources."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        torch.cuda.empty_cache()
        gc.collect()


# ============================================================================
# STANDALONE FUNCTION FOR REFINEMENT LOOP
# ============================================================================

# Global pipeline cache for refinement loop efficiency
_cached_pipe = None
_cached_config = None

def generate_image(
    reference_image_path: str,
    fingerprints: dict,
    output_path: str,
    negative_prompt: str = None,
    iteration: int = 1,
    sdxl_prompt: str = None,
    seed: int = None
):
    """
    Generate a single image using SDXL + IP-Adapter.
    
    This standalone function is designed for use in the refinement loop.
    It caches the pipeline between calls for efficiency.
    
    Args:
        reference_image_path: Path to the reference image
        fingerprints: Dictionary containing extracted attributes
        output_path: Path to save the generated image
        negative_prompt: Negative prompt (overrides Config default)
        iteration: Current iteration number (for logging)
        sdxl_prompt: Explicit SDXL prompt (if None, uses fingerprints['description'] or 'sdxl_prompt')
        
    Returns:
        str: Path to the generated image, or None on failure
    """
    global _cached_pipe, _cached_config
    
    print(f"   🎨 Generating image (iteration {iteration})...")
    
    # Determine prompt
    if sdxl_prompt is None:
        sdxl_prompt = fingerprints.get("sdxl_prompt") or fingerprints.get("description", "A product photo")
    
    # Determine negative prompt
    if negative_prompt is None:
        negative_prompt = Config.Generate.NEGATIVE_PROMPT

    # Check if we need to load/reload the pipeline
    current_config = (Config.Models.SDXL_MODEL, Config.Models.IP_ADAPTER_REPO, Config.Generate.USE_LAYERWISE_SCALING)

    if _cached_pipe is None or _cached_config != current_config:
        print(f"   🔌 Loading SDXL + IP-Adapter pipeline...")

        # Cleanup old pipeline if exists
        if _cached_pipe is not None:
            del _cached_pipe
            torch.cuda.empty_cache()
            gc.collect()

        try:
            from diffusers import StableDiffusionXLPipeline

            # Load base SDXL
            _cached_pipe = StableDiffusionXLPipeline.from_pretrained(
                Config.Models.SDXL_MODEL,
                torch_dtype=torch.float16 if Config.GPU.USE_FP16 else torch.float32,
                use_safetensors=True,
                variant="fp16" if Config.GPU.USE_FP16 else None
            ).to(Config.GPU.DEVICE)

            # Load IP-Adapter
            _cached_pipe.load_ip_adapter(
                Config.Models.IP_ADAPTER_REPO,
                subfolder=Config.Models.IP_ADAPTER_SUBFOLDER,
                weight_name=Config.Models.IP_ADAPTER_WEIGHT_NAME
            )

            # Set IP-Adapter scale
            if Config.Generate.USE_LAYERWISE_SCALING and Config.Generate.IP_ADAPTER_LAYER_WEIGHTS:
                _cached_pipe.set_ip_adapter_scale(Config.Generate.IP_ADAPTER_LAYER_WEIGHTS)
            else:
                _cached_pipe.set_ip_adapter_scale(Config.Generate.IP_ADAPTER_SCALE_GLOBAL)
            
            _cached_config = current_config
            print(f"   ✓ Pipeline loaded")
            
        except Exception as e:
            print(f"   ❌ Failed to load pipeline: {e}")
            _cached_pipe = None
            return None
    
    # Load and process reference image
    try:
        from PIL import Image
        
        ref_image = Image.open(reference_image_path).convert("RGB")
        
        # Resize if needed
        if Config.Images.REFERENCE_IMAGE_SIZE:
            ref_image = ref_image.resize(
                (Config.Images.REFERENCE_IMAGE_SIZE, Config.Images.REFERENCE_IMAGE_SIZE),
                Image.Resampling.LANCZOS
            )

        # Generate
        _seed = seed if seed is not None else Config.Generate.SEED
        gen = torch.Generator(device=Config.GPU.DEVICE).manual_seed(_seed)
        result = _cached_pipe(
            prompt=sdxl_prompt,
            negative_prompt=negative_prompt,
            ip_adapter_image=ref_image,
            num_inference_steps=Config.Generate.NUM_INFERENCE_STEPS,
            guidance_scale=Config.Generate.GUIDANCE_SCALE,
            height=Config.Images.OUTPUT_IMAGE_SIZE,
            width=Config.Images.OUTPUT_IMAGE_SIZE,
            generator=gen
        )
        
        # Save
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        result.images[0].save(output_path)
        print(f"   ✓ Saved: {os.path.basename(output_path)}")
        
        return output_path
        
    except Exception as e:
        print(f"   ❌ Generation error: {e}")
        import traceback
        traceback.print_exc()
        return None


def cleanup_generator_cache():
    """Release the cached pipeline to free GPU memory."""
    global _cached_pipe, _cached_config
    
    if _cached_pipe is not None:
        del _cached_pipe
        _cached_pipe = None
        _cached_config = None
        torch.cuda.empty_cache()
        gc.collect()
        print("   🧹 Generator cache cleared")


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

def main():
    """
    Standalone entry point for image generation.
    
    Creates a Generator instance with default configuration and
    generates images for all concepts in the database.
    
    This function:
    1. Changes working directory to project root (for consistent paths)
    2. Uses default database and output paths at project root
    3. Runs full generation pipeline
    
    Returns:
        dict: Generation statistics
    """
    # Change to project root for consistent relative paths
    # This ensures all relative paths work correctly when running standalone
    os.chdir(PROJECT_ROOT)
    print(f"📂 Working directory: {os.getcwd()}")
    
    # Default paths relative to project root
    db_filename = "database.json" if Config.Database.CANONICAL_NAME else "database_perva_train_clip.json"
    default_database = os.path.join(PROJECT_ROOT, "database", db_filename)
    default_output = os.path.join(PROJECT_ROOT, "output", "generated_images")
    
    # Check if database exists
    if not os.path.exists(default_database):
        print(f"❌ Database not found: {default_database}")
        print("\n   Available databases:")
        db_dir = os.path.join(PROJECT_ROOT, "database")
        if os.path.exists(db_dir):
            for f in os.listdir(db_dir):
                if f.endswith('.json'):
                    print(f"      - {f}")
        else:
            print("      (database/ folder does not exist)")
        print("\n   Run build_database.py first to create a database.")
        return {"error": "Database not found"}
    
    print(f"\n{'='*70}")
    print("GENERATOR - Standalone Execution")
    print(f"{'='*70}")
    print(f"  Database: {default_database}")
    print(f"  Output:   {default_output}")
    print(f"{'='*70}\n")
    
    # Create and run generator
    generator = Generator(
        database_path=default_database,
        output_dir=default_output
    )
    
    try:
        stats = generator.generate_all()
    finally:
        generator.cleanup()
    
    return stats


if __name__ == "__main__":
    main()