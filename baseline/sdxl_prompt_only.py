# baseline/sdxl_prompt_only.py
"""
BASELINE 2: Solo SDXL Prompt (senza IP-Adapter)

Generazione usando SOLO il prompt SDXL estratto,
senza injection di identità via IP-Adapter.

Testa la qualità del prompt da solo (text-to-image puro).

Usage:
    python baseline/sdxl_prompt_only.py --category bag --num 5
    python baseline/sdxl_prompt_only.py --category bottle --num 10
"""

import os
import sys
import json
import torch
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from baseline.config_baseline import BaselineConfig
from pipeline.utils2 import cleanup_gpu, ensure_output_dir


class SDXLPromptOnlyGenerator:
    """
    Generatore baseline: solo prompt SDXL (NO IP-Adapter).
    
    Non usa:
    - IP-Adapter
    - Immagine di riferimento
    - Layer-wise scaling
    
    Usa:
    - Solo il prompt SDXL estratto dal database
    """
    
    def __init__(self, category: str = None, num_concepts: int = None):
        """
        Args:
            category: Categoria da testare (default: da config)
            num_concepts: Numero di concetti da testare (default: da config)
        """
        self.category = category or BaselineConfig.CATEGORY
        self.num_concepts = num_concepts or BaselineConfig.NUM_CONCEPTS
        
        # Validate category
        if not BaselineConfig.validate_category(self.category):
            raise ValueError(
                f"Invalid category: {self.category}. "
                f"Available: {BaselineConfig.get_available_categories()}"
            )
        
        self.output_dir = os.path.join(
            BaselineConfig.OUTPUT_BASE,
            "sdxl_prompt_only",
            self.category
        )
        ensure_output_dir(self.output_dir)
        
        self.pipe = None
        self.database = None
        
        self._print_header()
    
    def _print_header(self):
        """Print experiment header."""
        print(f"\n{'='*70}")
        print("📝 BASELINE 2: SDXL Prompt Only (No IP-Adapter)")
        print(f"{'='*70}")
        print(f"   Category:    {self.category}")
        print(f"   Num Concepts: {self.num_concepts}")
        print(f"   Output:      {self.output_dir}")
        print(f"   Mode:        Text-to-Image ONLY")
        print(f"   Device:      {BaselineConfig.DEVICE}")
        print(f"{'='*70}\n")
    
    def load_database(self):
        """Load database and filter by category."""
        print(f"📂 Loading database from {BaselineConfig.DATABASE_PATH}...")
        
        if not os.path.exists(BaselineConfig.DATABASE_PATH):
            raise FileNotFoundError(f"Database not found: {BaselineConfig.DATABASE_PATH}")
        
        with open(BaselineConfig.DATABASE_PATH, 'r', encoding='utf-8') as f:
            db = json.load(f)
        
        concept_dict = db.get("concept_dict", {})
        
        # Filter by category
        filtered = {
            k: v for k, v in concept_dict.items()
            if v.get("category") == self.category
        }
        
        if not filtered:
            raise ValueError(f"No concepts found for category: {self.category}")
        
        # Take first N concepts
        concept_ids = list(filtered.keys())[:self.num_concepts]
        self.database = {k: filtered[k] for k in concept_ids}
        
        print(f"   ✅ Loaded {len(self.database)} concepts from category '{self.category}'")
        print(f"   📝 Concepts: {list(self.database.keys())}")
    
    def initialize_pipeline(self):
        """Initialize SDXL pipeline WITHOUT IP-Adapter."""
        print("\n🔌 Loading SDXL pipeline (NO IP-Adapter)...")
        
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            BaselineConfig.SDXL_MODEL,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        ).to(BaselineConfig.DEVICE)
        
        # NO IP-ADAPTER - questo è text-to-image puro!
        
        # Memory optimizations
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_slicing()
        
        print("   ✅ Pipeline ready (text-to-image only, no IP-Adapter)")
    
    def generate_all(self) -> dict:
        """
        Generate images for all concepts.
        
        Returns:
            dict: Statistics with success/failed counts
        """
        self.load_database()
        self.initialize_pipeline()
        
        stats = {
            "total": len(self.database),
            "success": 0,
            "failed": 0,
            "results": []
        }
        
        print(f"\n🖼️  Generating {stats['total']} images...\n")
        
        for concept_id, content in tqdm(self.database.items(), desc="Generating"):
            result = self._generate_single(concept_id, content)
            
            if result["success"]:
                stats["success"] += 1
            else:
                stats["failed"] += 1
            
            stats["results"].append(result)
        
        self.cleanup()
        self._print_summary(stats)
        
        return stats
    
    def _generate_single(self, concept_id: str, content: dict) -> dict:
        """Generate a single image."""
        result = {
            "concept_id": concept_id,
            "success": False,
            "output_path": None,
            "prompt_used": None,
            "error": None
        }
        
        try:
            # Get SDXL prompt from database
            info = content.get("info", {})
            sdxl_prompt = info.get("sdxl_prompt", "")
            
            if not sdxl_prompt:
                result["error"] = "No SDXL prompt in database entry"
                return result
            
            result["prompt_used"] = sdxl_prompt[:100] + "..."  # Truncate for logging
            
            # Generate (text-to-image only!)
            generated = self.pipe(
                prompt=sdxl_prompt,
                negative_prompt=BaselineConfig.NEGATIVE_PROMPT,
                num_inference_steps=BaselineConfig.NUM_INFERENCE_STEPS,
                guidance_scale=BaselineConfig.GUIDANCE_SCALE,
                height=BaselineConfig.OUTPUT_IMAGE_SIZE,
                width=BaselineConfig.OUTPUT_IMAGE_SIZE,
                generator=torch.Generator(device=BaselineConfig.DEVICE).manual_seed(BaselineConfig.SEED)
            ).images[0]
            
            # Save
            safe_id = concept_id.replace("<", "").replace(">", "")
            output_path = os.path.join(self.output_dir, f"{safe_id}_promptonly.png")
            generated.save(output_path)
            
            result["success"] = True
            result["output_path"] = output_path
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _print_summary(self, stats: dict):
        """Print generation summary."""
        print(f"\n{'='*70}")
        print("📊 GENERATION COMPLETE")
        print(f"{'='*70}")
        print(f"   ✅ Success: {stats['success']}/{stats['total']}")
        print(f"   ❌ Failed:  {stats['failed']}/{stats['total']}")
        print(f"   📁 Output:  {self.output_dir}")
        
        # Print failures if any
        failures = [r for r in stats["results"] if not r["success"]]
        if failures:
            print(f"\n   ⚠️  Failures:")
            for f in failures:
                print(f"      - {f['concept_id']}: {f['error']}")
        
        print(f"{'='*70}\n")
    
    def cleanup(self):
        """Release GPU memory."""
        print("\n🧹 Cleaning up GPU memory...")
        if self.pipe:
            del self.pipe
            self.pipe = None
        cleanup_gpu()


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Baseline 2: SDXL Prompt Only Generation (No IP-Adapter)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python baseline/sdxl_prompt_only.py --category bag --num 5
    python baseline/sdxl_prompt_only.py --category bottle --num 10
    python baseline/sdxl_prompt_only.py --category cup
        """
    )
    parser.add_argument(
        "--category", "-c",
        type=str,
        default=BaselineConfig.CATEGORY,
        choices=BaselineConfig.get_available_categories(),
        help=f"Category to test (default: {BaselineConfig.CATEGORY})"
    )
    parser.add_argument(
        "--num", "-n",
        type=int,
        default=BaselineConfig.NUM_CONCEPTS,
        help=f"Number of concepts to test (default: {BaselineConfig.NUM_CONCEPTS})"
    )
    
    args = parser.parse_args()
    
    generator = SDXLPromptOnlyGenerator(
        category=args.category,
        num_concepts=args.num
    )
    generator.generate_all()


if __name__ == "__main__":
    main()
