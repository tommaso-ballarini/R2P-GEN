# test/anydoor_with_fingerprints.py
"""
TEST 2: AnyDoor with Fingerprints

Generate images using AnyDoor combined with extracted fingerprints
from the R2P-GEN database.

This tests whether fingerprints can improve AnyDoor's output quality.

Usage:
    python test/anydoor_with_fingerprints.py --category bag --num 5
    python test/anydoor_with_fingerprints.py --category bottle --num 10
"""

import os
import sys
import json
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datetime import datetime

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from test.config_test import TestConfig
from test.utils_test import (
    cleanup_gpu, ensure_output_dir, load_database,
    load_reference_image, save_experiment_metadata, 
    ExperimentLogger, get_concept_info
)
from test.anydoor_wrapper import AnyDoorWrapper


class AnyDoorFingerprintsGenerator:
    """
    AnyDoor generator WITH fingerprints.
    
    Uses:
    - Reference image
    - Extracted fingerprints to build detailed prompt
    - SDXL-style prompt formatting
    
    Tests whether detailed descriptions improve AnyDoor output.
    """
    
    def __init__(self, category: str = None, num_concepts: int = None):
        """
        Args:
            category: Category to test (default: from config)
            num_concepts: Number of concepts to test (default: from config)
        """
        self.category = category or TestConfig.CATEGORY
        self.num_concepts = num_concepts or TestConfig.NUM_CONCEPTS
        
        # Validate category
        if not TestConfig.validate_category(self.category):
            raise ValueError(
                f"Invalid category: {self.category}. "
                f"Available: {TestConfig.get_available_categories()}"
            )
        
        # Setup output directory
        self.output_dir = os.path.join(
            TestConfig.OUTPUT_BASE,
            "anydoor_fingerprints",
            self.category
        )
        ensure_output_dir(self.output_dir)
        
        # Initialize logger
        self.logger = ExperimentLogger(
            self.output_dir,
            f"AnyDoor + Fingerprints - {self.category}"
        )
        
        # Initialize components
        self.anydoor = None
        self.database = None
        
        self._print_header()
    
    def _print_header(self):
        """Print experiment header."""
        print(f"\n{'='*70}")
        print("🚪 TEST: AnyDoor + Fingerprints")
        print(f"{'='*70}")
        print(f"   Category:     {self.category}")
        print(f"   Num Concepts: {self.num_concepts}")
        print(f"   Output:       {self.output_dir}")
        print(f"   Mode:         Reference image + fingerprint-based prompt")
        print(f"   Device:       {TestConfig.DEVICE}")
        print(f"{'='*70}\n")
    
    def load_database(self):
        """Load database and filter by category."""
        self.logger.info(f"Loading database from {TestConfig.DATABASE_PATH}")
        
        self.database = load_database(
            TestConfig.DATABASE_PATH,
            category=self.category,
            num_concepts=self.num_concepts
        )
        
        self.logger.success(f"Loaded {len(self.database)} concepts")
        self.logger.info(f"Concepts: {list(self.database.keys())}")
    
    def initialize_model(self):
        """Initialize AnyDoor model."""
        self.logger.info("Initializing AnyDoor model...")
        
        self.anydoor = AnyDoorWrapper(device=TestConfig.DEVICE)
        
        if not self.anydoor.setup_complete:
            self.logger.error("AnyDoor setup incomplete!")
            self.logger.error("Run: .\\test\\setup_anydoor.ps1")
            raise RuntimeError("AnyDoor not properly set up")
        
        # Load model
        self.anydoor.load_model()
        self.logger.success("AnyDoor model loaded")
    
    def _build_fingerprint_prompt(self, content: dict) -> str:
        """
        Build a detailed prompt from fingerprints.
        
        Args:
            content: Concept content from database
            
        Returns:
            Detailed prompt string
        """
        info = get_concept_info(content)
        
        # Try to use SDXL prompt if available
        if info["sdxl_prompt"]:
            return info["sdxl_prompt"]
        
        # Otherwise, build from fingerprints
        parts = []
        
        # Add general description
        if info["general"]:
            parts.append(info["general"])
        
        # Add fingerprint details
        fingerprints = info["fingerprints"]
        
        # Priority order for fingerprints
        priority_keys = ["color", "material", "shape", "texture", "pattern", "style"]
        
        for key in priority_keys:
            if key in fingerprints and fingerprints[key]:
                value = fingerprints[key]
                if isinstance(value, list):
                    value = ", ".join(value)
                parts.append(f"{key}: {value}")
        
        # Add remaining fingerprints
        for key, value in fingerprints.items():
            if key not in priority_keys and value:
                if isinstance(value, list):
                    value = ", ".join(value)
                parts.append(f"{key}: {value}")
        
        # Combine into prompt
        prompt = ", ".join(parts)
        
        # Add quality modifiers
        prompt += ", high quality, professional photo, detailed"
        
        return prompt
    
    def generate_all(self) -> dict:
        """
        Generate images for all concepts.
        
        Returns:
            dict: Statistics with success/failed counts
        """
        self.load_database()
        self.initialize_model()
        
        stats = {
            "total": len(self.database),
            "success": 0,
            "failed": 0,
            "results": [],
            "config": {
                "category": self.category,
                "num_concepts": self.num_concepts,
                "model": "AnyDoor",
                "mode": "with_fingerprints"
            }
        }
        
        self.logger.info(f"Generating {stats['total']} images...")
        
        for concept_id, content in tqdm(self.database.items(), desc="Generating"):
            result = self._generate_single(concept_id, content)
            
            if result["success"]:
                stats["success"] += 1
                self.logger.success(f"{concept_id}: Generated")
            else:
                stats["failed"] += 1
                self.logger.error(f"{concept_id}: {result['error']}")
            
            stats["results"].append(result)
        
        # Save metadata
        save_experiment_metadata(
            self.output_dir,
            "anydoor_fingerprints",
            stats["config"],
            {"success": stats["success"], "failed": stats["failed"]}
        )
        
        self.cleanup()
        self._print_summary(stats)
        self.logger.finish()
        
        return stats
    
    def _generate_single(self, concept_id: str, content: dict) -> dict:
        """Generate a single image with AnyDoor + fingerprints."""
        result = {
            "concept_id": concept_id,
            "success": False,
            "output_path": None,
            "prompt_used": None,
            "error": None
        }
        
        try:
            # Get reference image
            images = content.get("image", [])
            if not images:
                result["error"] = "No images in database entry"
                return result
            
            ref_image_path = images[0]
            
            if not os.path.exists(ref_image_path):
                result["error"] = f"Image not found: {ref_image_path}"
                return result
            
            # Load reference image
            ref_img = load_reference_image(
                ref_image_path,
                size=512
            )
            
            # Build fingerprint-based prompt
            prompt = self._build_fingerprint_prompt(content)
            result["prompt_used"] = prompt[:100] + "..."  # Truncate for logging
            
            # Generate with AnyDoor
            generated = self.anydoor.generate(
                reference_image=ref_img,
                prompt=prompt,
                num_inference_steps=TestConfig.NUM_INFERENCE_STEPS,
                guidance_scale=TestConfig.GUIDANCE_SCALE,
                seed=TestConfig.SEED
            )
            
            # Save output
            safe_id = concept_id.replace("<", "").replace(">", "")
            output_path = os.path.join(self.output_dir, f"{safe_id}_anydoor_fp.png")
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
        
        # Print failures
        failures = [r for r in stats["results"] if not r["success"]]
        if failures:
            print(f"\n   ⚠️  Failures:")
            for f in failures[:5]:
                print(f"      - {f['concept_id']}: {f['error']}")
            if len(failures) > 5:
                print(f"      ... and {len(failures) - 5} more")
        
        print(f"{'='*70}\n")
    
    def cleanup(self):
        """Release resources."""
        self.logger.info("Cleaning up...")
        if self.anydoor:
            self.anydoor.cleanup()
        cleanup_gpu()


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test 2: AnyDoor with Fingerprints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python test/anydoor_with_fingerprints.py --category bag --num 5
    python test/anydoor_with_fingerprints.py --category bottle --num 10
    python test/anydoor_with_fingerprints.py --category cup
    
Note: 
    Run setup first: .\\test\\setup_anydoor.ps1
        """
    )
    parser.add_argument(
        "--category", "-c",
        type=str,
        default=TestConfig.CATEGORY,
        choices=TestConfig.get_available_categories(),
        help=f"Category to test (default: {TestConfig.CATEGORY})"
    )
    parser.add_argument(
        "--num", "-n",
        type=int,
        default=TestConfig.NUM_CONCEPTS,
        help=f"Number of concepts to test (default: {TestConfig.NUM_CONCEPTS})"
    )
    
    args = parser.parse_args()
    
    generator = AnyDoorFingerprintsGenerator(
        category=args.category,
        num_concepts=args.num
    )
    generator.generate_all()


if __name__ == "__main__":
    main()
