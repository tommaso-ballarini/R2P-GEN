# test/instantstyle_scaling.py
"""
TEST 3: InstantStyle Layer-Wise Scaling Experiments

Test different layer-wise weight configurations for IP-Adapter.
Compare multiple configurations to find the optimal settings.

Usage:
    # Test a single configuration
    python test/instantstyle_scaling.py --category bag --num 5 --config v2_high_identity
    
    # Test all configurations
    python test/instantstyle_scaling.py --category bag --num 5 --all-configs
    
    # List available configurations
    python test/instantstyle_scaling.py --list-configs
"""

import os
import sys
import json
import torch
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from test.config_test import TestConfig, InstantStyleConfigs
from test.utils_test import (
    cleanup_gpu, ensure_output_dir, load_database,
    load_reference_image, save_experiment_metadata,
    ExperimentLogger, create_comparison_grid, format_time
)


class InstantStyleScalingTester:
    """
    Test different layer-wise scaling configurations.
    
    For each configuration:
    1. Apply the layer-wise weights to IP-Adapter
    2. Generate images for selected concepts
    3. Save results in config-specific folders
    4. Generate comparison grids
    """
    
    def __init__(
        self,
        category: str = None,
        num_concepts: int = None,
        config_names: List[str] = None
    ):
        """
        Args:
            category: Category to test (default: from config)
            num_concepts: Number of concepts to test (default: from config)
            config_names: List of config names to test. If None, tests all.
        """
        self.category = category or TestConfig.CATEGORY
        self.num_concepts = num_concepts or TestConfig.NUM_CONCEPTS
        
        # Validate category
        if not TestConfig.validate_category(self.category):
            raise ValueError(
                f"Invalid category: {self.category}. "
                f"Available: {TestConfig.get_available_categories()}"
            )
        
        # Get configurations to test
        if config_names is None:
            self.config_names = InstantStyleConfigs.list_configs()
        else:
            # Validate config names
            available = InstantStyleConfigs.list_configs()
            for name in config_names:
                if name not in available:
                    raise ValueError(f"Unknown config: {name}. Available: {available}")
            self.config_names = config_names
        
        # Setup base output directory
        self.output_base = os.path.join(
            TestConfig.OUTPUT_BASE,
            "instantstyle_scaling"
        )
        ensure_output_dir(self.output_base)
        
        # Initialize components
        self.pipe = None
        self.database = None
        self.all_results = {}
        
        self._print_header()
    
    def _print_header(self):
        """Print experiment header."""
        print(f"\n{'='*70}")
        print("🎨 TEST: InstantStyle Layer-Wise Scaling")
        print(f"{'='*70}")
        print(f"   Category:       {self.category}")
        print(f"   Num Concepts:   {self.num_concepts}")
        print(f"   Configurations: {len(self.config_names)}")
        print(f"   Output Base:    {self.output_base}")
        print(f"   Device:         {TestConfig.DEVICE}")
        print(f"\n   Configs to test:")
        for name in self.config_names:
            cfg = InstantStyleConfigs.get_config(name)
            print(f"      - {name}: {cfg['description']}")
        print(f"{'='*70}\n")
    
    def load_database(self):
        """Load database and filter by category."""
        print(f"📂 Loading database from {TestConfig.DATABASE_PATH}...")
        
        self.database = load_database(
            TestConfig.DATABASE_PATH,
            category=self.category,
            num_concepts=self.num_concepts
        )
        
        print(f"   ✅ Loaded {len(self.database)} concepts from '{self.category}'")
        print(f"   📝 Concepts: {list(self.database.keys())}")
    
    def initialize_pipeline(self):
        """Initialize SDXL + IP-Adapter pipeline."""
        print("\n🔌 Loading SDXL + IP-Adapter pipeline...")
        
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            TestConfig.SDXL_MODEL,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        ).to(TestConfig.DEVICE)
        
        # Load IP-Adapter
        print("   📥 Loading IP-Adapter weights...")
        self.pipe.load_ip_adapter(
            TestConfig.IP_ADAPTER_REPO,
            subfolder=TestConfig.IP_ADAPTER_SUBFOLDER,
            weight_name=TestConfig.IP_ADAPTER_WEIGHT_NAME
        )
        
        # Memory optimizations
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_slicing()
        
        print("   ✅ Pipeline ready")
    
    def _apply_layer_weights(self, weights: Dict[str, Any]):
        """
        Apply layer-wise weights to IP-Adapter.
        
        Args:
            weights: Layer-wise weights dictionary
        """
        self.pipe.set_ip_adapter_scale(weights)
    
    def run_all_configs(self) -> Dict[str, Any]:
        """
        Run tests for all configurations.
        
        Returns:
            Dictionary with results for each configuration
        """
        self.load_database()
        self.initialize_pipeline()
        
        total_start = datetime.now()
        
        for i, config_name in enumerate(self.config_names):
            print(f"\n{'='*70}")
            print(f"📊 Testing Configuration {i+1}/{len(self.config_names)}: {config_name}")
            print(f"{'='*70}")
            
            config = InstantStyleConfigs.get_config(config_name)
            print(f"   Description: {config['description']}")
            print(f"   Motivation: {config['motivation'][:60]}...")
            
            # Run this configuration
            results = self._run_single_config(config_name, config)
            self.all_results[config_name] = results
        
        # Generate comparison
        self._generate_comparison()
        
        # Summary
        total_time = (datetime.now() - total_start).total_seconds()
        self._print_final_summary(total_time)
        
        self.cleanup()
        
        return self.all_results
    
    def run_single_config(self, config_name: str) -> Dict[str, Any]:
        """
        Run test for a single configuration.
        
        Args:
            config_name: Name of configuration to test
            
        Returns:
            Results dictionary
        """
        self.load_database()
        self.initialize_pipeline()
        
        config = InstantStyleConfigs.get_config(config_name)
        results = self._run_single_config(config_name, config)
        
        self.cleanup()
        
        return results
    
    def _run_single_config(
        self,
        config_name: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run generation for a single configuration."""
        # Setup output directory for this config
        output_dir = os.path.join(
            self.output_base,
            config_name,
            self.category
        )
        ensure_output_dir(output_dir)
        
        # Apply weights
        weights = config["weights"]
        self._apply_layer_weights(weights)
        
        print(f"\n   Applied weights: mid={weights['mid']}, "
              f"up.b1.max={max(weights['up']['block_1'])}")
        
        stats = {
            "config_name": config_name,
            "description": config["description"],
            "total": len(self.database),
            "success": 0,
            "failed": 0,
            "results": [],
            "output_dir": output_dir
        }
        
        for concept_id, content in tqdm(self.database.items(), 
                                        desc=f"   Generating ({config_name})"):
            result = self._generate_single(concept_id, content, output_dir)
            
            if result["success"]:
                stats["success"] += 1
            else:
                stats["failed"] += 1
            
            stats["results"].append(result)
        
        # Save config metadata
        save_experiment_metadata(
            output_dir,
            f"instantstyle_{config_name}",
            {
                "config_name": config_name,
                "description": config["description"],
                "motivation": config["motivation"],
                "weights": weights,
                "category": self.category,
                "num_concepts": self.num_concepts
            },
            {
                "success": stats["success"],
                "failed": stats["failed"]
            }
        )
        
        print(f"\n   ✅ {config_name}: {stats['success']}/{stats['total']} successful")
        
        return stats
    
    def _generate_single(
        self,
        concept_id: str,
        content: dict,
        output_dir: str
    ) -> dict:
        """Generate a single image."""
        result = {
            "concept_id": concept_id,
            "success": False,
            "output_path": None,
            "error": None
        }
        
        try:
            # Get reference image
            images = content.get("image", [])
            if not images:
                result["error"] = "No images"
                return result
            
            ref_image_path = images[0]
            
            if not os.path.exists(ref_image_path):
                result["error"] = "Image not found"
                return result
            
            # Load reference
            ref_img = load_reference_image(
                ref_image_path,
                size=TestConfig.REFERENCE_IMAGE_SIZE
            )
            
            # Get SDXL prompt
            info = content.get("info", {})
            sdxl_prompt = info.get("sdxl_prompt", "")
            
            if not sdxl_prompt:
                # Fallback to general description
                sdxl_prompt = info.get("general", f"a {content.get('category', 'object')}")
            
            # Generate
            generated = self.pipe(
                prompt=sdxl_prompt,
                negative_prompt=TestConfig.NEGATIVE_PROMPT,
                ip_adapter_image=ref_img,
                num_inference_steps=TestConfig.NUM_INFERENCE_STEPS,
                guidance_scale=TestConfig.GUIDANCE_SCALE,
                height=TestConfig.OUTPUT_IMAGE_SIZE,
                width=TestConfig.OUTPUT_IMAGE_SIZE,
                generator=torch.Generator(device=TestConfig.DEVICE).manual_seed(TestConfig.SEED)
            ).images[0]
            
            # Save
            safe_id = concept_id.replace("<", "").replace(">", "")
            output_path = os.path.join(output_dir, f"{safe_id}.png")
            generated.save(output_path)
            
            result["success"] = True
            result["output_path"] = output_path
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _generate_comparison(self):
        """Generate comparison images across configurations."""
        print("\n📊 Generating comparison grids...")
        
        comparison_dir = os.path.join(self.output_base, "comparison", self.category)
        ensure_output_dir(comparison_dir)
        
        # For each concept, create a comparison grid
        for concept_id in self.database.keys():
            safe_id = concept_id.replace("<", "").replace(">", "")
            
            images = []
            labels = []
            
            # Collect images from each config
            for config_name in self.config_names:
                img_path = os.path.join(
                    self.output_base,
                    config_name,
                    self.category,
                    f"{safe_id}.png"
                )
                
                if os.path.exists(img_path):
                    images.append(Image.open(img_path))
                    labels.append(config_name[:15])  # Truncate label
            
            if len(images) > 1:
                # Create grid
                grid = create_comparison_grid(
                    images,
                    labels,
                    cols=min(4, len(images)),
                    cell_size=256
                )
                
                grid_path = os.path.join(comparison_dir, f"{safe_id}_comparison.png")
                grid.save(grid_path)
        
        print(f"   ✅ Saved comparison grids to {comparison_dir}")
    
    def _print_final_summary(self, total_time: float):
        """Print final summary of all configurations."""
        print(f"\n{'='*70}")
        print("📊 FINAL SUMMARY - ALL CONFIGURATIONS")
        print(f"{'='*70}")
        print(f"\n   Total time: {format_time(total_time)}")
        print(f"   Category: {self.category}")
        print(f"   Concepts tested: {self.num_concepts}")
        print(f"\n   Results by configuration:")
        print(f"   {'-'*50}")
        print(f"   {'Config':<25} {'Success':>10} {'Failed':>10}")
        print(f"   {'-'*50}")
        
        for config_name, stats in self.all_results.items():
            success = stats["success"]
            failed = stats["failed"]
            print(f"   {config_name:<25} {success:>10} {failed:>10}")
        
        print(f"   {'-'*50}")
        print(f"\n   Output: {self.output_base}")
        print(f"   Comparison grids: {os.path.join(self.output_base, 'comparison', self.category)}")
        print(f"{'='*70}\n")
    
    def cleanup(self):
        """Release GPU memory."""
        print("\n🧹 Cleaning up...")
        if self.pipe:
            del self.pipe
            self.pipe = None
        cleanup_gpu()


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test InstantStyle Layer-Wise Scaling Configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test a single configuration
    python test/instantstyle_scaling.py --category bag --num 5 --config v2_high_identity
    
    # Test multiple configurations
    python test/instantstyle_scaling.py --category bag --num 5 --config v1_current_baseline v2_high_identity
    
    # Test all configurations
    python test/instantstyle_scaling.py --category bag --num 5 --all-configs
    
    # List available configurations
    python test/instantstyle_scaling.py --list-configs
    
    # Compare configurations
    python test/instantstyle_scaling.py --compare-configs
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
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        default=None,
        help="Configuration name(s) to test"
    )
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Test all available configurations"
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List all available configurations and exit"
    )
    parser.add_argument(
        "--compare-configs",
        action="store_true",
        help="Print configuration comparison table and exit"
    )
    
    args = parser.parse_args()
    
    # Handle info-only commands
    if args.list_configs:
        InstantStyleConfigs.print_configs()
        return
    
    if args.compare_configs:
        InstantStyleConfigs.compare_configs()
        return
    
    # Determine which configs to test
    if args.all_configs:
        config_names = None  # Will test all
    elif args.config:
        config_names = args.config
    else:
        # Default: test the current baseline
        config_names = ["v1_current_baseline"]
    
    # Run tests
    tester = InstantStyleScalingTester(
        category=args.category,
        num_concepts=args.num,
        config_names=config_names
    )
    
    if len(tester.config_names) == 1:
        tester.run_single_config(tester.config_names[0])
    else:
        tester.run_all_configs()


if __name__ == "__main__":
    main()
