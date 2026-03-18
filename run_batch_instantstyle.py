# run_batch_instantstyle.py
"""
Batch runner for InstantStyle layer-wise scaling experiments.

Tests all 8 InstantStyle configurations across all categories:
- v1_current_baseline: Current R2P-GEN weights
- v2_instantstyle_original: Original InstantStyle
- v3_uniform_light: Light uniform weights
- v4_uniform_medium: Medium uniform weights
- v5_uniform_heavy: Heavy uniform weights
- v6_style_focus: Focus on style blocks
- v7_content_focus: Focus on content blocks
- v8_shape_focus: Focus on shape preservation

Usage:
    # Run all configs on all categories
    python run_batch_instantstyle.py
    
    # Run specific configs
    python run_batch_instantstyle.py --configs v1_current_baseline v3_uniform_light
    
    # Run on specific categories
    python run_batch_instantstyle.py --categories bag bottle cup
    
    # Generate comparison grids
    python run_batch_instantstyle.py --compare
    
    # Parallel execution
    python run_batch_instantstyle.py --parallel 2
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# ============================================
# CONFIGURATION - MODIFY HERE
# ============================================

# Database path (change manually if needed)
DATABASE_PATH = "database/database_perva_train_1_clip.json"

# Number of concepts per category
NUM_CONCEPTS = 5

# Random seed
SEED = 42

# All available categories
ALL_CATEGORIES = [
    "bag", "book", "bottle", "bowl", "clothe", "cup", "decoration",
    "headphone", "pillow", "plant", "plate", "remote", "retail",
    "telephone", "tie", "towel", "toy", "tro_bag", "tumbler", "umbrella", "veg"
]

# All InstantStyle configurations
ALL_CONFIGS = [
    "v1_current_baseline",
    "v2_instantstyle_original",
    "v3_uniform_light",
    "v4_uniform_medium",
    "v5_uniform_heavy",
    "v6_style_focus",
    "v7_content_focus",
    "v8_shape_focus"
]

# Script path
INSTANTSTYLE_SCRIPT = "test/instantstyle_scaling.py"

# ============================================
# END CONFIGURATION
# ============================================

PROJECT_ROOT = Path(__file__).parent


class InstantStyleBatchRunner:
    """Batch runner for InstantStyle scaling experiments."""
    
    def __init__(
        self,
        categories: List[str] = None,
        configs: List[str] = None,
        num_concepts: int = NUM_CONCEPTS,
        parallel: int = 1,
        generate_comparison: bool = False
    ):
        self.categories = categories or ALL_CATEGORIES
        self.configs = configs or ALL_CONFIGS
        self.num_concepts = num_concepts
        self.parallel = parallel
        self.generate_comparison = generate_comparison
        
        # Validate configs
        invalid_configs = [c for c in self.configs if c not in ALL_CONFIGS]
        if invalid_configs:
            raise ValueError(f"Invalid configs: {invalid_configs}. Valid: {ALL_CONFIGS}")
        
        # Setup paths
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = PROJECT_ROOT / "output" / "test" / "instantstyle_scaling" / f"run_{self.timestamp}"
        self.log_dir = PROJECT_ROOT / "logs"
        
        # Results tracking
        self.results = {
            "start_time": None,
            "end_time": None,
            "total_runtime": None,
            "config": {
                "categories": self.categories,
                "instantstyle_configs": self.configs,
                "num_concepts": self.num_concepts,
                "database": DATABASE_PATH,
                "parallel": self.parallel,
                "seed": SEED,
                "generate_comparison": generate_comparison
            },
            "summary": {
                "total_tasks": 0,
                "successful": 0,
                "failed": 0
            },
            "tasks": []
        }
    
    def setup(self):
        """Create necessary directories."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file
        self.log_file = self.log_dir / f"batch_instantstyle_{self.timestamp}.log"
    
    def log(self, message: str, also_print: bool = True):
        """Log message to file and optionally print."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_line + "\n")
        
        if also_print:
            print(log_line)
    
    def preflight_check(self) -> bool:
        """Run pre-flight checks and show summary."""
        print("\n" + "=" * 70)
        print("🔍 PRE-FLIGHT CHECKS")
        print("=" * 70)
        
        errors = []
        
        # Check database
        db_path = PROJECT_ROOT / DATABASE_PATH
        if db_path.exists():
            print(f"   ✅ Database found: {DATABASE_PATH}")
            
            # Load and check categories
            with open(db_path, 'r', encoding='utf-8') as f:
                db = json.load(f)
            
            concept_dict = db.get("concept_dict", {})
            available_cats = set(v.get("category") for v in concept_dict.values())
            
            for cat in self.categories:
                if cat not in available_cats:
                    errors.append(f"Category '{cat}' not found in database")
                else:
                    count = sum(1 for v in concept_dict.values() if v.get("category") == cat)
                    print(f"   ✅ Category '{cat}': {count} concepts available")
        else:
            errors.append(f"Database not found: {DATABASE_PATH}")
        
        # Check script exists
        script_path = PROJECT_ROOT / INSTANTSTYLE_SCRIPT
        if script_path.exists():
            print(f"   ✅ Script found: {INSTANTSTYLE_SCRIPT}")
        else:
            errors.append(f"Script not found: {INSTANTSTYLE_SCRIPT}")
        
        # Check GPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"   ✅ GPU available: {gpu_name} ({gpu_mem:.1f} GB)")
            else:
                errors.append("No GPU available")
        except ImportError:
            errors.append("PyTorch not installed")
        
        # Check IP-Adapter (required for InstantStyle)
        try:
            from diffusers import StableDiffusionXLPipeline
            print(f"   ✅ Diffusers installed")
        except ImportError:
            errors.append("Diffusers not installed")
        
        # Print errors
        if errors:
            print(f"\n   ❌ ERRORS:")
            for e in errors:
                print(f"      - {e}")
            return False
        
        # Calculate estimates
        total_tasks = len(self.categories) * len(self.configs)
        total_images = total_tasks * self.num_concepts
        est_time_per_image = 30  # seconds
        est_total_time = total_images * est_time_per_image / 60  # minutes
        
        if self.parallel > 1:
            est_total_time /= self.parallel
        
        # Add time for comparisons
        if self.generate_comparison:
            est_total_time += len(self.categories) * 0.5  # ~30s per comparison grid
        
        print("\n" + "=" * 70)
        print("📊 BATCH SUMMARY")
        print("=" * 70)
        print(f"   Categories:           {len(self.categories)}")
        print(f"   Concepts per cat:     {self.num_concepts}")
        print(f"   InstantStyle configs: {len(self.configs)}")
        for cfg in self.configs:
            print(f"      - {cfg}")
        print(f"   Total tasks:          {total_tasks}")
        print(f"   Total images:         {total_images}")
        print(f"   Parallel workers:     {self.parallel}")
        print(f"   Generate comparisons: {'Yes' if self.generate_comparison else 'No'}")
        print(f"   Estimated time:       ~{est_total_time:.0f} minutes")
        print(f"   Output directory:     {self.run_dir}")
        print("=" * 70)
        
        self.results["summary"]["total_tasks"] = total_tasks
        
        # Ask for confirmation
        response = input("\n🚀 Continue? [y/N]: ").strip().lower()
        return response in ['y', 'yes']
    
    def run_single_task(self, config_name: str, category: str) -> Dict[str, Any]:
        """Run InstantStyle script for a single config and category."""
        task_result = {
            "config": config_name,
            "category": category,
            "success": False,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "runtime_seconds": None,
            "error": None
        }
        
        start_time = time.time()
        
        try:
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / INSTANTSTYLE_SCRIPT),
                "--category", category,
                "--config", config_name,
                "--num", str(self.num_concepts)
            ]
            
            self.log(f"▶️  Running: {config_name} --category {category}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT)
            )
            
            end_time = time.time()
            runtime = end_time - start_time
            
            task_result["end_time"] = datetime.now().isoformat()
            task_result["runtime_seconds"] = round(runtime, 2)
            
            if result.returncode == 0:
                task_result["success"] = True
                self.log(f"   ✅ {config_name} [{category}] completed in {runtime:.1f}s")
            else:
                task_result["error"] = result.stderr[:500] if result.stderr else "Unknown error"
                self.log(f"   ❌ {config_name} [{category}] FAILED: {task_result['error'][:100]}")
            
        except Exception as e:
            task_result["error"] = str(e)
            task_result["end_time"] = datetime.now().isoformat()
            self.log(f"   ❌ {config_name} [{category}] EXCEPTION: {e}")
        
        return task_result
    
    def run_all_configs_for_category(self, category: str):
        """Run all configs for a single category (allows comparison generation)."""
        self.log(f"\n{'='*50}")
        self.log(f"📂 Category: {category}")
        self.log(f"{'='*50}")
        
        for config_name in self.configs:
            task_result = self.run_single_task(config_name, category)
            self.results["tasks"].append(task_result)
            
            if task_result["success"]:
                self.results["summary"]["successful"] += 1
            else:
                self.results["summary"]["failed"] += 1
        
        # Generate comparison grid if requested
        if self.generate_comparison:
            self.generate_comparison_grid(category)
    
    def generate_comparison_grid(self, category: str):
        """Generate comparison grid for a category."""
        self.log(f"   📊 Generating comparison grid for {category}...")
        
        try:
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / INSTANTSTYLE_SCRIPT),
                "--compare",
                "--category", category
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT)
            )
            
            if result.returncode == 0:
                self.log(f"   ✅ Comparison grid created for {category}")
            else:
                self.log(f"   ⚠️  Comparison grid failed for {category}")
                
        except Exception as e:
            self.log(f"   ⚠️  Comparison grid error for {category}: {e}")
    
    def run_sequential(self):
        """Run all tasks sequentially."""
        for category in self.categories:
            self.run_all_configs_for_category(category)
    
    def run_parallel(self):
        """Run tasks in parallel (configs in parallel, categories sequential)."""
        # For comparison generation to work, we process all configs per category
        # but can parallelize configs within each category
        
        for category in self.categories:
            self.log(f"\n{'='*50}")
            self.log(f"📂 Category: {category}")
            self.log(f"{'='*50}")
            
            with ThreadPoolExecutor(max_workers=self.parallel) as executor:
                futures = {
                    executor.submit(self.run_single_task, cfg, category): cfg
                    for cfg in self.configs
                }
                
                for future in as_completed(futures):
                    task_result = future.result()
                    self.results["tasks"].append(task_result)
                    
                    if task_result["success"]:
                        self.results["summary"]["successful"] += 1
                    else:
                        self.results["summary"]["failed"] += 1
            
            # Generate comparison after all configs complete
            if self.generate_comparison:
                self.generate_comparison_grid(category)
    
    def run(self):
        """Main run method."""
        self.setup()
        
        if not self.preflight_check():
            print("\n❌ Batch cancelled by user or failed pre-flight checks.")
            return False
        
        self.results["start_time"] = datetime.now().isoformat()
        start_time = time.time()
        
        self.log(f"\n🚀 Starting InstantStyle batch run at {self.results['start_time']}")
        self.log(f"   Log file: {self.log_file}")
        
        try:
            if self.parallel > 1:
                self.run_parallel()
            else:
                self.run_sequential()
        except KeyboardInterrupt:
            self.log("\n⚠️  Batch interrupted by user!")
        
        end_time = time.time()
        self.results["end_time"] = datetime.now().isoformat()
        self.results["total_runtime"] = f"{(end_time - start_time) / 60:.1f} minutes"
        
        # Save summary
        self.save_summary()
        
        # Print final summary
        self.print_summary()
        
        return True
    
    def save_summary(self):
        """Save results summary to JSON."""
        summary_path = self.run_dir / "batch_summary.json"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        self.log(f"\n💾 Summary saved to: {summary_path}")
    
    def print_summary(self):
        """Print final summary."""
        print("\n" + "=" * 70)
        print("📊 INSTANTSTYLE BATCH COMPLETE")
        print("=" * 70)
        print(f"   Total runtime:    {self.results['total_runtime']}")
        print(f"   Total tasks:      {self.results['summary']['total_tasks']}")
        print(f"   Successful:       {self.results['summary']['successful']}")
        print(f"   Failed:           {self.results['summary']['failed']}")
        print(f"   Output:           {self.run_dir}")
        print(f"   Log:              {self.log_file}")
        
        # Results by config
        print(f"\n   Results by config:")
        for cfg in self.configs:
            cfg_tasks = [t for t in self.results["tasks"] if t["config"] == cfg]
            success = sum(1 for t in cfg_tasks if t["success"])
            total = len(cfg_tasks)
            print(f"      - {cfg}: {success}/{total}")
        
        # Print failures if any
        failures = [t for t in self.results["tasks"] if not t["success"]]
        if failures:
            print(f"\n   ⚠️  Failed tasks:")
            for f in failures[:10]:
                print(f"      - {f['config']} [{f['category']}]: {f['error'][:50]}...")
            if len(failures) > 10:
                print(f"      ... and {len(failures) - 10} more")
        
        print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Batch runner for InstantStyle layer-wise scaling experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all 8 configs on all categories
    python run_batch_instantstyle.py
    
    # Run specific configs only
    python run_batch_instantstyle.py --configs v1_current_baseline v3_uniform_light v5_uniform_heavy
    
    # Run on specific categories
    python run_batch_instantstyle.py --categories bag bottle cup plant
    
    # Generate comparison grids after each category
    python run_batch_instantstyle.py --compare
    
    # Run with 2 parallel workers (per category)
    python run_batch_instantstyle.py --parallel 2
    
Available configs:
    v1_current_baseline     - Current R2P-GEN weights
    v2_instantstyle_original - Original InstantStyle paper
    v3_uniform_light        - Light uniform weights (0.3)
    v4_uniform_medium       - Medium uniform weights (0.5)
    v5_uniform_heavy        - Heavy uniform weights (0.8)
    v6_style_focus          - Focus on style blocks
    v7_content_focus        - Focus on content blocks
    v8_shape_focus          - Focus on shape preservation
        """
    )
    
    parser.add_argument(
        "--categories", "-c",
        nargs="+",
        default=None,
        choices=ALL_CATEGORIES,
        help="Categories to run (default: all)"
    )
    
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        choices=ALL_CONFIGS,
        help="InstantStyle configs to test (default: all 8)"
    )
    
    parser.add_argument(
        "--parallel", "-p",
        type=int,
        default=1,
        help="Number of parallel workers per category (default: 1 = sequential)"
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate comparison grids after each category"
    )
    
    args = parser.parse_args()
    
    runner = InstantStyleBatchRunner(
        categories=args.categories,
        configs=args.configs,
        parallel=args.parallel,
        generate_comparison=args.compare
    )
    
    runner.run()


if __name__ == "__main__":
    main()
