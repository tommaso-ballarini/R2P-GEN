# run_batch_anydoor.py
"""
Batch runner for AnyDoor experiments.

Runs both AnyDoor scripts:
- anydoor_vanilla.py (without fingerprints)
- anydoor_with_fingerprints.py (with fingerprints)

On all categories with specified number of concepts.

Usage:
    # Run on all categories
    python run_batch_anydoor.py
    
    # Run on specific categories
    python run_batch_anydoor.py --categories bag bottle cup
    
    # Run only vanilla version
    python run_batch_anydoor.py --vanilla-only
    
    # Run only fingerprints version
    python run_batch_anydoor.py --fingerprints-only
    
    # Parallel execution
    python run_batch_anydoor.py --parallel 2
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

# AnyDoor parameters (fixed)
CONTROLNET_SCALE = 0.5
USE_OPTIMIZED_PROMPT = True
SEED = 42

# All available categories
ALL_CATEGORIES = [
    "bag", "book", "bottle", "bowl", "clothe", "cup", "decoration",
    "headphone", "pillow", "plant", "plate", "remote", "retail",
    "telephone", "tie", "towel", "toy", "tro_bag", "tumbler", "umbrella", "veg"
]

# AnyDoor scripts
ANYDOOR_SCRIPTS = {
    "vanilla": "test/anydoor_vanilla.py",
    "fingerprints": "test/anydoor_with_fingerprints.py"
}

# ============================================
# END CONFIGURATION
# ============================================

PROJECT_ROOT = Path(__file__).parent


class AnyDoorBatchRunner:
    """Batch runner for AnyDoor experiments."""
    
    def __init__(
        self,
        categories: List[str] = None,
        num_concepts: int = NUM_CONCEPTS,
        parallel: int = 1,
        run_vanilla: bool = True,
        run_fingerprints: bool = True
    ):
        self.categories = categories or ALL_CATEGORIES
        self.num_concepts = num_concepts
        self.parallel = parallel
        self.run_vanilla = run_vanilla
        self.run_fingerprints = run_fingerprints
        
        # Determine which scripts to run
        self.scripts_to_run = []
        if run_vanilla:
            self.scripts_to_run.append(("vanilla", ANYDOOR_SCRIPTS["vanilla"]))
        if run_fingerprints:
            self.scripts_to_run.append(("fingerprints", ANYDOOR_SCRIPTS["fingerprints"]))
        
        # Setup paths
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = PROJECT_ROOT / "output" / "test" / f"anydoor_run_{self.timestamp}"
        self.log_dir = PROJECT_ROOT / "logs"
        
        # Results tracking
        self.results = {
            "start_time": None,
            "end_time": None,
            "total_runtime": None,
            "config": {
                "categories": self.categories,
                "num_concepts": self.num_concepts,
                "database": DATABASE_PATH,
                "parallel": self.parallel,
                "controlnet_scale": CONTROLNET_SCALE,
                "use_optimized_prompt": USE_OPTIMIZED_PROMPT,
                "seed": SEED,
                "run_vanilla": run_vanilla,
                "run_fingerprints": run_fingerprints
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
        self.log_file = self.log_dir / f"batch_anydoor_{self.timestamp}.log"
    
    def log(self, message: str, also_print: bool = True):
        """Log message to file and optionally print."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_line + "\n")
        
        if also_print:
            print(log_line)
    
    def check_anydoor_setup(self) -> List[str]:
        """Check if AnyDoor is properly set up."""
        errors = []
        
        # Check AnyDoor repo
        anydoor_path = PROJECT_ROOT / "test" / "external" / "AnyDoor"
        if not anydoor_path.exists():
            errors.append(f"AnyDoor repo not found at: {anydoor_path}")
            errors.append("Run: .\\test\\setup_anydoor.ps1")
        
        # Check checkpoint
        checkpoint_path = PROJECT_ROOT / "checkpoints" / "anydoor"
        if not checkpoint_path.exists():
            errors.append(f"AnyDoor checkpoints not found at: {checkpoint_path}")
        
        return errors
    
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
        
        # Check AnyDoor setup
        anydoor_errors = self.check_anydoor_setup()
        if anydoor_errors:
            for e in anydoor_errors:
                errors.append(e)
            print(f"   ⚠️  AnyDoor setup incomplete")
        else:
            print(f"   ✅ AnyDoor setup OK")
        
        # Check scripts exist
        for name, script in self.scripts_to_run:
            script_path = PROJECT_ROOT / script
            if script_path.exists():
                print(f"   ✅ Script found: {script}")
            else:
                errors.append(f"Script not found: {script}")
        
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
        
        # Print errors
        if errors:
            print(f"\n   ❌ ERRORS:")
            for e in errors:
                print(f"      - {e}")
            return False
        
        # Calculate estimates
        total_tasks = len(self.categories) * len(self.scripts_to_run)
        total_images = total_tasks * self.num_concepts
        est_time_per_image = 45  # seconds (AnyDoor is slower)
        est_total_time = total_images * est_time_per_image / 60  # minutes
        
        if self.parallel > 1:
            est_total_time /= self.parallel
        
        print("\n" + "=" * 70)
        print("📊 BATCH SUMMARY")
        print("=" * 70)
        print(f"   Categories:           {len(self.categories)}")
        print(f"   Concepts per cat:     {self.num_concepts}")
        print(f"   AnyDoor scripts:      {len(self.scripts_to_run)}")
        print(f"      - Vanilla:         {'Yes' if self.run_vanilla else 'No'}")
        print(f"      - Fingerprints:    {'Yes' if self.run_fingerprints else 'No'}")
        print(f"   Total tasks:          {total_tasks}")
        print(f"   Total images:         {total_images}")
        print(f"   Parallel workers:     {self.parallel}")
        print(f"   Estimated time:       ~{est_total_time:.0f} minutes")
        print(f"   Output directory:     {self.run_dir}")
        print(f"\n   AnyDoor Parameters:")
        print(f"      - ControlNet scale:     {CONTROLNET_SCALE}")
        print(f"      - Use optimized prompt: {USE_OPTIMIZED_PROMPT}")
        print(f"      - Seed:                 {SEED}")
        print("=" * 70)
        
        self.results["summary"]["total_tasks"] = total_tasks
        
        # Ask for confirmation
        response = input("\n🚀 Continue? [y/N]: ").strip().lower()
        return response in ['y', 'yes']
    
    def run_single_task(self, script_info: tuple, category: str) -> Dict[str, Any]:
        """Run a single AnyDoor script for a category."""
        script_name, script_path = script_info
        
        task_result = {
            "script": script_name,
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
                str(PROJECT_ROOT / script_path),
                "--category", category,
                "--num", str(self.num_concepts)
            ]
            
            self.log(f"▶️  Running: anydoor_{script_name} --category {category}")
            
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
                self.log(f"   ✅ anydoor_{script_name} [{category}] completed in {runtime:.1f}s")
            else:
                task_result["error"] = result.stderr[:500] if result.stderr else "Unknown error"
                self.log(f"   ❌ anydoor_{script_name} [{category}] FAILED: {task_result['error'][:100]}")
            
        except Exception as e:
            task_result["error"] = str(e)
            task_result["end_time"] = datetime.now().isoformat()
            self.log(f"   ❌ anydoor_{script_name} [{category}] EXCEPTION: {e}")
        
        return task_result
    
    def run_sequential(self):
        """Run all tasks sequentially."""
        for category in self.categories:
            self.log(f"\n{'='*50}")
            self.log(f"📂 Category: {category}")
            self.log(f"{'='*50}")
            
            for script_info in self.scripts_to_run:
                task_result = self.run_single_task(script_info, category)
                self.results["tasks"].append(task_result)
                
                if task_result["success"]:
                    self.results["summary"]["successful"] += 1
                else:
                    self.results["summary"]["failed"] += 1
    
    def run_parallel(self):
        """Run tasks in parallel."""
        tasks = [
            (script_info, category)
            for category in self.categories
            for script_info in self.scripts_to_run
        ]
        
        with ThreadPoolExecutor(max_workers=self.parallel) as executor:
            futures = {
                executor.submit(self.run_single_task, script_info, cat): (script_info, cat)
                for script_info, cat in tasks
            }
            
            for future in as_completed(futures):
                task_result = future.result()
                self.results["tasks"].append(task_result)
                
                if task_result["success"]:
                    self.results["summary"]["successful"] += 1
                else:
                    self.results["summary"]["failed"] += 1
    
    def run(self):
        """Main run method."""
        self.setup()
        
        if not self.preflight_check():
            print("\n❌ Batch cancelled by user or failed pre-flight checks.")
            return False
        
        self.results["start_time"] = datetime.now().isoformat()
        start_time = time.time()
        
        self.log(f"\n🚀 Starting AnyDoor batch run at {self.results['start_time']}")
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
        print("📊 ANYDOOR BATCH COMPLETE")
        print("=" * 70)
        print(f"   Total runtime:    {self.results['total_runtime']}")
        print(f"   Total tasks:      {self.results['summary']['total_tasks']}")
        print(f"   Successful:       {self.results['summary']['successful']}")
        print(f"   Failed:           {self.results['summary']['failed']}")
        print(f"   Output:           {self.run_dir}")
        print(f"   Log:              {self.log_file}")
        
        # Print failures if any
        failures = [t for t in self.results["tasks"] if not t["success"]]
        if failures:
            print(f"\n   ⚠️  Failed tasks:")
            for f in failures[:10]:
                print(f"      - {f['script']} [{f['category']}]: {f['error'][:50]}...")
            if len(failures) > 10:
                print(f"      ... and {len(failures) - 10} more")
        
        print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Batch runner for AnyDoor experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run both vanilla and fingerprints on all categories
    python run_batch_anydoor.py
    
    # Run on specific categories only
    python run_batch_anydoor.py --categories bag bottle cup plant
    
    # Run only vanilla version
    python run_batch_anydoor.py --vanilla-only
    
    # Run only fingerprints version
    python run_batch_anydoor.py --fingerprints-only
    
    # Run with 2 parallel workers
    python run_batch_anydoor.py --parallel 2
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
        "--parallel", "-p",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 = sequential)"
    )
    
    parser.add_argument(
        "--vanilla-only",
        action="store_true",
        help="Run only vanilla AnyDoor (no fingerprints)"
    )
    
    parser.add_argument(
        "--fingerprints-only",
        action="store_true",
        help="Run only AnyDoor with fingerprints"
    )
    
    args = parser.parse_args()
    
    # Determine which versions to run
    run_vanilla = True
    run_fingerprints = True
    
    if args.vanilla_only and args.fingerprints_only:
        # Both flags = run both (default)
        pass
    elif args.vanilla_only:
        run_fingerprints = False
    elif args.fingerprints_only:
        run_vanilla = False
    
    runner = AnyDoorBatchRunner(
        categories=args.categories,
        parallel=args.parallel,
        run_vanilla=run_vanilla,
        run_fingerprints=run_fingerprints
    )
    
    runner.run()


if __name__ == "__main__":
    main()
