"""
R2P-GEN Full Pipeline Orchestrator

This module orchestrates the complete R2P-GEN pipeline:
1. Build Database (fingerprint extraction)
2. Generate (image generation with SDXL + IP-Adapter)
3. Verify (generated image verification with V5 logit-based confidence)
4. Refine (iterative refinement loop)

The pipeline can run in two modes:
- Full Database Mode: Process all images from a database JSON
- Single Image Mode: Process a single target image with optional refinement
"""

import os
import json
import torch
import gc
from pathlib import Path

from pipeline.build_database import DatabaseBuilder
from pipeline.generate import Generator
from pipeline.verify import verify_generation_r2p
from pipeline.r2p_tools import ClipScoreCalculator
from pipeline.utils2 import cleanup_gpu, ensure_output_dir
from r2p_core.models.mini_cpm_reasoning import MiniCPMReasoning
from config import Config


def run_database_pipeline(
    database_path: str,
    output_dir: str = None,
    skip_generation: bool = False,
    skip_verification: bool = False
):
    """
    Run the full R2P-GEN pipeline on a pre-built database.
    
    This is the main batch processing mode:
    1. Load database (built by build_database.py)
    2. Generate images for all concepts
    3. Verify each generated image
    
    Args:
        database_path: Path to fingerprint database JSON
        output_dir: Output directory (default: from Config)
        skip_generation: If True, skip generation (use existing images)
        skip_verification: If True, skip verification step
        
    Returns:
        dict: Results with generation and verification stats
    """
    print(f"\n{'='*70}")
    print(f"🚀 R2P-GEN DATABASE PIPELINE")
    print(f"{'='*70}")
    print(f"   Database: {database_path}")
    print(f"   Output: {output_dir or Config.OUTPUT_DIR}")
    print(f"{'='*70}\n")
    
    if not os.path.exists(database_path):
        print(f"❌ Database not found: {database_path}")
        print("   Run: python pipeline/build_database.py first")
        return None
    
    output_dir = output_dir or Config.OUTPUT_DIR
    ensure_output_dir(output_dir)
    
    results = {
        "database_path": database_path,
        "output_dir": output_dir,
        "generation_stats": None,
        "verification_results": {}
    }
    
    # ═══════════════════════════════════════════════════════
    # PHASE 1: GENERATION
    # ═══════════════════════════════════════════════════════
    if not skip_generation:
        print(f"{'─'*70}")
        print("📍 PHASE 1: IMAGE GENERATION")
        print(f"{'─'*70}")
        
        generator = Generator(
            database_path=database_path,
            output_dir=output_dir
        )
        
        generation_stats = generator.generate_all()
        results["generation_stats"] = generation_stats
        
        generator.cleanup()
        cleanup_gpu()
    else:
        print("⏭️  Skipping generation (using existing images)")
    
    # ═══════════════════════════════════════════════════════
    # PHASE 2: VERIFICATION
    # ═══════════════════════════════════════════════════════
    if not skip_verification:
        print(f"\n{'─'*70}")
        print("📍 PHASE 2: IMAGE VERIFICATION (V5)")
        print(f"{'─'*70}")
        
        # Load database for verification
        with open(database_path, 'r', encoding='utf-8') as f:
            database = json.load(f)
        
        concept_dict = database.get("concept_dict", {})
        
        # Initialize verification models
        print("   Loading verification models...")
        reasoner = MiniCPMReasoning(
            model_path=Config.VLM_MODEL,
            device="cuda",
            torch_dtype=torch.float16 if Config.USE_FP16 else torch.float32,
            attn_implementation="sdpa",
            seed=Config.SEED
        )
        clip_calculator = ClipScoreCalculator(device="cuda")
        
        verified_count = 0
        failed_count = 0
        
        for concept_id, content in concept_dict.items():
            # Find generated image
            gen_image_path = os.path.join(output_dir, f"{concept_id}_generated.png")
            
            if not os.path.exists(gen_image_path):
                print(f"   ⚠️  {concept_id}: Generated image not found, skipping")
                continue
            
            # Get reference image (first selected image)
            selected_images = content.get("selected_images", [])
            if not selected_images:
                print(f"   ⚠️  {concept_id}: No reference images, skipping")
                continue
            
            ref_image_path = selected_images[0]
            fingerprints = content.get("info", {})
            
            # Run verification
            verification = verify_generation_r2p(
                reasoner=reasoner,
                clip_calculator=clip_calculator,
                gen_image_path=gen_image_path,
                ref_image_path=ref_image_path,
                fingerprints=fingerprints
            )
            
            results["verification_results"][concept_id] = verification
            
            if verification["is_verified"]:
                verified_count += 1
                print(f"   ✅ {concept_id}: PASS ({verification['method']})")
            else:
                failed_count += 1
                print(f"   ❌ {concept_id}: FAIL ({verification['method']})")
        
        # Cleanup
        del reasoner
        del clip_calculator
        cleanup_gpu()
        
        # Summary
        total = verified_count + failed_count
        print(f"\n📊 Verification Summary: {verified_count}/{total} passed ({100*verified_count/max(1,total):.1f}%)")
    else:
        print("⏭️  Skipping verification")
    
    return results


def run_single_image_pipeline(
    target_image_path: str,
    fingerprints: dict = None,
    output_dir: str = None,
    use_refinement: bool = True
):
    """
    Run the R2P-GEN pipeline on a single target image.
    
    This mode is useful for:
    - Testing with a single image
    - Iterative refinement experiments
    - Quick prototyping
    
    Args:
        target_image_path: Path to the target image
        fingerprints: Pre-extracted fingerprints (optional, will extract if None)
        output_dir: Output directory
        use_refinement: If True, use iterative refinement loop
        
    Returns:
        dict: Result with best_image, best_score, verification details
    """
    print(f"\n{'='*70}")
    print(f"🚀 R2P-GEN SINGLE IMAGE PIPELINE")
    print(f"{'='*70}")
    print(f"   Target: {target_image_path}")
    print(f"   Mode: {'Iterative Refinement' if use_refinement else 'Single-Shot'}")
    print(f"{'='*70}\n")
    
    if not os.path.exists(target_image_path):
        print(f"❌ Error: File {target_image_path} not found.")
        return None
    
    output_dir = output_dir or Config.OUTPUT_DIR
    ensure_output_dir(output_dir)
    
    # ═══════════════════════════════════════════════════════
    # PHASE 1: EXTRACTION (if fingerprints not provided)
    # ═══════════════════════════════════════════════════════
    if fingerprints is None:
        print(f"{'─'*70}")
        print("📍 PHASE 1: FINGERPRINT EXTRACTION")
        print(f"{'─'*70}")
        print("⚠️  Single-image extraction not yet implemented")
        print("   Please provide fingerprints dict or use database mode")
        print("   Run: python pipeline/build_database.py first")
        return None
    
    print(f"✅ Using provided fingerprints ({len(fingerprints)} keys)")
    
    # ═══════════════════════════════════════════════════════
    # PHASE 2: GENERATION & VERIFICATION
    # ═══════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print("📍 PHASE 2: GENERATION & VERIFICATION")
    print(f"{'─'*70}")
    
    if use_refinement:
        # Import refinement module
        from pipeline.refine import iterative_refinement
        
        result = iterative_refinement(
            reference_image_path=target_image_path,
            fingerprints_dict=fingerprints,
            output_dir=output_dir
        )
        
        final_image = result.get("best_image")
        final_score = result.get("best_score", 0)
        is_verified = result.get("best_verification", {}).get("is_verified", False)
        iterations_used = result.get("iterations", 0)
        
    else:
        # Single-shot mode
        from pipeline.generate import Generator
        
        # Create temporary database for single image
        temp_db = {
            "concept_dict": {
                "single_target": {
                    "selected_images": [target_image_path],
                    "info": fingerprints,
                    "sdxl_prompt": fingerprints.get("description", "A product image")
                }
            }
        }
        
        temp_db_path = os.path.join(output_dir, "_temp_single_db.json")
        with open(temp_db_path, 'w') as f:
            json.dump(temp_db, f)
        
        # Generate
        generator = Generator(
            database_path=temp_db_path,
            output_dir=output_dir
        )
        generator.generate_all()
        generator.cleanup()
        
        final_image = os.path.join(output_dir, "single_target_generated.png")
        
        # Verify
        print("   Loading verification models...")
        reasoner = MiniCPMReasoning(
            model_path=Config.VLM_MODEL,
            device="cuda",
            torch_dtype=torch.float16 if Config.USE_FP16 else torch.float32,
            attn_implementation="sdpa",
            seed=Config.SEED
        )
        clip_calculator = ClipScoreCalculator(device="cuda")
        
        verification = verify_generation_r2p(
            reasoner=reasoner,
            clip_calculator=clip_calculator,
            gen_image_path=final_image,
            ref_image_path=target_image_path,
            fingerprints=fingerprints
        )
        
        final_score = verification["score"]
        is_verified = verification["is_verified"]
        iterations_used = 1
        
        # Cleanup
        del reasoner
        del clip_calculator
        cleanup_gpu()
        os.remove(temp_db_path)  # Remove temp database
        
        result = {
            "best_image": final_image,
            "best_score": final_score,
            "is_verified": is_verified,
            "iterations": 1,
            "verification": verification
        }
    
    # ═══════════════════════════════════════════════════════
    # FINAL REPORT
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("🏁 PIPELINE COMPLETED")
    print(f"{'='*70}")
    print(f"   📁 Input:  {Path(target_image_path).name}")
    print(f"   📁 Output: {Path(final_image).name if final_image else 'None'}")
    print(f"   📊 Score:  {final_score:.2f}")
    print(f"   ✓ Verified: {is_verified}")
    print(f"   🔄 Iterations: {iterations_used}")
    
    if is_verified:
        print(f"   ✅ SUCCESS - Verification passed!")
    elif final_score >= 0.6:
        print(f"   ⚠️  PARTIAL - Acceptable but not verified")
    else:
        print(f"   ❌ FAILURE - Insufficient quality")
    
    print(f"{'='*70}\n")
    
    return result


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="R2P-GEN Full Pipeline")
    parser.add_argument("--mode", choices=["database", "single"], default="database",
                       help="Pipeline mode: 'database' for batch, 'single' for one image")
    parser.add_argument("--database", type=str, default="database/database_perva_train_1_clip.json",
                       help="Path to fingerprint database (for database mode)")
    parser.add_argument("--image", type=str, default=None,
                       help="Path to target image (for single mode)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory")
    parser.add_argument("--skip-generation", action="store_true",
                       help="Skip generation (use existing images)")
    parser.add_argument("--skip-verification", action="store_true",
                       help="Skip verification step")
    parser.add_argument("--no-refinement", action="store_true",
                       help="Disable iterative refinement (single mode only)")
    
    args = parser.parse_args()
    
    if args.mode == "database":
        run_database_pipeline(
            database_path=args.database,
            output_dir=args.output,
            skip_generation=args.skip_generation,
            skip_verification=args.skip_verification
        )
    else:
        if not args.image:
            print("❌ --image is required for single mode")
        else:
            # For single mode, you'd need to provide fingerprints
            # This is a placeholder - in practice you'd load from database or extract
            print("⚠️  Single mode requires pre-extracted fingerprints")
            print("   Use database mode or provide fingerprints programmatically")
            
            # Example usage:
            # run_single_image_pipeline(
            #     target_image_path=args.image,
            #     fingerprints={"description": "...", "color": "...", ...},
            #     output_dir=args.output,
            #     use_refinement=not args.no_refinement
            # )