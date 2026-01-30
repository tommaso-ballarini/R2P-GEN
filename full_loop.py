"""
R2P-GEN Full Pipeline Orchestrator

This module orchestrates the complete R2P-GEN pipeline:
1. Build Database (fingerprint extraction)
2. Generate (image generation with SDXL + IP-Adapter)
3. Verify (generated image verification)
4. Refine (iterative refinement loop)
"""

import os
import torch
import gc
from pathlib import Path
from pipeline.build_database import DatabaseBuilder
from pipeline.generate import Generator
from pipeline.refine import iterative_refinement
from pipeline.utils2 import cleanup_gpu, ensure_output_dir
from config import Config

def run_r2p_gen_pipeline(target_image_path, use_refinement=True):
    """
    Complete R2P-GEN Pipeline with Iterative Refinement.
    
    Args:
        target_image_path: Image to reproduce
        use_refinement: If True, use iterative loop; otherwise single-shot
        
    Returns:
        dict: Result containing best_image, best_score, iterations, history
    """
    print(f"\n{'='*70}")
    print(f"🚀 R2P-GEN PIPELINE")
    print(f"{'='*70}")
    print(f"   Target: {target_image_path}")
    print(f"   Mode: {'Iterative Refinement' if use_refinement else 'Single-Shot'}")
    print(f"{'='*70}\n")
    
    if not os.path.exists(target_image_path):
        print(f"❌ Error: File {target_image_path} not found.")
        return None
    
    output_dir = ensure_output_dir(Config.OUTPUT_DIR)
    
    # ═══════════════════════════════════════════════════════
    # PHASE 1: EXTRACTION (using DatabaseBuilder)
    # ═══════════════════════════════════════════════════════
    print(f"{'─'*70}")
    print("📍 PHASE 1/2: EXTRACTION")
    print(f"{'─'*70}")
    
    # TODO: Replace with DatabaseBuilder when integrated
    # For now, extraction is done separately via build_database.py
    # fingerprints_dict = builder.extract_single_image(target_image_path)
    print("⚠️  Extraction should be done via build_database.py first")
    print("    This function expects a pre-built database.")
    
    # Placeholder for integration
    fingerprints_dict = None
    vlm_model = None
    
    if not fingerprints_dict:
        print("❌ Extraction failed - Pipeline interrupted")
        print("   Run build_database.py first to create the fingerprint database.")
        return None
    
    print(f"\n✅ Fingerprints extracted:")
    for k, v in fingerprints_dict.items():
        if k != "description":
            print(f"   • {k}: {v}")
    
    # Free VLM to make space for SDXL
    if vlm_model:
        del vlm_model
    cleanup_gpu()
    
    # ═══════════════════════════════════════════════════════
    # PHASE 2: GENERATION (with or without refinement)
    # ═══════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print("📍 PHASE 2/2: GENERATION & VERIFICATION")
    print(f"{'─'*70}")
    
    if use_refinement:
        # Iterative mode (CORE of R2P)
        result = iterative_refinement(
            target_image_path,
            fingerprints_dict,
            output_dir=output_dir
        )
        
        final_image = result["best_image"]
        final_score = result["best_score"]
        iterations_used = result["iterations"]
        
    else:
        # Single-shot mode (for comparison)
        from pipeline.generate import Generator
        from pipeline.verify import verify_generation_r2p
        from pipeline.r2p_tools import ClipScoreCalculator
        from r2p_core.models.mini_cpm_reasoning import MiniCPMReasoning
        
        final_image = f"{output_dir}/singleshot_{Path(target_image_path).stem}.png"
        
        # TODO: Use Generator class when integrated
        # generator = Generator()
        # generator.generate_image(...)
        print("⚠️  Single-shot generation not yet integrated with Generator class")
        
        # Load models for V5 verify
        print("   Loading verification models...")
        reasoner = MiniCPMReasoning(
            model_path=Config.VLM_MODEL,
            device="cuda",
            torch_dtype=torch.float16 if Config.USE_FP16 else torch.float32,
            attn_implementation="sdpa",
            seed=Config.SEED
        )
        clip_calculator = ClipScoreCalculator(device="cuda")
        
        verification_result = verify_generation_r2p(
            reasoner=reasoner,
            clip_calculator=clip_calculator,
            gen_image_path=final_image,
            ref_image_path=target_image_path,
            fingerprints=fingerprints_dict
        )
        
        final_score = verification_result["score"]
        is_verified = verification_result["is_verified"]
        
        # Cleanup
        del reasoner
        del clip_calculator
        cleanup_gpu()
        
        iterations_used = 1
        result = {
            "best_image": final_image,
            "best_score": final_score,
            "is_verified": is_verified,
            "iterations": 1,
            "history": [],
            "verification": verification_result
        }
    
    # ═══════════════════════════════════════════════════════
    # FINAL REPORT
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("🏁 PIPELINE COMPLETED")
    print(f"{'='*70}")
    print(f"   📁 Input:  {Path(target_image_path).name}")
    print(f"   📁 Output: {Path(final_image).name}")
    print(f"   📊 Score:  {final_score:.1%}")
    print(f"   🔄 Iterations: {iterations_used}")
    
    if final_score >= Config.TARGET_ACCURACY:
        print(f"   ✅ SUCCESS - Target reached!")
    elif final_score >= 0.7:
        print(f"   ⚠️  PARTIAL - Acceptable result")
    else:
        print(f"   ❌ FAILURE - Insufficient quality")
    
    print(f"{'='*70}\n")
    
    return result


if __name__ == "__main__":
    # Test with example image
    test_image = "data/perva-data/test/bag/alx/1.jpg"
    
    if os.path.exists(test_image):
        # Test with refinement
        result = run_r2p_gen_pipeline(test_image, use_refinement=True)
        
        if result:
            print("\n📊 ITERATION DETAILS:")
            for h in result["history"]:
                print(f"   Iter {h['iteration']}: {h['score']:.1%}")
    else:
        print(f"❌ Test file not found: {test_image}")
        print("   Create the 'data/perva-data/' folder and add a test image")