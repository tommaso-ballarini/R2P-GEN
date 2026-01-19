# full_loop.py
import os
import torch
import gc
from pathlib import Path

from extract import extract_fingerprints
from refine import iterative_refinement
from config import Config
from utils2 import cleanup_gpu, ensure_output_dir

import time

def run_r2p_gen_pipeline(target_image_path, use_refinement=True):
    """
    Pipeline completa R2P-GEN con Iterative Refinement
    
    Args:
        target_image_path: Immagine da riprodurre
        use_refinement: Se True, usa loop iterativo; altrimenti single-shot
    """

    start_time = time.time()

    print(f"\n{'='*70}")
    print(f"🚀 R2P-GEN PIPELINE")
    print(f"{'='*70}")
    print(f"   Target: {target_image_path}")
    print(f"   Mode: {'Iterative Refinement' if use_refinement else 'Single-Shot'}")
    print(f"{'='*70}\n")
    
    if not os.path.exists(target_image_path):
        print(f"❌ Errore: File {target_image_path} non trovato.")
        return None
    
    output_dir = ensure_output_dir(Config.OUTPUT_DIR)
    
    # ═══════════════════════════════════════════════════════
    # FASE 1: EXTRACTION
    # ═══════════════════════════════════════════════════════

    print(f"{'─'*70}")
    print("📍 FASE 1/2: EXTRACTION")
    print(f"{'─'*70}")
    
    fingerprints_dict, vlm_model = extract_fingerprints(target_image_path)
    
    if not fingerprints_dict:
        print("❌ Extraction fallita - Pipeline interrotta")
        return None
    
    print(f"\n✅ Fingerprints estratti:")
    for k, v in fingerprints_dict.items():
        if k != "description":
            print(f"   • {k}: {v}")
    
    # Libera VLM per fare spazio a SDXL
    del vlm_model
    cleanup_gpu()
    
    extraction_time = time.time() - start_time
    
    # ═══════════════════════════════════════════════════════
    # FASE 2: GENERATION (con o senza refinement)
    # ═══════════════════════════════════════════════════════

    gen_start = time.time()

    print(f"\n{'─'*70}")
    print("📍 FASE 2/2: GENERATION & VERIFICATION")
    print(f"{'─'*70}")
    
    if use_refinement:
        # Modalità iterativa (CORE di R2P)
        result = iterative_refinement(
            target_image_path,
            fingerprints_dict,
            output_dir=output_dir
        )
        
        final_image = result["best_image"]
        final_score = result["best_score"]
        iterations_used = result["iterations"]
        
    else:
        # Modalità single-shot (per confronto)
        from generate import generate_image
        from verify import verify_generation
        
        final_image = f"{output_dir}/singleshot_{Path(target_image_path).stem}.png"
        
        generate_image(
            target_image_path,
            fingerprints_dict,
            output_path=final_image,
            iteration=1
        )
        
        final_score, _ = verify_generation(
            final_image,
            target_image_path,
            fingerprints_dict
        )
        
        iterations_used = 1
        result = {
            "best_image": final_image,
            "best_score": final_score,
            "iterations": 1,
            "history": []
        }

        generation_time = time.time() - gen_start
    
    # ═══════════════════════════════════════════════════════
    # REPORT FINALE
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("🏁 PIPELINE COMPLETATA")
    print(f"{'='*70}")
    print(f"   📁 Input:  {Path(target_image_path).name}")
    print(f"   📁 Output: {Path(final_image).name}")
    print(f"   📊 Score:  {final_score:.1%}")
    print(f"   🔄 Iterations: {iterations_used}")
    
    if final_score >= Config.TARGET_ACCURACY:
        print(f"   ✅ SUCCESSO - Target raggiunto!")
    elif final_score >= 0.7:
        print(f"   ⚠️  PARZIALE - Risultato accettabile")
    else:
        print(f"   ❌ FALLIMENTO - Qualità insufficiente")
    
    print(f"{'='*70}\n")

    result['timings'] = {
        'extraction': extraction_time,
        'generation': generation_time,
        'total': time.time() - start_time
    }
    
    return result


if __name__ == "__main__":
    # Test con immagine esempio
    test_image = "data/perva_test/11.jpg"
    
    if os.path.exists(test_image):
        # Test con refinement
        result = run_r2p_gen_pipeline(test_image, use_refinement=True)
        
        if result:
            print("\n📊 DETTAGLIO ITERAZIONI:")
            for h in result["history"]:
                print(f"   Iter {h['iteration']}:{h['score']:.1%}")
else:
    print(f"❌ File test non trovato: {test_image}")
    print("   Crea la cartella 'data/perva_test/' e inserisci un'immagine di test")