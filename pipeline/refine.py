# refine.py
"""
Iterative Refinement Loop con Feedback del VLM Judge
Questo è il CORE del sistema R2P-GEN

Updated for Verify V5 API
"""
import torch
from pathlib import Path
from config import Config
from pipeline.utils2 import cleanup_gpu, ensure_output_dir, get_iteration_filename, print_memory_stats
from generate import generate_image
from pipeline.verify import verify_generation_r2p
from pipeline.r2p_tools import ClipScoreCalculator

# Import reasoner - needed for V5 verify
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))
from r2p_core.models.mini_cpm_reasoning import MiniCPMReasoning


def iterative_refinement(reference_image_path, fingerprints_dict, output_dir="output"):
    """
    Loop di raffinamento iterativo con feedback VLM
    
    Algoritmo:
    1. Genera immagine candidata
    2. VLM Judge verifica attributi (V5: uses CLIP + VLM + Pairwise)
    3. Se accuracy < target:
       - Identifica attributi mancanti (from failed_attributes)
       - Aggiorna negative prompt
       - Rigenera (max MAX_ITERATIONS volte)
    4. Ritorna best result
    
    Args:
        reference_image_path: Immagine target
        fingerprints_dict: Attributi da riprodurre
        output_dir: Cartella output
    
    Returns:
        dict: {
            "best_image": path,
            "best_score": float,
            "iterations": int,
            "history": [...]
        }
    """
    print(f"\n{'='*60}")
    print(f"🔁 ITERATIVE REFINEMENT LOOP (V5)")
    print(f"{'='*60}")
    print(f"   Max Iterazioni: {Config.MAX_ITERATIONS}")
    print(f"   Target Accuracy: {Config.TARGET_ACCURACY:.0%}")
    print(f"   Min Improvement: {Config.MIN_IMPROVEMENT:.0%}")
    
    ensure_output_dir(output_dir)
    
    # Stato del loop
    best_image_path = None
    best_score = 0.0
    best_verification = None
    iteration = 0
    
    # Negative prompt dinamico
    negative_additions = []
    
    # Storia iterazioni
    history = []
    
    # =========================================================================
    # INITIALIZE MODELS FOR V5 VERIFY
    # =========================================================================
    print(f"\n   📦 Loading verification models...")
    
    # Load VLM Reasoner (MiniCPM)
    reasoner = MiniCPMReasoning(
        model_path=Config.VLM_MODEL,
        device="cuda",
        torch_dtype=torch.float16 if Config.USE_FP16 else torch.float32,
        attn_implementation="sdpa",
        seed=Config.SEED
    )
    
    # Load CLIP Calculator
    clip_calculator = ClipScoreCalculator(device="cuda")
    
    print(f"   ✓ Models loaded")
    # =========================================================================
    
    while iteration < Config.MAX_ITERATIONS:
        iteration += 1
        
        print(f"\n{'─'*60}")
        print(f"🔄 ITERAZIONE {iteration}/{Config.MAX_ITERATIONS}")
        print(f"{'─'*60}")
        
        # === STEP 1: GENERAZIONE ===
        output_filename = get_iteration_filename(
            f"{output_dir}/candidate_{Path(reference_image_path).stem}.png",
            iteration
        )
        
        # Costruisci negative prompt cumulativo
        current_negative = Config.NEGATIVE_BASE
        if negative_additions:
            current_negative += ", " + ", ".join(negative_additions)
            print(f"   🚫 Negative aggiunti: {', '.join(negative_additions)}")
        
        # Genera
        try:
            generate_image(
                reference_image_path,
                fingerprints_dict,
                output_path=output_filename,
                negative_prompt=current_negative,
                iteration=iteration
            )
        except Exception as e:
            print(f"   ❌ Errore generazione: {e}")
            break
        
        # === STEP 2: VERIFICA (V5 API) ===
        verification = verify_generation_r2p(
            reasoner=reasoner,
            clip_calculator=clip_calculator,
            gen_image_path=output_filename,
            ref_image_path=reference_image_path,
            fingerprints=fingerprints_dict
        )
        
        # V5 returns is_verified (bool) and score (float)
        # Convert to accuracy-style metric for compatibility
        current_score = verification["score"]
        is_verified = verification["is_verified"]
        failed_attrs = verification.get("failed_attributes", [])
        
        # Build missing list from failed_attributes for history compatibility
        # V5 failed_attributes is a list of attribute strings
        missing_list = [(attr, attr) for attr in failed_attrs]  # (key, value) format
        present_count = len(verification.get("vlm_history", [])) - len(failed_attrs)
        
        # Salva nella storia
        history.append({
            "iteration": iteration,
            "image": output_filename,
            "score": current_score,
            "is_verified": is_verified,
            "method": verification.get("method", "unknown"),
            "failed_attributes": failed_attrs,
            "reason": verification.get("reason", "")
        })
        
        total_attrs = len(verification.get("vlm_history", [])) // 2  # Rough estimate (single + pairwise)
        if total_attrs == 0:
            total_attrs = max(1, len(failed_attrs))
        
        print(f"\n   📊 Score: {current_score:.2f} | Verified: {is_verified} | Method: {verification.get('method', '?')}")
        
        # === STEP 3: AGGIORNA BEST ===
        improvement = current_score - best_score
        
        if current_score > best_score:
            best_score = current_score
            best_image_path = output_filename
            best_verification = verification
            print(f"   🏆 NUOVO BEST! (+{improvement:.2f})")
        else:
            print(f"   📉 Nessun miglioramento ({improvement:+.2f})")
        
        # === STEP 4: CONDIZIONI DI USCITA ===
        
        # Successo: verification passed
        if is_verified:
            print(f"\n   ✅ VERIFIED! Method: {verification.get('method', '?')}")
            break
        
        # Convergenza: nessun miglioramento significativo
        if iteration > 1 and improvement < Config.MIN_IMPROVEMENT:
            print(f"\n   🔻 Convergenza: miglioramento < {Config.MIN_IMPROVEMENT:.2f}")
            break
        
        # Nessun attributo mancante (caso edge)
        if not failed_attrs:
            print(f"\n   ✅ Nessun attributo mancante rilevato")
            break
        
        # === STEP 5: PREPARA PROSSIMA ITERAZIONE ===
        if iteration < Config.MAX_ITERATIONS:
            print(f"\n   🔧 Preparazione iterazione {iteration + 1}:")
            print(f"      Attributi da correggere: {len(failed_attrs)}")
            
            # Analizza attributi mancanti e aggiorna negative prompt
            # V5: failed_attrs is list of strings, convert to (key, value) format
            new_negatives = build_negative_from_missing(missing_list)
            
            for neg in new_negatives:
                if neg not in negative_additions:
                    negative_additions.append(neg)
                    print(f"      • Aggiungo negative: '{neg}'")
            
            # Cleanup tra iterazioni
            cleanup_gpu()
    
    # === CLEANUP FINALE ===
    del reasoner
    del clip_calculator
    cleanup_gpu()
    
    # === REPORT FINALE ===
    print(f"\n{'='*60}")
    print(f"🏁 REFINEMENT COMPLETATO (V5)")
    print(f"{'='*60}")
    print(f"   Best Image: {Path(best_image_path).name if best_image_path else 'None'}")
    print(f"   Best Score: {best_score:.2f}")
    print(f"   Iterazioni: {iteration}")
    
    if best_verification:
        print(f"\n   Method: {best_verification.get('method', 'unknown')}")
        print(f"   Verified: {best_verification.get('is_verified', False)}")
        print(f"   Reason: {best_verification.get('reason', 'N/A')}")
        
        failed = best_verification.get('failed_attributes', [])
        if failed:
            print(f"\n   ❌ Failed Attributes ({len(failed)}):")
            for attr in failed[:5]:
                print(f"      • {attr[:50]}...")
            if len(failed) > 5:
                print(f"      ... and {len(failed) - 5} more")
    
    print(f"{'='*60}")
    
    return {
        "best_image": best_image_path,
        "best_score": best_score,
        "iterations": iteration,
        "history": history,
        "verification": best_verification
    }


def build_negative_from_missing(missing_attributes):
    """
    Costruisce negative prompts da attributi mancanti
    
    Strategia:
    - Se manca "color: red" → aggiungi "not red, wrong color"
    - Se manca "brand: Nike" → aggiungi "without logo, wrong brand"
    """
    negatives = []
    
    for attr_key, attr_value in missing_attributes:
        if attr_key == "color":
            negatives.append(f"wrong color, not {attr_value}")
        
        elif attr_key == "brand":
            negatives.append(f"without brand logo, incorrect branding")
        
        elif attr_key == "material":
            negatives.append(f"wrong material, not {attr_value}")
        
        elif attr_key == "packaging":
            negatives.append(f"incorrect packaging")
        
        elif attr_key == "product_type":
            negatives.append(f"wrong product category")
        
        else:
            # Generico
            negatives.append(f"missing {attr_key}")
    
    return negatives


if __name__ == "__main__":
    # Test standalone
    import os
    
    test_img = "data/perva_test/1.jpg"
    test_fp = {
        "brand": "Test Brand",
        "color": "blue",
        "product_type": "bottle",
        "packaging": "plastic bottle with label"
    }
    
    if os.path.exists(test_img):
        result = iterative_refinement(test_img, test_fp)
        
        print("\n" + "="*60)
        print("📊 HISTORY:")
        for h in result["history"]:
            print(f"   Iter {h['iteration']}: {h['score']:.1%} - {h['image']}")
    else:
        print(f"❌ File test non trovato: {test_img}")