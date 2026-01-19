# refine.py
"""
Iterative Refinement Loop con Feedback del VLM Judge
Questo è il CORE del sistema R2P-GEN
"""
import torch
from pathlib import Path
from config import Config
from utils2 import cleanup_gpu, ensure_output_dir, get_iteration_filename, print_memory_stats
from generate import generate_image
from verify import verify_generation_detailed


def iterative_refinement(reference_image_path, fingerprints_dict, output_dir="output"):
    """
    Loop di raffinamento iterativo con feedback VLM
    
    Algoritmo:
    1. Genera immagine candidata
    2. VLM Judge verifica attributi
    3. Se accuracy < target:
       - Identifica attributi mancanti
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
    print(f"🔁 ITERATIVE REFINEMENT LOOP")
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
    
    # VLM Judge (lo manteniamo caricato tra iterazioni)
    vlm_judge = None
    
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
        
        # === STEP 2: VERIFICA ===
        verification, vlm_judge = verify_generation_detailed(
            output_filename,
            fingerprints_dict,
            model=vlm_judge
        )
        
        current_score = verification["accuracy"]
        
        # Salva nella storia
        history.append({
            "iteration": iteration,
            "image": output_filename,
            "score": current_score,
            "missing": verification["missing"],
            "present": verification["present"]
        })
        
        print(f"\n   📊 Score: {current_score:.1%} ({len(verification['present'])}/{len(verification['present']) + len(verification['missing'])} attributi)")
        
        # === STEP 3: AGGIORNA BEST ===
        improvement = current_score - best_score
        
        if current_score > best_score:
            best_score = current_score
            best_image_path = output_filename
            best_verification = verification
            print(f"   🏆 NUOVO BEST! (+{improvement:.1%})")
        else:
            print(f"   📉 Nessun miglioramento ({improvement:+.1%})")
        
        # === STEP 4: CONDIZIONI DI USCITA ===
        
        # Successo: accuracy raggiunta
        if current_score >= Config.TARGET_ACCURACY:
            print(f"\n   ✅ TARGET RAGGIUNTO! ({current_score:.1%} >= {Config.TARGET_ACCURACY:.1%})")
            break
        
        # Convergenza: nessun miglioramento significativo
        if iteration > 1 and improvement < Config.MIN_IMPROVEMENT:
            print(f"\n   🔻 Convergenza: miglioramento < {Config.MIN_IMPROVEMENT:.1%}")
            break
        
        # Nessun attributo mancante (caso edge)
        if not verification["missing"]:
            print(f"\n   ✅ Nessun attributo mancante rilevato")
            break
        
        # === STEP 5: PREPARA PROSSIMA ITERAZIONE ===
        if iteration < Config.MAX_ITERATIONS:
            print(f"\n   🔧 Preparazione iterazione {iteration + 1}:")
            print(f"      Attributi da correggere: {len(verification['missing'])}")
            
            # Analizza attributi mancanti e aggiorna negative prompt
            new_negatives = build_negative_from_missing(verification["missing"])
            
            for neg in new_negatives:
                if neg not in negative_additions:
                    negative_additions.append(neg)
                    print(f"      • Aggiungo negative: '{neg}'")
            
            # Cleanup tra iterazioni
            cleanup_gpu()
    
    # === CLEANUP FINALE ===
    del vlm_judge
    cleanup_gpu()
    
    # === REPORT FINALE ===
    print(f"\n{'='*60}")
    print(f"🏁 REFINEMENT COMPLETATO")
    print(f"{'='*60}")
    print(f"   Best Image: {Path(best_image_path).name}")
    print(f"   Best Score: {best_score:.1%}")
    print(f"   Iterazioni: {iteration}")
    
    if best_verification:
        print(f"\n   ✅ Attributi Presenti ({len(best_verification['present'])}):")
        for attr in best_verification['present']:
            print(f"      • {attr}")
        
        if best_verification['missing']:
            print(f"\n   ❌ Attributi Mancanti ({len(best_verification['missing'])}):")
            for attr_key, attr_val in best_verification['missing']:
                print(f"      • {attr_key}: {attr_val[:50]}...")
    
    print(f"{'='*60}")
    
    return {
        "best_image": best_image_path,
        "best_score": best_score,
        "iterations": iteration,
        "history": history,
        "verification": best_verification
    }


def build_negative_from_missing(missing_attributes):
    """USARE SCORES invece di lista binaria"""
    
    # Ordina per confidence (attributi con score < 0.3 priorità max)
    sorted_attrs = sorted(
        missing_attributes,
        key=lambda x: x[1],  # x = (attr_name, score)
        reverse=False  # Lower score = higher priority
    )
    
    negatives = []
    for attr_name, score in sorted_attrs[:5]:  # Top-5 peggiori
        if score < 0.3:
            negatives.append(f"incorrect {attr_name}, wrong {attr_name}")
    
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