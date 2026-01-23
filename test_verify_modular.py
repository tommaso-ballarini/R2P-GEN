import os
import sys
import json
import torch
from PIL import Image
from datetime import datetime

# --- CONFIGURAZIONE PATH ROBUSTA ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

print(f"📂 Working Directory set to: {os.getcwd()}")

# 3. SETUP PERCORSI R2P
project_sam_path = os.path.join(SCRIPT_DIR, "project_sam")
r2p_core_path = os.path.join(project_sam_path, "r2p_core")

if os.path.exists(project_sam_path) and project_sam_path not in sys.path:
    sys.path.append(project_sam_path)

if os.path.exists(r2p_core_path):
    print(f"✅ Found R2P Core path: {r2p_core_path}")
    if r2p_core_path not in sys.path:
        sys.path.insert(0, r2p_core_path)
else:
    print(f"⚠️ WARNING: Cartella r2p_core non trovata in {r2p_core_path}")

try:
    # --- IMPORT MODULI PIPELINE ---
    # Assicurati che verify_generation_r2p sia la versione v3 aggiornata!
    from pipeline2.verify import verify_generation_r2p, _extract_attributes_for_clip
    from pipeline2.r2p_tools import ClipScoreCalculator
    
    # --- IMPORT MINICPM ---
    try:
        from models.mini_cpm_reasoning import MiniCPMReasoning
        print("🔹 Imported MiniCPMReasoning from 'models'")
    except ImportError:
        try:
            from r2p_core.models.mini_cpm_reasoning import MiniCPMReasoning
            print("🔹 Imported MiniCPMReasoning from 'r2p_core.models'")
        except ImportError:
            sys.modules['src'] = __import__('r2p_core')
            from src.models.mini_cpm_reasoning import MiniCPMReasoning
            print("🔹 Imported MiniCPMReasoning via 'src' alias override")

except ImportError as e:
    print(f"\n❌ ERRORE CRITICO DI IMPORT: {e}")
    sys.exit(1)

# --- DEFINIZIONE DEI CASI DI TEST ---
TEST_DATA_DIR = "test_sandbox" 
JSON_PATH = os.path.join(TEST_DATA_DIR, "fingerprints.json")

TEST_CASES = [
    {"id": "Test_Bag_1", "ref_img": os.path.join(TEST_DATA_DIR, "ref_1.jpeg"), "gen_img": os.path.join(TEST_DATA_DIR, "gen_1.jpeg"), "json_key": "<bxa_1>"},
    {"id": "Test_Bag_2", "ref_img": os.path.join(TEST_DATA_DIR, "ref_2.jpeg"), "gen_img": os.path.join(TEST_DATA_DIR, "gen_2.jpeg"), "json_key": "<ash_1>"},
    {"id": "Test_Bag_3", "ref_img": os.path.join(TEST_DATA_DIR, "ref_3.jpeg"), "gen_img": os.path.join(TEST_DATA_DIR, "gen_3.jpeg"), "json_key": "<dbi_1>"},
    {"id": "Test_Bag_4", "ref_img": os.path.join(TEST_DATA_DIR, "ref_4.jpeg"), "gen_img": os.path.join(TEST_DATA_DIR, "gen_4.jpeg"), "json_key": "<bkq_1>"},
    {"id": "Test_Bag_5", "ref_img": os.path.join(TEST_DATA_DIR, "ref_5.jpeg"), "gen_img": os.path.join(TEST_DATA_DIR, "gen_5.jpeg"), "json_key": "<alx_1>"}
]



# --- FUNZIONE REPORT DETTAGLIATA (V4 - AGGIORNATA PER PIPELINE V3) ---
def write_report_entry(f, case_id, result, attributes_list):
    f.write(f"\n{'='*80}\n")
    f.write(f"CASE: {case_id}\n")
    f.write(f"{'='*80}\n")
    f.write(f"VERDICT: {'✅ PASS' if result['is_verified'] else '❌ FAIL'}\n")
    f.write(f"METHOD:  {result['method']}\n")
    f.write(f"SCORE:   {result['score']:.4f}\n")
    f.write(f"REASON:  {result.get('reason', 'N/A')}\n\n")

    f.write("--- 1. VLM & PAIRWISE REASONING HISTORY ---\n")
    
    # Recuperiamo la storia completa
    history = result.get('vlm_history', [])
    
    if history:
        for i, item in enumerate(history):
            phase = item.get("phase", "unknown")
            attr_name = item.get("attribute", "Unknown")
            score = item.get("score", 0.0)
            
            # FORMATTAZIONE IN BASE ALLA FASE
            
            # FASE 4: Pairwise Check (Confronto tra due immagini)
            if phase == "pairwise":
                marker = "⭐" if score > 0.60 else "⚠️"
                f.write(f"Step {i+1} [PAIRWISE CHECK]:\n")
                f.write(f"   Target: {attr_name}\n")
                f.write(f"   Match Score: {score:.4f} {marker}\n")
                f.write("-" * 40 + "\n")

            # FASE 1: Single Check (Nuovo formato con Confidence Score)
            elif phase == "single_check":
                prompt_text = item.get('prompt', 'N/A')
                response_text = item.get('response', '').strip()
                # Colore indicativo per la confidence
                conf_marker = "🟢" if score > 0.8 else "🟡" if score > 0.5 else "🔴"
                
                f.write(f"Step {i+1} [SINGLE IMAGE CONFIDENCE]:\n")
                f.write(f"   Attribute: {attr_name}\n")
                f.write(f"   Model Ans: {response_text[:50]}...\n")
                f.write(f"   Confidence: {score:.4f} {conf_marker}\n")
                f.write("-" * 40 + "\n")
            
            # Fallback per vecchi formati o errori
            else:
                f.write(f"Step {i+1} [RAW]: {item}\n")
                f.write("-" * 40 + "\n")
    else:
        f.write("No VLM history available.\n")

    f.write("\n--- 2. CLIP ATTRIBUTE BREAKDOWN ---\n")
    f.write(f"{'ATTRIBUTE':<40} | {'REF':<8} | {'GEN':<8} | {'DELTA'}\n")
    f.write(f"{'-'*40}-|-{'-'*8}-|-{'-'*8}-|-{'-'*8}\n")
    
    details = result.get('clip_details', {})
    gen_scores = details.get('gen', {})
    ref_scores = details.get('ref', {})
    
    table_rows = []
    # Se la lista attributi è vuota (fallback), prendiamo le chiavi dal dizionario
    display_attrs = attributes_list if attributes_list else list(gen_scores.keys())

    for attr in display_attrs:
        r = ref_scores.get(attr, 0.0)
        g = gen_scores.get(attr, 0.0)
        delta = g - r
        table_rows.append((attr, r, g, delta))
    
    # Ordiniamo per drop peggiore
    table_rows.sort(key=lambda x: x[3])

    for attr, r, g, delta in table_rows:
        attr_display = (attr[:37] + '..') if len(attr) > 37 else attr
        marker = ""
        if delta < -0.05: marker = "🔴"
        elif delta < -0.02: marker = "⚠️"
        elif delta > 0.02: marker = "🟢"
        
        f.write(f"{marker:<2} {attr_display:<37} | {r:.4f}   | {g:.4f}   | {delta:+.4f}\n")
    
    f.write("\n")

# --- MAIN LOOP ---
def main():
    print("=========================================================")
    print("   🧪 R2P VERIFICATION MODULE TESTER (PIPELINE V3) 🧪   ")
    print("=========================================================")

    # 1. Caricamento JSON
    if not os.path.exists(JSON_PATH):
        print(f"❌ File JSON non trovato: {JSON_PATH}")
        return

    with open(JSON_PATH, 'r') as f:
        full_json_data = json.load(f)
        fingerprints_db = full_json_data.get("concept_dict", full_json_data)
        print(f"✅ JSON caricato. Trovati {len(fingerprints_db)} concetti.")

    # 2. Inizializzazione Modelli
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n⚙️  Inizializzazione Modelli su {device}...")
    
    try:
        print("   -> Loading MiniCPM-V (Wait...)...")
        reasoner = MiniCPMReasoning(model_path="openbmb/MiniCPM-o-2_6", device=device)
        
        print("   -> Loading CLIP (Wait...)...")
        clip_calculator = ClipScoreCalculator(device=device)
        print("✅ Modelli Caricati con successo.\n")
    except Exception as e:
        print(f"\n❌ Errore nel caricamento modelli: {e}")
        return

    # 3. Esecuzione Test Loop con Report
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = f"verification_report_v3_{timestamp}.txt"
    
    print("=========================================================")
    print("              STARTING VERIFICATION LOOP                 ")
    print(f"      📄 Report File: {report_path}")
    print("=========================================================")

    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(f"R2P VERIFICATION V3 DEBUG REPORT - {timestamp}\n")
        
        for i, case in enumerate(TEST_CASES):
            print(f"\n🔹 CASE {i+1}: {case['id']}")
            
            if not os.path.exists(case['ref_img']) or not os.path.exists(case['gen_img']):
                print("   ⚠️  SKIPPING: Immagini non trovate su disco.")
                continue
            
            key = case['json_key']
            if key not in fingerprints_db:
                print(f"   ⚠️  SKIPPING: Chiave '{key}' non trovata.")
                continue
                
            fingerprints = fingerprints_db[key]
            if "info" in fingerprints: fingerprints = fingerprints["info"]
            if "name" not in fingerprints: fingerprints["name"] = key

            attrs = _extract_attributes_for_clip(fingerprints)

            try:
                print("\n   --- 🚀 RUNNING verify_generation_r2p (V3) 🚀 ---")
                
                # --- UPDATE ARGOMENTI PER V3 ---
                result = verify_generation_r2p(
                    reasoner=reasoner,
                    clip_calculator=clip_calculator,
                    gen_image_path=case['gen_img'],
                    ref_image_path=case['ref_img'],
                    fingerprints=fingerprints,
                    # Nuovi Parametri per la logica Disagreement-Based
                    vlm_high_confidence=0.85,  # Auto-Pass se > 0.85
                    vlm_low_confidence=0.40,   # Auto-Fail se < 0.40
                    clip_hard_floor=0.15,
                    max_drop_threshold=-0.03
                )
                
                write_report_entry(report_file, case['id'], result, attrs)
                print(f"   📝 Logged detailed report for {case['id']}")

                status = "✅ VERIFIED" if result['is_verified'] else "❌ REJECTED"
                print(f"   🏁 RESULT: {status} | Score: {result['score']:.4f} | Method: {result['method']}")
                    
            except Exception as e:
                print(f"   ❌ CRASH during verification function: {e}")
                import traceback
                traceback.print_exc()

    print("\n=========================================================")
    print(f"   TEST COMPLETED - REPORT SAVED TO:")
    print(f"   📂 {os.path.abspath(report_path)}")
    print("=========================================================")

if __name__ == "__main__":
    main()