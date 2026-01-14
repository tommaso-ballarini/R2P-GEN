import os
import torch
import gc

# Import dei nostri 3 moduli
from extract import extract_fingerprints
from generate import generate_image
from verify import verify_generation

def run_r2p_gen_pipeline(target_image_path):
    print(f"🚀 AVVIO R2P-GEN PIPELINE SU: {target_image_path}")
    print("="*50)
    
    if not os.path.exists(target_image_path):
        print(f"❌ Errore: File {target_image_path} non trovato.")
        return

    # --- FASE 1: EXTRACTION ---
    # Usiamo il VLM per estrarre le features
    raw_fingerprints, vlm_model = extract_fingerprints(target_image_path)
    
    if not raw_fingerprints:
        print("❌ Pipeline interrotta allo Step 1.")
        return

    # Liberiamo la memoria del VLM per fare spazio a SDXL
    # (È importante sul cluster se la GPU ha < 40GB)
    del vlm_model
    torch.cuda.empty_cache()
    gc.collect()
    print("🧹 VRAM pulita per Generazione.")

    # --- FASE 2: GENERATION ---
    output_filename = f"gen_{os.path.basename(target_image_path)}"
    generate_image(target_image_path, raw_fingerprints, output_path=output_filename)

    # --- FASE 3: VERIFICATION ---
    # Ricarichiamo il VLM (o usiamo un modello diverso se volessimo)
    score, _ = verify_generation(output_filename, target_image_path, raw_fingerprints)

    print("="*50)
    print(f"🏁 REPORT FINALE")
    print(f"   Fingerprints: {raw_fingerprints[:100]}...")
    print(f"   File Generato: {output_filename}")
    print(f"   Quality Score: {score}")
    
    if score > 0.5:
        print("✅ SUCCESS")
    else:
        print("⚠️ FAIL - Refinement necessario")

if __name__ == "__main__":
    # Cambia questo con il file che caricherai sul cluster
    img = "data/perva_test/1.jpg" 
    run_r2p_gen_pipeline(img)