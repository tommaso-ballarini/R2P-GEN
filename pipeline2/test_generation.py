import json
import os
import torch
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm

# --- CONFIGURAZIONE ---
DATABASE_PATH = "database/database_perva.json"  # Il file generato da build_database.py
OUTPUT_DIR = "test_results"
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEVICE = "cuda"

def main():
    # 1. Carica il Database
    print(f"📂 Loading database from {DATABASE_PATH}...")
    if not os.path.exists(DATABASE_PATH):
        print("❌ Database not found! Run build_database.py first.")
        return

    with open(DATABASE_PATH, 'r') as f:
        data = json.load(f)

    concept_dict = data.get("concept_dict", {})
    if not concept_dict:
        print("❌ Concept dictionary is empty.")
        return

    print(f"✅ Found {len(concept_dict)} concepts. Preparing SDXL...")

    # 2. Carica Stable Diffusion XL
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.float16, 
            use_safetensors=True, 
            variant="fp16"
        )
        pipe.to(DEVICE)
    except Exception as e:
        print(f"❌ Error loading SDXL: {e}")
        print("Tip: Ensure you have 'diffusers', 'transformers', 'accelerate' installed and a GPU.")
        return

    # Crea cartella output
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3. Generazione
    print(f"\n🎨 Starting Generation for {len(concept_dict)} items...")
    
    for concept_id, content in tqdm(concept_dict.items()):
        name = content.get('name', 'unknown')
        info = content.get('info', {})
        prompt = info.get('sdxl_prompt', '')

        if not prompt:
            print(f"⚠️ Skipping {name}: No prompt found.")
            continue

        print(f"\n🔹 Generating: {name}")
        print(f"   Prompt: {prompt[:100]}...")  # Stampa solo l'inizio per pulizia

        # Genera immagine (50 step è standard per SDXL)
        image = pipe(
            prompt=prompt,
            num_inference_steps=40,
            guidance_scale=7.5
        ).images[0]

        # Salva
        save_path = os.path.join(OUTPUT_DIR, f"{name}_test.png")
        image.save(save_path)
        print(f"   💾 Saved to: {save_path}")

    print(f"\n✅ All done! Check the '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    main()