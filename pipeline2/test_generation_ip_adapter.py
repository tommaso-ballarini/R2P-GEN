import json
import os
import torch
import gc
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm

# --- CONFIGURAZIONE ---
DATABASE_PATH = "database/database_perva.json"
OUTPUT_DIR = "test_results_ip_adapter"
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
IP_ADAPTER_ID = "h94/IP-Adapter"
IP_ADAPTER_SCALE = 0.6
DEVICE = "cuda"

def main():
    # Pulizia memoria preventiva
    torch.cuda.empty_cache()
    gc.collect()

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

    print(f"✅ Found {len(concept_dict)} concepts.")

    # 2. Carica SDXL + IP-Adapter (MODALITÀ RISPARMIO MEMORIA)
    print(f"🔌 Loading SDXL with IP-Adapter (Scale: {IP_ADAPTER_SCALE})...")
    try:
        # Carica pipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.float16, 
            use_safetensors=True, 
            variant="fp16"
        )
        
        # Carica IP-Adapter
        pipe.load_ip_adapter(
            IP_ADAPTER_ID, 
            subfolder="sdxl_models", 
            weight_name="ip-adapter_sdxl.bin"
        )
        
        pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)

        # --- OTTIMIZZAZIONI CRITICHE ---
        # Invece di pipe.to("cuda"), usiamo l'offloading.
        # Questo carica i pezzi del modello su GPU solo quando servono.
        print("   -> Enabling Model CPU Offload (Fix for Bus Error)...")
        pipe.enable_model_cpu_offload()
        
        # Abilita VAE Slicing per evitare crash mentre decodifica l'immagine 1024x1024
        pipe.enable_vae_slicing()
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3. Generazione
    print(f"\n🎨 Starting Generation with IP-Adapter...")
    
    for concept_id, content in tqdm(concept_dict.items()):
        name = content.get('name', 'unknown')
        info = content.get('info', {})
        prompt = info.get('sdxl_prompt', '')
        
        img_list = content.get('image', [])
        if not img_list:
            continue
            
        ref_img_path = img_list[0]
        
        if not os.path.exists(ref_img_path):
            print(f"⚠️ Skipping {name}: Reference image not found.")
            continue

        if not prompt:
            print(f"⚠️ Skipping {name}: No prompt found.")
            continue

        # Carica immagine di riferimento
        try:
            ref_image = Image.open(ref_img_path).convert("RGB")
            # Ridimensiona l'immagine di input per non sovraccaricare l'encoder visivo
            ref_image = ref_image.resize((512, 512)) 
        except Exception as e:
            print(f"Error loading image {ref_img_path}: {e}")
            continue

        print(f"\n🔹 Generating: {name}")

        with torch.inference_mode():
            generated_image = pipe(
                prompt=prompt,
                ip_adapter_image=ref_image,
                height=1024,
                width=1024,
                num_inference_steps=40,
                guidance_scale=7.5
            ).images[0]

        save_path = os.path.join(OUTPUT_DIR, f"{name}_ipa.png")
        generated_image.save(save_path)
        print(f"   💾 Saved to: {save_path}")
        
        # Pulizia dopo ogni generazione
        torch.cuda.empty_cache()

    print(f"\n✅ All done! Check '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()