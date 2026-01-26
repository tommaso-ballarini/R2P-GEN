import json
import os
import torch
import gc
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm

# --- CONFIGURAZIONE ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(SCRIPT_DIR, "database/database_perva_train_1_clip.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "test_results_ip_adapter")
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
IP_ADAPTER_ID = "h94/IP-Adapter"
DEVICE = "cuda"

# IP-Adapter Scaling Strategy
USE_LAYERWISE_SCALING = True  # True = Layer-wise (ignore background); False = Global scale

# Global scale (used if USE_LAYERWISE_SCALING = False)
IP_ADAPTER_SCALE_GLOBAL = 0.6

# Layer-wise weights (used if USE_LAYERWISE_SCALING = True)
# Strategy: Minimize background contamination from reference, maximize identity preservation
# Based on InstantStyle research: separate spatial structure (down) from texture/color (up)
# IP_ADAPTER_LAYER_WEIGHTS = {
#     "down": {
#         "block_0": [0.0, 0.0],      # NO background/layout from reference
#         "block_1": [0.0, 0.0],      # NO composition from reference
#         "block_2": [0.1, 0.1],      # Minimal structural hint only
#     },
#     "mid": 0.5,                     # Balanced semantic features
#     "up": {
#         "block_0": [0.9, 0.9, 0.9], # MAXIMUM texture/color injection (fingerprints!)
#         "block_1": [0.8, 0.8, 0.8], # High material fidelity
#         "block_2": [0.7, 0.7],      # Fine details preservation
#         "block_3": [0.6],           # Final refinement
#     }
# }

# Optimized for R2P: Maximum identity preservation, zero background leakage
# Based on Gemini proposal + architectural corrections for SDXL
IP_ADAPTER_LAYER_WEIGHTS = {
    "down": {
        "block_0": [0.0, 0.0],         # Zero background from reference
        "block_1": [0.0, 0.0],         # Zero composition from reference
        "block_2": [0.4, 0.7],         # Object shape preservation (smooth gradient)
    },
    "mid": 0.9,                        # Very high semantic identity (flexible)
    "up": {
        "block_0": [0.6, 0.8, 0.9],    # Texture/color injection (3 layers, smooth gradient)
        "block_1": [0.95, 0.95, 0.95], # Maximum material fidelity (3 layers)
        "block_2": [0.85, 0.85],       # Fine details preservation (2 layers)
        "block_3": [0.7],              # Final refinement (1 layer)
    }
}



# Negative Prompt Configuration
# Prevents background contamination from reference image and common SDXL artifacts
# Critical for layer-wise scaling: reinforces down blocks=0.0 to eliminate leakage
NEGATIVE_PROMPT = (
    "blurry, low quality, low resolution, distorted, deformed, "
    "(background contamination:1.3), (reference background leakage:1.2), "
    "(original background visible:1.2), "
    "artifact, watermark, text overlay, logo overlay, signature, "
    "oversaturated, overexposed, underexposed, noise, grain, "
    "worst quality, jpeg artifacts, duplicate, cropped, "
    "unrealistic proportions, anatomical errors, "
    "blur, out of focus"
)

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
    print(f"🔌 Loading SDXL with IP-Adapter...")
    
    if USE_LAYERWISE_SCALING:
        print(f"   🎨 Strategy: Layer-Wise Scaling (R2P Optimized)")
        print(f"      → Down Blocks: 0.0-0.7 (zero background, shape preservation)")
        print(f"      → Mid Block: 0.9 (very high semantic identity)")
        print(f"      → Up Blocks: 0.6-0.95 (maximum texture/material fidelity)")
    else:
        print(f"   🎨 Strategy: Global Scale {IP_ADAPTER_SCALE_GLOBAL}")
    
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
        
        # 🔥 APPLY SCALING STRATEGY
        if USE_LAYERWISE_SCALING:
            pipe.set_ip_adapter_scale(IP_ADAPTER_LAYER_WEIGHTS)
            print("   ✅ Layer-wise weights applied successfully")
        else:
            pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE_GLOBAL)
            print(f"   ✅ Global scale {IP_ADAPTER_SCALE_GLOBAL} applied")

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
        print(f"   Prompt: {prompt[:100]}...")

        with torch.inference_mode():
            generated_image = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                ip_adapter_image=ref_image,
                height=1024,
                width=1024,
                num_inference_steps=40,
                guidance_scale=7.5
            ).images[0]

        # Add method suffix to filename for A/B comparison
        method_suffix = "layerwise" if USE_LAYERWISE_SCALING else "global"
        save_path = os.path.join(OUTPUT_DIR, f"{name}_ipa_{method_suffix}.png")
        generated_image.save(save_path)
        print(f"   💾 Saved to: {save_path}")
        
        # Pulizia dopo ogni generazione
        torch.cuda.empty_cache()

    print(f"\n✅ All done! Check '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()