import torch
import os
from datetime import datetime
from diffusers import StableDiffusionXLPipeline
from diffusers.utils import load_image
from PIL import Image

# --- CONFIGURAZIONE UTENTE ---
# Inserisci qui il percorso della tua immagine (può essere un URL o un percorso locale)
# Esempio URL: "https://raw.githubusercontent.com/tencent-ailab/IP-Adapter/main/assets/images/woman.png"
# Esempio Locale: "C:/Users/NomeUtente/Pictures/mia_foto.jpg"
input_image_path = r"C:\Users\Sport Tech Student\PYTHON_DIRECTORY\FM\4.jpg" 

# Configurazione cartelle modelli
sdxl_local_path = "./models/sdxl-base-1.0"
ip_adapter_local_path = "./models/IP-Adapter"
output_folder = "result_sdxl_IP"

# --- 1. PREPARAZIONE ENVIRONMENT ---
# Crea la cartella di output se non esiste
os.makedirs(output_folder, exist_ok=True)

device = "cuda"
dtype = torch.float16

# --- 2. CARICAMENTO MODELLI ---
print(f"Caricamento SDXL da locale: {sdxl_local_path}...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    sdxl_local_path,
    torch_dtype=dtype,
    variant="fp16",
    use_safetensors=True
)

print(f"Caricamento IP-Adapter da locale: {ip_adapter_local_path}...")
pipe.load_ip_adapter(
    ip_adapter_local_path, 
    subfolder="sdxl_models", 
    weight_name="ip-adapter_sdxl.bin"
)

pipe.enable_model_cpu_offload()

# --- 3. CARICAMENTO IMMAGINE DI RIFERIMENTO ---
print(f"Caricamento immagine di riferimento da: {input_image_path}")

try:
    if input_image_path.startswith("http"):
        # Se è un URL
        ref_image = load_image(input_image_path)
    else:
        # Se è un file locale
        if not os.path.exists(input_image_path):
            raise FileNotFoundError(f"Immagine non trovata nel percorso: {input_image_path}")
        ref_image = Image.open(input_image_path).convert("RGB")
except Exception as e:
    print(f"ERRORE nel caricamento dell'immagine: {e}")
    exit()

# --- 4. DEFINIZIONE PROMPT (Simulazione R2P) ---
category = "clothes"
fingerprint_attributes = ["high quality", "gray", "towel", "floor"]
prompt_text = f"photo of a {category}, {', '.join(fingerprint_attributes)}"
print(f"Prompt Costruito: {prompt_text}")

# --- 5. GENERAZIONE ---
scale = 0.7 
pipe.set_ip_adapter_scale(scale)

print("Generazione in corso...")
generator = torch.Generator(device="cpu").manual_seed(42)

images = pipe(
    prompt=prompt_text,
    ip_adapter_image=ref_image,
    negative_prompt="blurry, low quality, distortion, ugly",
    num_inference_steps=30,
    guidance_scale=7.5,
    generator=generator,
).images

# --- 6. SALVATAGGIO CON TIMESTAMP ---
# Genera timestamp (AnnoMeseGiorno_OraMinutiSecondi)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"output_{timestamp}.png"
output_path = os.path.join(output_folder, filename)

images[0].save(output_path)
print(f"Fatto! Immagine salvata in: {output_path}")