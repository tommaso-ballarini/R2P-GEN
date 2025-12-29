import torch
from diffusers import StableDiffusionXLPipeline, EDMDPMSolverMultistepScheduler
from diffusers.utils import load_image
from PIL import Image

# --- 1. CONFIGURAZIONE HARDWARE ---
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 # Usiamo fp16 per risparmiare VRAM e velocizzare

# --- 2. CARICAMENTO MODELLO BASE (SDXL 1.0) ---
print("Caricamento SDXL...")
model_id = "stabilityai/stable-diffusion-xl-base-1.0"

# Carichiamo la pipeline standard
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    torch_dtype=dtype,
    variant="fp16",
    use_safetensors=True
)

# --- 3. CARICAMENTO IP-ADAPTER ---
print("Caricamento IP-Adapter...")
# Hugging Face ha i pesi ufficiali pronti per SDXL
# "h94/IP-Adapter" è la repo ufficiale convertita per diffusers
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")

# OTTIMIZZAZIONE MEMORIA (CRITICO per la tua 4090 Mobile)
# Questo sposta i modelli tra CPU e GPU dinamicamente. Rallenta leggermente
# ma ti permette di non andare Out Of Memory (OOM).
pipe.enable_model_cpu_offload() 

# --- 4. SIMULAZIONE LOGICA R2P ---

# A) L'immagine di riferimento (Il "Visual Prompt")
# Per il test usiamo un'immagine URL, poi userai le tue locali
image_url = "https://raw.githubusercontent.com/tencent-ailab/IP-Adapter/main/assets/images/woman.png"
ref_image = load_image(image_url)

# B) Gli attributi R2P (Il "Textual Fingerprint")
# Immagina che questi vengano dal tuo JSON R2P
category = "woman"
fingerprint_attributes = ["high quality", "red dress", "looking at camera", "studio lighting"]

# Costruzione del prompt combinato
prompt_text = f"photo of a {category}, {', '.join(fingerprint_attributes)}"
print(f"Prompt Costruito: {prompt_text}")

# --- 5. GENERAZIONE ---
# ip_adapter_scale: Quanto deve "pesare" l'immagine rispetto al testo.
# 0.5 = ibrido bilanciato
# 1.0 = massima fedeltà all'immagine di riferimento
scale = 0.7 
pipe.set_ip_adapter_scale(scale)

print("Generazione in corso...")
generator = torch.Generator(device="cpu").manual_seed(42) # Seed fisso per riproducibilità

images = pipe(
    prompt=prompt_text,
    ip_adapter_image=ref_image,
    negative_prompt="blurry, low quality, distortion, ugly",
    num_inference_steps=30,
    guidance_scale=7.5,
    generator=generator,
).images

# --- 6. SALVATAGGIO ---
images[0].save("lab_output.png")
print("Fatto! Immagine salvata come lab_output.png")