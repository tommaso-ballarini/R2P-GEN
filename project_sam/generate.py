import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline

def generate_image(reference_image_path, fingerprints_text, output_path="candidate.png"):
    """
    Genera immagine usando Reference Image (IP-Adapter) + Fingerprints (Prompt).
    """
    print(f"\n🎨 [STEP 2] Generazione Ibrida SDXL + IP-Adapter")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Configurazione Modelli
    base_model = "stabilityai/stable-diffusion-xl-base-1.0"
    ip_adapter_repo = "h94/IP-Adapter"
    
    print("   -> Caricamento Pipeline Generativa...")
    # Usiamo float16 per velocità e memoria su GPU Cluster
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    ).to(device)
    
    # Carica IP-Adapter specifico per SDXL
    pipe.load_ip_adapter(
        ip_adapter_repo, 
        subfolder="sdxl_models", 
        weight_name="ip-adapter_sdxl.bin"
    )
    # Scala 0.6: Buon compromesso tra fedeltà visiva e obbedienza al testo
    pipe.set_ip_adapter_scale(0.6)
    
    # 2. Preparazione Input
    ref_image = Image.open(reference_image_path).convert("RGB")
    
    # Costruiamo il prompt unendo la richiesta generica con i dettagli estratti
    prompt = (
        f"A high quality professional studio photo of a product, "
        f"distinctive features: {fingerprints_text}, "
        f"photorealistic, 8k, sharp focus"
    )
    neg_prompt = "blurry, distortion, bad anatomy, watermark, text overlay, low quality, deformation"
    
    # 3. Generazione
    print("   -> Generazione in corso...")
    generator = torch.Generator(device).manual_seed(42)
    
    image = pipe(
        prompt=prompt,
        ip_adapter_image=ref_image,
        negative_prompt=neg_prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=generator
    ).images[0]
    
    image.save(output_path)
    print(f"   ✅ Immagine generata salvata in: {output_path}")
    
    # Pulizia memoria GPU (Cruciale per non crashare nello step successivo)
    del pipe
    torch.cuda.empty_cache()
    
    return image

if __name__ == "__main__":
    # Test Standalone
    dummy_fingerprints = "blue package, white logo, code RVG"
    test_img = "data/perva_test/1.jpg"
    if os.path.exists(test_img):
        generate_image(test_img, dummy_fingerprints, "test_step2.png")