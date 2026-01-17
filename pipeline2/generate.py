# generate.py
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from config import Config
from utils2 import cleanup_gpu, print_memory_stats
import os

def generate_image(reference_image_path, fingerprints_dict, output_path="candidate.png", 
                   negative_prompt=None, iteration=1):
    """
    Genera immagine con SDXL + IP-Adapter
    
    Args:
        reference_image_path: Path immagine di riferimento
        fingerprints_dict: Dizionario attributi estratti
        output_path: Dove salvare l'output
        negative_prompt: Negative prompt custom (opzionale)
        iteration: Numero iterazione (per logging)
    """
    print(f"\n🎨 [GENERATE] Generazione Immagine (Iter {iteration})")
    print_memory_stats("PRIMA caricamento pipeline")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Caricamento Pipeline
    print("   -> Caricamento SDXL + IP-Adapter...")
    
    dtype = torch.float16 if Config.USE_FP16 else torch.float32
    
    pipe = StableDiffusionXLPipeline.from_pretrained(
        Config.SDXL_MODEL,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if Config.USE_FP16 else None
    ).to(device)
    
    # Carica IP-Adapter
    pipe.load_ip_adapter(
        Config.IP_ADAPTER_REPO,
        subfolder="sdxl_models",
        weight_name="ip-adapter_sdxl.bin"
    )
    pipe.set_ip_adapter_scale(Config.IP_ADAPTER_SCALE)
    
    print_memory_stats("DOPO caricamento pipeline")
    
    # 2. Preparazione Input
    ref_image = Image.open(reference_image_path).convert("RGB")
    
    # Costruisci prompt
    prompt = build_smart_prompt(fingerprints_dict)
    
    # Usa negative prompt custom o default
    if negative_prompt is None:
        negative_prompt = Config.NEGATIVE_BASE
    
    print(f"   📝 Prompt ({len(prompt)} chars):")
    print(f"      {prompt[:150]}...")
    print(f"   🚫 Negative: {negative_prompt[:100]}...")
    
    # 3. Generazione
    print("   -> Inferenza in corso...")
    generator = torch.Generator(device).manual_seed(Config.SEED)
    
    try:
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                ip_adapter_image=ref_image,
                negative_prompt=negative_prompt,
                num_inference_steps=Config.NUM_INFERENCE_STEPS,
                guidance_scale=Config.GUIDANCE_SCALE,
                generator=generator
            ).images[0]
        
        image.save(output_path)
        print(f"   ✅ Salvata: {output_path}")
        
    except torch.cuda.OutOfMemoryError:
        print("   ⚠️ OOM durante generazione, retry con steps ridotti...")
        cleanup_gpu()
        
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                ip_adapter_image=ref_image,
                negative_prompt=negative_prompt,
                num_inference_steps=20,  # Ridotto
                guidance_scale=Config.GUIDANCE_SCALE,
                generator=generator
            ).images[0]
        
        image.save(output_path)
        print(f"   ✅ Salvata (reduced steps): {output_path}")
    
    # 4. Cleanup
    del pipe
    cleanup_gpu()
    print_memory_stats("DOPO cleanup")
    
    return image


def build_smart_prompt(fp_dict):
    """Costruisce prompt strutturato da fingerprints"""
    prompt_parts = ["A professional studio photograph of a retail product"]
    
    # Attributi principali
    if "brand" in fp_dict and fp_dict["brand"]:
        prompt_parts.append(f"brand: {fp_dict['brand']}")
    
    if "product_type" in fp_dict and fp_dict["product_type"]:
        prompt_parts.append(f"type: {fp_dict['product_type']}")
    
    if "color" in fp_dict and fp_dict["color"]:
        prompt_parts.append(f"color: {fp_dict['color']}")
    
    if "material" in fp_dict and fp_dict["material"]:
        prompt_parts.append(f"material: {fp_dict['material']}")
    
    if "packaging" in fp_dict and fp_dict["packaging"]:
        prompt_parts.append(f"packaging: {fp_dict['packaging']}")
    
    if "distinctive_features" in fp_dict and fp_dict["distinctive_features"]:
        prompt_parts.append(f"features: {fp_dict['distinctive_features']}")
    
    # Fallback su descrizione
    if "description" in fp_dict and len(prompt_parts) == 1:
        prompt_parts.append(fp_dict["description"][:150])
    
    # Qualifiers
    prompt_parts.append("photorealistic, 8k resolution, sharp focus, professional product photography")
    
    return ", ".join(prompt_parts)


if __name__ == "__main__":
    # Test
    dummy_fp = {
        "brand": "Test Brand",
        "color": "blue and white",
        "product_type": "bottle"
    }
    
    test_img = "data/perva_test/1.jpg"
    if os.path.exists(test_img):
        generate_image(test_img, dummy_fp, "test_generate.png")