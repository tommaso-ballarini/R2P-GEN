# generation_v2.py
"""
Sistema di generazione con prompt gerarchici e IP-Adapter adattivo
"""
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from config import Config

class AdaptiveGenerator:
    """Generatore che bilancia IP-Adapter e text guidance"""
    
    # Pesi IP-Adapter per categoria (tuning empirico necessario)
    IP_ADAPTER_WEIGHTS = {
        "retail_packaged": 0.5,   # Basso: privilegia text (brand/text)
        "clothing": 0.7,          # Alto: privilegia visual (pattern/texture)
        "electronics": 0.6,       # Medio: bilancia forma e dettagli
        "furniture": 0.65,        # Medio-alto: privilegia materiale/stile
        "generic": 0.6            # Default
    }
    
    def __init__(self):
        self.pipe = None
        self.current_category = None
        
    def load_pipeline(self, category: str):
        """Carica pipeline con IP-Adapter weight adattivo"""
        dtype = torch.float16 if Config.USE_FP16 else torch.float32
        
        if self.pipe is None:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                Config.SDXL_MODEL,
                torch_dtype=dtype,
                use_safetensors=True,
                variant="fp16" if Config.USE_FP16 else None
            ).to("cuda")
            
            self.pipe.load_ip_adapter(
                Config.IP_ADAPTER_REPO,
                subfolder="sdxl_models",
                weight_name="ip-adapter_sdxl.bin"
            )
        
        # Aggiorna peso IP-Adapter
        ip_weight = self.IP_ADAPTER_WEIGHTS.get(category, 0.6)
        self.pipe.set_ip_adapter_scale(ip_weight)
        self.current_category = category
        
        print(f"   ⚙️ IP-Adapter weight: {ip_weight} (category: {category})")
    
    def build_hierarchical_prompt(self, fingerprints: dict, category: str) -> str:
        """Costruisce prompt con priorità gerarchica"""
        
        # Template per categoria
        category_prefixes = {
            "retail_packaged": "Professional product photography of a packaged retail item",
            "clothing": "High-quality fashion photography of a garment",
            "electronics": "Clean product shot of an electronic device",
            "furniture": "Interior design photograph of a furniture piece",
            "generic": "Professional studio photograph of an object"
        }
        
        base = category_prefixes.get(category, category_prefixes["generic"])
        
        # Attributi critici (massima priorità)
        critical_parts = []
        critical_attrs = ["brand", "product_type", "garment_type", "device_type", 
                         "furniture_type", "object_type", "packaging"]
        
        for attr in critical_attrs:
            if attr in fingerprints and fingerprints[attr]:
                critical_parts.append(f"{fingerprints[attr]}")
        
        # Attributi importanti (media priorità)
        important_parts = []
        important_attrs = ["color", "pattern", "material", "primary_color", 
                          "fabric_texture", "style"]
        
        for attr in important_attrs:
            if attr in fingerprints and fingerprints[attr]:
                important_parts.append(f"{attr.replace('_', ' ')}: {fingerprints[attr]}")
        
        # Attributi distintivi
        distinctive = []
        if "distinctive_features" in fingerprints and fingerprints["distinctive_features"]:
            distinctive.append(fingerprints["distinctive_features"])
        
        if "text_visible" in fingerprints and fingerprints["text_visible"]:
            distinctive.append(f"with visible text: '{fingerprints['text_visible']}'")
        
        if "logo_position" in fingerprints and fingerprints["logo_position"]:
            distinctive.append(f"logo positioned {fingerprints['logo_position']}")
        
        # Assembla prompt gerarchico
        prompt_parts = [base]
        
        if critical_parts:
            prompt_parts.append(", ".join(critical_parts))
        
        if important_parts:
            prompt_parts.append(", ".join(important_parts))
        
        if distinctive:
            prompt_parts.append(", ".join(distinctive))
        
        # Qualifiers finali
        prompt_parts.append("photorealistic, 8k, sharp focus, professional product photography")
        
        final_prompt = ", ".join(prompt_parts)
        
        # Limita lunghezza (SDXL max ~77 tokens)
        if len(final_prompt) > 400:
            final_prompt = final_prompt[:400] + "..."
        
        return final_prompt
    
    def generate(
        self,
        reference_image_path: str,
        fingerprints: dict,
        output_path: str,
        negative_prompt: str = None,
        iteration: int = 1
    ):
        """Genera con parametri adattivi"""
        
        # Estrai categoria dai fingerprints
        category = fingerprints.get('_category', 'generic')
        
        # Carica pipeline con pesi adattivi
        self.load_pipeline(category)
        
        # Costruisci prompt
        prompt = self.build_hierarchical_prompt(fingerprints, category)
        
        # Negative prompt adattivo
        if negative_prompt is None:
            negative_prompt = self._get_category_negative(category)
        
        print(f"\n   📝 Prompt ({len(prompt)} chars):")
        print(f"      {prompt[:150]}...")
        print(f"   🚫 Negative: {negative_prompt[:100]}...")
        
        # Carica reference image
        ref_image = Image.open(reference_image_path).convert("RGB")
        
        # Resize se necessario
        max_dim = 1024  # SDXL native resolution
        if max(ref_image.size) > max_dim:
            ratio = max_dim / max(ref_image.size)
            new_size = tuple(int(dim * ratio) for dim in ref_image.size)
            ref_image = ref_image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Genera
        generator = torch.Generator("cuda").manual_seed(Config.SEED)
        
        try:
            with torch.no_grad():
                result = self.pipe(
                    prompt=prompt,
                    ip_adapter_image=ref_image,
                    negative_prompt=negative_prompt,
                    num_inference_steps=Config.NUM_INFERENCE_STEPS,
                    guidance_scale=Config.GUIDANCE_SCALE,
                    generator=generator,
                    height=1024,
                    width=1024
                ).images[0]
            
            result.save(output_path)
            print(f"   ✅ Saved: {output_path}")
            
        except torch.cuda.OutOfMemoryError:
            print("   ⚠️ OOM, retrying with lower resolution...")
            
            # Fallback
            with torch.no_grad():
                result = self.pipe(
                    prompt=prompt,
                    ip_adapter_image=ref_image,
                    negative_prompt=negative_prompt,
                    num_inference_steps=20,
                    guidance_scale=Config.GUIDANCE_SCALE,
                    generator=generator,
                    height=768,
                    width=768
                ).images[0]
            
            result.save(output_path)
            print(f"   ✅ Saved (reduced): {output_path}")
        
        return result
    
    def _get_category_negative(self, category: str) -> str:
        """Negative prompts specifici per categoria"""
        base = Config.NEGATIVE_BASE
        
        category_negatives = {
            "retail_packaged": "damaged packaging, missing labels, blurry text, wrong brand",
            "clothing": "wrong pattern, incorrect fabric, mismatched colors, stains",
            "electronics": "wrong model, missing buttons, incorrect screen, broken",
            "furniture": "wrong material, incorrect style, damaged surface",
            "generic": "wrong object type, incorrect shape"
        }
        
        specific = category_negatives.get(category, "")
        return f"{base}, {specific}" if specific else base
    
    def cleanup(self):
        """Libera memoria"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            torch.cuda.empty_cache()


# Test
if __name__ == "__main__":
    generator = AdaptiveGenerator()
    
    test_fp = {
        "_category": "retail_packaged",
        "brand": "Lakme",
        "product_type": "sunscreen tube",
        "packaging": "orange plastic tube",
        "color": "orange, white",
        "text_visible": "SUN EXPERT SPF 50+",
        "distinctive_features": "UV protection branding"
    }
    
    generator.generate(
        "data/perva_test/1.jpg",
        test_fp,
        "output/test_adaptive.png"
    )