# extraction_v2.py
"""
Sistema di estrazione fingerprints con categorizzazione CLIP
Ispirato a R2P ma ottimizzato per generazione
"""
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from config import Config

class CategoryAwareExtractor:
    """Estrae fingerprints adattivi alla categoria dell'oggetto"""
    
    # Template gerarchici per categoria
    CATEGORY_TEMPLATES = {
        "retail_packaged": {
            "critical": ["brand", "packaging", "product_type"],
            "important": ["color", "text_visible", "logo_position"],
            "optional": ["size_appearance", "material"]
        },
        "clothing": {
            "critical": ["garment_type", "pattern", "primary_color"],
            "important": ["fabric_texture", "neckline", "sleeve_length"],
            "optional": ["brand_visible", "size_label"]
        },
        "electronics": {
            "critical": ["device_type", "brand", "model_indicators"],
            "important": ["color", "screen_presence", "button_layout"],
            "optional": ["cables_visible", "condition"]
        },
        "furniture": {
            "critical": ["furniture_type", "material", "primary_color"],
            "important": ["style", "legs_shape", "surface_texture"],
            "optional": ["brand_visible", "hardware_color"]
        },
        "generic": {
            "critical": ["object_type", "primary_color", "shape"],
            "important": ["material", "distinctive_features"],
            "optional": ["brand", "text_visible"]
        }
    }
    
    def __init__(self, vlm_model, tokenizer):
        self.vlm = vlm_model
        self.tokenizer = tokenizer
        
        # CLIP per categorizzazione
        self.clip_model = CLIPModel.from_pretrained(
            'openai/clip-vit-large-patch14-336'
        ).to("cuda")
        self.clip_processor = CLIPProcessor.from_pretrained(
            'openai/clip-vit-large-patch14-336'
        )
        
    def detect_category(self, image: Image.Image) -> str:
        """Classifica immagine in macro-categoria"""
        categories = list(self.CATEGORY_TEMPLATES.keys())
        category_prompts = [
            f"A photo of a {cat.replace('_', ' ')}" 
            for cat in categories
        ]
        
        inputs = self.clip_processor(
            text=category_prompts,
            images=image,
            return_tensors="pt",
            padding=True
        ).to("cuda")
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
        
        best_idx = probs.argmax().item()
        confidence = probs[0, best_idx].item()
        
        detected_cat = categories[best_idx]
        
        # Fallback a generic se confidenza bassa
        if confidence < 0.4:
            detected_cat = "generic"
            
        return detected_cat
    
    def extract_fingerprints(self, image_path: str) -> dict:
        """Estrazione adattiva con prompt gerarchico"""
        image = Image.open(image_path).convert('RGB')
        
        # 1. Rileva categoria
        category = self.detect_category(image)
        template = self.CATEGORY_TEMPLATES[category]
        
        print(f"   🔍 Detected category: {category}")
        
        # 2. Costruisci prompt JSON gerarchico
        json_structure = {
            level: {attr: f"<{attr} value>" for attr in attrs}
            for level, attrs in template.items()
        }
        
        prompt = f"""Analyze this {category.replace('_', ' ')} image.
Extract visual attributes following this HIERARCHICAL structure:

CRITICAL attributes (must be accurate):
{self._format_attributes(template['critical'])}

IMPORTANT attributes (good to have):
{self._format_attributes(template['important'])}

OPTIONAL attributes (if clearly visible):
{self._format_attributes(template['optional'])}

Return ONLY valid JSON:
{json_structure}

Be specific and objective. Focus on visual facts, not interpretations."""
        
        msgs = [{'role': 'user', 'content': [image, prompt]}]
        
        # 3. VLM Inference
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                result = self.vlm.chat(
                    image=image,
                    msgs=msgs,
                    tokenizer=self.tokenizer
                )
        
        raw_response = result[-1] if isinstance(result, tuple) else str(result)
        
        # 4. Parsing robusto
        fingerprints = self._parse_hierarchical_json(raw_response, template)
        fingerprints['_category'] = category  # Metadata
        
        return fingerprints
    
    def _format_attributes(self, attrs: list) -> str:
        """Formatta lista attributi con descrizioni"""
        descriptions = {
            "brand": "Brand name or logo text",
            "packaging": "Type of packaging (box, bottle, bag, etc.)",
            "product_type": "Specific product category",
            "color": "Dominant colors (comma-separated)",
            "text_visible": "Any visible text or labels",
            "logo_position": "Location of brand logo",
            "garment_type": "Type of clothing item",
            "pattern": "Visual pattern (solid, striped, floral, etc.)",
            "fabric_texture": "Texture appearance (smooth, ribbed, etc.)",
            "device_type": "Type of electronic device",
            "model_indicators": "Model number or distinguishing features",
            "furniture_type": "Specific furniture category",
            "style": "Design style (modern, vintage, etc.)",
            "object_type": "General object category",
            "shape": "Overall geometric shape",
            "distinctive_features": "Unique identifying characteristics"
        }
        
        return "\n".join([
            f"  - {attr}: {descriptions.get(attr, 'Describe this attribute')}"
            for attr in attrs
        ])
    
    def _parse_hierarchical_json(self, raw_text: str, template: dict) -> dict:
        """Parse con fallback gerarchico"""
        import json
        import re
        
        # Rimuovi markdown
        clean_text = re.sub(r'```json|```', '', raw_text).strip()
        
        try:
            parsed = json.loads(clean_text)
            
            # Valida struttura gerarchica
            result = {}
            for level in ['critical', 'important', 'optional']:
                if level in parsed:
                    result.update(parsed[level])
            
            return result if result else parsed
            
        except json.JSONDecodeError:
            # Fallback: estrai coppie key-value manualmente
            print("   ⚠️ JSON parsing failed, using regex fallback")
            result = {}
            
            for level, attrs in template.items():
                for attr in attrs:
                    pattern = rf'"{attr}":\s*"([^"]+)"'
                    match = re.search(pattern, clean_text)
                    if match:
                        result[attr] = match.group(1)
            
            return result


# Test usage
if __name__ == "__main__":
    from models.mini_cpm_reasoning import MiniCPMReasoning
    
    reasoner = MiniCPMReasoning()
    extractor = CategoryAwareExtractor(
        reasoner.model_interface.model,
        reasoner.model_interface.tokenizer
    )
    
    test_img = "data/perva_test/1.jpg"
    fingerprints = extractor.extract_fingerprints(test_img)
    
    print("\n📋 Extracted Fingerprints:")
    for k, v in fingerprints.items():
        print(f"   {k}: {v}")