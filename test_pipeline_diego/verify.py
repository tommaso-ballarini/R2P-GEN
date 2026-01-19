# verification_v2.py
"""
Sistema di verifica multi-metrica con scoring quantitativo
"""
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from torchmetrics.image.lpips import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
import numpy as np

class MultiMetricVerifier:
    """Verifica generazione con metriche complementari"""
    
    def __init__(self, vlm_model, tokenizer):
        self.vlm = vlm_model
        self.tokenizer = tokenizer
        
        # CLIP per similarity
        self.clip = CLIPModel.from_pretrained(
            'openai/clip-vit-large-patch14-336'
        ).to("cuda")
        self.clip_processor = CLIPProcessor.from_pretrained(
            'openai/clip-vit-large-patch14-336'
        )
        
        # LPIPS per perceptual similarity
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to("cuda")
        
        # FID (opzionale, richiede batch)
        self.fid = FrechetInceptionDistance(feature=2048).to("cuda")
        
    def verify(
        self,
        generated_image_path: str,
        reference_image_path: str,
        fingerprints: dict
    ) -> dict:
        """Verifica completa con scoring multi-dimensionale"""
        
        gen_img = Image.open(generated_image_path).convert('RGB')
        ref_img = Image.open(reference_image_path).convert('RGB')
        
        # Resize per metriche
        target_size = (512, 512)
        gen_resized = gen_img.resize(target_size, Image.Resampling.LANCZOS)
        ref_resized = ref_img.resize(target_size, Image.Resampling.LANCZOS)
        
        results = {}
        
        # === 1. CLIP Similarity (0-1, higher better) ===
        clip_score = self._compute_clip_similarity(gen_resized, ref_resized)
        results['clip_similarity'] = clip_score
        
        # === 2. LPIPS Distance (0-1, lower better) ===
        lpips_dist = self._compute_lpips(gen_resized, ref_resized)
        results['lpips_distance'] = lpips_dist
        
        # === 3. Attribute Verification (VLM-based) ===
        attr_scores = self._verify_attributes_batch(gen_img, fingerprints)
        results['attribute_scores'] = attr_scores
        results['attribute_accuracy'] = np.mean(list(attr_scores.values()))
        
        # === 4. Text Presence (per retail products) ===
        if 'text_visible' in fingerprints and fingerprints['text_visible']:
            text_score = self._verify_text_presence(gen_img, fingerprints['text_visible'])
            results['text_verification'] = text_score
        
        # === 5. Composite Score ===
        results['composite_score'] = self._compute_composite_score(results)
        
        # === 6. Missing Attributes ===
        results['missing_attributes'] = [
            attr for attr, score in attr_scores.items() if score < 0.5
        ]
        
        return results
    
    def _compute_clip_similarity(self, gen_img: Image.Image, ref_img: Image.Image) -> float:
        """CLIP cosine similarity tra immagini"""
        inputs = self.clip_processor(
            images=[gen_img, ref_img],
            return_tensors="pt",
            padding=True
        ).to("cuda")
        
        with torch.no_grad():
            image_features = self.clip.get_image_features(**inputs)
            image_features = F.normalize(image_features, p=2, dim=1)
        
        similarity = F.cosine_similarity(
            image_features[0:1],
            image_features[1:2]
        ).item()
        
        return similarity
    
    def _compute_lpips(self, gen_img: Image.Image, ref_img: Image.Image) -> float:
        """LPIPS perceptual distance"""
        import torchvision.transforms as T
        
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        gen_tensor = transform(gen_img).unsqueeze(0).to("cuda")
        ref_tensor = transform(ref_img).unsqueeze(0).to("cuda")
        
        with torch.no_grad():
            distance = self.lpips(gen_tensor, ref_tensor).item()
        
        return distance
    
    def _verify_attributes_batch(self, gen_img: Image.Image, fingerprints: dict) -> dict:
        """Verifica attributi con singola chiamata VLM"""
        
        # Filtra attributi critici
        critical_attrs = {
            k: v for k, v in fingerprints.items()
            if k in ['brand', 'product_type', 'color', 'pattern', 
                    'garment_type', 'device_type', 'furniture_type',
                    'packaging', 'text_visible']
            and v and k != '_category'
        }
        
        if not critical_attrs:
            return {}
        
        # Costruisci prompt batch
        attrs_list = "\n".join([
            f"- {k.replace('_', ' ')}: {v}"
            for k, v in critical_attrs.items()
        ])
        
        prompt = f"""Verify if this image shows these attributes:

{attrs_list}

For each attribute, answer with a confidence score (0.0 to 1.0):
- 1.0 = clearly present and accurate
- 0.5 = partially present or uncertain
- 0.0 = absent or incorrect

Return ONLY a JSON object with scores:
{{
{chr(10).join([f'  "{k}": <score>' for k in critical_attrs.keys()])}
}}"""
        
        msgs = [{'role': 'user', 'content': [gen_img, prompt]}]
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                result = self.vlm.chat(
                    image=gen_img,
                    msgs=msgs,
                    tokenizer=self.tokenizer
                )
        
        response = result[-1] if isinstance(result, tuple) else str(result)
        
        # Parse scores
        import json
        import re
        
        try:
            clean_response = re.sub(r'```json|```', '', response).strip()
            scores = json.loads(clean_response)
            
            # Converti stringhe in float
            for k in scores:
                try:
                    scores[k] = float(scores[k])
                except:
                    scores[k] = 0.5  # Default
            
            return scores
            
        except:
            print("   ⚠️ Failed to parse VLM scores, using fallback")
            # Fallback: assegna 0.5 a tutti
            return {k: 0.5 for k in critical_attrs.keys()}
    
    def _verify_text_presence(self, gen_img: Image.Image, expected_text: str) -> float:
        """Verifica presenza text con OCR o VLM"""
        
        prompt = f"""Does this image contain the text: "{expected_text}"?

Answer with:
- 1.0 if the exact text is clearly visible
- 0.7 if similar text is visible
- 0.3 if text is partially visible or unclear
- 0.0 if no matching text is present

Return ONLY a number (0.0 to 1.0)."""
        
        msgs = [{'role': 'user', 'content': [gen_img, prompt]}]
        
        with torch.no_grad():
            result = self.vlm.chat(
                image=gen_img,
                msgs=msgs,
                tokenizer=self.tokenizer
            )
        
        response = result[-1] if isinstance(result, tuple) else str(result)
        
        # Estrai score
        import re
        match = re.search(r'(0\.\d+|1\.0)', response)
        if match:
            return float(match.group(1))
        
        return 0.5  # Default
    
    def _compute_composite_score(self, results: dict) -> float:
        """Score composito pesato"""
        
        weights = {
            'clip_similarity': 0.3,      # Visual similarity
            'lpips_distance': 0.2,       # Perceptual quality (inverso)
            'attribute_accuracy': 0.4,   # Attribute correctness
            'text_verification': 0.1     # Text presence (se presente)
        }
        
        score = 0.0
        total_weight = 0.0
        
        # CLIP similarity
        score += results['clip_similarity'] * weights['clip_similarity']
        total_weight += weights['clip_similarity']
        
        # LPIPS (inverti: lower is better)
        lpips_score = 1.0 - min(results['lpips_distance'], 1.0)
        score += lpips_score * weights['lpips_distance']
        total_weight += weights['lpips_distance']
        
        # Attributes
        score += results['attribute_accuracy'] * weights['attribute_accuracy']
        total_weight += weights['attribute_accuracy']
        
        # Text (opzionale)
        if 'text_verification' in results:
            score += results['text_verification'] * weights['text_verification']
            total_weight += weights['text_verification']
        
        return score / total_weight if total_weight > 0 else 0.0


# Test
if __name__ == "__main__":
    from models.mini_cpm_reasoning import MiniCPMReasoning
    
    reasoner = MiniCPMReasoning()
    verifier = MultiMetricVerifier(
        reasoner.model_interface.model,
        reasoner.model_interface.tokenizer
    )
    
    test_fp = {
        "brand": "Lakme",
        "product_type": "sunscreen",
        "color": "orange",
        "text_visible": "SPF 50+"
    }
    
    results = verifier.verify(
        "output/test.png",
        "data/perva_test/1.jpg",
        test_fp
    )
    
    print("\n📊 Verification Results:")
    for k, v in results.items():
        if k != 'attribute_scores':
            print(f"   {k}: {v}")