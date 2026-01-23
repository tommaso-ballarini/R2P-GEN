import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# =============================================================================
# SEZIONE 1: CLIP SCORE CALCULATOR (Per Cross-Modal Attribute Verification - Eq. 5)
# =============================================================================

class ClipScoreCalculator:
    def __init__(self, device="cuda", model_name="openai/clip-vit-large-patch14-336"):
        """
        Gestisce il caricamento e il calcolo degli score con CLIP.
        """
        self.device = device
        print(f"Loading CLIP model: {model_name}...")
        self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.feature_extractor = CLIPProcessor.from_pretrained(model_name)
        self.clip_model.eval()

    def format_text_for_clip(self, sentence, category):
        """
        Formats text into CLIP-compatible prompts.
        Original logic from R2P to maintain consistency.
        """
        # Se la frase è una lista di attributi separati da virgola
        return [f"A photo of a {category} with {item.strip()}" for item in sentence.split(',')]

    def get_clip_image_feature(self, image_path_or_obj):
        """Extracts and normalizes CLIP image features."""
        if isinstance(image_path_or_obj, str):
            image = Image.open(image_path_or_obj).convert('RGB')
        else:
            image = image_path_or_obj.convert('RGB')

        inputs = self.feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(inputs['pixel_values'].to(self.device))
        return image_features / image_features.norm(p=2, dim=-1, keepdim=True)

    def get_text_features(self, formatted_sentences):
        """Extracts and normalizes CLIP text features."""
        inputs = self.feature_extractor(text=formatted_sentences, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(inputs["input_ids"], inputs["attention_mask"])
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)   
        return text_features.detach()

    def compute_attribute_score(self, image, attribute_list):
        """
        Computes the average CLIP similarity score AND breakdown per attribute.
        """
        if not attribute_list:
            return 0.0, {}
            
        if len(attribute_list) > 20: # Aumentato limite per sicurezza
            attribute_list = attribute_list[:20]

        try:
            # 1. Get Text Embeddings
            text_features = self.get_text_features(attribute_list) 
            
            # 2. Get Image Embeddings
            image_features = self.get_clip_image_feature(image)

            # 3. Expand image features
            image_features = image_features.expand(text_features.size(0), -1)

            # 4. Compute Cosine Similarity
            similarities = torch.sum(image_features * text_features, dim=1)
            
            # 5. Prepare Breakdown
            # Convert tensor to list of floats
            scores_list = similarities.detach().cpu().tolist()
            
            # Create dictionary { "attribute text": 0.245 }
            breakdown = {attr: score for attr, score in zip(attribute_list, scores_list)}
            
            # Return Mean AND Breakdown
            return torch.mean(similarities).item(), breakdown
            
        except Exception as e:
            print(f"Error in CLIP calculation: {e}")
            return 0.0, {}
            
    

# =============================================================================
# SEZIONE 2: CONFIDENCE CALCULATOR (Per Pairwise Reasoning - Eq. 7 & 8)
# =============================================================================

import torch
import numpy as np

class ConfidenceCalculator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
        # Pre-compute token IDs for common answers
        self.yes_token_id = self.tokenizer.convert_tokens_to_ids("Yes")
        self.no_token_id = self.tokenizer.convert_tokens_to_ids("No")
        
        # Alternative capitalizations (alcuni tokenizer sono case-sensitive)
        self.yes_variants = [
            self.tokenizer.convert_tokens_to_ids(v) 
            for v in ["Yes", "yes", "YES", "▁Yes", "▁yes"]
            if v in self.tokenizer.get_vocab()
        ]
        self.no_variants = [
            self.tokenizer.convert_tokens_to_ids(v) 
            for v in ["No", "no", "NO", "▁No", "▁no"]
            if v in self.tokenizer.get_vocab()
        ]
        
        print(f"🔧 ConfidenceCalculator initialized:")
        print(f"   Yes token IDs: {self.yes_variants}")
        print(f"   No token IDs: {self.no_variants}")

    def calculate_binary_confidence(self, outputs, response_text=None):
        """
        Calcola la confidence per risposte binarie Yes/No.
        
        Args:
            outputs: Output del modello (può essere dict, ModelOutput, o altro)
            response_text: Testo della risposta (usato come fallback)
            
        Returns:
            dict con:
                - yes_confidence: probabilità di "Yes" (0-1)
                - no_confidence: probabilità di "No" (0-1)
                - confidence_score: probabilità del token scelto
                - chosen_answer: "yes" o "no"
                - method: "logits" o "text_fallback"
        """
        
        # =====================================================================
        # STEP 1: Estrai sequences e scores/logits
        # =====================================================================
        sequences = None
        scores = None
        
        # Gestione vari formati di output
        if isinstance(outputs, dict):
            sequences = outputs.get("sequences")
            scores = outputs.get("scores") or outputs.get("logits")
        elif hasattr(outputs, "sequences"):
            sequences = outputs.sequences
            scores = getattr(outputs, "scores", None) or getattr(outputs, "logits", None)
        elif isinstance(outputs, torch.Tensor):
            sequences = outputs
            
        # Se non abbiamo gli scores, usa il fallback testuale
        if scores is None:
            return self._text_fallback(response_text)
        
        # =====================================================================
        # STEP 2: Trova l'indice del primo token generato
        # =====================================================================
        try:
            # Gli scores sono una tupla/lista di tensors, uno per ogni step di generazione
            # Prendiamo il PRIMO step (la prima parola generata)
            first_token_logits = scores[0]  # Shape: [batch_size, vocab_size]
            
            # Se batch_size > 1, prendiamo il primo elemento
            if first_token_logits.dim() > 1:
                first_token_logits = first_token_logits[0]  # Shape: [vocab_size]
            
            # =====================================================================
            # STEP 3: Calcola probabilità per Yes e No
            # =====================================================================
            
            # Ottieni i logits per tutti i token Yes/No variants
            yes_logits = [first_token_logits[tid].item() for tid in self.yes_variants]
            no_logits = [first_token_logits[tid].item() for tid in self.no_variants]
            
            # Prendi il massimo per ogni categoria (gestisce varianti)
            max_yes_logit = max(yes_logits) if yes_logits else -float('inf')
            max_no_logit = max(no_logits) if no_logits else -float('inf')
            
            # Calcola softmax solo su Yes e No (binary choice)
            logits_tensor = torch.tensor([max_yes_logit, max_no_logit])
            probs = torch.softmax(logits_tensor, dim=0)
            
            yes_prob = probs[0].item()
            no_prob = probs[1].item()
            
            # Determina la risposta scelta
            chosen = "yes" if yes_prob > no_prob else "no"
            confidence = max(yes_prob, no_prob)
            
            return {
                "yes_confidence": yes_prob,
                "no_confidence": no_prob,
                "confidence_score": confidence,
                "chosen_answer": chosen,
                "method": "logits",
                "yes_logit": max_yes_logit,
                "no_logit": max_no_logit
            }
            
        except Exception as e:
            print(f"⚠️ Error in logit-based confidence calculation: {e}")
            return self._text_fallback(response_text)
    
    def _text_fallback(self, response_text):
        """Fallback basato su parsing del testo quando i logits non sono disponibili."""
        if not response_text:
            return {
                "yes_confidence": 0.5,
                "no_confidence": 0.5,
                "confidence_score": 0.5,
                "chosen_answer": "unknown",
                "method": "text_fallback_no_text"
            }
        
        text_lower = response_text.lower().strip()
        
        # Check primi 10 caratteri per risposta immediata
        first_chars = text_lower[:10]
        
        if "yes" in first_chars:
            return {
                "yes_confidence": 0.95,
                "no_confidence": 0.05,
                "confidence_score": 0.95,
                "chosen_answer": "yes",
                "method": "text_fallback"
            }
        elif "no" in first_chars:
            return {
                "yes_confidence": 0.05,
                "no_confidence": 0.95,
                "confidence_score": 0.95,
                "chosen_answer": "no",
                "method": "text_fallback"
            }
        else:
            # Ambiguous
            return {
                "yes_confidence": 0.5,
                "no_confidence": 0.5,
                "confidence_score": 0.5,
                "chosen_answer": "unknown",
                "method": "text_fallback_ambiguous"
            }
    
    # =========================================================================
    # METODI LEGACY (Mantieni per compatibilità, ma non usare)
    # =========================================================================
    
    def calculate_pairwise_confidence(self, outputs):
        """
        DEPRECATED: Usa calculate_binary_confidence() invece.
        Mantenuto solo per backward compatibility.
        """
        return self.calculate_binary_confidence(outputs)
    
    def compute_confidence(self, outputs, candidate_indices, token_labels):
        """
        DEPRECATED: Metodo originale R2P, troppo complesso per questo use case.
        """
        # Implementazione originale mantenuta per compatibilità...
        # (puoi mantenere il codice esistente se serve per altri scopi)
        pass
            
# =============================================================================
# SEZIONE 3: MODEL ADAPTERS (Per Formattazione Prompt Multi-Immagine)
# =============================================================================

class ModelAdapter:
    """Base class for model-specific message format adapters."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

class MiniCPMAdapter(ModelAdapter):
    """Adapter for MiniCPM model message format (v2.6 strict)."""

    def format_image2image_plus_text_comparison_msgs(self, test_image, ret_image, prompt):
        """
        Prepara il payload per il Pairwise Reasoning: 2 Immagini + Prompt.
        """
        # Load images if paths are provided
        if isinstance(test_image, str):
            test_image = Image.open(test_image).convert('RGB')
        if isinstance(ret_image, str):
            ret_image = Image.open(ret_image).convert('RGB')
            
        # MiniCPM accetta una lista di oggetti nel content
        return [{
            'role': 'user', 
            'content': [test_image, ret_image, prompt]
        }]
    
    def format_text_options_msgs(self, test_image, prompt):
        """
        Prepara il payload standard: 1 Immagine + Prompt.
        """
        if isinstance(test_image, str):
            test_image = Image.open(test_image).convert('RGB')

        return [{
            'role': 'user', 
            'content': [test_image, prompt]
        }]