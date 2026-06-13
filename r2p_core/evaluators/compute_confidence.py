import torch
import numpy as np


import torch

import numpy as np
# from scipy.stats import spearmanr

def compute_agreement(scores1, scores2):
    """
    Computes the agreement between two sets of scores for three options.
    
    It returns:
    - A boolean indicating if the top candidate (the option with the highest score)
      is the same for both methods.
      
    Args:
        scores1 (list or np.array): Scores from method 1 for three options.
        scores2 (list or np.array): Scores from method 2 for three options.
        
    Returns:
        tuple: (rho, top_agree)
            - top_agree (bool): True if both methods agree on the top candidate, otherwise False.
    """
    # Convert scores to numpy arrays.
    scores1 = np.array(scores1)
    scores2 = np.array(scores2)
    
    # # Compute Spearman's rank correlation.
    # rho, _ = spearmanr(scores1, scores2)
    
    # Determine the index of the top candidate for each set.
    top1 = np.argmax(scores1)
    top2 = np.argmax(scores2)
    
    top_agree = (top1 == top2)
    
    return top_agree


class ClipScoreCalculator:
    def __init__(self, feature_extractor, clip_model):
        self.feature_extractor = feature_extractor
        self.clip_model = clip_model
        self.device = "cuda"

    def format_text_for_clip(self, sentence, category):
        """Formats text into CLIP-compatible prompts."""
        return [f"A photo of a {category} with {item.strip()}" for item in sentence.split(',')]

    def get_clip_image_feature(self, image):
        """Extracts and normalizes CLIP image features."""
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        image_features = self.clip_model.get_image_features(inputs['pixel_values'].to(self.device))
        return image_features / image_features.norm(p=2, dim=-1, keepdim=True)

    def get_text_features(self, formatted_sentences):
        """Extracts and normalizes CLIP text features."""
        inputs = self.feature_extractor(text=formatted_sentences, return_tensors="pt", padding=True).to(self.device)
        text_features = self.clip_model.get_text_features(inputs["input_ids"], inputs["attention_mask"])
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)  
        return text_features.detach()  # Keep as a PyTorch tensor

    def clip_score(self, image, text_list):
        """Computes the CLIP similarity score between image and text options."""
        if len(text_list) > 10:
            text_list  = text_list[:10]
        try:
            text_features = self.get_text_features(text_list)  # Now remains a tensor
        except:
            print(text_list)
        image_features = self.get_clip_image_feature(image)

        # Expand image features to match text features shape
        image_features = image_features.expand(text_features.size(0), -1)

        # Compute cosine similarity
        text_similarities = torch.sum(image_features * text_features, dim=1)
        return text_similarities, torch.mean(text_similarities)


class ConfidenceCalculator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def extract_candidate_indices(self, tokens, target_tokens, trigger_tokens=None):
        """
        Extracts candidate indices where the next token matches the given target tokens.
        """
        candidate_indices = []
        for i, token in enumerate(tokens):
            token_clean = token.lstrip("Ġ").lower()
            if token_clean in trigger_tokens:
                j = i + 1
                while j < len(tokens):
                    next_token = tokens[j].lstrip("Ġ").lower()
                    if next_token in ['":', '"', ',', 'Ġ']:  # Skip punctuation-like tokens
                        j += 2 # changed it to 2 originally 1
                        continue
                    if next_token in target_tokens:
                        candidate_indices.append(j)
                    break  # Stop after finding the first valid token
        if not candidate_indices:
            if tokens[len(tokens) - 3] in target_tokens:
                candidate_indices = [len(tokens) - 3]
        return candidate_indices

    def compute_confidence(self, outputs, candidate_indices, token_labels):
        """
        Computes the confidence scores for the given token labels.
        """
        if not candidate_indices:
            return {label + "_confidence": 0.0 for label in token_labels} | \
                   {label + "_score": 0.0 for label in token_labels} | \
                   {"margin": 0.0}

        confidences = []
        key = 'scores' if 'scores' in outputs else 'logits'
        for idx in candidate_indices:
            logits = outputs[key][idx][0]  # shape: (vocab_size,)
            token_ids = [self.tokenizer.convert_tokens_to_ids(label) for label in token_labels]
            token_scores = [logits[token_id].item() for token_id in token_ids]
            # token_scores = [score if score != -np.inf else 0.01 for score in token_scores]
            probs = torch.softmax(torch.tensor(token_scores), dim=-1)
            confidences.append((*probs.tolist(), *token_scores))
        # Compute the average confidence over all occurrences
        avg_probs = [sum(pair[i] for pair in confidences) / len(confidences) for i in range(len(token_labels))]
        avg_scores = [sum(pair[i + len(token_labels)] for pair in confidences) / sum(confidences[0][len(token_labels):]) for i in range(len(token_labels))]
        # Compute margin (difference between top-1 and top-2 probabilities)
        sorted_scores = sorted(avg_scores, reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        return {label + "_confidence": avg_probs[i] for i, label in enumerate(token_labels)} | \
               {label + "_score": avg_scores[i] for i, label in enumerate(token_labels)} | \
               {"margin": margin}

    def calculate_confidence(self, outputs, task="recognition", target_tokens=["yes", "no"]):
        """
        Calculates the confidence scores for either 'recognition' (yes/no) or 'recall' (A/B/C) tasks.
        """
        tokens = self.tokenizer.convert_ids_to_tokens(outputs["sequences"][0])
        trigger_tokens = ["answer"]
        if task == "recognition":
            candidate_indices = self.extract_candidate_indices(tokens, ["yes", "no"], trigger_tokens)
            return self.compute_confidence(outputs, candidate_indices, ["yes", "no"])

        elif task == "recall_hard":
            candidate_indices = self.extract_candidate_indices(tokens, target_tokens, trigger_tokens)
            return self.compute_confidence(outputs, candidate_indices, target_tokens)

        else:
            raise ValueError(f"Invalid task type: {task}. Choose either 'recognition' or 'recall'.")
