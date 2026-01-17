import torch
from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoImageProcessor

class ModelInterface:
    """Base interface for different language models."""

    def __init__(self, model_path: str, device: str = "cuda", attn_implementation: str = "sdpa", torch_dtype=torch.bfloat16):
        self.device = device
        if model_path == 'openbmb/MiniCPM-o-2_6':
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                torch_dtype=torch_dtype
            ).eval().to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)

    def chat(self, msgs):
        raise NotImplementedError("Subclasses must implement this method.")