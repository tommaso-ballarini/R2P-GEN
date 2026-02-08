# test/anydoor_wrapper.py
"""
Wrapper for AnyDoor model.

This provides a clean interface to AnyDoor, isolating its dependencies
from the rest of the project.

AnyDoor: https://github.com/ali-vilab/AnyDoor
Paper: "Anydoor: Zero-shot Object-level Image Customization" (CVPR 2024)

Setup:
    1. Run: .\\test\\setup_anydoor.ps1
    2. Or manually:
       - Clone: git clone https://github.com/ali-vilab/AnyDoor.git test/external/AnyDoor
       - Download checkpoint from: https://huggingface.co/xichenhku/AnyDoor
       - Place at: checkpoints/anydoor/anydoor_model.pth
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import torch
from PIL import Image
import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from test.config_test import TestConfig


class AnyDoorWrapper:
    """
    Clean interface for AnyDoor model.
    
    AnyDoor uses:
    - DINOv2 for global identity encoding
    - High-frequency detail extractor for local features
    - Stable Diffusion for generation
    
    Key features:
    - Zero-shot object customization
    - Better fine-grained detail preservation than IP-Adapter
    - Frequency decomposition to prevent background leakage
    """
    
    def __init__(
        self,
        checkpoint_path: str = None,
        device: str = None,
        use_fp16: bool = True
    ):
        """
        Initialize AnyDoor wrapper.
        
        Args:
            checkpoint_path: Path to AnyDoor checkpoint. If None, uses default.
            device: Device to use. If None, uses config default.
            use_fp16: Whether to use FP16 for memory efficiency.
        """
        self.device = device or TestConfig.DEVICE
        self.use_fp16 = use_fp16
        
        # Set paths
        self.anydoor_path = Path(TestConfig.ANYDOOR_REPO_PATH)
        self.checkpoint_path = checkpoint_path or os.path.join(
            TestConfig.ANYDOOR_CHECKPOINT_PATH,
            TestConfig.ANYDOOR_MODEL_NAME
        )
        
        # Validate setup
        self._validate_setup()
        
        # Will be loaded on first use
        self.model = None
        self.is_loaded = False
    
    def _validate_setup(self):
        """Validate that AnyDoor is properly set up."""
        errors = []
        
        if not self.anydoor_path.exists():
            errors.append(
                f"AnyDoor repository not found at: {self.anydoor_path}\n"
                f"Run: .\\test\\setup_anydoor.ps1"
            )
        
        if not os.path.exists(self.checkpoint_path):
            errors.append(
                f"AnyDoor checkpoint not found at: {self.checkpoint_path}\n"
                f"Download from: https://huggingface.co/xichenhku/AnyDoor"
            )
        
        if errors:
            self.setup_complete = False
            self.setup_errors = errors
            print(f"\n⚠️  AnyDoor setup incomplete:")
            for e in errors:
                print(f"   - {e}")
            print()
        else:
            self.setup_complete = True
            self.setup_errors = []
    
    def load_model(self):
        """
        Load AnyDoor model.
        
        This imports from the cloned AnyDoor repository.
        """
        if not self.setup_complete:
            raise RuntimeError(
                "AnyDoor setup incomplete. Errors:\n" + 
                "\n".join(self.setup_errors)
            )
        
        if self.is_loaded:
            return
        
        print("🚪 Loading AnyDoor model...")
        
        # Add AnyDoor to path
        sys.path.insert(0, str(self.anydoor_path))
        
        try:
            # Import AnyDoor components
            # Note: The exact imports depend on AnyDoor's structure
            # This is a template - adjust based on actual AnyDoor code
            
            from omegaconf import OmegaConf
            
            # Try to import AnyDoor's model
            # The actual import path may vary based on AnyDoor's structure
            try:
                from ldm.models.diffusion.ddpm import LatentDiffusion
                from ldm.util import instantiate_from_config
            except ImportError:
                # Alternative import structure
                print("   ⚠️  Using alternative import structure...")
                from cldm.model import load_state_dict
                from cldm.cldm import ControlLDM
            
            # Load config
            config_path = self.anydoor_path / "configs" / "anydoor.yaml"
            if config_path.exists():
                config = OmegaConf.load(config_path)
            else:
                # Use default config
                print("   ⚠️  Config not found, using defaults...")
                config = self._get_default_config()
            
            # Initialize model
            # This is a template - actual initialization depends on AnyDoor code
            self.model = self._initialize_model(config)
            
            # Load checkpoint
            print(f"   📥 Loading checkpoint from {self.checkpoint_path}...")
            state_dict = torch.load(self.checkpoint_path, map_location='cpu')
            self.model.load_state_dict(state_dict, strict=False)
            
            # Move to device
            self.model = self.model.to(self.device)
            
            if self.use_fp16:
                self.model = self.model.half()
            
            self.model.eval()
            self.is_loaded = True
            
            print("   ✅ AnyDoor model loaded successfully")
            
        except Exception as e:
            print(f"   ❌ Error loading AnyDoor: {e}")
            print(f"   Please check the AnyDoor repository structure.")
            raise
        
        finally:
            # Remove AnyDoor from path to avoid conflicts
            if str(self.anydoor_path) in sys.path:
                sys.path.remove(str(self.anydoor_path))
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default AnyDoor configuration."""
        return {
            "model": {
                "target": "cldm.cldm.ControlLDM",
                "params": {
                    "linear_start": 0.00085,
                    "linear_end": 0.0120,
                    "num_timesteps_cond": 1,
                    "log_every_t": 200,
                    "timesteps": 1000,
                    "first_stage_key": "jpg",
                    "cond_stage_key": "txt",
                    "image_size": 64,
                    "channels": 4,
                    "cond_stage_trainable": False,
                    "conditioning_key": "crossattn",
                    "scale_factor": 0.18215,
                }
            }
        }
    
    def _initialize_model(self, config):
        """Initialize model from config."""
        # This is a placeholder - actual implementation depends on AnyDoor code
        # The model initialization will follow AnyDoor's pattern
        from omegaconf import OmegaConf
        
        try:
            from ldm.util import instantiate_from_config
            model = instantiate_from_config(config.model)
        except:
            # Fallback: manual initialization
            print("   Using manual model initialization...")
            model = None  # Will be set up based on actual AnyDoor structure
        
        return model
    
    def generate(
        self,
        reference_image: Image.Image,
        target_image: Optional[Image.Image] = None,
        mask: Optional[Image.Image] = None,
        prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: int = None
    ) -> Image.Image:
        """
        Generate image with AnyDoor.
        
        Args:
            reference_image: Image of the object to transfer
            target_image: Optional target scene image (for inpainting)
            mask: Optional mask for target region
            prompt: Text prompt to guide generation
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            seed: Random seed for reproducibility
            
        Returns:
            Generated PIL Image
        """
        if not self.is_loaded:
            self.load_model()
        
        # Set seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Preprocess inputs
        ref_tensor = self._preprocess_image(reference_image)
        
        target_tensor = None
        if target_image is not None:
            target_tensor = self._preprocess_image(target_image)
        
        mask_tensor = None
        if mask is not None:
            mask_tensor = self._preprocess_mask(mask)
        
        # Run inference
        with torch.no_grad():
            # The actual inference call depends on AnyDoor's API
            # This is a template structure
            
            if self.model is None:
                raise RuntimeError("Model not properly initialized")
            
            # Prepare inputs for AnyDoor
            inputs = {
                "reference": ref_tensor.to(self.device),
                "prompt": prompt,
            }
            
            if target_tensor is not None:
                inputs["target"] = target_tensor.to(self.device)
            
            if mask_tensor is not None:
                inputs["mask"] = mask_tensor.to(self.device)
            
            # Generate
            # Note: actual method name may vary
            output = self.model.sample(
                **inputs,
                steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
        
        # Postprocess
        result = self._postprocess_output(output)
        
        return result
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input."""
        # Resize to model's expected size
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
        
        # Convert to tensor
        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        
        # Normalize to [-1, 1]
        img_tensor = img_tensor * 2.0 - 1.0
        
        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)
        
        if self.use_fp16:
            img_tensor = img_tensor.half()
        
        return img_tensor
    
    def _preprocess_mask(self, mask: Image.Image) -> torch.Tensor:
        """Preprocess mask for model input."""
        # Resize
        mask = mask.resize((512, 512), Image.Resampling.NEAREST)
        
        # Convert to tensor
        mask_array = np.array(mask.convert('L')).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0)
        
        if self.use_fp16:
            mask_tensor = mask_tensor.half()
        
        return mask_tensor
    
    def _postprocess_output(self, output: torch.Tensor) -> Image.Image:
        """Postprocess model output to PIL Image."""
        # Remove batch dimension
        if output.dim() == 4:
            output = output[0]
        
        # Convert from [-1, 1] to [0, 1]
        output = (output + 1.0) / 2.0
        
        # Clamp
        output = torch.clamp(output, 0, 1)
        
        # Convert to numpy
        output_np = output.cpu().float().permute(1, 2, 0).numpy()
        
        # Convert to uint8
        output_np = (output_np * 255).astype(np.uint8)
        
        return Image.fromarray(output_np)
    
    def cleanup(self):
        """Release GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        self.is_loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Simple test
if __name__ == "__main__":
    print("\n🧪 Testing AnyDoor Wrapper\n")
    
    wrapper = AnyDoorWrapper()
    
    print(f"Setup complete: {wrapper.setup_complete}")
    
    if not wrapper.setup_complete:
        print("\n⚠️  Please run the setup script first:")
        print("   .\\test\\setup_anydoor.ps1")
    else:
        print("\n✅ AnyDoor is ready to use!")
        print(f"   Repo: {wrapper.anydoor_path}")
        print(f"   Checkpoint: {wrapper.checkpoint_path}")
