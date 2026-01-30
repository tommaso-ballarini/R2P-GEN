# config.py
"""
Centralized configuration for the R2P-GEN pipeline.
All configurable parameters should be defined here.
"""

class Config:
    # === MODELS ===
    VLM_MODEL = "openbmb/MiniCPM-o-2_6"
    SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
    IP_ADAPTER_REPO = "h94/IP-Adapter"
    IP_ADAPTER_SUBFOLDER = "sdxl_models"
    IP_ADAPTER_WEIGHT_NAME = "ip-adapter_sdxl.bin"
    
    # === GENERATION ===
    IP_ADAPTER_SCALE_GLOBAL = 0.6
    NUM_INFERENCE_STEPS = 40
    GUIDANCE_SCALE = 7.5
    SEED = 42
    DEVICE = "cuda"
    
    # === IP-ADAPTER LAYER-WISE SCALING ===
    # Strategy: Minimize background contamination, maximize identity preservation
    # Based on InstantStyle research + R2P optimization
    USE_LAYERWISE_SCALING = True
    IP_ADAPTER_LAYER_WEIGHTS = {
        "down": {
            "block_0": [0.0, 0.0],         # Zero background from reference
            "block_1": [0.0, 0.0],         # Zero composition from reference
            "block_2": [0.4, 0.7],         # Object shape preservation (smooth gradient)
        },
        "mid": 0.9,                        # Very high semantic identity
        "up": {
            "block_0": [0.6, 0.8, 0.9],    # Texture/color injection (3 layers)
            "block_1": [0.95, 0.95, 0.95], # Maximum material fidelity (3 layers)
            "block_2": [0.85, 0.85],       # Fine details preservation (2 layers)
            "block_3": [0.7],              # Final refinement (1 layer)
        }
    }
    
    # === REFINEMENT LOOP ===
    MAX_ITERATIONS = 3
    TARGET_ACCURACY = 0.95  # 95% accuracy to exit loop
    MIN_IMPROVEMENT = 0.05  # Minimum improvement between iterations
    
    # === IMAGES ===
    MAX_IMAGE_DIM = 896
    FALLBACK_DIM = 448      # Reduced dimension in case of OOM
    REFERENCE_IMAGE_SIZE = 512  # Size to resize reference images for IP-Adapter
    OUTPUT_IMAGE_SIZE = 1024    # Generated image size
    
    # === PATHS ===
    OUTPUT_DIR = "output"
    DATASET_DIR = "data/perva-data"
    DATABASE_DIR = "database"
    
    # === NEGATIVE PROMPTS ===
    # Prevents background contamination and common SDXL artifacts
    # Critical for layer-wise scaling: reinforces down blocks=0.0
    NEGATIVE_PROMPT = (
        "blurry, low quality, low resolution, distorted, deformed, "
        "(background contamination:1.3), (reference background leakage:1.2), "
        "(original background visible:1.2), "
        "artifact, watermark, text overlay, logo overlay, signature, "
        "oversaturated, overexposed, underexposed, noise, grain, "
        "worst quality, jpeg artifacts, duplicate, cropped, "
        "unrealistic proportions, anatomical errors, "
        "blur, out of focus"
    )
    
    # === GPU ===
    MEMORY_FRACTION = 0.9
    USE_FP16 = True