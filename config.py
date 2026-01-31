# config.py
"""
Centralized configuration for the R2P-GEN pipeline.
All configurable parameters should be defined here.
"""
import os

class Config:
    # === CLUSTER MODE ===
    # Set to True when running on SLURM cluster
    CLUSTER_MODE = os.environ.get("R2P_CLUSTER_MODE", "false").lower() == "true"
    
    # === BASE PATHS (auto-detect based on mode) ===
    if CLUSTER_MODE:
        HOME_DIR = os.environ.get("HOME", "/home/tommaso.ballarini-1")
        BASE_DIR = os.path.join(HOME_DIR, "R2P-GEN")
        DATA_BASE = os.path.join(HOME_DIR, "data")
    else:
        BASE_DIR = "."
        DATA_BASE = "data"
    
    # === MODELS ===
    VLM_MODEL = "openbmb/MiniCPM-o-2_6"
    SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
    IP_ADAPTER_REPO = "h94/IP-Adapter"
    IP_ADAPTER_SUBFOLDER = "sdxl_models"
    IP_ADAPTER_WEIGHT_NAME = "ip-adapter_sdxl.bin"
    QWEN_MODEL = "Qwen/Qwen2-VL-7B-Instruct"  # For Final Judge
    
    # === GENERATION ===
    IP_ADAPTER_SCALE_GLOBAL = 0.6
    NUM_INFERENCE_STEPS = 40
    GUIDANCE_SCALE = 7.5
    SEED = 42
    DEVICE = "cuda"
    
    # === IP-ADAPTER LAYER-WISE SCALING ===
    USE_LAYERWISE_SCALING = True
    IP_ADAPTER_LAYER_WEIGHTS = {
        "down": {
            "block_0": [0.0, 0.0],
            "block_1": [0.0, 0.0],
            "block_2": [0.4, 0.7],
        },
        "mid": 0.9,
        "up": {
            "block_0": [0.6, 0.8, 0.9],
            "block_1": [0.95, 0.95, 0.95],
            "block_2": [0.85, 0.85],
            "block_3": [0.7],
        }
    }
    
    # === REFINEMENT LOOP ===
    MAX_ITERATIONS = 3
    TARGET_ACCURACY = 0.95
    MIN_IMPROVEMENT = 0.05
    
    # === IMAGES ===
    MAX_IMAGE_DIM = 896
    FALLBACK_DIM = 448
    REFERENCE_IMAGE_SIZE = 512
    OUTPUT_IMAGE_SIZE = 1024
    
    # === PATHS ===
    OUTPUT_DIR = os.path.join(BASE_DIR, "output") if CLUSTER_MODE else "output"
    DATASET_DIR = os.path.join(DATA_BASE, "perva-data") if CLUSTER_MODE else "data/perva-data"
    DATABASE_DIR = os.path.join(BASE_DIR, "database") if CLUSTER_MODE else "database"
    
    # === NEGATIVE PROMPTS ===
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