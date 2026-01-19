# config.py
"""
Configurazione centralizzata della pipeline R2P-GEN
"""

class Config:
    # === MODELLI ===
    VLM_MODEL = "openbmb/MiniCPM-o-2_6"
    SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
    IP_ADAPTER_REPO = "h94/IP-Adapter"
    
    # EXTRACTION
    CATEGORY_DETECTION_THRESHOLD = 0.4  # Soglia CLIP per categoria
    
    # GENERATION  
    IP_ADAPTER_WEIGHTS = {  # Da tuning empirico
        "retail_packaged": 0.5,
        "clothing": 0.7,
        "electronics": 0.6,
        "furniture": 0.65,
        "generic": 0.6
    }
    
    # VERIFICATION
    ATTRIBUTE_WEIGHT = 0.4  # Peso accuracy attributi
    CLIP_WEIGHT = 0.3       # Peso CLIP similarity
    LPIPS_WEIGHT = 0.2      # Peso LPIPS
    TEXT_WEIGHT = 0.1       # Peso text verification
    
    # CONVERGENCE
    TARGET_ACCURACY = 0.90  # Era 0.95, troppo alto
    MIN_IMPROVEMENT = 0.03  # Era 0.05
    
    # === IMMAGINI ===
    MAX_IMAGE_DIM = 896
    FALLBACK_DIM = 448  # Dimensione ridotta in caso OOM
    
    # === PATHS ===
    OUTPUT_DIR = "output"
    DATASET_DIR = "data/perva_test"
    
    # === NEGATIVE PROMPTS ===
    NEGATIVE_BASE = "blurry, low quality, distorted, deformed, bad anatomy, watermark, text overlay"
    
    # === GPU ===
    MEMORY_FRACTION = 0.9
    USE_FP16 = True