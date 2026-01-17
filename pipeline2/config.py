# config.py
"""
Configurazione centralizzata della pipeline R2P-GEN
"""

class Config:
    # === MODELLI ===
    VLM_MODEL = "openbmb/MiniCPM-o-2_6"
    SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
    IP_ADAPTER_REPO = "h94/IP-Adapter"
    
    # === GENERAZIONE ===
    IP_ADAPTER_SCALE = 0.6
    NUM_INFERENCE_STEPS = 30
    GUIDANCE_SCALE = 7.5
    SEED = 42
    
    # === REFINEMENT LOOP ===
    MAX_ITERATIONS = 3
    TARGET_ACCURACY = 0.95  # 95% accuratezza per uscire dal loop
    MIN_IMPROVEMENT = 0.05   # Miglioramento minimo tra iterazioni
    
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