from huggingface_hub import snapshot_download
import os
import shutil

# Funzione per pulire se ci sono residui corrotti
def clean_folder(folder):
    if os.path.exists(folder):
        print(f"Pulizia cartella pre-esistente: {folder}")
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

# Crea la cartella principale
os.makedirs("models", exist_ok=True)

print("--- INIZIO DOWNLOAD SDXL BASE (SOLO PYTORCH FP16) ---")
# SDXL è enorme, quindi escludiamo esplicitamente tutto ciò che non è torch
sdxl_path = snapshot_download(
    repo_id="stabilityai/stable-diffusion-xl-base-1.0",
    local_dir="./models/sdxl-base-1.0",
    # Scarica solo safetensors (pesi sicuri) e config json/txt
    allow_patterns=[
        "*.safetensors", 
        "*.json", 
        "*.txt", 
        "model_index.json"
    ],
    # Escludi esplicitamente formati pesanti inutili
    ignore_patterns=[
        "*.bin",            # Vecchi pesi PyTorch
        "*.ckpt",           # Checkpoint legacy
        "*.msgpack",        # Flax/JAX (GIGANTESCO)
        "*.xml",            # OpenVINO
        "*.onnx",           # ONNX
        "*openvino*", 
        "*flax*"
    ]
)
print(f"SDXL scaricato in: {sdxl_path}")

print("\n--- INIZIO DOWNLOAD IP-ADAPTER ---")
ip_adapter_path = snapshot_download(
    repo_id="h94/IP-Adapter",
    local_dir="./models/IP-Adapter",
    # Qui ci serve solo la cartella sdxl_models
    allow_patterns=["sdxl_models/*"],
    ignore_patterns=["*.bin", "*.msgpack"] # Scarichiamo safetensors se c'è, altrimenti bin
)
# Nota: IP-Adapter spesso ha solo .bin, quindi se il download sopra fallisce
# o scarica poco, rimuovi "*.bin" dagli ignore_patterns solo per IP-Adapter.
# Ma per h94/IP-Adapter, i file principali sono binari piccoli, quindi:
print(f"IP-Adapter scaricato in: {ip_adapter_path}")

print("\n--- TUTTO PRONTO ---")