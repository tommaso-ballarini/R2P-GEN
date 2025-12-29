from huggingface_hub import snapshot_download
import os

print("--- FIX DOWNLOAD IP-ADAPTER ---")

# Scarichiamo la cartella sdxl_models permettendo i file .bin
ip_adapter_path = snapshot_download(
    repo_id="h94/IP-Adapter",
    local_dir="./models/IP-Adapter",
    allow_patterns=["sdxl_models/*"], 
    # RIMOSSO "*.bin" dagli ignore patterns!
    ignore_patterns=["*.msgpack", "*.safetensors"] 
)
print(f"IP-Adapter aggiornato in: {ip_adapter_path}")
print("\n--- RIPARAZIONE COMPLETATA ---")