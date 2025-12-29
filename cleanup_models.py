import os
import shutil

def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024 * 1024) # Ritorna in GB

print(f"Dimensione attuale: {get_size('./models'):.2f} GB")
print("--- INIZIO PULIZIA ---")

# 1. ELIMINA I FILE "MONOLITICI" NELLA ROOT DI SDXL
# Questi file non servono perché diffusers usa le sottocartelle
files_to_delete = [
    "./models/sdxl-base-1.0/sd_xl_base_1.0.safetensors",
    "./models/sdxl-base-1.0/sd_xl_base_1.0_0.9vae.safetensors",
    "./models/sdxl-base-1.0/sd_xl_offset_example-lora_1.0.safetensors"
]

for f in files_to_delete:
    if os.path.exists(f):
        print(f"Eliminazione file monolitico inutile: {f}")
        os.remove(f)

# 2. ELIMINA I PESI FP32 (Full Precision) SE ESISTE LA VERSIONE FP16
# Cerchiamo nelle sottocartelle
subfolders = ["unet", "vae", "text_encoder", "text_encoder_2"]
base_path = "./models/sdxl-base-1.0"

for sub in subfolders:
    folder_path = os.path.join(base_path, sub)
    if not os.path.exists(folder_path): continue
    
    files = os.listdir(folder_path)
    for file in files:
        # Se il file è un .safetensors ma NON ha "fp16" nel nome...
        if ".safetensors" in file and "fp16" not in file:
            # ...e se esiste il suo fratello gemello "fp16"...
            fp16_version = file.replace(".safetensors", ".fp16.safetensors")
            if fp16_version in files:
                full_path = os.path.join(folder_path, file)
                print(f"Eliminazione peso FP32 ridondante: {sub}/{file}")
                os.remove(full_path)

# 3. PULIZIA CACHE DI HUGGINGFACE
# Eliminiamo le cartelle .cache che contengono solo metadati
for root, dirs, files in os.walk("./models"):
    if ".cache" in dirs:
        cache_path = os.path.join(root, ".cache")
        print(f"Eliminazione cache metadati: {cache_path}")
        shutil.rmtree(cache_path)

print("--- PULIZIA COMPLETATA ---")
print(f"Nuova dimensione: {get_size('./models'):.2f} GB")