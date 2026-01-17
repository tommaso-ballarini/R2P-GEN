# utils.py
import torch
import gc
import os
from pathlib import Path

def cleanup_gpu():
    """Libera memoria GPU in modo aggressivo"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def print_memory_stats(label=""):
    """Stampa statistiche memoria GPU"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - reserved
        print(f"   📊 {label}")
        print(f"      Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Free: {free:.2f}GB")

def ensure_output_dir(output_dir="output"):
    """Crea cartella output se non esiste"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir

def get_iteration_filename(base_name, iteration):
    """Genera nome file per iterazione"""
    stem = Path(base_name).stem
    ext = Path(base_name).suffix
    return f"{stem}_iter{iteration}{ext}"