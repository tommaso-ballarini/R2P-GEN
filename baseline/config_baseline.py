# baseline/config_baseline.py
"""
Configurazione condivisa per esperimenti baseline.
Eredita dalla Config principale e aggiunge parametri specifici per i test.
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config as MainConfig


class BaselineConfig:
    """
    Configuration for baseline experiments.
    
    Attributes:
        CATEGORY: Category to test (bag, book, bottle, bowl, clothe, cup, decoration, etc.)
        NUM_CONCEPTS: Number of concepts to test per category (5 or 10 recommended)
        DATABASE_PATH: Path to the database JSON file
    """
    
    # === SELECTION ===
    # Available categories: bag, book, bottle, bowl, clothe, cup, decoration, 
    #                       headphone, pillow, plant, plate, remote, retail,
    #                       telephone, tie, towel, toy, tro_bag, tumbler, umbrella, veg
    CATEGORY = "bag"
    NUM_CONCEPTS = 5  # Number of concepts to test (5 or 10)
    
    # === DATABASE ===
    DATABASE_PATH = os.path.join(PROJECT_ROOT, "database", "database_perva_train_1_clip.json")
    
    # === OUTPUT ===
    OUTPUT_BASE = os.path.join(PROJECT_ROOT, "output", "baseline")
    
    # === GENERATION (inherited from MainConfig) ===
    DEVICE = MainConfig.DEVICE
    SEED = MainConfig.SEED
    NUM_INFERENCE_STEPS = MainConfig.NUM_INFERENCE_STEPS
    GUIDANCE_SCALE = MainConfig.GUIDANCE_SCALE
    OUTPUT_IMAGE_SIZE = MainConfig.OUTPUT_IMAGE_SIZE
    REFERENCE_IMAGE_SIZE = MainConfig.REFERENCE_IMAGE_SIZE
    NEGATIVE_PROMPT = MainConfig.NEGATIVE_PROMPT
    
    # === MODELS (inherited from MainConfig) ===
    SDXL_MODEL = MainConfig.SDXL_MODEL
    IP_ADAPTER_REPO = MainConfig.IP_ADAPTER_REPO
    IP_ADAPTER_SUBFOLDER = MainConfig.IP_ADAPTER_SUBFOLDER
    IP_ADAPTER_WEIGHT_NAME = MainConfig.IP_ADAPTER_WEIGHT_NAME
    
    # === IP-ADAPTER VANILLA SCALE ===
    # Global scale for baseline (no layer-wise scaling)
    IP_ADAPTER_SCALE = 0.6
    
    @classmethod
    def get_available_categories(cls) -> list:
        """Return list of available categories from dataset."""
        return [
            "bag", "book", "bottle", "bowl", "clothe", "cup", "decoration",
            "headphone", "pillow", "plant", "plate", "remote", "retail",
            "telephone", "tie", "towel", "toy", "tro_bag", "tumbler", "umbrella", "veg"
        ]
    
    @classmethod
    def validate_category(cls, category: str) -> bool:
        """Validate if category exists."""
        return category in cls.get_available_categories()
    
    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("\n" + "=" * 60)
        print("📋 BASELINE CONFIGURATION")
        print("=" * 60)
        print(f"   Category: {cls.CATEGORY}")
        print(f"   Num Concepts: {cls.NUM_CONCEPTS}")
        print(f"   Database: {cls.DATABASE_PATH}")
        print(f"   Output Dir: {cls.OUTPUT_BASE}")
        print(f"   Device: {cls.DEVICE}")
        print(f"   Seed: {cls.SEED}")
        print(f"   Steps: {cls.NUM_INFERENCE_STEPS}")
        print(f"   Guidance: {cls.GUIDANCE_SCALE}")
        print(f"   IP-Adapter Scale: {cls.IP_ADAPTER_SCALE}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    # Test configuration
    BaselineConfig.print_config()
    print(f"Available categories: {BaselineConfig.get_available_categories()}")
