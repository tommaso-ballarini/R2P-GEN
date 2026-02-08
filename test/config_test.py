# test/config_test.py
"""
Configuration for test experiments.
Contains InstantStyle layer-wise configurations and test settings.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config as MainConfig


class TestConfig:
    """
    Configuration for test experiments.
    
    Attributes:
        CATEGORY: Category to test
        NUM_CONCEPTS: Number of concepts to test per category
        DATABASE_PATH: Path to the database JSON file
    """
    
    # === SELECTION ===
    CATEGORY = "bag"
    NUM_CONCEPTS = 5
    
    # === DATABASE ===
    DATABASE_PATH = os.path.join(PROJECT_ROOT, "database", "database_perva_train_1_clip.json")
    
    # === OUTPUT ===
    OUTPUT_BASE = os.path.join(PROJECT_ROOT, "output", "test")
    
    # === GENERATION (inherited from MainConfig) ===
    DEVICE = MainConfig.DEVICE
    SEED = MainConfig.SEED
    NUM_INFERENCE_STEPS = MainConfig.NUM_INFERENCE_STEPS
    GUIDANCE_SCALE = MainConfig.GUIDANCE_SCALE
    OUTPUT_IMAGE_SIZE = MainConfig.OUTPUT_IMAGE_SIZE
    REFERENCE_IMAGE_SIZE = MainConfig.REFERENCE_IMAGE_SIZE
    NEGATIVE_PROMPT = MainConfig.NEGATIVE_PROMPT
    
    # === MODELS ===
    SDXL_MODEL = MainConfig.SDXL_MODEL
    IP_ADAPTER_REPO = MainConfig.IP_ADAPTER_REPO
    IP_ADAPTER_SUBFOLDER = MainConfig.IP_ADAPTER_SUBFOLDER
    IP_ADAPTER_WEIGHT_NAME = MainConfig.IP_ADAPTER_WEIGHT_NAME
    
    # === ANYDOOR ===
    ANYDOOR_REPO_PATH = os.path.join(PROJECT_ROOT, "test", "external", "AnyDoor")
    ANYDOOR_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "anydoor")
    ANYDOOR_MODEL_NAME = "anydoor_model.pth"
    
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


class InstantStyleConfigs:
    """
    Predefined layer-wise configurations for InstantStyle testing.
    
    Each configuration has:
    - name: Unique identifier
    - description: Human-readable description
    - motivation: Why this configuration might work well
    - weights: Layer-wise weights dictionary
    
    Usage:
        config = InstantStyleConfigs.get_config("v2_high_identity")
        weights = config["weights"]
    """
    
    CONFIGS = {
        # ══════════════════════════════════════════════════════════════════
        # V1: CURRENT BASELINE (configurazione attuale R2P-GEN)
        # ══════════════════════════════════════════════════════════════════
        "v1_current_baseline": {
            "description": "Current R2P-GEN configuration",
            "motivation": "Baseline configuration from original implementation. "
                         "Balanced approach with moderate background suppression.",
            "weights": {
                "down": {
                    "block_0": [0.0, 0.0],         # Zero background
                    "block_1": [0.0, 0.0],         # Zero composition
                    "block_2": [0.4, 0.7],         # Object shape
                },
                "mid": 0.9,                        # High semantic identity
                "up": {
                    "block_0": [0.6, 0.8, 0.9],    # Texture/color
                    "block_1": [0.95, 0.95, 0.95], # Material fidelity
                    "block_2": [0.85, 0.85],       # Fine details
                    "block_3": [0.7],              # Final refinement
                }
            }
        },
        
        # ══════════════════════════════════════════════════════════════════
        # V2: HIGH IDENTITY (massima fedeltà identità)
        # ══════════════════════════════════════════════════════════════════
        "v2_high_identity": {
            "description": "Maximum identity preservation",
            "motivation": "mid=1.0 + up.block_1=[0.90,0.95,1.0] for peak material fidelity. "
                         "Best for objects with distinctive textures and patterns.",
            "weights": {
                "down": {
                    "block_0": [0.0, 0.0],
                    "block_1": [0.0, 0.0],
                    "block_2": [0.3, 0.5],         # Slightly reduced
                },
                "mid": 1.0,                        # 🔥 MAX semantic identity
                "up": {
                    "block_0": [0.5, 0.7, 0.85],
                    "block_1": [0.90, 0.95, 1.0],  # 🔥 PEAK material/texture
                    "block_2": [0.80, 0.75],       # Slight decay
                    "block_3": [0.6],
                }
            }
        },
        
        # ══════════════════════════════════════════════════════════════════
        # V3: EXTREME BACKGROUND SUPPRESSION
        # ══════════════════════════════════════════════════════════════════
        "v3_zero_background": {
            "description": "Complete background suppression",
            "motivation": "All down blocks = 0.0 to eliminate ANY background contamination. "
                         "Compensates with full up block injection. Best for clean cutouts.",
            "weights": {
                "down": {
                    "block_0": [0.0, 0.0],
                    "block_1": [0.0, 0.0],
                    "block_2": [0.0, 0.0],         # 🔥 ANCHE block_2 a zero
                },
                "mid": 1.0,
                "up": {
                    "block_0": [0.6, 0.8, 0.9],
                    "block_1": [1.0, 1.0, 1.0],    # 🔥 Compensate with full up
                    "block_2": [0.9, 0.85],
                    "block_3": [0.7],
                }
            }
        },
        
        # ══════════════════════════════════════════════════════════════════
        # V4: GRADUAL RAMP (crescita graduale)
        # ══════════════════════════════════════════════════════════════════
        "v4_gradual_ramp": {
            "description": "Smooth gradient from down to up",
            "motivation": "Linear increase to avoid sudden transitions. "
                         "May produce more natural-looking results with softer identity.",
            "weights": {
                "down": {
                    "block_0": [0.0, 0.0],
                    "block_1": [0.1, 0.2],         # Start light injection
                    "block_2": [0.3, 0.4],
                },
                "mid": 0.6,                        # Lower mid
                "up": {
                    "block_0": [0.7, 0.8, 0.85],
                    "block_1": [0.90, 0.95, 0.97],
                    "block_2": [0.93, 0.90],
                    "block_3": [0.85],
                }
            }
        },
        
        # ══════════════════════════════════════════════════════════════════
        # V5: INSTANTSTYLE PAPER (dal paper originale)
        # ══════════════════════════════════════════════════════════════════
        "v5_instantstyle_paper": {
            "description": "Configuration from InstantStyle paper",
            "motivation": "Reproduce paper results for comparison. "
                         "Only uses up blocks, mid=0.0 for style-only transfer.",
            "weights": {
                "down": {
                    "block_0": [0.0, 0.0],
                    "block_1": [0.0, 0.0],
                    "block_2": [0.0, 0.0],
                },
                "mid": 0.0,                        # Paper usa 0.0 per mid!
                "up": {
                    "block_0": [1.0, 1.0, 1.0],    # Solo up blocks
                    "block_1": [1.0, 1.0, 1.0],
                    "block_2": [1.0, 1.0],
                    "block_3": [1.0],
                }
            }
        },
        
        # ══════════════════════════════════════════════════════════════════
        # V6: BALANCED (bilanciato)
        # ══════════════════════════════════════════════════════════════════
        "v6_balanced": {
            "description": "Balanced between identity and flexibility",
            "motivation": "Mid-range values for natural results. "
                         "Good starting point for fine-tuning.",
            "weights": {
                "down": {
                    "block_0": [0.0, 0.0],
                    "block_1": [0.0, 0.0],
                    "block_2": [0.5, 0.6],
                },
                "mid": 0.75,
                "up": {
                    "block_0": [0.8, 0.85, 0.9],
                    "block_1": [0.9, 0.9, 0.9],
                    "block_2": [0.85, 0.8],
                    "block_3": [0.75],
                }
            }
        },
        
        # ══════════════════════════════════════════════════════════════════
        # V7: TEXTURE FOCUS (focus su texture)
        # ══════════════════════════════════════════════════════════════════
        "v7_texture_focus": {
            "description": "Focus on texture and material preservation",
            "motivation": "Higher weights on later up blocks for fine details. "
                         "Best for objects with complex patterns (fabrics, woven items).",
            "weights": {
                "down": {
                    "block_0": [0.0, 0.0],
                    "block_1": [0.0, 0.0],
                    "block_2": [0.2, 0.4],
                },
                "mid": 0.7,
                "up": {
                    "block_0": [0.5, 0.6, 0.7],    # Lower early up
                    "block_1": [0.85, 0.9, 0.95],
                    "block_2": [1.0, 1.0],         # 🔥 MAX fine details
                    "block_3": [0.95],             # 🔥 High final refinement
                }
            }
        },
        
        # ══════════════════════════════════════════════════════════════════
        # V8: SHAPE FOCUS (focus su forma)
        # ══════════════════════════════════════════════════════════════════
        "v8_shape_focus": {
            "description": "Focus on shape and structure preservation",
            "motivation": "Higher weights on mid and early up blocks for structure. "
                         "Best for objects with distinctive shapes.",
            "weights": {
                "down": {
                    "block_0": [0.0, 0.0],
                    "block_1": [0.0, 0.0],
                    "block_2": [0.6, 0.8],         # 🔥 Higher shape
                },
                "mid": 1.0,                        # 🔥 MAX semantic
                "up": {
                    "block_0": [0.9, 0.95, 1.0],   # 🔥 High structure
                    "block_1": [0.8, 0.8, 0.8],    # Lower material
                    "block_2": [0.7, 0.65],
                    "block_3": [0.6],
                }
            }
        },
    }
    
    @classmethod
    def get_config(cls, name: str) -> Dict[str, Any]:
        """
        Get configuration by name.
        
        Args:
            name: Configuration name (e.g., "v2_high_identity")
            
        Returns:
            Configuration dictionary with description, motivation, and weights
            
        Raises:
            ValueError: If configuration name is not found
        """
        if name not in cls.CONFIGS:
            available = list(cls.CONFIGS.keys())
            raise ValueError(
                f"Unknown config: '{name}'.\n"
                f"Available configurations: {available}"
            )
        return cls.CONFIGS[name]
    
    @classmethod
    def get_weights(cls, name: str) -> Dict[str, Any]:
        """
        Get only the weights dictionary from a configuration.
        
        Args:
            name: Configuration name
            
        Returns:
            Layer-wise weights dictionary
        """
        return cls.get_config(name)["weights"]
    
    @classmethod
    def list_configs(cls) -> List[str]:
        """Return list of all configuration names."""
        return list(cls.CONFIGS.keys())
    
    @classmethod
    def print_configs(cls):
        """Print all available configurations with descriptions."""
        print("\n" + "=" * 70)
        print("📊 AVAILABLE INSTANTSTYLE CONFIGURATIONS")
        print("=" * 70)
        
        for name, cfg in cls.CONFIGS.items():
            print(f"\n🔹 {name}")
            print(f"   Description: {cfg['description']}")
            print(f"   Motivation:  {cfg['motivation'][:80]}...")
            
            # Print summary of weights
            weights = cfg["weights"]
            mid_val = weights["mid"]
            up_block_1_max = max(weights["up"]["block_1"])
            down_block_2_max = max(weights["down"]["block_2"]) if any(weights["down"]["block_2"]) else 0.0
            
            print(f"   Key values:  mid={mid_val}, up.block_1.max={up_block_1_max}, down.block_2.max={down_block_2_max}")
        
        print("\n" + "=" * 70)
    
    @classmethod
    def compare_configs(cls, config_names: List[str] = None):
        """
        Print comparison table of configurations.
        
        Args:
            config_names: List of config names to compare. If None, compare all.
        """
        if config_names is None:
            config_names = cls.list_configs()
        
        print("\n" + "=" * 90)
        print("📊 CONFIGURATION COMPARISON")
        print("=" * 90)
        print(f"{'Config Name':<25} {'mid':>6} {'down.b2':>10} {'up.b0 max':>10} {'up.b1 max':>10} {'up.b2 max':>10}")
        print("-" * 90)
        
        for name in config_names:
            if name not in cls.CONFIGS:
                continue
                
            w = cls.CONFIGS[name]["weights"]
            mid = w["mid"]
            down_b2 = max(w["down"]["block_2"]) if any(w["down"]["block_2"]) else 0.0
            up_b0 = max(w["up"]["block_0"])
            up_b1 = max(w["up"]["block_1"])
            up_b2 = max(w["up"]["block_2"])
            
            print(f"{name:<25} {mid:>6.2f} {down_b2:>10.2f} {up_b0:>10.2f} {up_b1:>10.2f} {up_b2:>10.2f}")
        
        print("=" * 90 + "\n")


if __name__ == "__main__":
    # Demo
    print("\n🧪 InstantStyle Configuration Demo\n")
    
    # List all configs
    InstantStyleConfigs.print_configs()
    
    # Compare all
    InstantStyleConfigs.compare_configs()
    
    # Get specific config
    config = InstantStyleConfigs.get_config("v2_high_identity")
    print(f"\nExample config 'v2_high_identity':")
    print(f"  Description: {config['description']}")
    print(f"  Mid weight: {config['weights']['mid']}")
