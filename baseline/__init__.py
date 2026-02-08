# baseline/__init__.py
"""
Baseline experiments for R2P-GEN comparison.

This module provides baseline generators to compare against the main R2P-GEN pipeline:
- IP-Adapter Only: Generation using only IP-Adapter with generic prompts
- SDXL Prompt Only: Text-to-image generation using extracted prompts (no IP-Adapter)
- General Description: IP-Adapter with general description (not optimized SDXL prompt)

Usage:
    # Command line
    python baseline/ip_adapter_only.py --category bag --num 5
    python baseline/sdxl_prompt_only.py --category bottle --num 10
    python baseline/general_desc_only.py --category cup --num 5

    # Python API
    from baseline import IPAdapterOnlyGenerator, SDXLPromptOnlyGenerator, GeneralDescOnlyGenerator
"""

from baseline.config_baseline import BaselineConfig
from baseline.ip_adapter_only import IPAdapterOnlyGenerator
from baseline.sdxl_prompt_only import SDXLPromptOnlyGenerator
from baseline.general_desc_only import GeneralDescOnlyGenerator

__all__ = [
    "BaselineConfig",
    "IPAdapterOnlyGenerator",
    "SDXLPromptOnlyGenerator", 
    "GeneralDescOnlyGenerator"
]
