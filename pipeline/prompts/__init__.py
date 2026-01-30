# pipeline/prompts/__init__.py
"""
SDXL Prompt Templates for R2P-GEN Pipeline.

This module contains various prompt engineering strategies for 
generating SDXL-compatible prompts from object fingerprints.
"""

from .sdxl_prompts import (
    SYSTEM_PROMPT_SIMPLE,
    SYSTEM_PROMPT_GEMINI,
    SYSTEM_PROMPT_OPTIMIZED,
    HARDCODED_STYLE
)

__all__ = [
    'SYSTEM_PROMPT_SIMPLE',
    'SYSTEM_PROMPT_GEMINI', 
    'SYSTEM_PROMPT_OPTIMIZED',
    'HARDCODED_STYLE'
]
