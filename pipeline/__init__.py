# pipeline/__init__.py
"""
R2P-GEN Pipeline Module

This module contains the core components of the R2P-GEN pipeline:
- build_database: Database building and fingerprint extraction
- generate: Image generation using SDXL + IP-Adapter
- verify: Generated image verification (maintained by team)
- refine: Iterative refinement loop (maintained by team)
- utils2: Utility functions for the pipeline
"""

from .build_database import DatabaseBuilder
from .generate import Generator
# from .verify import Verifier  # TODO: Uncomment when verify class is ready
# from .refine import Refiner  # TODO: Uncomment when refine class is ready
from .utils2 import cleanup_gpu, print_memory_stats, ensure_output_dir, get_iteration_filename

__all__ = [
    'DatabaseBuilder',
    'Generator',
    # 'Verifier', 
    # 'Refiner',
    'cleanup_gpu',
    'print_memory_stats',
    'ensure_output_dir',
    'get_iteration_filename',
]
