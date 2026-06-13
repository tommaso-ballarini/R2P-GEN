# pipeline/__init__.py
"""
R2P-GEN Pipeline Module (FLUX Edition)

Contains only the components actually used by the active workflow
orchestrated by flux_loop.py:

- generate: Image generation with FLUX (Img2Img)
- verify: Verification of generated images (MiniCPM + CLIP, V5)
- judge: Independent final evaluation (CLIP/DINO/VQA)
- metrics: Quantitative metrics (CLIP-I, CLIP-T, DINO-I, TIFA)
- utils2: Utilities for GPU memory management and file handling

NOTE: build_database and refine have been moved to their respective
*_legacy.py files and are NOT imported here, because:
  - build_database_legacy depends on modules (database.mini_cpm_info,
    database.create_train_test_perva_split) that may not exist
    in the current environment, and would cause `import pipeline` to fail.
  - refine_legacy references Generator._initialize_pipeline(),
    a method of the old SDXL Generator, which is not present in the FLUX Generator.
"""

from .generate import Generator
from .verify import verify_generation_r2p
from .judge import FinalJudge, JudgeResult
from .metrics import MetricsCalculator, MetricsResult, quick_evaluate
from .utils2 import cleanup_gpu, print_memory_stats, ensure_output_dir, get_iteration_filename

__all__ = [
    # Core pipeline (FLUX)
    'Generator',

    # Verification
    'verify_generation_r2p',

    # Final Judge & Metrics
    'FinalJudge',
    'JudgeResult',
    'MetricsCalculator',
    'MetricsResult',
    'quick_evaluate',

    # Utilities
    'cleanup_gpu',
    'print_memory_stats',
    'ensure_output_dir',
    'get_iteration_filename',
]