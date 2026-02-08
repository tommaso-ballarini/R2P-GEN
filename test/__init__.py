# test/__init__.py
"""
Test module for R2P-GEN advanced experiments.

This module provides:
- AnyDoor integration tests (vanilla and with fingerprints)
- InstantStyle layer-wise scaling experiments
- Comparison utilities

Usage:
    # AnyDoor tests
    python test/anydoor_vanilla.py --category bag --num 5
    python test/anydoor_with_fingerprints.py --category bottle --num 5

    # InstantStyle scaling tests
    python test/instantstyle_scaling.py --category bag --num 5 --config v2_high_identity
    python test/instantstyle_scaling.py --category cup --num 10 --all-configs

    # Python API
    from test import AnyDoorVanillaGenerator, AnyDoorFingerprintsGenerator
    from test import InstantStyleScalingTester
    from test.config_test import InstantStyleConfigs, TestConfig
"""

from test.config_test import TestConfig, InstantStyleConfigs

__all__ = [
    "TestConfig",
    "InstantStyleConfigs",
]
