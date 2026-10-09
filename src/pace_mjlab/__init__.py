"""PACE joint and actuator identification on mjlab (MuJoCo Warp)."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PACKAGE_DIR / "assets"
PROJECT_ROOT = PACKAGE_DIR.parents[1]
"""Repository root; ``data/`` and ``logs/`` live here, with the same layout as pace-sim2real."""
