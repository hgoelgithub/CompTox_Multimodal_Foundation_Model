"""Centralized absolute project paths.

Every command can be launched from the repository root with:
    python scripts/<script_name>.py
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATA_SOURCE = DATA_DIR / "source"
DATA_NORMALIZED = DATA_DIR / "normalized"
DATA_PROCESSED = DATA_DIR / "processed"
MEA_RAW = DATA_DIR / "mea"
MEA_PROCESSED = DATA_DIR / "mea_processed"
ENDPOINT_DATA = DATA_DIR / "endpoints"
CHECKPOINTS = PROJECT_ROOT / "checkpoints"
RESULTS = PROJECT_ROOT / "results"
CONFIG = PROJECT_ROOT / "config.yaml"
NOTEBOOKS = PROJECT_ROOT / "notebooks"

# Compatibility aliases used by the validated modeling/query modules.
RAW_COMPTox = DATA_NORMALIZED
PROCESSED_COMPTox = DATA_PROCESSED
OUTPUTS = RESULTS
