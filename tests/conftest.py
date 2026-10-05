"""Shared test helpers. The step scripts have names like `03_train_foundation_model.py`, which cannot be
imported with a normal `import`, so tests load them by path."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
_cache = {}


def load_script(name):
    """Import scripts/<name>.py as a module (cached)."""
    if name not in _cache:
        spec = importlib.util.spec_from_file_location(f"step_{name}", SCRIPTS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _cache[name] = module
    return _cache[name]


@pytest.fixture(scope="session")
def train():
    return load_script("03_train_foundation_model")


@pytest.fixture(scope="session")
def evaluate():
    return load_script("05_evaluate_foundation_model")


@pytest.fixture(scope="session")
def transfer():
    return load_script("10_transfer_learning")
