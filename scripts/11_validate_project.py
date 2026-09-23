"""STEP 11 -- Quick local validation of the repository's structure and health:
every required script exists and compiles, every script has a matching
notebook (and vice versa -- no orphans), each notebook's implementation cells
still match its script (via scripts/sync_notebooks.py --check), and the
automated test suite passes. Run this after any edit to scripts/ or
notebooks/ to catch drift or breakage immediately."""
from pathlib import Path
import json
import py_compile
import subprocess
import sys

from core.paths import PROJECT_ROOT


REQUIRED_SCRIPTS = [
    "00_download_epa_data.py", "01_normalize_epa_data.py", "02_build_training_cohort.py",
    "03_train_foundation_model.py", "04_export_embeddings.py", "05_evaluate_foundation_model.py",
    "06_prepare_mea_data.py", "07_query_chemical.py", "08_build_knowledge_graph.py",
    "09_register_toxicity_endpoints.py", "10_transfer_learning.py", "11_validate_project.py",
]
AUXILIARY_SCRIPTS = [
    "00a_import_toxcast_sql.py", "00b_download_structures.py",
    "01a_export_toxcast_mysql.py", "01b_prepare_structures.py",
]


def main():
    """Run every check in order, printing a PASS line for each, and raising
    (nonzero exit) on the first failure so CI/local runs fail loudly."""
    scripts = PROJECT_ROOT / "scripts"
    missing = [name for name in REQUIRED_SCRIPTS if not (scripts / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected scripts: {missing}")
    missing_aux = [name for name in AUXILIARY_SCRIPTS if not (scripts / name).exists()]
    if missing_aux:
        raise FileNotFoundError(f"Missing auxiliary data-pipeline scripts: {missing_aux}")

    py_files = sorted(scripts.rglob("*.py"))
    for path in py_files:
        py_compile.compile(str(path), doraise=True)
    print(f"Python syntax: PASS ({len(py_files)} files)")

    notebooks_dir = PROJECT_ROOT / "notebooks"
    expected_notebooks = [Path(name).with_suffix(".ipynb").name for name in REQUIRED_SCRIPTS + AUXILIARY_SCRIPTS]
    missing_notebooks = [name for name in expected_notebooks if not (notebooks_dir / name).exists()]
    extra_notebooks = [
        p.name for p in notebooks_dir.glob("*.ipynb")
        if p.name not in expected_notebooks
    ]
    if missing_notebooks:
        raise FileNotFoundError(f"Missing matching notebooks: {missing_notebooks}")
    if extra_notebooks:
        raise ValueError(f"Unexpected extra notebooks: {extra_notebooks}")

    notebooks = sorted(notebooks_dir.glob("*.ipynb"))
    for path in notebooks:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("nbformat") != 4:
            raise ValueError(f"Unexpected notebook format: {path}")
    print(f"Notebook mirror: PASS ({len(notebooks)} notebooks for {len(REQUIRED_SCRIPTS) + len(AUXILIARY_SCRIPTS)} workflow scripts)")

    sync_result = subprocess.run(
        [sys.executable, str(scripts / "sync_notebooks.py"), "--check"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    print(sync_result.stdout)
    if sync_result.returncode != 0:
        print(sync_result.stderr)
        raise SystemExit(sync_result.returncode)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(scripts / "tests"), "-q"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit(result.returncode)
    print("Project tests: PASS")


if __name__ == "__main__":
    main()
