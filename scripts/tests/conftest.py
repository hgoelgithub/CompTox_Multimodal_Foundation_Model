from pathlib import Path
import subprocess
import sys

SCRIPTS = Path(__file__).resolve().parents[1]
ROOT = SCRIPTS.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def pytest_sessionstart(session):
    workbook = ROOT / "data" / "mea" / "MEA_T_DATA.xlsx"
    summary = ROOT / "data" / "mea_processed" / "mea_compound_summary.csv"
    if workbook.exists() and not summary.exists():
        subprocess.run([sys.executable, str(SCRIPTS / "06_prepare_mea_data.py")], cwd=ROOT, check=True)
