"""Step 0b - Download the DSSTox chemical structures

The ToxCast database identifies chemicals by DTXSID but has no molecular structures. This step downloads
EPA's DSSTox structure dump (DTXSID -> SMILES, name, CASRN), which step 01b joins onto the chemical list.
The download size is checked against the known size of the release, so a truncated or interrupted download is
never mistaken for a complete one.

Output: data/source/dsstox/DSSTox_CCD_dump_12092025_CSVs.zip (+ .provenance.json)
Run   : python scripts/00b_download_structures.py
"""

# %%
import json
import zipfile
from pathlib import Path

import requests

# %% [markdown]
# ## 1. Settings

# %%
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESTINATION = PROJECT_ROOT / "data" / "source" / "dsstox" / "DSSTox_CCD_dump_12092025_CSVs.zip"

FILE_ID = "69529775e4b0731a616efc4b"           # file id on EPA's Clowder data server
URL = f"https://clowder.edap-cluster.com/api/files/{FILE_ID}/blob"
EXPECTED_SIZE = 289_824_966                    # bytes, for the December 2025 release

# %% [markdown]
# ## 2. Download and verify
# The file is streamed to a `.part` file, its size is checked, and only then is it renamed. The archive's
# contents are listed and a small provenance file records where the data came from.

# %%
def download_structures():
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    if not DESTINATION.exists() or DESTINATION.stat().st_size != EXPECTED_SIZE:
        part = DESTINATION.with_suffix(".zip.part")
        with requests.get(URL, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            with part.open("wb") as out:
                for chunk in response.iter_content(8 * 1024 * 1024):
                    out.write(chunk)
        if part.stat().st_size != EXPECTED_SIZE:
            raise IOError("Incomplete DSSTox download; run the step again.")
        part.replace(DESTINATION)
    with zipfile.ZipFile(DESTINATION) as archive:
        print("Downloaded:", DESTINATION)
        for item in archive.infolist():
            print(" ", item.filename, item.file_size)
    DESTINATION.with_suffix(".provenance.json").write_text(
        json.dumps({"url": URL, "release": "December 2025", "size": EXPECTED_SIZE}, indent=2))

# %% Command line
if __name__ == "__main__":
    download_structures()
