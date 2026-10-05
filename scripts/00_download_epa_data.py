"""Step 0 - Download the EPA source data

Downloads the three public EPA datasets the pipeline is built on, one sub-folder each under data/source/:
  toxcast   ToxCast / invitrodb   in-vitro bioactivity of ~10,000 chemicals (a large MySQL dump)
  toxval    ToxValDB              hazard / toxicity values from many studies
  cpdat     CPDat                 consumer-product and use data (exposure)

File locations are looked up at run time from the public Figshare API (so the script follows new article
versions instead of hard-coding URLs). Some articles only link out to the EPA "Clowder" data server; those
archives are fetched from there. No API key is needed for these bulk downloads. The key in `EPA_API_KEY` (if
set) is only reported, never written to disk; it is meant for targeted lookups later.

Downloads are large (several GB). Start with a listing (`--list-only`) to see what would be fetched.
Files that are already on disk with the right size are skipped, so re-running is safe.

Output: data/source/<dataset>/...
Run   : python scripts/00_download_epa_data.py --list-only
        python scripts/00_download_epa_data.py --datasets toxval cpdat
"""

# %%
import argparse
import os
import re
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import requests

# %% [markdown]
# ## 1. Settings

# %%
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "data" / "source"

# Official EPA Figshare article ids (the API returns the latest version of each article).
ARTICLES = {"toxcast": 6062623, "toxval": 20394501, "cpdat": 5352997}

USEFUL_EXTENSIONS = (".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet", ".zip", ".gz", ".tgz", ".tar.gz",
                     ".sql", ".sqlite", ".db")
SKIP_WORDS = ("plot", "figure", "image", "poster", "presentation", "manual", "guide")

CLOWDER_BASE = "https://clowder.edap-cluster.com"
CLOWDER_LINKS = {  # articles whose real data lives on the Clowder server
    "toxval": f"{CLOWDER_BASE}/datasets/61147fefe4b0856fdc65639b#folderId=62e184ebe4b055edffbfc22b&page=0",
}

# %% [markdown]
# ## 2. Finding what to download
# `figshare_metadata` asks Figshare for an article's file list. `is_useful` keeps tabular / database / archive
# files and skips figures and documentation.

# %%
def figshare_metadata(article_id, timeout=30):
    response = requests.get(f"https://api.figshare.com/v2/articles/{article_id}", timeout=timeout)
    response.raise_for_status()
    return response.json()


def is_useful(file_record):
    name = str(file_record.get("name", "")).lower()
    return not any(word in name for word in SKIP_WORDS) and name.endswith(USEFUL_EXTENSIONS)

# %% [markdown]
# ## 3. Downloading
# * `download_file` streams one file to a `.part` file and renames it when complete, and skips files already
#   on disk with the expected size.
# * `download_clowder_dataset` fetches a whole Clowder dataset or folder as one archive. It cannot resume
#   an interrupted download, and it does **not** check whether the archive already exists, so do not re-run it
#   for a dataset you already have.

# %%
def download_file(url, destination, expected_size=None, chunk_mb=8):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (not expected_size or destination.stat().st_size == int(expected_size)):
        print(f"  exists: {destination.name}")
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_mb * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
    temporary.replace(destination)
    print(f"  saved:  {destination}")


def clowder_dataset_id(link_url, timeout=30):
    """Dataset id from a Clowder link; for a *space* link, the dataset that looks like the database bundle."""
    match = re.search(r"/datasets/([0-9a-f]{24})", link_url)
    if match:
        return match.group(1)
    space = re.search(r"/spaces/([0-9a-f]{24})", link_url)
    if not space:
        return None
    response = requests.get(f"{CLOWDER_BASE}/spaces/{space.group(1)}", timeout=timeout)
    response.raise_for_status()
    datasets = re.findall(r'/datasets/([0-9a-f]{24})\?space=[0-9a-f]{24}[^>]*>\s*([^<]+)', response.text)
    if not datasets:
        return None
    preferred = [d for d in datasets if "mysql" in d[1].lower() or "database" in d[1].lower()]
    return (preferred or datasets)[0][0]


def download_clowder_dataset(link_url, destination, max_bytes):
    dataset_id = clowder_dataset_id(link_url)
    if not dataset_id:
        print("    skipped: could not resolve the linked Clowder dataset")
        return False
    folder_id = parse_qs(urlsplit(link_url).fragment).get("folderId", [None])[0]
    endpoint = f"{CLOWDER_BASE}/api/datasets/{dataset_id}/" + ("downloadFolder" if folder_id else "download")
    params = {"bagit": "false", "tracking": "false"}
    if folder_id:
        params["folderId"] = folder_id
    response = requests.get(endpoint, params=params, stream=True, timeout=(30, 900))
    response.raise_for_status()
    size = int(response.headers.get("content-length") or 0)
    if size and size > max_bytes:
        response.close()
        print("    skipped: the archive is larger than the size limit")
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with response, temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
    except requests.exceptions.RequestException as exc:
        print(f"    download interrupted: {exc}")
        print(f"    partial file kept: {temporary} (the server cannot resume; the next run restarts this archive)")
        return False
    temporary.replace(destination)
    print(f"  saved:  {destination}")
    return True

# %% [markdown]
# ## 4. Run
# For each selected dataset: fetch its file list, pick the useful files, and download those that are not on
# disk yet (or just list them with `list_only=True`).

# %%
def download(datasets=tuple(ARTICLES), mode="model", list_only=True, max_file_gb=5.0):
    print("EPA_API_KEY:", "detected" if os.getenv("EPA_API_KEY") else "not set (not needed for these public downloads)")
    print("Download directory:", SOURCE_DIR)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    max_bytes = max_file_gb * 1024 ** 3
    reached, failed = 0, False
    for dataset in datasets:
        print(f"\n[{dataset.upper()}] Figshare article {ARTICLES[dataset]}")
        try:
            metadata = figshare_metadata(ARTICLES[dataset])
        except requests.RequestException as exc:
            print("  Could not reach the Figshare API. Check your connection and try again.\n  Details:", exc)
            continue
        reached += 1
        print("Title:", metadata.get("title", ""), "| version:", metadata.get("version", "latest"))
        files = metadata.get("files", [])
        selected = [f for f in files if is_useful(f)] if mode == "model" else files
        if mode == "model" and not selected and len(files) <= 5:
            selected = files
        if not selected:
            print("No downloadable files were selected.")
            continue
        for record in selected:
            name = record.get("name", f"file_{record.get('id', 'unknown')}")
            size = int(record.get("size") or 0)
            print(f"  {name} ({size / 1024 ** 3:.2f} GB)" if size else f"  {name}")
            if list_only:
                continue
            if size and size > max_bytes:
                print(f"    skipped: larger than max_file_gb={max_file_gb}")
                continue
            url = CLOWDER_LINKS.get(dataset, record.get("download_url"))
            if not url:
                print("    skipped: no download url in the metadata")
                continue
            if record.get("is_link_only"):
                archive = SOURCE_DIR / dataset / f"{dataset}_clowder_dataset.zip"
                failed |= not download_clowder_dataset(url, archive, max_bytes)
            else:
                download_file(url, SOURCE_DIR / dataset / name, size or None)
    if reached == 0:
        print("\nNo remote metadata could be retrieved; nothing was downloaded.")
    elif failed:
        print("\nA download was interrupted. Delete the .part file before retrying if disk space is short.")
    elif list_only:
        print("\nListing complete. Run again with list_only=False (or without --list-only) to download.")
    else:
        print("\nDone. Next: step 01 (or 00a / 01a for the ToxCast database).")

# %% Command line
def main():
    parser = argparse.ArgumentParser(description="Download EPA bulk data from the public Figshare / Clowder releases.")
    parser.add_argument("--datasets", nargs="+", choices=sorted(ARTICLES), default=list(ARTICLES))
    parser.add_argument("--mode", choices=["model", "all"], default="model",
                        help="model = tabular/database files only; all = every file in the article")
    parser.add_argument("--list-only", action="store_true", help="list remote files without downloading")
    parser.add_argument("--max-file-gb", type=float, default=5.0, help="skip files larger than this")
    args = parser.parse_args()
    download(args.datasets, args.mode, args.list_only, args.max_file_gb)


if __name__ == "__main__":
    main()
