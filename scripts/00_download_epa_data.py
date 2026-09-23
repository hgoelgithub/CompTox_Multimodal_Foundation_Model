"""STEP 0 -- Download the three EPA source datasets the pipeline is built on:
ToxCast/invitrodb (bioactivity), ToxValDB (hazard), and CPDat (exposure).
Output lands in data/source/<dataset>/, one subfolder per dataset.

Bulk training data do not require the CTX API key. The key is detected from
EPA_API_KEY for later targeted API workflows, but it is never written to disk.

The script resolves file URLs dynamically from the public Figshare API, so
download URLs are not hard-coded and can follow updated article versions.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
import requests

from core.paths import DATA_SOURCE


# Official EPA Figshare article IDs referenced from EPA's download pages.
ARTICLES = {
    "toxcast": 6062623,   # ToxCast invitroDB package (latest article version)
    "toxval": 20394501,  # ToxValDB article (latest article version)
    "cpdat": 5352997,    # CPDat bulk-data article
}

MODEL_EXTENSIONS = (
    ".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet",
    ".zip", ".gz", ".tgz", ".tar.gz", ".sql", ".sqlite", ".db"
)
SKIP_WORDS = ("plot", "figure", "image", "poster", "presentation", "manual", "guide")
CLOWDER_BASE = "https://clowder.edap-cluster.com"
CLOWDER_LINKS = {
    "toxval": "https://clowder.edap-cluster.com/datasets/61147fefe4b0856fdc65639b#folderId=62e184ebe4b055edffbfc22b&page=0",
}


def figshare_metadata(article_id: int, timeout: int = 30) -> dict:
    """Fetch an EPA Figshare article's metadata (title, version, file list/URLs)."""
    url = f"https://api.figshare.com/v2/articles/{article_id}"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def model_relevant(file_record: dict) -> bool:
    """True for tabular/database files the model can consume; false for figures, guides, etc."""
    name = str(file_record.get("name", "")).lower()
    if any(word in name for word in SKIP_WORDS):
        return False
    return name.endswith(MODEL_EXTENSIONS)


def download_file(url: str, destination: Path, expected_size=None, chunk_mb: int = 8):
    """Stream one file to disk. Idempotent: skips the download if a file of the
    expected size already exists there, so re-running this script is safe."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and expected_size and destination.stat().st_size == int(expected_size):
        print(f"  exists: {destination.name}")
        return
    if destination.exists() and not expected_size:
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


def clowder_dataset_id(link_url: str, timeout: int = 30) -> str | None:
    """Resolve a Clowder dataset id from a direct dataset link, or from a space
    link by picking the dataset in that space whose name looks like the MySQL
    database bundle (some EPA articles only link to a Clowder *space*, which
    can contain several datasets)."""
    match = re.search(r"/datasets/([0-9a-f]{24})", link_url)
    if match:
        return match.group(1)

    space_match = re.search(r"/spaces/([0-9a-f]{24})", link_url)
    if not space_match:
        return None

    response = requests.get(f"{CLOWDER_BASE}/spaces/{space_match.group(1)}", timeout=timeout)
    response.raise_for_status()
    datasets = re.findall(
        r'/datasets/([0-9a-f]{24})\?space=[0-9a-f]{24}[^>]*>\s*([^<]+)',
        response.text,
    )
    if not datasets:
        return None
    preferred = [item for item in datasets if "mysql" in item[1].lower() or "database" in item[1].lower()]
    return (preferred or datasets)[0][0]


def download_clowder_dataset(link_url: str, destination: Path, max_bytes: float) -> bool:
    """Download an entire Clowder dataset/folder as one archive (used for
    articles, like ToxValDB, that link out to Clowder instead of hosting the
    file on Figshare directly).

    Unlike download_file(), this does NOT check whether `destination` already
    exists before downloading -- re-running it always re-fetches the archive.
    Skip re-running this for a dataset you already have, or it will silently
    duplicate a multi-GB download."""
    dataset_id = clowder_dataset_id(link_url)
    if not dataset_id:
        print("    skipped: could not resolve the linked Clowder dataset")
        return False
    folder_id = parse_qs(urlsplit(link_url).fragment).get("folderId", [None])[0]
    endpoint = (
        f"{CLOWDER_BASE}/api/datasets/{dataset_id}/downloadFolder"
        if folder_id
        else f"{CLOWDER_BASE}/api/datasets/{dataset_id}/download"
    )
    params = {"bagit": "false", "tracking": "false"}
    if folder_id:
        params["folderId"] = folder_id
    response = requests.get(
        endpoint,
        params=params,
        stream=True,
        timeout=(30, 900),
    )
    response.raise_for_status()
    content_length = int(response.headers.get("content-length") or 0)
    if content_length and content_length > max_bytes:
        response.close()
        print("    skipped: linked dataset archive exceeds --max-file-gb")
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
        print(f"    partial file retained: {temporary}")
        print("    The server does not support resume; the next run must restart this archive.")
        return False
    temporary.replace(destination)
    print(f"  saved:  {destination}")
    return True


def main():
    """CLI entry point: for each selected dataset, fetch its Figshare metadata,
    pick the relevant files, and download whichever aren't already on disk."""
    parser = argparse.ArgumentParser(description="Download EPA bulk data from public Figshare releases.")
    parser.add_argument(
        "--datasets", nargs="+", choices=sorted(ARTICLES), default=list(ARTICLES),
        help="Datasets to inspect/download. Default: toxcast toxval cpdat."
    )
    parser.add_argument(
        "--mode", choices=["model", "all"], default="model",
        help="model = tabular/database files only; all = every file in the article."
    )
    parser.add_argument("--list-only", action="store_true", help="List remote files without downloading.")
    parser.add_argument(
        "--max-file-gb", type=float, default=5.0,
        help="Safety limit per file. Increase explicitly for very large database archives."
    )
    args = parser.parse_args()

    key = os.getenv("EPA_API_KEY")
    print("EPA_API_KEY:", "detected" if key else "not set (not required for these public bulk downloads)")
    print("Download directory:", DATA_SOURCE)

    DATA_SOURCE.mkdir(parents=True, exist_ok=True)
    max_bytes = args.max_file_gb * (1024 ** 3)

    successful_metadata = 0
    download_failed = False
    for dataset in args.datasets:
        article_id = ARTICLES[dataset]
        print(f"\n[{dataset.upper()}] Figshare article {article_id}")
        try:
            metadata = figshare_metadata(article_id)
        except requests.RequestException as exc:
            print("  Could not reach the public Figshare API.")
            print("  Check your internet connection and try again.")
            print("  Details:", exc)
            continue
        successful_metadata += 1
        print("Title:", metadata.get("title", ""))
        print("Version:", metadata.get("version", "latest"))
        files = metadata.get("files", [])
        if args.mode == "model":
            selected = [f for f in files if model_relevant(f)]
            # If metadata has only a small number of files, keep all non-document files.
            if not selected and len(files) <= 5:
                selected = files
        else:
            selected = files

        if not selected:
            print("No downloadable files were selected from this article metadata.")
            continue

        for record in selected:
            name = record.get("name", f"file_{record.get('id', 'unknown')}")
            size = int(record.get("size") or 0)
            size_gb = size / (1024 ** 3) if size else 0
            print(f"  {name} ({size_gb:.2f} GB)" if size else f"  {name}")
            if args.list_only:
                continue
            if size and size > max_bytes:
                print(f"    skipped: exceeds --max-file-gb={args.max_file_gb}")
                continue
            url = CLOWDER_LINKS.get(dataset, record.get("download_url"))
            if not url:
                print("    skipped: no download_url in Figshare metadata")
                continue
            destination = DATA_SOURCE / dataset / name
            if record.get("is_link_only"):
                if not download_clowder_dataset(
                    url, DATA_SOURCE / dataset / f"{dataset}_clowder_dataset.zip", max_bytes
                ):
                    download_failed = True
            else:
                download_file(url, destination, size or None)

    if successful_metadata == 0:
        print("\nNo remote metadata could be retrieved. Nothing was downloaded.")
        print("The script itself is ready; retry when network access is available.")
    elif download_failed:
        print("\nDownload interrupted. Remove the .part file before retrying if disk space is limited.")
    elif args.list_only:
        print("\nList-only complete. Re-run without --list-only to download files.")
    else:
        print("\nDownload step complete. Next: python scripts/01_normalize_epa_data.py")


if __name__ == "__main__":
    main()
