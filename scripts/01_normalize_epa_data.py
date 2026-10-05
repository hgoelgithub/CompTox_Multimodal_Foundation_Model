"""Step 1 - Normalize the EPA tables

Turns the raw downloads under data/source/ into four clean tables with fixed column names (the "schemas"
the rest of the pipeline expects):
  chemicals.csv     DTXSID, smiles, preferred_name, CASRN
  bioactivity.csv   DTXSID, program, assay, hitcall, ac50_uM, efficacy, signed_hitcall
  hazard.csv        DTXSID, metric, value, units (+ study context columns when present)
  exposure.csv      DTXSID, product_id, use_category, function_category

The normalizer is *generic*. Rather than knowing the exact file layout of ToxValDB or CPDat, it looks at the
header of every table-like file under data/source/, and treats a file as a source for a schema when it has all of
that schema's required columns under any of their usual names (`ALIASES`, e.g. `hitc` or `chit` both mean the
hit call). That copes with ToxValDB's many differently named spreadsheets without a hand-written file list.
The trade-off: an unrelated new file with similar column names could be picked up by accident. (The raw DSSTox
dump in data/source/dsstox/ is explicitly excluded, so it cannot overwrite the RDKit-canonicalised chemicals
table from step 01b.) Point to exact files with --chemicals / --bioactivity / --hazard / --exposure if needed.

If no compatible source exists for a table, an existing output is kept rather than deleted, so re-running is
safe. For ToxCast bioactivity and structures use steps 00a + 01a and 00b + 01b.

Input : data/source/**
Output: data/normalized/{chemicals,bioactivity,hazard,exposure}.csv, normalization_report.json
Run   : python scripts/01_normalize_epa_data.py [--no-extract] [--only hazard exposure]
"""

# %%
import argparse
import json
import re
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

# %% [markdown]
# ## 1. Paths and column aliases

# %%
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "data" / "source"
NORMALIZED = PROJECT_ROOT / "data" / "normalized"

TABLE_SUFFIXES = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet", ".sdf"}
ALIASES = {
    "DTXSID": ["dtxsid", "dsstox_substance_id", "dsstoxsid", "dsstox_sid", "sid"],
    "smiles": ["smiles", "qsar_ready_smiles", "canonicalsmiles", "canonical_smiles", "smiles_qsar"],
    "preferred_name": ["preferred_name", "chemical_name", "name", "chnm", "substance_name"],
    "CASRN": ["casrn", "cas", "cas_number", "casn"],
    "spid": ["spid", "sample_id", "sampleid"],
    "assay": ["assay", "aenm", "assay_endpoint_name", "assay_component_endpoint_name", "aeid", "endpoint"],
    "hitcall": ["hitcall", "hitc", "chit"],
    "ac50_uM": ["ac50_um", "ac50", "oldstyle_ac50"],
    "log10_ac50_uM": ["logac50", "log10_ac50", "log10_ac50_um"],
    "efficacy": ["efficacy", "modl_tp", "top", "response_top"],
    "program": ["program", "assay_program", "assay_source", "asnm"],
    "hazard_metric": ["metric", "toxval_type", "toxvaltype", "endpoint", "point_of_departure_type"],
    "hazard_value": ["value", "toxval_numeric", "toxval", "numeric_value", "point_estimate"],
    "hazard_units": ["units", "unit", "toxval_units"],
    "product_id": ["product_id", "productid", "prod_id"],
    "use_category": ["use_category", "product_use_category", "puc", "general_use", "use"],
    "function_category": ["function_category", "functional_use", "function"],
}
# Columns a file must contain (under any alias) to be considered a source for each table.
REQUIRED = {"chemicals": ["DTXSID", "smiles"], "bioactivity": ["assay", "hitcall"],
            "hazard": ["DTXSID", "hazard_metric", "hazard_value"], "exposure": ["DTXSID"]}
TABLES = list(REQUIRED)

# %% [markdown]
# ## 2. Finding and reading source files
# * `extract_archives` unpacks zip / tar archives next to themselves (one level deep, so documentation bundles
#   are not unpacked recursively).
# * `discover_tables` lists candidate table files, skipping GitHub dumps, QC-failed extracts, documentation and
#   the raw DSSTox folder.
# * `read_any` loads a file by extension; `columns_of` reads just the header cheaply.

# %%
def norm(text):
    """Fold a column name: lower case, runs of non-alphanumerics -> '_' ('DTXSID ' == 'dtxsid')."""
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


def alias_column(columns, canonical):
    """The column in `columns` that matches one of the aliases of `canonical`, or None."""
    lookup = {norm(c): c for c in columns}
    for alias in ALIASES[canonical]:
        if norm(alias) in lookup:
            return lookup[norm(alias)]
    return None


def extract_archives(root, max_depth=1):
    for path in list(root.rglob("*")):
        if not path.is_file() or sum(part.endswith("_extracted") for part in path.parts) >= max_depth:
            continue
        target = path.parent / (path.name + "_extracted")
        if target.exists():
            continue
        name = path.name.lower()
        try:
            if name.endswith(".zip"):
                target.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(path) as archive:
                    archive.extractall(target)
            elif name.endswith((".tar.gz", ".tgz", ".tar")):
                target.mkdir(parents=True, exist_ok=True)
                with tarfile.open(path) as archive:
                    archive.extractall(target, filter="data")
        except (zipfile.BadZipFile, tarfile.TarError) as exc:
            raise RuntimeError(f"Could not extract {path}. Remove the incomplete extraction and retry.") from exc


def discover_tables(root):
    excluded = ("github input files", "qc_status fail", "documentation", f"{root.name}/dsstox/")
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in TABLE_SUFFIXES and not any(x in str(p).lower() for x in excluded))


def read_any(path, nrows=None):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, nrows=nrows, low_memory=False)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep=None, engine="python", nrows=nrows)   # auto-detect the separator
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, nrows=nrows)
    if suffix == ".parquet":
        df = pd.read_parquet(path)
        return df.head(nrows) if nrows else df
    raise ValueError(f"Unsupported table file: {path}")


def columns_of(path):
    try:
        return {"DTXSID", "smiles"} if path.suffix.lower() == ".sdf" else set(read_any(path, nrows=5).columns)
    except Exception:
        return set()


def sdf_chemicals(path):
    """DTXSID and canonical SMILES from a structure-data file (for releases that ship structures as SDF)."""
    rows = []
    for mol in Chem.SDMolSupplier(str(path), removeHs=False):
        if mol is None:
            continue
        props = mol.GetPropsAsDict()
        keys = {norm(k): k for k in props}
        key = next((keys[x] for x in ["dtxsid", "dsstox_substance_id", "pubchem_external_data_source"] if x in keys), None)
        if not key:
            continue
        match = re.search(r"DTXSID\d+", str(props[key]))
        rows.append({"DTXSID": match.group(0) if match else str(props[key]), "smiles": Chem.MolToSmiles(mol)})
    return pd.DataFrame(rows)


def load_source(path):
    return sdf_chemicals(path) if path.suffix.lower() == ".sdf" else read_any(path)

# %% [markdown]
# ## 3. One normalizer per table
# Each function reshapes a raw table into its schema, or returns an empty table if the file lacks the needed
# columns. Notes:
# * **Bioactivity**: ToxCast v4 hit calls are continuous in [-1, 1]. Values outside that range are discarded;
#   `signed_hitcall` keeps the original and `hitcall` is clipped at 0. Potency and efficacy are masked for rows
#   below the activity threshold (0.9), since they are not meaningful for inactive assays. Some tables give only
#   a sample id, which is translated to a DTXSID through a sample-to-chemical table.
# * **Hazard**: keeps the metric, numeric value and units, plus study type / species / route / duration /
#   source when available.
# * **Exposure**: the use category falls back to the function category if there is no use column.

# %%
def normalize_chemicals(df):
    dtxsid, smiles = alias_column(df.columns, "DTXSID"), alias_column(df.columns, "smiles")
    if not dtxsid or not smiles:
        return pd.DataFrame()
    out = pd.DataFrame({"DTXSID": df[dtxsid].astype(str), "smiles": df[smiles]})
    name, cas = alias_column(df.columns, "preferred_name"), alias_column(df.columns, "CASRN")
    if name:
        out["preferred_name"] = df[name]
    if cas:
        out["CASRN"] = df[cas]
    out = out.replace({"nan": np.nan}).dropna(subset=["DTXSID", "smiles"])
    return out[out["DTXSID"].str.startswith("DTXSID", na=False)].drop_duplicates("DTXSID")


def build_spid_map(files):
    """The largest table that maps a sample id (spid) to a DTXSID, or an empty frame."""
    best, best_rows = None, 0
    for path in files:
        if path.suffix.lower() == ".sdf":
            continue
        try:
            head = read_any(path, nrows=5)
            spid, dtxsid = alias_column(head.columns, "spid"), alias_column(head.columns, "DTXSID")
            if not spid or not dtxsid:
                continue
            df = read_any(path)
            if len(df) > best_rows:
                best_rows, best = len(df), df[[spid, dtxsid]].rename(columns={spid: "spid", dtxsid: "DTXSID"})
        except Exception:
            continue
    return best.dropna().drop_duplicates("spid") if best is not None else pd.DataFrame()


def normalize_bioactivity(df, spid_map=None, active_threshold=0.90):
    assay, hit = alias_column(df.columns, "assay"), alias_column(df.columns, "hitcall")
    if not assay or not hit:
        return pd.DataFrame()
    dtxsid, spid = alias_column(df.columns, "DTXSID"), alias_column(df.columns, "spid")
    work = df.copy()
    if not dtxsid:
        if spid and spid_map is not None and not spid_map.empty:
            work = work.merge(spid_map, left_on=spid, right_on="spid", how="left")
            dtxsid = "DTXSID"
        else:
            return pd.DataFrame()

    out = pd.DataFrame({"DTXSID": work[dtxsid].astype(str), "assay": work[assay].astype("string"),
                        "hitcall": pd.to_numeric(work[hit], errors="coerce")})
    out.loc[~out["hitcall"].between(-1, 1, inclusive="both"), "hitcall"] = np.nan
    out["signed_hitcall"] = out["hitcall"]
    out["hitcall"] = out["hitcall"].clip(lower=0)
    program = alias_column(work.columns, "program")
    out["program"] = work[program].astype(str) if program else "ToxCast/Tox21"

    ac50, log_ac50 = alias_column(work.columns, "ac50_uM"), alias_column(work.columns, "log10_ac50_uM")
    if ac50:
        out["ac50_uM"] = pd.to_numeric(work[ac50], errors="coerce")
    elif log_ac50:
        out["ac50_uM"] = np.power(10.0, pd.to_numeric(work[log_ac50], errors="coerce"))   # log10 micromolar -> uM
    efficacy = alias_column(work.columns, "efficacy")
    if efficacy:
        out["efficacy"] = pd.to_numeric(work[efficacy], errors="coerce")

    inactive = out["hitcall"].notna() & (out["hitcall"] < active_threshold)
    for column in ["ac50_uM", "efficacy"]:
        if column in out:
            out.loc[inactive, column] = np.nan
    out = out[out["DTXSID"].str.startswith("DTXSID", na=False)]
    return out.dropna(subset=["DTXSID", "assay", "hitcall"])


def normalize_hazard(df):
    dtxsid, metric, value = (alias_column(df.columns, k) for k in ("DTXSID", "hazard_metric", "hazard_value"))
    if not dtxsid or not metric or not value:
        return pd.DataFrame()
    out = pd.DataFrame({"DTXSID": df[dtxsid].astype(str), "metric": df[metric].astype(str),
                        "value": pd.to_numeric(df[value], errors="coerce")})
    units = alias_column(df.columns, "hazard_units")
    if units:
        out["units"] = df[units]
    for extra in ["study_type", "species", "route", "duration", "source"]:
        column = next((c for c in df.columns if norm(c) == extra), None)
        if column:
            out[extra] = df[column]
    return out[out["DTXSID"].str.startswith("DTXSID", na=False)].dropna(subset=["value"])


def normalize_exposure(df):
    dtxsid, use = alias_column(df.columns, "DTXSID"), alias_column(df.columns, "use_category")
    product, function = alias_column(df.columns, "product_id"), alias_column(df.columns, "function_category")
    if not dtxsid or (not use and not function):
        return pd.DataFrame()
    out = pd.DataFrame({"DTXSID": df[dtxsid].astype(str)})
    out["product_id"] = df[product].astype("string") if product else pd.NA
    out["use_category"] = df[use].astype("string") if use else df[function].astype("string")
    if function:
        out["function_category"] = df[function]
    return out[out["DTXSID"].str.startswith("DTXSID", na=False)]


NORMALIZERS = {"chemicals": normalize_chemicals, "bioactivity": normalize_bioactivity,
               "hazard": normalize_hazard, "exposure": normalize_exposure}

# %% [markdown]
# ## 4. Run
# For each requested table, `normalize` picks every source file (or the one you named) with the required
# columns, normalizes and concatenates them, drops duplicates and writes the CSV atomically. A
# `normalization_report.json` records which files fed which table and any warnings. The step fails at the end
# only if `chemicals.csv` or `bioactivity.csv` is still missing, since the model cannot be trained without them.

# %%
def normalize(only=tuple(TABLES), extract=True, extract_depth=1, manual=None, output_dir=NORMALIZED):
    manual = manual or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if extract:
        extract_archives(SOURCE_DIR, extract_depth)
    files = discover_tables(SOURCE_DIR)
    columns = {p: columns_of(p) for p in files}
    report = {"source_files_seen": len(files), "outputs": {}, "sources": {}, "warnings": []}
    spid_map = None

    for kind in only:
        chosen = [Path(manual[kind])] if manual.get(kind) else [
            p for p, c in columns.items() if all(alias_column(c, key) for key in REQUIRED[kind])]
        if kind == "bioactivity" and not manual.get(kind):
            chosen = [p for p in chosen if alias_column(columns[p], "DTXSID") or alias_column(columns[p], "spid")]
        if kind == "exposure" and not manual.get(kind):
            chosen = [p for p in chosen if alias_column(columns[p], "use_category") or alias_column(columns[p], "function_category")]
        frames, sources = [], []
        for path in chosen:
            try:
                raw = load_source(path)
                if kind == "bioactivity" and not alias_column(raw.columns, "DTXSID"):
                    spid_map = build_spid_map(files) if spid_map is None else spid_map
                    frame = normalize_bioactivity(raw, spid_map)
                else:
                    frame = NORMALIZERS[kind](raw)
                if not frame.empty:
                    frame["source_file"] = str(path.relative_to(SOURCE_DIR)) if path.is_relative_to(SOURCE_DIR) else str(path)
                    frames.append(frame)
                    sources.append(str(path))
            except Exception as exc:
                if manual.get(kind):
                    raise
                report["warnings"].append(f"{path.name}: {exc}")
        destination = output_dir / f"{kind}.csv"
        if frames:
            table = pd.concat(frames, ignore_index=True).drop_duplicates()
            if kind == "chemicals":
                table = table.drop_duplicates("DTXSID")
            temporary = destination.with_suffix(".csv.part")
            table.to_csv(temporary, index=False)
            temporary.replace(destination)
            report["outputs"][kind], report["sources"][kind] = len(table), sources
            print(kind, len(table), "rows from", len(sources), "files", flush=True)
        elif destination.exists():
            report["warnings"].append(f"{kind}: kept the existing output; no new compatible source found.")
        else:
            report["warnings"].append(f"{kind}: missing. For ToxCast run 00a then 01a; for structures run 00b then 01b.")

    (output_dir / "normalization_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    missing = [name for name in ("chemicals", "bioactivity") if not (output_dir / f"{name}.csv").exists()]
    if missing:
        raise SystemExit("Training prerequisites still missing: " + ", ".join(missing))

# %% Command line
def main():
    parser = argparse.ArgumentParser(description="Normalize EPA source tables into the four pipeline schemas.")
    for name in TABLES:
        parser.add_argument(f"--{name}", help=f"use this file as the {name} source instead of auto-discovery")
    parser.add_argument("--only", nargs="+", choices=TABLES, default=TABLES)
    parser.add_argument("--no-extract", action="store_true", help="do not unpack archives under data/source")
    parser.add_argument("--extract-depth", type=int, default=1)
    args = parser.parse_args()
    if args.extract_depth < 0:
        raise ValueError("Extraction depth must not be negative.")
    normalize(args.only, not args.no_extract, args.extract_depth, {name: getattr(args, name) for name in TABLES})


if __name__ == "__main__":
    main()
