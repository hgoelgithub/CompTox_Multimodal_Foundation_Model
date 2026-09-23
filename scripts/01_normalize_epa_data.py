"""STEP 1 -- Normalize downloaded EPA tables into the four schemas used by the model.

This is a *generic* normalizer: rather than knowing ToxVal's or CPDat's exact
file layout, it scans every table-like file under data/source/, scores each
one by how many of a schema's expected columns it has (via ALIASES), and
picks the best match(es) per schema. That is what lets it handle ToxValDB's
~50 differently-named source spreadsheets and CPDat's files without a
hand-written file list -- but it also means a new, unrelated file dropped
under data/source/ with similarly-named columns could get picked up by
accident, which is why data/source/dsstox/ is explicitly excluded (see
discover_tables): that folder's raw DSSTox dump would otherwise outrank
01b_prepare_structures.py's carefully matched/canonicalized chemicals.csv.

Outputs:
  data/normalized/chemicals.csv
  data/normalized/bioactivity.csv
  data/normalized/hazard.csv       (when recognized)
  data/normalized/exposure.csv     (when recognized)

The normalizer can auto-discover common EPA/ToxCast/ToxVal/CPDat column names,
or you can point to exact files with command-line arguments.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from core.paths import DATA_SOURCE, DATA_NORMALIZED


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


def norm(text) -> str:
    """Fold a column name to a comparable form: lowercase, non-alnum runs -> '_'.
    So 'DTXSID', 'dtxsid', and 'DTXSID ' all normalize the same way."""
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


def alias_column(columns, canonical):
    """Find which of `columns` matches one of ALIASES[canonical]'s known names
    for this field (e.g. 'hitc' or 'chit' both mean the canonical 'hitcall')."""
    mapping = {norm(c): c for c in columns}
    for alias in ALIASES[canonical]:
        if norm(alias) in mapping:
            return mapping[norm(alias)]
    return None


def extract_archives(root: Path, max_depth: int = 1):
    """Extract top-level archives without recursively unpacking document bundles."""
    for path in list(root.rglob("*")):
        if not path.is_file():
            continue
        archive_depth = sum(part.endswith("_extracted") for part in path.parts)
        if archive_depth >= max_depth:
            continue
        name = path.name.lower()
        target = path.parent / (path.name + "_extracted")
        if target.exists():
            continue
        try:
            if name.endswith(".zip"):
                target.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(path) as zf:
                    zf.extractall(target)
            elif name.endswith((".tar.gz", ".tgz", ".tar")):
                target.mkdir(parents=True, exist_ok=True)
                with tarfile.open(path) as tf:
                    tf.extractall(target, filter="data")
        except (zipfile.BadZipFile, tarfile.TarError) as exc:
            raise RuntimeError(f"Archive extraction failed: {path}. Remove incomplete extraction before retrying.") from exc


def discover_tables(root: Path):
    """List every candidate table file under data/source/ (recursively), excluding
    GitHub repo dumps, QC-failed extracts, plain documentation, and the raw
    DSSTox dump (owned by 00b/01b -- see the module docstring)."""
    files = []
    # data/source/dsstox is owned by 00b_download_structures.py /
    # 01b_prepare_structures.py: it holds the raw, unfiltered DSSTox dump and
    # must never be auto-discovered here, or a "chemicals" pass could
    # silently overwrite 01b's RDKit-canonicalized, MEA-matched chemicals.csv
    # with unfiltered rows.
    excluded = ("github input files", "qc_status fail", "documentation", f"{root.name}/dsstox/")
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in TABLE_SUFFIXES and not any(x in str(path).lower() for x in excluded):
            files.append(path)
    return sorted(files)


def read_table(path: Path, nrows=None):
    """Load one table file by extension (csv/tsv/txt/xlsx/xls/parquet).
    `nrows` lets callers peek at just the header + a few rows cheaply."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, nrows=nrows, low_memory=False)
    if suffix in {".tsv", ".txt"}:
        # Auto-separator works well for heterogeneous EPA text exports.
        return pd.read_csv(path, sep=None, engine="python", nrows=nrows)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, nrows=nrows)
    if suffix == ".parquet":
        df = pd.read_parquet(path)
        return df.head(nrows) if nrows else df
    raise ValueError(f"Unsupported tabular file: {path}")


def sdf_chemicals(path: Path):
    """Extract DTXSID + canonical SMILES from an SDF (structure-data file),
    for the rare EPA release that ships structures as SDF instead of a table."""
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    rows = []
    for mol in supplier:
        if mol is None:
            continue
        props = mol.GetPropsAsDict()
        keys = {norm(k): k for k in props}
        dtx_key = next((keys[x] for x in ["dtxsid", "dsstox_substance_id", "pubchem_external_data_source"] if x in keys), None)
        if not dtx_key:
            continue
        dtxsid = str(props[dtx_key])
        match = re.search(r"DTXSID\d+", dtxsid)
        if match:
            dtxsid = match.group(0)
        rows.append({"DTXSID": dtxsid, "smiles": Chem.MolToSmiles(mol)})
    return pd.DataFrame(rows)


def inspect_columns(path: Path):
    """Cheaply return a file's column names (and a small row count) without
    loading the whole table -- used to score/select candidate source files."""
    try:
        if path.suffix.lower() == ".sdf":
            return {"DTXSID", "smiles"}, 0
        df = read_table(path, nrows=5)
        return set(df.columns), len(df)
    except Exception:
        return set(), 0


def score_file(path: Path, groups):
    """Weighted count of how many `groups` (canonical, weight) columns this
    file has -- higher means a better candidate match for a schema.
    Currently unused: main() below selects *every* file that has all of a
    schema's required columns rather than ranking by score (see `chosen` in
    main()); score_file/best_file were an earlier "pick just the single best
    file" approach kept here in case a future caller wants it."""
    cols, _ = inspect_columns(path)
    score = 0
    for canonical, weight in groups:
        if alias_column(cols, canonical):
            score += weight
    return score


def best_file(files, groups):
    """Return the single highest-scoring file for `groups`, or None if nothing
    matches at all. See score_file's docstring: not currently called by main()."""
    scored = sorted(((score_file(p, groups), p) for p in files), reverse=True)
    return scored[0][1] if scored and scored[0][0] > 0 else None


def normalize_chemicals(df):
    """Reshape a raw source table into the chemicals.csv schema (DTXSID,
    smiles, preferred_name, CASRN), if it has the required columns."""
    d = alias_column(df.columns, "DTXSID")
    s = alias_column(df.columns, "smiles")
    if not d or not s:
        return pd.DataFrame()
    out = pd.DataFrame({"DTXSID": df[d].astype(str), "smiles": df[s]})
    n = alias_column(df.columns, "preferred_name")
    c = alias_column(df.columns, "CASRN")
    if n: out["preferred_name"] = df[n]
    if c: out["CASRN"] = df[c]
    out = out.replace({"nan": np.nan}).dropna(subset=["DTXSID", "smiles"])
    out = out[out["DTXSID"].str.startswith("DTXSID", na=False)]
    return out.drop_duplicates("DTXSID")


def build_spid_map(files):
    """Some bioactivity tables identify chemicals by sample id (spid) rather
    than DTXSID directly. Find the largest file that maps spid -> DTXSID, so
    normalize_bioactivity() can join through it when a bioactivity file lacks
    its own DTXSID column."""
    best = None
    best_rows = 0
    for path in files:
        if path.suffix.lower() == ".sdf":
            continue
        try:
            head = read_table(path, nrows=5)
            sp = alias_column(head.columns, "spid")
            dt = alias_column(head.columns, "DTXSID")
            if not sp or not dt:
                continue
            df = read_table(path)
            if len(df) > best_rows:
                best_rows = len(df)
                best = df[[sp, dt]].rename(columns={sp: "spid", dt: "DTXSID"})
        except Exception:
            continue
    return best.dropna().drop_duplicates("spid") if best is not None else pd.DataFrame()


def normalize_bioactivity(df, spid_map=None, active_threshold=0.90):
    """Reshape a raw source table into the bioactivity.csv schema (DTXSID,
    program, assay, hitcall, ac50_uM, efficacy, signed_hitcall). Resolves
    DTXSID via `spid_map` when the table only has a sample id, and masks
    potency/efficacy for rows below `active_threshold` since those aren't
    treated as real active-assay points of departure."""
    assay = alias_column(df.columns, "assay")
    hit = alias_column(df.columns, "hitcall")
    if not assay or not hit:
        return pd.DataFrame()

    dt = alias_column(df.columns, "DTXSID")
    sp = alias_column(df.columns, "spid")
    work = df.copy()
    if not dt:
        if sp and spid_map is not None and not spid_map.empty:
            work = work.merge(spid_map, left_on=sp, right_on="spid", how="left")
            dt = "DTXSID"
        else:
            return pd.DataFrame()

    out = pd.DataFrame()
    out["DTXSID"] = work[dt].astype(str)
    out["assay"] = work[assay].astype("string")
    out["hitcall"] = pd.to_numeric(work[hit], errors="coerce")
    # Invitrodb v4+ uses continuous hit calls in [0, 1]. Preserve the
    # continuous value, but discard invalid values outside the documented range.
    out.loc[~out["hitcall"].between(-1, 1, inclusive="both"), "hitcall"] = np.nan
    out["signed_hitcall"]=out["hitcall"]
    out["hitcall"]=out["hitcall"].clip(lower=0)

    program = alias_column(work.columns, "program")
    out["program"] = work[program].astype(str) if program else "ToxCast/Tox21"

    ac = alias_column(work.columns, "ac50_uM")
    logac = alias_column(work.columns, "log10_ac50_uM")
    if ac:
        out["ac50_uM"] = pd.to_numeric(work[ac], errors="coerce")
    elif logac:
        # tcpl/invitrodb matrix concentrations are log10 micromolar.
        values = pd.to_numeric(work[logac], errors="coerce")
        out["ac50_uM"] = np.power(10.0, values)

    eff = alias_column(work.columns, "efficacy")
    if eff:
        out["efficacy"] = pd.to_numeric(work[eff], errors="coerce")

    # EPA examples for invitrodb v4+ commonly use hitc >= 0.90 as active.
    # Potency/response-top values below that threshold are not treated as active
    # assay PODs and are therefore masked for those rows.
    inactive = out["hitcall"].notna() & (out["hitcall"] < active_threshold)
    for column in ["ac50_uM", "efficacy"]:
        if column in out:
            out.loc[inactive, column] = np.nan

    out = out[out["DTXSID"].str.startswith("DTXSID", na=False)]
    return out.dropna(subset=["DTXSID", "assay", "hitcall"])


def normalize_hazard(df):
    """Reshape a raw ToxValDB-style table into the hazard.csv schema (DTXSID,
    metric, value, units, plus study context columns when present)."""
    dt = alias_column(df.columns, "DTXSID")
    metric = alias_column(df.columns, "hazard_metric")
    value = alias_column(df.columns, "hazard_value")
    if not dt or not metric or not value:
        return pd.DataFrame()
    out = pd.DataFrame({
        "DTXSID": df[dt].astype(str),
        "metric": df[metric].astype(str),
        "value": pd.to_numeric(df[value], errors="coerce"),
    })
    units = alias_column(df.columns, "hazard_units")
    if units: out["units"] = df[units]
    for extra in ["study_type", "species", "route", "duration", "source"]:
        source = next((c for c in df.columns if norm(c) == extra), None)
        if source: out[extra] = df[source]
    return out[out["DTXSID"].str.startswith("DTXSID", na=False)].dropna(subset=["value"])


def normalize_exposure(df):
    """Reshape a raw CPDat-style table into the exposure.csv schema (DTXSID,
    product_id, use_category, function_category)."""
    dt = alias_column(df.columns, "DTXSID")
    use = alias_column(df.columns, "use_category")
    product = alias_column(df.columns, "product_id")
    function = alias_column(df.columns, "function_category")
    if not dt or (not use and not function):
        return pd.DataFrame()
    out = pd.DataFrame({"DTXSID": df[dt].astype(str)})
    out["product_id"] = df[product].astype("string") if product else pd.NA
    out["use_category"] = df[use].astype("string") if use else df[function].astype("string")
    if function: out["function_category"] = df[function]
    return out[out["DTXSID"].str.startswith("DTXSID", na=False)]


def load_manual(path):
    """Load a file the user pointed to explicitly (--chemicals/--bioactivity/
    --hazard/--exposure), bypassing auto-discovery entirely."""
    p = Path(path)
    if p.suffix.lower() == ".sdf":
        return sdf_chemicals(p)
    return read_table(p)


def main():
    """CLI entry point: for each of chemicals/bioactivity/hazard/exposure (or
    just the ones in --only), find every matching source file, normalize and
    concatenate them, and atomically overwrite the corresponding CSV. Safe to
    re-run: if no matching source is found, the existing output is kept
    as-is rather than being deleted."""
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['chemicals','bioactivity','hazard','exposure']:
        parser.add_argument('--'+name)
    parser.add_argument('--only',nargs='+',choices=['chemicals','bioactivity','hazard','exposure'],default=['chemicals','bioactivity','hazard','exposure'])
    parser.add_argument('--no-extract',action='store_true')
    parser.add_argument('--extract-depth',type=int,default=1)
    args=parser.parse_args()
    if args.extract_depth<0: raise ValueError('Extraction depth must be nonnegative')
    DATA_NORMALIZED.mkdir(parents=True,exist_ok=True)
    if not args.no_extract: extract_archives(DATA_SOURCE,args.extract_depth)
    files=discover_tables(DATA_SOURCE)
    columns={p:inspect_columns(p)[0] for p in files}
    schemas={'chemicals':['DTXSID','smiles'],'bioactivity':['assay','hitcall'],
             'hazard':['DTXSID','hazard_metric','hazard_value'],'exposure':['DTXSID']}
    funcs={'chemicals':normalize_chemicals,'bioactivity':normalize_bioactivity,'hazard':normalize_hazard,'exposure':normalize_exposure}
    report={'source_files_seen':len(files),'outputs':{},'sources':{},'warnings':[]}
    for kind in args.only:
        manual=getattr(args,kind)
        chosen=[Path(manual)] if manual else [p for p,c in columns.items() if all(alias_column(c,key) for key in schemas[kind])]
        if kind=='bioactivity' and not manual:
            chosen=[p for p in chosen if alias_column(columns[p],'DTXSID') or alias_column(columns[p],'spid')]
        if kind=='exposure' and not manual:
            chosen=[p for p in chosen if alias_column(columns[p],'use_category') or alias_column(columns[p],'function_category')]
        chunks=[]; sources=[]
        for path in chosen:
            try:
                raw=load_manual(path)
                frame=funcs[kind](raw,build_spid_map(files)) if kind=='bioactivity' and not alias_column(raw.columns,'DTXSID') else funcs[kind](raw)
                if not frame.empty:
                    frame['source_file']=str(path.relative_to(DATA_SOURCE)) if path.is_relative_to(DATA_SOURCE) else str(path)
                    chunks.append(frame); sources.append(str(path))
            except Exception as exc:
                if manual: raise
                report['warnings'].append(f'{path.name}: {exc}')
        destination=DATA_NORMALIZED/(kind+'.csv')
        if chunks:
            frame=pd.concat(chunks,ignore_index=True).drop_duplicates()
            if kind=='chemicals': frame=frame.drop_duplicates('DTXSID')
            temporary=destination.with_suffix('.csv.part'); frame.to_csv(temporary,index=False); temporary.replace(destination)
            report['outputs'][kind]=len(frame); report['sources'][kind]=sources
            print(kind,len(frame),'rows from',len(sources),'files',flush=True)
        elif destination.exists():
            report['warnings'].append(f'{kind}: retained existing normalized output; no new compatible source.')
        else:
            report['warnings'].append(f'{kind}: missing. For ToxCast run 00a then 01a; for structures run 00b then 01b.')
    (DATA_NORMALIZED/'normalization_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    missing=[name for name in ['chemicals','bioactivity'] if not (DATA_NORMALIZED/(name+'.csv')).exists()]
    if missing: raise SystemExit('Training prerequisites still missing: '+', '.join(missing))


if __name__=='__main__': main()
