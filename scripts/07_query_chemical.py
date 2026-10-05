"""Step 7 - Query one chemical

Given a chemical name or DTXSID, gathers everything the project knows about it into one profile, printed to
the console and saved as JSON plus a small knowledge graph (GraphML, PNG, interactive HTML).

The profile keeps evidence types separate and labelled, so *measured*, *predicted* and *not available locally*
are never confused:
  structure, chemical class, literature targets      identity and annotation
  ToxProfiler targets                                 PREDICTED (model output, not measurement)
  ToxCast/Tox21 bioactivity, hazard, exposure         measured records from EPA data
  MEA phenotypes and concentration-response          measured in-house microelectrode-array screen
  registered toxicity endpoints                       measured hERG / DILI / ... from step 09
  foundation-model embedding neighbours               similar chemicals according to the pretrained model

Input : data/mea_processed/*.csv (06), data/processed/* (02, 04), data/normalized/* (01), data/endpoints/*
Output: results/<name>_multimodal_profile.json, _knowledge_graph.graphml / .png / .html
Run   : python scripts/07_query_chemical.py --chemical Permethrin --no-pubchem
"""

# %%
import argparse
import json
import re
from difflib import get_close_matches
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# %% [markdown]
# ## 1. Paths

# %%
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data"
NORMALIZED = DATA / "normalized"
MEA_PROCESSED = DATA / "mea_processed"
ENDPOINTS = DATA / "endpoints"
COHORT = DATA / "processed" / "comptox_v3.parquet"
QUERY_INDEX = DATA / "processed" / "comptox_query_index.csv"
EMBEDDINGS = DATA / "processed" / "comptox_embeddings.parquet"
RESULTS = PROJECT_ROOT / "results"

# %% [markdown]
# ## 2. Reading the data

# %% [markdown]
# ### Reading the project's tables
# `load_tables` reads the MEA tables (step 06), the endpoint registry (step 09) and the cohort index (step 02)
# into a dictionary. Bioactivity, hazard and exposure records for one chemical are read from the normalized
# Parquet mirrors with DuckDB (fast, no need to load millions of rows), falling back to the CSV files.

# %%
HIT_DIRECTION = {0: "no_hit", 1: "increase", 2: "decrease"}
TOXCAST_ACTIVE_THRESHOLD = 0.90


def read_if_exists(path):
    """A CSV as a DataFrame, or an empty DataFrame if it is missing or empty."""
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def load_tables():
    files = {"summary": "mea_compound_summary.csv", "class": "mea_molecule_class.csv",
             "literature": "mea_literature_targets_long.csv", "toxprofiler": "mea_toxprofiler_long.csv",
             "metric": "mea_metric_activity_summary.csv", "zscore": "mea_zscore_long.csv",
             "mapping": "mea_to_comptox_mapping_template.csv", "structure": "mea_structure_pubchem.csv"}
    tables = {key: read_if_exists(MEA_PROCESSED / name) for key, name in files.items()}
    tables["endpoint_registry"] = read_if_exists(ENDPOINTS / "endpoint_results.csv")
    tables["comptox_query_index"] = read_if_exists(QUERY_INDEX)
    return tables


def read_by_dtxsid(table, dtxsid):
    """All rows of a normalized table (bioactivity / hazard / exposure) for one chemical."""
    parquet = NORMALIZED / "parquet" / f"{table}.parquet"
    if parquet.exists():
        import duckdb
        return duckdb.connect().execute("SELECT * FROM read_parquet(?) WHERE DTXSID = ?",
                                        [parquet.as_posix(), str(dtxsid)]).df()
    csv_path = NORMALIZED / f"{table}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, low_memory=False)
        return df[df["DTXSID"].astype(str).eq(str(dtxsid))] if "DTXSID" in df else pd.DataFrame()
    return pd.DataFrame()


@lru_cache(maxsize=1)
def load_cohort():
    """The pretraining cohort table, read once (or None if step 02 has not been run)."""
    return pd.read_parquet(COHORT) if COHORT.exists() else None


def cohort_row(dtxsid):
    """The chemical's row in the pretraining cohort table (or None)."""
    frame = load_cohort()
    if frame is None or not dtxsid:
        return None
    rows = frame[frame["DTXSID"].astype(str).eq(str(dtxsid))]
    return rows.iloc[0] if not rows.empty else None


def toxcast_tox21_profile(dtxsid, max_assays=25):
    """The chemical's active ToxCast/Tox21 assays (hit call >= 0.9), strongest first.

    Preference order: the measured per-assay records in the normalized bioactivity table; otherwise the
    harmonised pretraining-cohort row."""
    if not dtxsid:
        return {"status": "no_dtxsid", "DTXSID": None, "active_assays": []}
    rows = read_by_dtxsid("bioactivity", dtxsid)
    if not rows.empty:
        for column in ["hitcall", "ac50_uM", "efficacy"]:
            if column in rows:
                rows[column] = pd.to_numeric(rows[column], errors="coerce")
        hit = rows["hitcall"] if "hitcall" in rows else pd.Series(np.nan, index=rows.index)
        active = rows[hit.fillna(-1) >= TOXCAST_ACTIVE_THRESHOLD].copy()
        if "ac50_uM" in active:
            active = active.sort_values(["hitcall", "ac50_uM"], ascending=[False, True], na_position="last")
        records = []
        for _, r in active.head(max_assays).iterrows():
            records.append({"program": str(r.get("program", "ToxCast/Tox21")), "assay": str(r.get("assay", "")),
                            "hitcall": float(r["hitcall"]) if pd.notna(r.get("hitcall")) else None,
                            "AC50_uM": float(r["ac50_uM"]) if pd.notna(r.get("ac50_uM")) else None,
                            "efficacy": float(r["efficacy"]) if pd.notna(r.get("efficacy")) else None})
        return {"status": "ok", "source": "normalized_raw_bioactivity", "DTXSID": dtxsid,
                "n_measured_assays": int(hit.notna().sum()),
                "n_active_assays": int((hit.fillna(-1) >= TOXCAST_ACTIVE_THRESHOLD).sum()), "active_assays": records}
    row = cohort_row(dtxsid)
    if row is not None:
        records, measured = [], 0
        for hit_column in [c for c in row.index if str(c).startswith("biohit__")]:
            assay = hit_column.removeprefix("biohit__")
            hit = pd.to_numeric(pd.Series([row[hit_column]]), errors="coerce").iloc[0]
            if pd.notna(hit):
                measured += 1
            if pd.isna(hit) or hit < TOXCAST_ACTIVE_THRESHOLD:
                continue
            log_ac50, efficacy = row.get("bioac50__" + assay, np.nan), row.get("bioeff__" + assay, np.nan)
            records.append({"program": "ToxCast/Tox21 harmonized", "assay": assay, "hitcall": float(hit),
                            "AC50_uM": 10 ** float(log_ac50) if pd.notna(log_ac50) else None,
                            "log10_AC50_uM": float(log_ac50) if pd.notna(log_ac50) else None,
                            "efficacy": float(efficacy) if pd.notna(efficacy) else None})
        records.sort(key=lambda x: (-(x["hitcall"] or 0), x["AC50_uM"] if x["AC50_uM"] is not None else float("inf")))
        return {"status": "ok", "source": "foundation_pretraining_table", "DTXSID": dtxsid,
                "n_measured_assays": measured, "n_active_assays": len(records), "active_assays": records[:max_assays]}
    return {"status": "not_available_locally", "DTXSID": dtxsid, "active_assays": []}


def comptox_context_profile(dtxsid, max_rows=20):
    """Hazard and exposure context. Kept apart from bioactivity and from toxicity endpoints."""
    out = {"hazard": {"status": "not_available", "records": []}, "exposure": {"status": "not_available"}}
    if not dtxsid:
        return out
    hazard = read_by_dtxsid("hazard", dtxsid)
    if not hazard.empty:
        columns = [c for c in ["metric", "value", "units", "study_type", "species", "route", "duration", "source"] if c in hazard]
        out["hazard"] = {"status": "ok", "n_records": len(hazard),
                         "records": hazard[columns].head(max_rows).where(pd.notna(hazard[columns]), None).to_dict("records")}
    exposure = read_by_dtxsid("exposure", dtxsid)
    if not exposure.empty:
        uses = exposure["use_category"].dropna().astype(str).value_counts().head(10).to_dict() if "use_category" in exposure else {}
        out["exposure"] = {"status": "ok", "n_records": len(exposure),
                           "product_count": int(exposure["product_id"].nunique()) if "product_id" in exposure else None,
                           "top_use_categories": uses}
    if out["hazard"]["status"] != "ok" or out["exposure"]["status"] != "ok":  # fall back to the cohort's summary columns
        row = cohort_row(dtxsid)
        if row is not None:
            if out["hazard"]["status"] != "ok":
                out["hazard"] = {"status": "summary_only", "hazard_count": row.get("hazard_count"),
                                 "noael_log10": row.get("noael_log"), "loael_log10": row.get("loael_log")}
            if out["exposure"]["status"] != "ok":
                out["exposure"] = {"status": "summary_only", "exposure_count": row.get("exposure_count"),
                                   "product_count": row.get("product_count"), "use_category_count": row.get("use_category_count")}
    return out


def write_graphml(G, path):
    """Write GraphML with globally unique edge ids.

    networkx multigraph keys restart at 0 for every node pair, so many edges would share id="0" and Gephi
    would reject the file. Renumbering the edges e0, e1, ... avoids that."""
    import networkx as nx
    H = nx.MultiDiGraph() if G.is_directed() else nx.MultiGraph()
    H.graph.update(G.graph)
    H.add_nodes_from(G.nodes(data=True))
    for i, (u, v, data) in enumerate(G.edges(data=True)):
        H.add_edge(u, v, key=f"e{i}", **data)
    nx.write_graphml(H, path)

# %% [markdown]
# ## 3. Finding the chemical
# A query may be a name, a CAS number, a DTXSID or a slightly misspelled name. Names are folded to a
# comparable form (lower case, letters and digits only, a trailing plate-well suffix removed) and matched in
# three passes: exact, then substring, then fuzzy (`difflib`). The same is done against the MEA workbook and
# against the CompTox index. If a name matches more than one chemical the query is reported as ambiguous.

# %%
def normalize_name(text):
    text = str(text).lower().strip()
    text = re.sub(r"_y\d+p\d+$", "", text).replace("(+/-)", "")
    return re.sub(r"[^a-z0-9]+", "", text)


def slugify(text):
    """Filesystem-safe name for output files."""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_") or "chemical"


def name_match(query, candidates, name_columns):
    if candidates.empty:
        return candidates
    q = normalize_name(query)
    work = candidates.copy()
    norm_columns = []
    for column in name_columns:
        if column in work:
            work[f"__norm_{column}"] = work[column].fillna("").astype(str).map(normalize_name)
            norm_columns.append(f"__norm_{column}")
    if not norm_columns:
        return work.iloc[0:0]
    exact = pd.Series(False, index=work.index)
    for column in norm_columns:
        exact |= work[column].eq(q)
    if exact.any():
        return work[exact].drop(columns=norm_columns)
    if q:
        contains = pd.Series(False, index=work.index)
        for column in norm_columns:
            contains |= work[column].str.contains(q, na=False, regex=False)
        if contains.any():
            return work[contains].drop(columns=norm_columns)
    display = next((c for c in name_columns if c in work), None)
    names = work[display].dropna().astype(str).unique().tolist() if display else []
    normalized = {normalize_name(x): x for x in names}
    close = get_close_matches(q, list(normalized), n=5, cutoff=0.55)
    if close:
        return work[work[display].isin([normalized[x] for x in close])].drop(columns=norm_columns)
    return work.iloc[0:0].drop(columns=norm_columns)


def resolve_mea_chemical(query, summary):
    if summary.empty:
        return pd.DataFrame(columns=["compound_id", "base_compound_name"])
    return name_match(query, summary[["compound_id", "base_compound_name"]].drop_duplicates(),
                      ["base_compound_name", "compound_id"])


def resolve_comptox_query(query, index):
    if index.empty:
        return index
    q = str(query).strip()
    if "DTXSID" in index and q.upper().startswith("DTXSID"):
        exact = index[index["DTXSID"].astype(str).str.upper().eq(q.upper())]
        if not exact.empty:
            return exact
    return name_match(query, index, [c for c in ["preferred_name", "chemical_name", "name", "CASRN", "DTXSID"] if c in index])


def resolve_dtxsid(query, compound_ids, tables, comptox_match):
    """The chemical's DTXSID: the query itself if it is one, else the MEA mapping, else the CompTox match."""
    q = str(query).strip()
    if q.upper().startswith("DTXSID"):
        return q.upper()
    mapping = tables.get("mapping", pd.DataFrame())
    if not mapping.empty and {"DTXSID", "compound_id"}.issubset(mapping.columns):
        rows = mapping[mapping["compound_id"].isin(compound_ids)].copy()
        rows["DTXSID"] = rows["DTXSID"].fillna("").astype(str).str.strip()
        rows = rows[rows["DTXSID"] != ""]
        if not rows.empty:
            return rows.iloc[0]["DTXSID"]
    if comptox_match is not None and not comptox_match.empty and "DTXSID" in comptox_match:
        sid = str(comptox_match.iloc[0]["DTXSID"]).strip()
        return sid if sid and sid.lower() != "nan" else None
    return None

# %% [markdown]
# ## 4. Structure and measured endpoints
# * **Structure**: looked up locally first (MEA structure table, then the DTXSID mapping, then the CompTox
#   index). PubChem is only queried if nothing local exists and the internet lookup is allowed.
# * **Toxicity endpoints**: real measured results only. A summary of the compound's MEA phenotype (if it has
#   active MEA metrics) plus any rows registered in step 09. ToxProfiler *predictions* are deliberately not
#   included here.

# %%
def pubchem_properties(name, timeout=12):
    """Structure and properties from the live PubChem API (used only when no local structure exists)."""
    try:
        import requests
    except ImportError:
        return {"status": "requests_not_installed"}
    fields = "CanonicalSMILES,IsomericSMILES,InChIKey,MolecularFormula,MolecularWeight,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount"
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(str(name), safe='')}/property/{fields}/JSON"
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        properties = response.json()["PropertyTable"]["Properties"][0]
        properties["status"] = "ok"
        return properties
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc)}


def local_structure_profile(compound_ids, base_names, tables, comptox_match):
    structure = tables.get("structure", pd.DataFrame())
    mapping = tables.get("mapping", pd.DataFrame())
    if not structure.empty and "compound_id" in structure:
        mask = structure["compound_id"].isin(compound_ids)
        if "base_compound_name" in structure:
            mask |= structure["base_compound_name"].isin(base_names)
        if mask.any():
            return structure[mask].iloc[0].dropna().to_dict()
    if not mapping.empty and "compound_id" in mapping:
        rows = mapping[mapping["compound_id"].isin(compound_ids)]
        if not rows.empty:
            row = rows.iloc[0].dropna().to_dict()
            if str(row.get("SMILES", "")).strip() or str(row.get("DTXSID", "")).strip():
                row["status"] = "local_mapping"
                return row
    if comptox_match is not None and not comptox_match.empty:
        row = comptox_match.iloc[0].dropna().to_dict()
        smiles = row.get("smiles") or row.get("SMILES")
        if smiles and str(smiles).strip():
            return {"DTXSID": row.get("DTXSID", ""), "SMILES": smiles, "status": "comptox_index"}
    return {}


def toxicity_endpoint_profile(preferred_name, compound_ids, dtxsid, tables, mea_metrics):
    output = []
    if mea_metrics:
        output.append({"endpoint": "MEA neurotoxicity phenotype", "endpoint_group": "neurotoxicity",
                       "result_type": "experimental", "label": "MEA activity profile",
                       "n_active_metrics": len(mea_metrics),
                       "n_increase": sum(m.get("direction") == "increase" for m in mea_metrics),
                       "n_decrease": sum(m.get("direction") == "decrease" for m in mea_metrics),
                       "source": "uploaded MEA workbook"})
    registry = tables.get("endpoint_registry", pd.DataFrame())
    if registry.empty:
        return output
    mask = pd.Series(False, index=registry.index)
    if dtxsid and "DTXSID" in registry:
        mask |= registry["DTXSID"].fillna("").astype(str).str.strip().eq(str(dtxsid))
    if compound_ids and "compound_id" in registry:
        mask |= registry["compound_id"].fillna("").astype(str).isin([str(x) for x in compound_ids])
    if "chemical_name" in registry:
        mask |= registry["chemical_name"].fillna("").astype(str).map(normalize_name).eq(normalize_name(preferred_name))
    for _, row in registry[mask].iterrows():
        output.append({k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()
                       if pd.notna(v) and str(v).strip() != ""})
    return output

# %% [markdown]
# ## 5. Assembling the profile
# `mea_sections` extracts the MEA evidence: the metrics with an effect (increase or decrease, with logAC50) and,
# for each metric, the concentration at which the z-score was largest. `embedding_neighbors` lists the five
# closest chemicals in the foundation model's embedding space. `build_profile` resolves the query and calls
# every section in turn.

# %%
def mea_sections(compound_ids, tables, top_metrics):
    """(active metrics, strongest z-score responses) for the given MEA compound ids."""
    active, strongest = [], []
    metric = tables.get("metric", pd.DataFrame())
    if not metric.empty and compound_ids:
        rows = metric[metric["compound_id"].isin(compound_ids)].copy()
        rows["hit_code"] = pd.to_numeric(rows["hit_code"], errors="coerce")
        rows["logac50"] = pd.to_numeric(rows["logac50"], errors="coerce")
        rows = rows[rows["hit_code"].fillna(0) > 0].sort_values(["hit_code", "logac50"], ascending=[False, True])
        active = [{"metric": str(r["metric"]), "hit_code": int(r["hit_code"]),
                   "direction": HIT_DIRECTION.get(int(r["hit_code"])),
                   "logAC50": float(r["logac50"]) if pd.notna(r["logac50"]) else None} for _, r in rows.iterrows()]
    z = tables.get("zscore", pd.DataFrame())
    if not z.empty and compound_ids:
        rows = z[z["compound_id"].isin(compound_ids)].copy()
        rows["zscore"] = pd.to_numeric(rows["zscore"], errors="coerce")
        rows["concentration_uM"] = pd.to_numeric(rows["concentration_uM"], errors="coerce")
        for name, group in rows.dropna(subset=["zscore"]).groupby("metric"):
            best = group.loc[group["zscore"].abs().idxmax()]
            strongest.append({"metric": str(name), "zscore": float(best["zscore"]),
                              "concentration_uM": float(best["concentration_uM"])})
        strongest.sort(key=lambda x: abs(x["zscore"]), reverse=True)
    return active, strongest[:top_metrics]


def embedding_neighbors(dtxsid, k=5):
    if not EMBEDDINGS.exists() or not dtxsid:
        return []
    table = pd.read_parquet(EMBEDDINGS)
    ids = table["DTXSID"].astype(str).tolist()
    z_columns = [c for c in table if re.match(r"^z\d+$", c)]
    if not z_columns or str(dtxsid) not in ids:
        return []
    from sklearn.metrics import pairwise_distances
    matrix = table[z_columns].to_numpy()
    i = ids.index(str(dtxsid))
    distances = pairwise_distances(matrix[i:i + 1], matrix, metric="cosine")[0]
    order = [j for j in np.argsort(distances) if j != i][:k]
    return [{"DTXSID": ids[j], "cosine_distance": float(distances[j])} for j in order]


def build_profile(query, tables, top_targets=12, top_metrics=12, top_toxcast=25, use_pubchem=True):
    matches = resolve_mea_chemical(query, tables.get("summary", pd.DataFrame()))
    comptox = resolve_comptox_query(query, tables.get("comptox_query_index", pd.DataFrame()))
    if matches.empty and comptox.empty and not str(query).upper().startswith("DTXSID"):
        return {"query": query, "status": "not_found", "suggestions": []}
    names = matches["base_compound_name"].dropna().unique() if not matches.empty else []
    if len(names) > 1 or (matches.empty and len(comptox) > 1):
        suggestions = list(names) if len(names) > 1 else comptox["DTXSID"].astype(str).tolist()
        return {"query": query, "status": "ambiguous", "suggestions": suggestions}

    if not matches.empty:
        compound_ids = matches["compound_id"].drop_duplicates().tolist()
        base_names = matches["base_compound_name"].drop_duplicates().tolist()
        preferred = base_names[0]
    else:
        compound_ids, preferred = [], None
        if not comptox.empty:
            for column in ["preferred_name", "chemical_name", "name"]:
                if column in comptox.columns and pd.notna(comptox.iloc[0][column]) and str(comptox.iloc[0][column]).strip():
                    preferred = str(comptox.iloc[0][column])
                    break
        preferred = preferred or str(query)
        base_names = [preferred]
    dtxsid = resolve_dtxsid(query, compound_ids, tables, comptox)
    profile = {"query": query, "status": "ok", "preferred_name": preferred, "matched_compound_ids": compound_ids,
               "base_compound_names": base_names, "DTXSID": dtxsid}

    classes = tables.get("class", pd.DataFrame())
    if not classes.empty and compound_ids:
        rows = classes[classes["compound_id"].isin(compound_ids)]
        profile["chemical_class"] = [{"super_class": r.get("compounds_super_class"), "class": r.get("compounds_class")}
                                     for _, r in rows.iterrows()]
    literature = tables.get("literature", pd.DataFrame())
    if not literature.empty and compound_ids:
        rows = literature[literature["compound_id"].isin(compound_ids)]
        profile["literature_targets"] = sorted(rows["target"].dropna().astype(str).unique().tolist())
    toxprofiler = tables.get("toxprofiler", pd.DataFrame())
    if not toxprofiler.empty and compound_ids:
        rows = toxprofiler[toxprofiler["compound_id"].isin(compound_ids)].copy()
        rows["score"] = pd.to_numeric(rows["score"], errors="coerce")
        rows = rows.dropna(subset=["score"]).sort_values("score", ascending=False).head(top_targets)
        profile["toxprofiler_top_targets"] = [{"target": str(r["target"]), "score": float(r["score"])} for _, r in rows.iterrows()]

    active_metrics, strongest = mea_sections(compound_ids, tables, top_metrics)
    if active_metrics:
        profile["mea_active_metrics"] = active_metrics[:top_metrics]
    if strongest:
        profile["strongest_zscore_responses"] = strongest

    structure = local_structure_profile(compound_ids, base_names, tables, comptox)
    if not structure and use_pubchem:
        structure = pubchem_properties(preferred)
        structure["source"] = "PubChem"
    elif structure:
        structure["source"] = structure.get("source", "local")
    profile["structure"] = structure
    profile["toxcast_tox21"] = toxcast_tox21_profile(dtxsid, top_toxcast)
    profile["comptox_context"] = comptox_context_profile(dtxsid)
    profile["toxicity_endpoints"] = toxicity_endpoint_profile(preferred, compound_ids, dtxsid, tables, active_metrics)
    neighbors = embedding_neighbors(dtxsid)
    if neighbors:
        profile["foundation_embedding_neighbors"] = neighbors
    return profile

# %% [markdown]
# ## 6. Printing the profile

# %%
def print_profile(p):
    print("\n" + "=" * 76 + "\nMULTIMODAL CHEMICAL / TOXICOLOGY PROFILE\n" + "=" * 76)
    if p.get("status") == "ambiguous":
        print("Ambiguous chemical. Use an exact name or DTXSID:", p["suggestions"])
        return
    if p.get("status") != "ok":
        print("Chemical not found in the local MEA data or CompTox query index:", p.get("query"))
        return
    print("Chemical:", p["preferred_name"])
    if p.get("DTXSID"):
        print("DTXSID:", p["DTXSID"])
    if p.get("matched_compound_ids"):
        print("MEA workbook IDs:", ", ".join(p["matched_compound_ids"]))

    print("\n[STRUCTURE]")
    structure = p.get("structure", {})
    shown = False
    for key in ["DTXSID", "CID", "MolecularFormula", "MolecularWeight", "XLogP", "TPSA", "CanonicalSMILES",
                "ConnectivitySMILES", "IsomericSMILES", "SMILES", "InChIKey"]:
        if key in structure and str(structure[key]).strip() not in {"", "nan", "None"}:
            print(f"  {key}: {structure[key]}")
            shown = True
    print("  source:", structure.get("source", structure.get("status", "local"))) if shown \
        else print("  No resolved molecular structure available locally.")

    print("\n[CHEMICAL CLASS]")
    classes = p.get("chemical_class", [])
    for r in classes:
        print(" ", r.get("super_class"), "->", r.get("class"))
    if not classes:
        print("  None recorded")
    print("\n[LITERATURE TARGETS]\n ", ", ".join(p.get("literature_targets", [])) or "None recorded")
    print("\n[TOXPROFILER PREDICTED TARGETS]  (model predictions, not measurements)")
    for r in p.get("toxprofiler_top_targets", []):
        print(f"  {r['target']}: {r['score']:.3f}")

    toxcast = p.get("toxcast_tox21", {})
    print("\n[TOXCAST / TOX21 BIOACTIVITY]\n  status:", toxcast.get("status"))
    if toxcast.get("status") == "ok":
        print(f"  measured assays: {toxcast.get('n_measured_assays', 0)} | active assays: {toxcast.get('n_active_assays', 0)}")
        for r in toxcast.get("active_assays", []):
            ac50 = "NA" if r.get("AC50_uM") is None else f"{r['AC50_uM']:.4g} µM"
            efficacy = "NA" if r.get("efficacy") is None else f"{r['efficacy']:.4g}"
            print(f"  {r.get('program')} | {r.get('assay')} | hit={r.get('hitcall')} | AC50={ac50} | efficacy={efficacy}")
    context = p.get("comptox_context", {})
    print("\n[COMPTOX HAZARD CONTEXT]\n ", context.get("hazard", {}))
    print("\n[COMPTOX EXPOSURE CONTEXT]\n ", context.get("exposure", {}))

    print("\n[MEA EXPERIMENTAL PHENOTYPES]")
    for r in p.get("mea_active_metrics", []):
        ac50 = "NA" if r["logAC50"] is None else f"{r['logAC50']:.3f}"
        print(f"  {r['metric']} | {r['direction'].upper()} | hit={r['hit_code']} | logAC50={ac50}")
    print("\n[MEA STRONGEST CONCENTRATION-RESPONSE Z-SCORES]")
    for r in p.get("strongest_zscore_responses", []):
        print(f"  {r['metric']} | z={r['zscore']:.2f} at {r['concentration_uM']:g} µM")

    print("\n[TOXICITY / INJURY ENDPOINTS]")
    endpoints = p.get("toxicity_endpoints", [])
    if not endpoints:
        print("  No registered independent endpoint results.")
    for r in endpoints:
        core = [str(r.get("endpoint", "endpoint")), str(r.get("endpoint_group", "")), str(r.get("result_type", ""))]
        extras = [f"{k}={r[k]}" for k in ["label", "value", "probability", "unit", "model", "source",
                                          "n_active_metrics", "n_increase", "n_decrease"]
                  if k in r and str(r[k]).strip() not in {"", "nan"}]
        print("  " + " | ".join([x for x in core if x and x != "nan"] + extras))
    if p.get("foundation_embedding_neighbors"):
        print("\n[FOUNDATION-MODEL EMBEDDING NEIGHBORS]")
        for r in p["foundation_embedding_neighbors"]:
            print(f"  {r['DTXSID']} | cosine distance={r['cosine_distance']:.4f}")

# %% [markdown]
# ## 7. A small knowledge graph for the chemical
# The chemical sits in the centre with one edge out to each piece of evidence. Each edge is labelled with
# its relation and its `modality` (structure, literature, predicted biology, ToxCast/Tox21, MEA, ...), so the
# graph keeps evidence types distinguishable. It is written as GraphML (open in Gephi or Cytoscape), as a static
# PNG and as an interactive HTML page.

# %%
def add_node(G, node, node_type, label, **attrs):
    G.add_node(node, node_type=node_type, label=label, **attrs)


def build_query_graph(p):
    G = nx.MultiGraph()
    center = "chemical::" + p["preferred_name"]
    add_node(G, center, "chemical", p["preferred_name"], DTXSID=str(p.get("DTXSID") or ""))
    structure = p.get("structure", {})
    smiles = structure.get("CanonicalSMILES") or structure.get("ConnectivitySMILES") or structure.get("SMILES")
    if smiles:
        node = "structure::" + p["preferred_name"]
        add_node(G, node, "structure", "Structure", smiles=str(smiles), inchi_key=str(structure.get("InChIKey", "")))
        G.add_edge(center, node, relation="has_structure", modality="structure")
    for r in p.get("chemical_class", []):
        for kind in ("super_class", "class"):
            if r.get(kind):
                node = f"{kind}::{r[kind]}"
                add_node(G, node, kind, str(r[kind]))
                G.add_edge(center, node, relation=f"belongs_to_{kind}", modality="chemistry")
    for target in p.get("literature_targets", []):
        add_node(G, "lit::" + target, "literature_target", target)
        G.add_edge(center, "lit::" + target, relation="literature_support", modality="literature")
    for r in p.get("toxprofiler_top_targets", []):
        node = "pred::" + r["target"]
        add_node(G, node, "predicted_target", r["target"], score=r["score"])
        G.add_edge(center, node, relation="ToxProfiler_prediction", modality="predicted_biology", weight=r["score"])
    for r in p.get("toxcast_tox21", {}).get("active_assays", [])[:15]:
        node = f"toxcast::{r.get('program')}::{r.get('assay')}"
        add_node(G, node, "toxcast_tox21_assay", f"{r.get('program')}\n{r.get('assay')}",
                 AC50_uM=str(r.get("AC50_uM", "")), efficacy=str(r.get("efficacy", "")))
        G.add_edge(center, node, relation="active_bioassay", modality="ToxCast_Tox21", hitcall=str(r.get("hitcall", "")))
    context = p.get("comptox_context", {})
    for kind, label in (("hazard", "Hazard context"), ("exposure", "Exposure context")):
        if context.get(kind, {}).get("status") not in {None, "not_available"}:
            add_node(G, f"context::{kind}", f"{kind}_context", label, summary=str(context[kind]))
            G.add_edge(center, f"context::{kind}", relation=f"has_{kind}_context", modality=kind)
    for r in p.get("mea_active_metrics", []):
        node = "mea::" + r["metric"]
        add_node(G, node, "mea_metric", r["metric"], hit_code=r["hit_code"], direction=r["direction"], logAC50=str(r["logAC50"]))
        G.add_edge(center, node, relation=f"MEA_{r['direction']}", modality="MEA", hit_code=r["hit_code"], logAC50=str(r["logAC50"]))
    for r in p.get("strongest_zscore_responses", [])[:8]:
        node = "zscore::" + r["metric"]
        add_node(G, node, "zscore_response", r["metric"], zscore=r["zscore"], concentration_uM=r["concentration_uM"])
        G.add_edge(center, node, relation="concentration_response", modality="MEA_zscore",
                   zscore=r["zscore"], concentration_uM=r["concentration_uM"])
    for i, r in enumerate(p.get("toxicity_endpoints", [])):
        endpoint = str(r.get("endpoint", f"endpoint_{i}"))
        node = f"toxendpoint::{i}::{endpoint}"
        label = endpoint + ("\n" + str(r["label"]) if r.get("label") else "")
        add_node(G, node, "toxicity_endpoint", label,
                 **{k: str(v) for k, v in r.items() if k not in {"endpoint", "label"} and v is not None})
        G.add_edge(center, node, relation="toxicity_or_injury_endpoint", modality="downstream_endpoint")
    for r in p.get("foundation_embedding_neighbors", []):
        node = "embedding_neighbor::" + r["DTXSID"]
        add_node(G, node, "embedding_neighbor", r["DTXSID"])
        G.add_edge(center, node, relation="foundation_embedding_neighbor", modality="foundation_embedding",
                   cosine_distance=r["cosine_distance"])
    return G


def save_png(G, path):
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G, seed=42, k=1.1)
    shapes = {"chemical": "o", "structure": "s", "super_class": "h", "class": "h", "literature_target": "^",
              "predicted_target": "v", "toxcast_tox21_assay": "8", "hazard_context": "d", "exposure_context": "d",
              "mea_metric": "D", "zscore_response": "P", "toxicity_endpoint": "*", "embedding_neighbor": "X"}
    for node_type, shape in shapes.items():
        nodes = [n for n, a in G.nodes(data=True) if a.get("node_type") == node_type]
        if nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_shape=shape, node_size=[1500 if node_type == "chemical" else 650] * len(nodes))
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    nx.draw_networkx_labels(G, pos, labels={n: a.get("label", n) for n, a in G.nodes(data=True)}, font_size=7)
    plt.title("Multimodal toxicology knowledge graph")
    plt.axis("off")
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def save_interactive_html(G, path):
    """Interactive graph page (needs pyvis; returns False if it is not installed)."""
    try:
        from pyvis.network import Network
    except ImportError:
        return False
    net = Network(height="850px", width="100%", bgcolor="#ffffff", font_color="#222222", cdn_resources="in_line")
    net.barnes_hut()
    for node, attrs in G.nodes(data=True):
        title = [f"<b>{attrs.get('label', node)}</b>", f"Type: {attrs.get('node_type', '')}"]
        title += [f"{k}: {v}" for k, v in attrs.items() if k not in {"label", "node_type"} and str(v)]
        net.add_node(node, label=attrs.get("label", node), title="<br>".join(title))
    for u, v, attrs in G.edges(data=True):
        net.add_edge(u, v, title="<br>".join(f"{k}: {val}" for k, val in attrs.items()), label=str(attrs.get("relation", "")))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(path), open_browser=False)
    return True

# %% [markdown]
# ## 8. Run
# `query_chemical` builds the profile, prints it and, if the chemical was found, saves the JSON profile and
# the graph files under `results/`.

# %%
def query_chemical(chemical, top_targets=12, top_metrics=12, top_toxcast=25, use_pubchem=True):
    profile = build_profile(chemical, load_tables(), top_targets, top_metrics, top_toxcast, use_pubchem)
    print_profile(profile)
    if profile.get("status") != "ok":
        return profile
    RESULTS.mkdir(exist_ok=True)
    slug = slugify(profile["preferred_name"])
    json_path = RESULTS / f"{slug}_multimodal_profile.json"
    json_path.write_text(json.dumps(profile, indent=2, default=str), encoding="utf-8")
    G = build_query_graph(profile)
    graphml, png, html = (RESULTS / f"{slug}_knowledge_graph{ext}" for ext in (".graphml", ".png", ".html"))
    write_graphml(G, graphml)
    save_png(G, png)
    html_ok = save_interactive_html(G, html)
    print("\n[OUTPUT FILES]")
    for path in (json_path, graphml, png):
        print(" ", path)
    print(" ", html if html_ok else "Interactive HTML skipped (install pyvis).")
    return profile

# %% Command line
def main():
    parser = argparse.ArgumentParser(description="Show everything the project knows about one chemical.")
    parser.add_argument("--chemical", help="chemical name or DTXSID (asked for if omitted)")
    parser.add_argument("--top-targets", type=int, default=12)
    parser.add_argument("--top-metrics", type=int, default=12)
    parser.add_argument("--top-toxcast", type=int, default=25)
    parser.add_argument("--no-pubchem", action="store_true", help="never query PubChem")
    args = parser.parse_args()
    chemical = args.chemical or input("Enter chemical name or DTXSID: ").strip()
    query_chemical(chemical, args.top_targets, args.top_metrics, args.top_toxcast, not args.no_pubchem)


if __name__ == "__main__":
    main()
