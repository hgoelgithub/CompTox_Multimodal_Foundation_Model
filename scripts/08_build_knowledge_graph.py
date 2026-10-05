"""Step 8 - Build the MEA knowledge graph

One graph covering every MEA-tested compound (step 07 builds a graph for a single chemical). Node types:
compounds, chemical classes, structures, literature targets, ToxProfiler *predicted* targets, MEA metrics,
MEA concentration responses, ToxCast/Tox21 assays, hazard / exposure context, registered toxicity endpoints.
Edges carry a `relation` and a `modality`, and two chemical-to-chemical edge types connect similar compounds:
structural similarity (Tanimoto on Morgan fingerprints) and foundation-model embedding similarity.

Input : data/mea_processed/*.csv (06); optional: data/processed/* (02, 04), data/normalized/*, data/endpoints/*
Output: data/mea_processed/multimodal_knowledge_graph.graphml (open in Gephi / Cytoscape),
        multimodal_kg_nodes.csv, multimodal_kg_edges.csv
Run   : python scripts/08_build_knowledge_graph.py
"""

# %%
import argparse
import re
from functools import lru_cache
from pathlib import Path

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
OUTPUT = MEA_PROCESSED / "multimodal_knowledge_graph.graphml"

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
# ## 3. Building the graph
# Nodes are added in stages. Every MEA compound is a node first; evidence nodes are then attached to it.
# * MEA metrics: an edge for each metric with a hit (labelled increase or decrease, with the logAC50).
# * Concentration responses: an edge only if the compound's largest absolute z-score for a metric is >= 2.
# * ToxProfiler targets: predictions with score >= 1.0, marked as `predicted_biology`.
# * ToxCast/Tox21, hazard and exposure: only for compounds mapped to a DTXSID.
# * Structural similarity: each compound to its 3 most similar compounds with Tanimoto >= 0.55.
# * Embedding similarity: each mapped compound to its 3 nearest neighbours in the embedding space.

# %%
def add_node(G, node, node_type, label, **attrs):
    G.add_node(node, node_type=node_type, label=str(label), **attrs)


def add_structural_similarity(G, structure, threshold=0.55, top_k=3):
    smiles_column = next((c for c in ["CanonicalSMILES", "ConnectivitySMILES", "IsomericSMILES", "SMILES"]
                          if c in structure), None)
    if structure.empty or "compound_id" not in structure or not smiles_column:
        return
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = []
    for _, r in structure.iterrows():
        smiles = r.get(smiles_column)
        mol = Chem.MolFromSmiles(str(smiles)) if pd.notna(smiles) and str(smiles).strip() else None
        if mol is not None:
            fingerprints.append((r["compound_id"], generator.GetFingerprint(mol)))
    for i, (compound, fingerprint) in enumerate(fingerprints):
        scored = sorted(((float(DataStructs.TanimotoSimilarity(fingerprint, other_fp)), other)
                         for j, (other, other_fp) in enumerate(fingerprints) if j != i), reverse=True)
        for similarity, other in [s for s in scored if s[0] >= threshold][:top_k]:
            if compound in G and other in G:
                G.add_edge(compound, other, relation="structural_similarity", tanimoto=similarity, modality="structure")


def add_embedding_similarity(G, mapping, top_k=3):
    if not EMBEDDINGS.exists() or mapping.empty or not {"compound_id", "DTXSID"}.issubset(mapping.columns):
        return
    embeddings = pd.read_parquet(EMBEDDINGS)
    z_columns = [c for c in embeddings if re.match(r"^z\d+$", c)]
    mapped = mapping.copy()
    mapped["DTXSID"] = mapped["DTXSID"].fillna("").astype(str).str.strip()
    mapped = mapped[mapped["DTXSID"] != ""].merge(embeddings[["DTXSID"] + z_columns], on="DTXSID", how="inner")
    if not z_columns or len(mapped) < 2:
        return
    from sklearn.metrics import pairwise_distances
    distances = pairwise_distances(mapped[z_columns].to_numpy(), metric="cosine")
    for i in range(len(mapped)):
        added = 0
        for j in np.argsort(distances[i]):
            if i == j:
                continue
            a, b = mapped.iloc[i]["compound_id"], mapped.iloc[j]["compound_id"]
            if a in G and b in G:
                G.add_edge(a, b, relation="foundation_embedding_similarity", modality="foundation_embedding",
                           cosine_distance=float(distances[i, j]))
                added += 1
            if added == top_k:
                break


def build_graph(tables):
    G = nx.MultiGraph()
    for _, r in tables["summary"].iterrows():
        add_node(G, r["compound_id"], "compound", r["base_compound_name"])

    for _, r in tables["class"].iterrows():
        for kind, column in (("super_class", "compounds_super_class"), ("class", "compounds_class")):
            if pd.notna(r.get(column)):
                node = f"{kind}::{r[column]}"
                add_node(G, node, kind, r[column])
                G.add_edge(r["compound_id"], node, relation=f"belongs_to_{kind}", modality="chemistry")

    structure = tables["structure"]
    if not structure.empty and "compound_id" in structure:
        for _, r in structure.iterrows():
            smiles = next((str(r[c]) for c in ["CanonicalSMILES", "ConnectivitySMILES", "IsomericSMILES", "SMILES"]
                           if c in r and pd.notna(r[c]) and str(r[c]).strip()), "")
            if smiles:
                node = "structure::" + str(r["compound_id"])
                add_node(G, node, "structure", "molecular structure", smiles=smiles, inchi_key=str(r.get("InChIKey", "")))
                G.add_edge(r["compound_id"], node, relation="has_structure", modality="structure")

    for _, r in tables["literature"].iterrows():
        node = "literature_target::" + str(r["target"])
        add_node(G, node, "literature_target", r["target"])
        G.add_edge(r["compound_id"], node, relation="literature_target", modality="literature")

    toxprofiler = tables["toxprofiler"].copy()
    if not toxprofiler.empty:
        toxprofiler["score"] = pd.to_numeric(toxprofiler["score"], errors="coerce")
        for _, r in toxprofiler[toxprofiler["score"] >= 1.0].iterrows():
            node = "predicted_target::" + str(r["target"])
            add_node(G, node, "predicted_target", r["target"])
            G.add_edge(r["compound_id"], node, relation="ToxProfiler_prediction", modality="predicted_biology", score=float(r["score"]))

    metric = tables["metric"].copy()
    if not metric.empty:
        metric["hit_code"] = pd.to_numeric(metric["hit_code"], errors="coerce")
        for _, r in metric[metric["hit_code"].fillna(0) > 0].iterrows():
            node = "mea_metric::" + str(r["metric"])
            add_node(G, node, "mea_metric", r["metric"])
            code = int(r["hit_code"])
            G.add_edge(r["compound_id"], node, relation=f"MEA_{HIT_DIRECTION.get(code, 'hit')}", modality="MEA",
                       hit_code=code, logAC50="" if pd.isna(r.get("logac50")) else float(r["logac50"]))

    z = tables["zscore"].copy()
    if not z.empty:
        z["zscore"] = pd.to_numeric(z["zscore"], errors="coerce")
        for (compound, name), group in z.dropna(subset=["zscore"]).groupby(["compound_id", "metric"]):
            best = group.loc[group["zscore"].abs().idxmax()]
            if abs(float(best["zscore"])) >= 2:
                node = "zresponse::" + str(name)
                add_node(G, node, "zscore_response", name)
                G.add_edge(compound, node, relation="concentration_response", modality="MEA_zscore",
                           zscore=float(best["zscore"]), concentration_uM=float(best["concentration_uM"]))

    mapping = tables["mapping"]
    if not mapping.empty and {"compound_id", "DTXSID"}.issubset(mapping.columns):
        mapped = mapping.copy()
        mapped["DTXSID"] = mapped["DTXSID"].fillna("").astype(str).str.strip()
        for _, m in mapped[mapped["DTXSID"] != ""].iterrows():
            compound, sid = m["compound_id"], m["DTXSID"]
            if compound not in G:
                continue
            for assay in toxcast_tox21_profile(sid, max_assays=25).get("active_assays", []):
                node = f"bioassay::{assay.get('program')}::{assay.get('assay')}"
                add_node(G, node, "toxcast_tox21_assay", assay.get("assay"), program=str(assay.get("program")))
                G.add_edge(compound, node, relation="active_bioassay", modality="ToxCast_Tox21",
                           hitcall=str(assay.get("hitcall")), AC50_uM=str(assay.get("AC50_uM")), efficacy=str(assay.get("efficacy")))
            context = comptox_context_profile(sid)
            for kind, label in (("hazard", "Hazard context"), ("exposure", "Exposure context")):
                if context[kind].get("status") not in {None, "not_available"}:
                    node = f"{kind}::{sid}"
                    add_node(G, node, f"{kind}_context", label, summary=str(context[kind]))
                    G.add_edge(compound, node, relation=f"has_{kind}_context", modality=kind)

    registry = tables["endpoint_registry"]  # measured hERG / DILI / other endpoints registered in step 09
    for index, r in registry.iterrows():
        matched = []
        if "compound_id" in registry and pd.notna(r.get("compound_id")) and str(r["compound_id"]).strip() in G:
            matched = [str(r["compound_id"]).strip()]
        elif "chemical_name" in registry and pd.notna(r.get("chemical_name")):
            q = re.sub(r"[^a-z0-9]+", "", str(r["chemical_name"]).lower())
            matched = [n for n, a in G.nodes(data=True) if a.get("node_type") == "compound"
                       and re.sub(r"[^a-z0-9]+", "", str(a.get("label", "")).lower()) == q]
        for compound in matched:
            endpoint = str(r.get("endpoint", "endpoint"))
            node = f"endpoint::{index}::{endpoint}"
            add_node(G, node, "toxicity_endpoint", endpoint,
                     **{k: str(v) for k, v in r.items() if pd.notna(v) and str(v).strip() and k not in {"endpoint", "label"}})
            G.add_edge(compound, node, relation="toxicity_or_injury_endpoint", modality="downstream_endpoint")

    add_structural_similarity(G, structure)
    add_embedding_similarity(G, mapping)
    return G

# %% [markdown]
# ## 4. Run
# The graph is written as GraphML plus two flat CSV views (one row per node and per edge) for pandas or Excel.

# %%
def build_knowledge_graph():
    tables = load_tables()
    if tables["summary"].empty:
        raise FileNotFoundError("Run scripts/06_prepare_mea_data.py first.")
    G = build_graph(tables)
    write_graphml(G, OUTPUT)
    pd.DataFrame([{"node_id": n, **a, "degree": G.degree(n)} for n, a in G.nodes(data=True)]).to_csv(
        MEA_PROCESSED / "multimodal_kg_nodes.csv", index=False)
    pd.DataFrame([{"source": u, "target": v, **a} for u, v, a in G.edges(data=True)]).to_csv(
        MEA_PROCESSED / "multimodal_kg_edges.csv", index=False)
    print("Saved:", OUTPUT)
    print("Nodes:", G.number_of_nodes(), "| Edges:", G.number_of_edges())
    return G

# %% Command line
def main():
    argparse.ArgumentParser(description="Build the MEA multimodal knowledge graph.").parse_args()
    build_knowledge_graph()


if __name__ == "__main__":
    main()
