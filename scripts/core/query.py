"""Integrated chemical query helpers while keeping evidence modalities separate.

This is the shared backend for 07_query_chemical.py (one chemical at a time)
and 08_build_knowledge_graph.py (all MEA-mapped chemicals at once). The
central function is profile_from_tables(): given a name/DTXSID and the
loaded local tables, it resolves the chemical's identity and then pulls
together every modality this project tracks for it, each clearly labeled by
source/status so a caller never confuses "measured" with "predicted" or
"not available locally"."""
from __future__ import annotations
from difflib import get_close_matches
from urllib.parse import quote
import re
import numpy as np
import pandas as pd
from core.paths import MEA_PROCESSED, ENDPOINT_DATA, RAW_COMPTox, PROCESSED_COMPTox
from core.table_io import read_table,table_exists
from core import duckdb_io

HIT_DIRECTION={0:"no_hit",1:"increase",2:"decrease"}
TOXCAST_ACTIVE_THRESHOLD=0.90


def normalize_name(text):
    """Fold a chemical name to a comparable form for fuzzy matching: lowercase,
    strip a trailing MEA well suffix and stereo-descriptor noise, drop
    everything but letters/digits."""
    text=str(text).lower().strip(); text=re.sub(r"_y\d+p\d+$","",text); text=text.replace("(+/-)","")
    return re.sub(r"[^a-z0-9]+","",text)


def slugify(text):
    """Turn a chemical name into a filesystem-safe basename for output files."""
    text=re.sub(r"[^A-Za-z0-9]+","_",str(text)).strip("_"); return text or "chemical"


def read_if_exists(path):
    """Read a CSV if it exists and isn't empty; otherwise an empty DataFrame,
    so every table lookup below can be written unconditionally."""
    path=pd.io.common.stringify_path(path)
    try:
        return pd.read_csv(path)
    except (FileNotFoundError,pd.errors.EmptyDataError):
        return pd.DataFrame()


def load_tables():
    """Load every MEA/endpoint-registry/CompTox-index CSV this module can use
    into one dict, keyed by short name -- called once per query/graph-build run."""
    files={"summary":"mea_compound_summary.csv","class":"mea_molecule_class.csv",
      "literature":"mea_literature_targets_long.csv","toxprofiler":"mea_toxprofiler_long.csv",
      "metric":"mea_metric_activity_summary.csv","zscore":"mea_zscore_long.csv",
      "mapping":"mea_to_comptox_mapping_template.csv","structure":"mea_structure_pubchem.csv"}
    t={k:read_if_exists(MEA_PROCESSED/v) for k,v in files.items()}
    t["endpoint_registry"]=read_if_exists(ENDPOINT_DATA/"endpoint_results.csv")
    t["comptox_query_index"]=read_if_exists(PROCESSED_COMPTox/"comptox_query_index.csv")
    return t


def _name_match(query,candidates,name_columns):
    """Match `query` against `candidates`' `name_columns`, trying exact
    normalized match first, then substring, then fuzzy (difflib) as a last
    resort -- so a typo or partial name still finds something reasonable."""
    if candidates.empty: return candidates
    q=normalize_name(query); work=candidates.copy(); norm_cols=[]
    for col in name_columns:
        if col in work:
            n=f"__norm_{col}"; work[n]=work[col].fillna("").astype(str).map(normalize_name); norm_cols.append(n)
    if not norm_cols: return work.iloc[0:0]
    exact=pd.Series(False,index=work.index)
    for col in norm_cols: exact |= work[col].eq(q)
    if exact.any(): return work[exact].drop(columns=norm_cols)
    if q:
        contains=pd.Series(False,index=work.index)
        for col in norm_cols: contains |= work[col].str.contains(q,na=False,regex=False)
        if contains.any(): return work[contains].drop(columns=norm_cols)
    display=next((c for c in name_columns if c in work),None)
    names=work[display].dropna().astype(str).unique().tolist() if display else []
    normalized={normalize_name(x):x for x in names}; close=get_close_matches(q,list(normalized),n=5,cutoff=.55)
    if close:
        selected=[normalized[x] for x in close]; return work[work[display].isin(selected)].drop(columns=norm_cols)
    return work.iloc[0:0].drop(columns=norm_cols)


def resolve_mea_chemical(query,summary):
    """Find MEA workbook compound_id(s) matching `query` by name or id."""
    if summary.empty: return pd.DataFrame(columns=["compound_id","base_compound_name"])
    return _name_match(query,summary[["compound_id","base_compound_name"]].drop_duplicates(),["base_compound_name","compound_id"])


def resolve_comptox_query(query,index):
    """Find CompTox query-index row(s) matching `query`, by exact DTXSID first
    if it looks like one, else by name."""
    if index.empty: return index
    q=str(query).strip()
    if "DTXSID" in index and q.upper().startswith("DTXSID"):
        exact=index[index["DTXSID"].astype(str).str.upper().eq(q.upper())]
        if not exact.empty: return exact
    cols=[c for c in ["preferred_name","chemical_name","name","CASRN","DTXSID"] if c in index]
    return _name_match(query,index,cols)


def pubchem_properties(name,timeout=12):
    """Look up a chemical's structure/properties from the live PubChem API --
    only used as a fallback when no local structure is available (see
    local_structure_profile) and --no-pubchem wasn't passed."""
    try: import requests
    except ImportError: return {"status":"requests_not_installed"}
    fields="CanonicalSMILES,IsomericSMILES,InChIKey,MolecularFormula,MolecularWeight,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount"
    url="https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"+quote(str(name),safe="")+"/property/"+fields+"/JSON"
    try:
        r=requests.get(url,timeout=timeout); r.raise_for_status(); props=r.json()["PropertyTable"]["Properties"][0]; props["status"]="ok"; return props
    except Exception as exc: return {"status":"unavailable","error":str(exc)}


def local_structure_profile(compound_ids,base_names,tables,comptox_match=None):
    """Find a structure for this chemical from local data only, checked in
    order: the MEA structure-resolution table, then the MEA-to-CompTox
    mapping template, then the CompTox query index. Returns {} if nothing
    local has a usable SMILES (the caller then falls back to PubChem)."""
    structure=tables.get("structure",pd.DataFrame()); mapping=tables.get("mapping",pd.DataFrame())
    if not structure.empty and "compound_id" in structure:
        mask=structure["compound_id"].isin(compound_ids)
        if "base_compound_name" in structure: mask |= structure["base_compound_name"].isin(base_names)
        rows=structure[mask]
        if not rows.empty: return rows.iloc[0].dropna().to_dict()
    if not mapping.empty and "compound_id" in mapping:
        rows=mapping[mapping["compound_id"].isin(compound_ids)]
        if not rows.empty:
            row=rows.iloc[0].dropna().to_dict()
            if str(row.get("SMILES","")).strip() or str(row.get("DTXSID","")).strip(): row["status"]="local_mapping"; return row
    if comptox_match is not None and not comptox_match.empty:
        row=comptox_match.iloc[0].dropna().to_dict()
        smi=row.get("smiles") or row.get("SMILES")
        if smi and str(smi).strip():
            return {"DTXSID":row.get("DTXSID",""),"SMILES":smi,"status":"comptox_index"}
    return {}


def resolve_dtxsid(query,compound_ids,tables,comptox_match=None):
    """Find this chemical's DTXSID: the query itself if it already looks like
    one, else the MEA mapping template, else the CompTox index match."""
    q=str(query).strip()
    if q.upper().startswith("DTXSID"): return q.upper()
    mapping=tables.get("mapping",pd.DataFrame())
    if not mapping.empty and {"DTXSID","compound_id"}.issubset(mapping.columns):
        rows=mapping[mapping["compound_id"].isin(compound_ids)].copy(); rows["DTXSID"]=rows["DTXSID"].fillna("").astype(str).str.strip(); rows=rows[rows["DTXSID"]!=""]
        if not rows.empty: return rows.iloc[0]["DTXSID"]
    match=comptox_match if comptox_match is not None else resolve_comptox_query(query,tables.get("comptox_query_index",pd.DataFrame()))
    if match is not None and not match.empty and "DTXSID" in match:
        sid=str(match.iloc[0]["DTXSID"]).strip(); return sid if sid and sid.lower()!="nan" else None
    return None


def toxcast_tox21_profile(dtxsid,max_assays=25):
    """This chemical's active ToxCast/Tox21 assays: prefers the raw
    normalized bioactivity table (measured, per-assay AC50/efficacy) and
    falls back to the harmonized pretraining-cohort table if that's all
    that's available (e.g. a compound outside the training cohort cap)."""
    if not dtxsid: return {"status":"no_dtxsid","DTXSID":None,"active_assays":[]}
    # DuckDB pushes the DTXSID filter down to the Parquet mirror instead of
    # loading the full multi-million-row bioactivity CSV on every lookup.
    if duckdb_io.table_available("bioactivity"):
        rows=duckdb_io.query_df("SELECT * FROM bioactivity WHERE DTXSID = ?",[str(dtxsid)])
    else:
        raw=RAW_COMPTox/"bioactivity.csv"
        rows=pd.DataFrame()
        if raw.exists():
            bio=pd.read_csv(raw)
            if "DTXSID" in bio: rows=bio[bio["DTXSID"].astype(str).eq(str(dtxsid))].copy()
    if not rows.empty:
        for col in ["hitcall","ac50_uM","efficacy"]:
            if col in rows: rows[col]=pd.to_numeric(rows[col],errors="coerce")
        hit=rows["hitcall"] if "hitcall" in rows else pd.Series(np.nan,index=rows.index)
        active=rows[hit.fillna(-1)>=TOXCAST_ACTIVE_THRESHOLD].copy()
        if "ac50_uM" in active: active=active.sort_values(["hitcall","ac50_uM"],ascending=[False,True],na_position="last")
        records=[]
        for _,r in active.head(max_assays).iterrows():
            records.append({"program":str(r.get("program","ToxCast/Tox21")),"assay":str(r.get("assay","")),
              "hitcall":float(r["hitcall"]) if pd.notna(r.get("hitcall")) else None,
              "AC50_uM":float(r["ac50_uM"]) if pd.notna(r.get("ac50_uM")) else None,
              "efficacy":float(r["efficacy"]) if pd.notna(r.get("efficacy")) else None})
        return {"status":"ok","source":"normalized_raw_bioactivity","DTXSID":dtxsid,
          "n_measured_assays":int(hit.notna().sum()),"n_active_assays":int((hit.fillna(-1)>=TOXCAST_ACTIVE_THRESHOLD).sum()),"active_assays":records}
    wide=PROCESSED_COMPTox/"comptox_v3.parquet"
    if table_exists(wide):
        frame=read_table(wide); rows=frame[frame["DTXSID"].astype(str).eq(str(dtxsid))] if "DTXSID" in frame else pd.DataFrame()
        if not rows.empty:
            row=rows.iloc[0]; records=[]; measured=0
            for hcol in [c for c in frame if c.startswith("biohit__")]:
                assay=hcol.removeprefix("biohit__"); hit=pd.to_numeric(pd.Series([row.get(hcol)]),errors="coerce").iloc[0]
                if pd.notna(hit): measured+=1
                if pd.isna(hit) or hit<TOXCAST_ACTIVE_THRESHOLD: continue
                logac=row.get("bioac50__"+assay,np.nan); eff=row.get("bioeff__"+assay,np.nan)
                records.append({"program":"ToxCast/Tox21 harmonized","assay":assay,"hitcall":float(hit),
                  "AC50_uM":10**float(logac) if pd.notna(logac) else None,"log10_AC50_uM":float(logac) if pd.notna(logac) else None,
                  "efficacy":float(eff) if pd.notna(eff) else None})
            records.sort(key=lambda x:(-(x["hitcall"] or 0),x["AC50_uM"] if x["AC50_uM"] is not None else float("inf")))
            return {"status":"ok","source":"foundation_pretraining_table","DTXSID":dtxsid,"n_measured_assays":measured,"n_active_assays":len(records),"active_assays":records[:max_assays]}
    return {"status":"not_available_locally","DTXSID":dtxsid,"active_assays":[]}


def comptox_context_profile(dtxsid,max_rows=20):
    """Hazard and exposure context, kept separate from bioactivity and endpoints."""
    out={"hazard":{"status":"not_available","records":[]},"exposure":{"status":"not_available"}}
    if not dtxsid: return out
    if duckdb_io.table_available("hazard"):
        rows=duckdb_io.query_df("SELECT * FROM hazard WHERE DTXSID = ?",[str(dtxsid)])
    else:
        hpath=RAW_COMPTox/"hazard.csv"; rows=pd.DataFrame()
        if hpath.exists():
            h=pd.read_csv(hpath); rows=h[h["DTXSID"].astype(str).eq(str(dtxsid))] if "DTXSID" in h else pd.DataFrame()
    if not rows.empty:
        cols=[c for c in ["metric","value","units","study_type","species","route","duration","source"] if c in rows]
        out["hazard"]={"status":"ok","n_records":len(rows),"records":rows[cols].head(max_rows).where(pd.notna(rows[cols]),None).to_dict("records")}
    if duckdb_io.table_available("exposure"):
        rows=duckdb_io.query_df("SELECT * FROM exposure WHERE DTXSID = ?",[str(dtxsid)])
    else:
        epath=RAW_COMPTox/"exposure.csv"; rows=pd.DataFrame()
        if epath.exists():
            e=pd.read_csv(epath); rows=e[e["DTXSID"].astype(str).eq(str(dtxsid))] if "DTXSID" in e else pd.DataFrame()
    if not rows.empty:
        uses=rows["use_category"].dropna().astype(str).value_counts().head(10).to_dict() if "use_category" in rows else {}
        out["exposure"]={"status":"ok","n_records":len(rows),"product_count":int(rows["product_id"].nunique()) if "product_id" in rows else None,"top_use_categories":uses}
    wide=PROCESSED_COMPTox/"comptox_v3.parquet"
    if table_exists(wide) and (out["hazard"]["status"]!="ok" or out["exposure"]["status"]!="ok"):
        frame=read_table(wide); rows=frame[frame["DTXSID"].astype(str).eq(str(dtxsid))] if "DTXSID" in frame else pd.DataFrame()
        if not rows.empty:
            r=rows.iloc[0]
            if out["hazard"]["status"]!="ok": out["hazard"]={"status":"summary_only","hazard_count":r.get("hazard_count"),"noael_log10":r.get("noael_log"),"loael_log10":r.get("loael_log")}
            if out["exposure"]["status"]!="ok": out["exposure"]={"status":"summary_only","exposure_count":r.get("exposure_count"),"product_count":r.get("product_count"),"use_category_count":r.get("use_category_count")}
    return out


def toxicity_endpoint_profile(preferred_name,compound_ids,dtxsid,tables,mea_metrics):
    """Real, measured toxicity/injury endpoints for this chemical: a
    derived MEA neurotoxicity summary (if it has active MEA metrics) plus
    any matching rows from the 09_register_toxicity_endpoints.py registry.
    Deliberately does not include ToxProfiler *predicted* targets -- those
    are a separate, clearly-labeled modality elsewhere in the profile."""
    output=[]
    if mea_metrics:
        inc=sum(x.get("direction")=="increase" for x in mea_metrics); dec=sum(x.get("direction")=="decrease" for x in mea_metrics)
        output.append({"endpoint":"MEA neurotoxicity phenotype","endpoint_group":"neurotoxicity","result_type":"experimental","label":"MEA activity profile","n_active_metrics":len(mea_metrics),"n_increase":inc,"n_decrease":dec,"source":"uploaded MEA workbook"})
    reg=tables.get("endpoint_registry",pd.DataFrame())
    if reg.empty: return output
    mask=pd.Series(False,index=reg.index)
    if dtxsid and "DTXSID" in reg: mask |= reg["DTXSID"].fillna("").astype(str).str.strip().eq(str(dtxsid))
    if compound_ids and "compound_id" in reg: mask |= reg["compound_id"].fillna("").astype(str).isin([str(x) for x in compound_ids])
    if "chemical_name" in reg:
        q=normalize_name(preferred_name); mask |= reg["chemical_name"].fillna("").astype(str).map(normalize_name).eq(q)
    for _,r in reg[mask].iterrows():
        clean={k:(v.item() if hasattr(v,"item") else v) for k,v in r.items() if pd.notna(v) and str(v).strip()!=""}; output.append(clean)
    return output


def profile_from_tables(query,tables,top_targets=12,top_metrics=12,top_toxcast=25,use_pubchem=True):
    """The main entry point: resolve `query` (name or DTXSID) to a chemical
    identity, then assemble its full multimodal profile -- structure, class,
    literature/predicted targets, MEA phenotypes and z-scores, ToxCast/Tox21
    bioactivity, hazard/exposure context, registered endpoints, and
    foundation-model embedding neighbors (whichever of these are available
    locally). Returns {"status": "not_found"|"ambiguous"|"ok", ...}."""
    matches=resolve_mea_chemical(query,tables.get("summary",pd.DataFrame())); comptox=resolve_comptox_query(query,tables.get("comptox_query_index",pd.DataFrame()))
    if matches.empty and comptox.empty and not str(query).upper().startswith("DTXSID"):
        return {"query":query,"status":"not_found","suggestions":[]}
    names=matches["base_compound_name"].dropna().unique() if not matches.empty else []
    if len(names)>1 or (matches.empty and len(comptox)>1):
        suggestions=list(names) if len(names)>1 else comptox["DTXSID"].astype(str).tolist()
        return {"query":query,"status":"ambiguous","suggestions":suggestions}
    if not matches.empty:
        compound_ids=matches["compound_id"].drop_duplicates().tolist(); base_names=matches["base_compound_name"].drop_duplicates().tolist(); preferred=base_names[0]
    else:
        compound_ids=[]; row=comptox.iloc[0] if not comptox.empty else None; preferred=None
        if row is not None:
            for col in ["preferred_name","chemical_name","name"]:
                if col in row and pd.notna(row[col]) and str(row[col]).strip(): preferred=str(row[col]); break
        preferred=preferred or str(query); base_names=[preferred]
    dtxsid=resolve_dtxsid(query,compound_ids,tables,comptox)
    profile={"query":query,"status":"ok","preferred_name":preferred,"matched_compound_ids":compound_ids,"base_compound_names":base_names,"DTXSID":dtxsid}
    cls=tables.get("class",pd.DataFrame())
    if not cls.empty and compound_ids:
        rows=cls[cls["compound_id"].isin(compound_ids)]; profile["chemical_class"]=[{"super_class":r.get("compounds_super_class"),"class":r.get("compounds_class")} for _,r in rows.iterrows()]
    lit=tables.get("literature",pd.DataFrame())
    if not lit.empty and compound_ids:
        rows=lit[lit["compound_id"].isin(compound_ids)]; profile["literature_targets"]=sorted(rows["target"].dropna().astype(str).unique().tolist())
    tox=tables.get("toxprofiler",pd.DataFrame())
    if not tox.empty and compound_ids:
        rows=tox[tox["compound_id"].isin(compound_ids)].copy(); rows["score"]=pd.to_numeric(rows["score"],errors="coerce"); rows=rows.dropna(subset=["score"]).sort_values("score",ascending=False).head(top_targets)
        profile["toxprofiler_top_targets"]=[{"target":str(r["target"]),"score":float(r["score"])} for _,r in rows.iterrows()]
    mea=[]; all_mea=[]; metric=tables.get("metric",pd.DataFrame())
    if not metric.empty and compound_ids:
        rows=metric[metric["compound_id"].isin(compound_ids)].copy(); rows["hit_code"]=pd.to_numeric(rows["hit_code"],errors="coerce"); rows["logac50"]=pd.to_numeric(rows["logac50"],errors="coerce")
        active=rows[rows["hit_code"].fillna(0)>0].copy(); active["direction"]=active["hit_code"].map(HIT_DIRECTION); active=active.sort_values(["hit_code","logac50"],ascending=[False,True])
        mea=[{"metric":str(r["metric"]),"hit_code":int(r["hit_code"]),"direction":HIT_DIRECTION.get(int(r["hit_code"])),"logAC50":float(r["logac50"]) if pd.notna(r["logac50"]) else None} for _,r in active.iterrows()]
        all_mea=mea
        profile["mea_active_metrics"]=mea[:top_metrics]
    z=tables.get("zscore",pd.DataFrame())
    if not z.empty and compound_ids:
        rows=z[z["compound_id"].isin(compound_ids)].copy(); rows["zscore"]=pd.to_numeric(rows["zscore"],errors="coerce"); rows["concentration_uM"]=pd.to_numeric(rows["concentration_uM"],errors="coerce"); rows=rows.dropna(subset=["zscore"])
        strongest=[]
        for name,g in rows.groupby("metric"):
            r=g.loc[g["zscore"].abs().idxmax()]; strongest.append({"metric":str(name),"zscore":float(r["zscore"]),"concentration_uM":float(r["concentration_uM"])})
        strongest.sort(key=lambda x:abs(x["zscore"]),reverse=True); profile["strongest_zscore_responses"]=strongest[:top_metrics]
    structure=local_structure_profile(compound_ids,base_names,tables,comptox)
    if not structure and use_pubchem:
        structure=pubchem_properties(preferred); structure["source"]="PubChem"
    elif structure: structure["source"]=structure.get("source","local")
    profile["structure"]=structure
    profile["toxcast_tox21"]=toxcast_tox21_profile(dtxsid,top_toxcast)
    profile["comptox_context"]=comptox_context_profile(dtxsid)
    profile["toxicity_endpoints"]=toxicity_endpoint_profile(preferred,compound_ids,dtxsid,tables,all_mea)
    emb_path=PROCESSED_COMPTox/"comptox_embeddings.parquet"
    if table_exists(emb_path) and dtxsid:
        emb=read_table(emb_path); zcols=[c for c in emb if re.match(r"^z\d+$",c)]
        if zcols and "DTXSID" in emb and str(dtxsid) in set(emb["DTXSID"].astype(str)):
            from sklearn.metrics import pairwise_distances
            i=emb.index[emb["DTXSID"].astype(str).eq(str(dtxsid))][0]; matrix=emb[zcols].to_numpy(); dist=pairwise_distances(matrix[i:i+1],matrix,metric="cosine")[0]; nn=[]
            for j in np.argsort(dist):
                if j==i: continue
                nn.append({"DTXSID":str(emb.iloc[j]["DTXSID"]),"cosine_distance":float(dist[j])})
                if len(nn)==5: break
            profile["foundation_embedding_neighbors"]=nn
    return profile


def write_graphml_unique_edge_ids(G,path):
    """Write GraphML with globally unique edge ids. networkx MultiGraph keys
    restart at 0 for every node pair, so many edges get id="0"; Gephi treats
    the id as the edge key and rejects the file ("mergeUndirectedEdgeWithKey")."""
    import networkx as nx
    H=nx.MultiGraph() if not G.is_directed() else nx.MultiDiGraph()
    H.graph.update(G.graph); H.add_nodes_from(G.nodes(data=True))
    for i,(u,v,d) in enumerate(G.edges(data=True)): H.add_edge(u,v,key=f"e{i}",**d)
    nx.write_graphml(H,path)
