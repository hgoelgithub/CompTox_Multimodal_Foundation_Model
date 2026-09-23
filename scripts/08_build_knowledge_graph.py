"""STEP 8 -- Build one big knowledge graph covering every MEA-tested compound
(not just a single queried one, unlike 07_query_chemical.py): chemical class,
structure, literature/predicted targets, MEA phenotypes and concentration
responses, ToxCast/Tox21 bioactivity + hazard/exposure context (for compounds
mapped to a DTXSID), registered toxicity endpoints, structural similarity
(Tanimoto on Morgan fingerprints), and foundation-model embedding similarity
(if step 04 has been run). Output: multimodal_knowledge_graph.graphml plus
flat node/edge CSVs, under data/mea_processed/."""
import re
import numpy as np
import pandas as pd
import networkx as nx
from core.query import write_graphml_unique_edge_ids,load_tables,HIT_DIRECTION,toxcast_tox21_profile,comptox_context_profile
from core.paths import MEA_PROCESSED,PROCESSED_COMPTox
from core.table_io import read_table,table_exists

OUT=MEA_PROCESSED/"multimodal_knowledge_graph.graphml"


def add_node(G,node,node_type,label,**attrs):
    """Add a node to the knowledge graph with a consistent node_type/label attribute shape."""
    G.add_node(node,node_type=node_type,label=str(label),**attrs)


def add_structural_similarity(G,structure,threshold=.55,top_k=3):
    """Connect each compound to its `top_k` most structurally similar
    compounds (Tanimoto similarity >= `threshold` on Morgan fingerprints)."""
    if structure.empty or "compound_id" not in structure: return
    smiles_col=next((c for c in ["CanonicalSMILES","ConnectivitySMILES","IsomericSMILES","SMILES"] if c in structure),None)
    if not smiles_col: return
    from rdkit import Chem,DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048); records=[]
    for _,r in structure.iterrows():
        smi=r.get(smiles_col); mol=Chem.MolFromSmiles(str(smi)) if pd.notna(smi) and str(smi).strip() else None
        if mol is not None: records.append((r["compound_id"],gen.GetFingerprint(mol)))
    for i,(cid,fp) in enumerate(records):
        scored=sorted([(float(DataStructs.TanimotoSimilarity(fp,fp2)),other) for j,(other,fp2) in enumerate(records) if i!=j],reverse=True)
        for sim,other in [x for x in scored if x[0]>=threshold][:top_k]:
            if cid in G and other in G: G.add_edge(cid,other,relation="structural_similarity",tanimoto=sim,modality="structure")


def main():
    """CLI entry point: build every node/edge type described in the module
    docstring, in order, then write the graph and its flat CSV views."""
    t=load_tables(); summary=t["summary"]
    if summary.empty: raise FileNotFoundError("Run scripts/06_prepare_mea_data.py first.")
    G=nx.MultiGraph()
    for _,r in summary.iterrows(): add_node(G,r["compound_id"],"compound",r["base_compound_name"])
    for _,r in t["class"].iterrows():
        cid=r["compound_id"]
        for typ,col in [("super_class","compounds_super_class"),("class","compounds_class")]:
            if pd.notna(r.get(col)): n=f"{typ}::{r[col]}"; add_node(G,n,typ,r[col]); G.add_edge(cid,n,relation=f"belongs_to_{typ}",modality="chemistry")
    structure=t["structure"]
    if not structure.empty and "compound_id" in structure:
        for _,r in structure.iterrows():
            cid=r["compound_id"]; smi=next((str(r[c]) for c in ["CanonicalSMILES","ConnectivitySMILES","IsomericSMILES","SMILES"] if c in r and pd.notna(r[c]) and str(r[c]).strip()),"")
            if smi:
                n="structure::"+str(cid); add_node(G,n,"structure","molecular structure",smiles=smi,inchi_key=str(r.get("InChIKey",""))); G.add_edge(cid,n,relation="has_structure",modality="structure")
    for _,r in t["literature"].iterrows(): n="literature_target::"+str(r["target"]); add_node(G,n,"literature_target",r["target"]); G.add_edge(r["compound_id"],n,relation="literature_target",modality="literature")
    tox=t["toxprofiler"].copy()
    if not tox.empty:
        tox["score"]=pd.to_numeric(tox["score"],errors="coerce")
        for _,r in tox[tox["score"]>=1.0].iterrows(): n="predicted_target::"+str(r["target"]); add_node(G,n,"predicted_target",r["target"]); G.add_edge(r["compound_id"],n,relation="ToxProfiler_prediction",modality="predicted_biology",score=float(r["score"]))
    metric=t["metric"].copy()
    if not metric.empty:
        metric["hit_code"]=pd.to_numeric(metric["hit_code"],errors="coerce")
        for _,r in metric[metric["hit_code"].fillna(0)>0].iterrows():
            n="mea_metric::"+str(r["metric"]); add_node(G,n,"mea_metric",r["metric"]); code=int(r["hit_code"]); G.add_edge(r["compound_id"],n,relation=f"MEA_{HIT_DIRECTION.get(code,'hit')}",modality="MEA",hit_code=code,logAC50="" if pd.isna(r.get("logac50")) else float(r["logac50"]))
    z=t["zscore"].copy()
    if not z.empty:
        z["zscore"]=pd.to_numeric(z["zscore"],errors="coerce"); z=z.dropna(subset=["zscore"])
        for (cid,name),g in z.groupby(["compound_id","metric"]):
            r=g.loc[g["zscore"].abs().idxmax()]
            if abs(float(r["zscore"]))>=2: n="zresponse::"+str(name); add_node(G,n,"zscore_response",name); G.add_edge(cid,n,relation="concentration_response",modality="MEA_zscore",zscore=float(r["zscore"]),concentration_uM=float(r["concentration_uM"]))
    # ToxCast/Tox21 + hazard/exposure for mapped MEA chemicals.
    mapping=t["mapping"]
    if not mapping.empty and {"compound_id","DTXSID"}.issubset(mapping.columns):
        map2=mapping.copy(); map2["DTXSID"]=map2["DTXSID"].fillna("").astype(str).str.strip(); map2=map2[map2["DTXSID"]!=""]
        for _,m in map2.iterrows():
            cid=m["compound_id"]; sid=m["DTXSID"]
            if cid not in G: continue
            for a in toxcast_tox21_profile(sid,max_assays=25).get("active_assays",[]):
                n=f"bioassay::{a.get('program')}::{a.get('assay')}"; add_node(G,n,"toxcast_tox21_assay",a.get("assay"),program=str(a.get("program"))); G.add_edge(cid,n,relation="active_bioassay",modality="ToxCast_Tox21",hitcall=str(a.get("hitcall")),AC50_uM=str(a.get("AC50_uM")),efficacy=str(a.get("efficacy")))
            ctx=comptox_context_profile(sid)
            if ctx["hazard"].get("status") not in {None,"not_available"}: n="hazard::"+sid; add_node(G,n,"hazard_context","Hazard context",summary=str(ctx["hazard"])); G.add_edge(cid,n,relation="has_hazard_context",modality="hazard")
            if ctx["exposure"].get("status") not in {None,"not_available"}: n="exposure::"+sid; add_node(G,n,"exposure_context","Exposure context",summary=str(ctx["exposure"])); G.add_edge(cid,n,relation="has_exposure_context",modality="exposure")
    # Registered hERG/DILI/other endpoints.
    reg=t["endpoint_registry"]
    if not reg.empty:
        for idx,r in reg.iterrows():
            matched=[]
            if "compound_id" in reg and pd.notna(r.get("compound_id")) and str(r.get("compound_id")).strip() in G: matched=[str(r.get("compound_id")).strip()]
            elif "chemical_name" in reg and pd.notna(r.get("chemical_name")):
                q=re.sub(r"[^a-z0-9]+","",str(r["chemical_name"]).lower())
                matched=[n for n,a in G.nodes(data=True) if a.get("node_type")=="compound" and re.sub(r"[^a-z0-9]+","",str(a.get("label","")).lower())==q]
            for cid in matched:
                endpoint=str(r.get("endpoint","endpoint")); n=f"endpoint::{idx}::{endpoint}"; attrs={k:str(v) for k,v in r.items() if pd.notna(v) and str(v).strip() and k not in {"endpoint","label"}}; add_node(G,n,"toxicity_endpoint",endpoint,**attrs); G.add_edge(cid,n,relation="toxicity_or_injury_endpoint",modality="downstream_endpoint")
    add_structural_similarity(G,structure)
    # FM embedding similarity for mapped MEA chemicals.
    emb_path=PROCESSED_COMPTox/"comptox_embeddings.parquet"
    if table_exists(emb_path) and not mapping.empty and {"compound_id","DTXSID"}.issubset(mapping.columns):
        emb=read_table(emb_path); zcols=[c for c in emb if re.match(r"^z\d+$",c)]; mm=mapping.copy(); mm["DTXSID"]=mm["DTXSID"].fillna("").astype(str).str.strip(); mm=mm[mm["DTXSID"]!=""].merge(emb[["DTXSID"]+zcols],on="DTXSID",how="inner") if zcols else pd.DataFrame()
        if len(mm)>=2:
            from sklearn.metrics import pairwise_distances
            dist=pairwise_distances(mm[zcols].to_numpy(),metric="cosine")
            for i in range(len(mm)):
                added=0
                for j in np.argsort(dist[i]):
                    if i==j: continue
                    c1,c2=mm.iloc[i]["compound_id"],mm.iloc[j]["compound_id"]
                    if c1 in G and c2 in G: G.add_edge(c1,c2,relation="foundation_embedding_similarity",modality="foundation_embedding",cosine_distance=float(dist[i,j])); added+=1
                    if added==3: break
    write_graphml_unique_edge_ids(G,OUT)
    pd.DataFrame([{"node_id":n,**a,"degree":G.degree(n)} for n,a in G.nodes(data=True)]).to_csv(MEA_PROCESSED/"multimodal_kg_nodes.csv",index=False)
    pd.DataFrame([{"source":u,"target":v,**a} for u,v,a in G.edges(data=True)]).to_csv(MEA_PROCESSED/"multimodal_kg_edges.csv",index=False)
    print("Saved:",OUT); print("Nodes:",G.number_of_nodes()); print("Edges:",G.number_of_edges())


if __name__=="__main__": main()
