"""STEP 7 -- Interactive/CLI lookup: given a chemical name or DTXSID, pull
together everything the project knows about it -- structure, chemical class,
literature/predicted targets, ToxCast/Tox21 bioactivity, hazard/exposure
context, MEA phenotypes, registered toxicity endpoints, and (if available)
foundation-model embedding neighbors -- and print it plus save a JSON profile
and a small per-chemical knowledge graph. Most of the actual lookup logic
lives in core/query.py; this file is presentation (printing) and graph
rendering."""
import argparse,json
import matplotlib.pyplot as plt
import networkx as nx
from core.query import write_graphml_unique_edge_ids,load_tables,profile_from_tables,slugify
from core.paths import OUTPUTS


def print_profile(p):
    """Pretty-print a profile dict (from core.query.profile_from_tables) to
    the console, section by section."""
    print("\n"+"="*76); print("MULTIMODAL CHEMICAL / TOXICOLOGY PROFILE"); print("="*76)
    if p.get("status")=="ambiguous":
        print("Ambiguous chemical. Use an exact name or DTXSID:", p["suggestions"]); return
    if p.get("status")!="ok": print("Chemical not found in local MEA or CompTox query index:",p.get("query")); return
    print("Chemical:",p["preferred_name"])
    if p.get("DTXSID"): print("DTXSID:",p["DTXSID"])
    if p.get("matched_compound_ids"): print("MEA workbook IDs:",", ".join(p["matched_compound_ids"]))
    print("\n[STRUCTURE]"); s=p.get("structure",{})
    structure_keys=["DTXSID","CID","MolecularFormula","MolecularWeight","XLogP","TPSA","CanonicalSMILES","ConnectivitySMILES","IsomericSMILES","SMILES","InChIKey"]
    shown=False
    for key in structure_keys:
        if key in s and str(s[key]).strip() not in {"","nan","None"}: print(f"  {key}: {s[key]}"); shown=True
    if shown: print("  source:",s.get("source",s.get("status","local")))
    else: print("  No resolved molecular structure available locally.")
    print("\n[CHEMICAL CLASS]")
    classes=p.get("chemical_class",[]); [print(" ",r.get("super_class"),"->",r.get("class")) for r in classes]
    if not classes: print("  None recorded")
    print("\n[LITERATURE TARGETS]"); print(" ",", ".join(p.get("literature_targets",[])) or "None recorded")
    print("\n[TOXPROFILER PREDICTED TARGETS]")
    for r in p.get("toxprofiler_top_targets",[]): print(f"  {r['target']}: {r['score']:.3f}")
    print("\n[TOXCAST / TOX21 BIOACTIVITY]"); t=p.get("toxcast_tox21",{}); print("  status:",t.get("status"))
    if t.get("status")=="ok":
        print(f"  measured assays: {t.get('n_measured_assays',0)} | active assays: {t.get('n_active_assays',0)}")
        for r in t.get("active_assays",[]):
            ac="NA" if r.get("AC50_uM") is None else f"{r['AC50_uM']:.4g} µM"; eff="NA" if r.get("efficacy") is None else f"{r['efficacy']:.4g}"
            print(f"  {r.get('program')} | {r.get('assay')} | hit={r.get('hitcall')} | AC50={ac} | efficacy={eff}")
    ctx=p.get("comptox_context",{})
    print("\n[COMPTOX HAZARD CONTEXT]"); print(" ",ctx.get("hazard",{}))
    print("\n[COMPTOX EXPOSURE CONTEXT]"); print(" ",ctx.get("exposure",{}))
    print("\n[MEA EXPERIMENTAL PHENOTYPES]")
    for r in p.get("mea_active_metrics",[]):
        ac="NA" if r["logAC50"] is None else f"{r['logAC50']:.3f}"; print(f"  {r['metric']} | {r['direction'].upper()} | hit={r['hit_code']} | logAC50={ac}")
    print("\n[MEA STRONGEST CONCENTRATION-RESPONSE Z-SCORES]")
    for r in p.get("strongest_zscore_responses",[]): print(f"  {r['metric']} | z={r['zscore']:.2f} at {r['concentration_uM']:g} µM")
    print("\n[TOXICITY / INJURY ENDPOINTS]"); eps=p.get("toxicity_endpoints",[])
    if not eps: print("  No registered independent endpoint results.")
    for r in eps:
        core=[str(r.get("endpoint","endpoint")),str(r.get("endpoint_group","")),str(r.get("result_type",""))]; extras=[]
        for key in ["label","value","probability","unit","model","source","n_active_metrics","n_increase","n_decrease"]:
            if key in r and str(r[key]).strip() not in {"","nan"}: extras.append(f"{key}={r[key]}")
        print("  "+" | ".join([x for x in core if x and x!="nan"]+extras))
    if p.get("foundation_embedding_neighbors"):
        print("\n[FOUNDATION-MODEL EMBEDDING NEIGHBORS]")
        for r in p["foundation_embedding_neighbors"]: print(f"  {r['DTXSID']} | cosine distance={r['cosine_distance']:.4f}")


def add_node(G,node,node_type,label,**attrs):
    """Add a node to the query graph with a consistent node_type/label attribute shape."""
    G.add_node(node,node_type=node_type,label=label,**attrs)


def build_query_graph(p):
    """Turn one chemical's profile dict into a small star-shaped graph: the
    chemical at the center, with an edge out to every piece of evidence
    (structure, targets, assays, MEA metrics, endpoints, ...)."""
    G=nx.MultiGraph(); center="chemical::"+p["preferred_name"]; add_node(G,center,"chemical",p["preferred_name"],DTXSID=str(p.get("DTXSID") or ""))
    s=p.get("structure",{}); smi=s.get("CanonicalSMILES") or s.get("ConnectivitySMILES") or s.get("SMILES")
    if smi:
        n="structure::"+p["preferred_name"]; add_node(G,n,"structure","Structure",smiles=str(smi),inchi_key=str(s.get("InChIKey",""))); G.add_edge(center,n,relation="has_structure",modality="structure")
    for r in p.get("chemical_class",[]):
        for typ,key in [("super_class","super_class"),("class","class")]:
            if r.get(key): n=f"{typ}::{r[key]}"; add_node(G,n,typ,str(r[key])); G.add_edge(center,n,relation=f"belongs_to_{typ}",modality="chemistry")
    for target in p.get("literature_targets",[]): n="lit::"+target; add_node(G,n,"literature_target",target); G.add_edge(center,n,relation="literature_support",modality="literature")
    for r in p.get("toxprofiler_top_targets",[]): n="pred::"+r["target"]; add_node(G,n,"predicted_target",r["target"],score=r["score"]); G.add_edge(center,n,relation="ToxProfiler_prediction",modality="predicted_biology",weight=r["score"])
    for r in p.get("toxcast_tox21",{}).get("active_assays",[])[:15]:
        label=f"{r.get('program')}\n{r.get('assay')}"; n="toxcast::"+str(r.get("program"))+"::"+str(r.get("assay")); add_node(G,n,"toxcast_tox21_assay",label,AC50_uM=str(r.get("AC50_uM","")),efficacy=str(r.get("efficacy",""))); G.add_edge(center,n,relation="active_bioassay",modality="ToxCast_Tox21",hitcall=str(r.get("hitcall","")))
    ctx=p.get("comptox_context",{})
    if ctx.get("hazard",{}).get("status") not in {None,"not_available"}:
        n="context::hazard"; add_node(G,n,"hazard_context","Hazard context",summary=str(ctx["hazard"])); G.add_edge(center,n,relation="has_hazard_context",modality="hazard")
    if ctx.get("exposure",{}).get("status") not in {None,"not_available"}:
        n="context::exposure"; add_node(G,n,"exposure_context","Exposure context",summary=str(ctx["exposure"])); G.add_edge(center,n,relation="has_exposure_context",modality="exposure")
    for r in p.get("mea_active_metrics",[]): n="mea::"+r["metric"]; add_node(G,n,"mea_metric",r["metric"],hit_code=r["hit_code"],direction=r["direction"],logAC50=str(r["logAC50"])); G.add_edge(center,n,relation=f"MEA_{r['direction']}",modality="MEA",hit_code=r["hit_code"],logAC50=str(r["logAC50"]))
    for r in p.get("strongest_zscore_responses",[])[:8]: n="zscore::"+r["metric"]; add_node(G,n,"zscore_response",r["metric"],zscore=r["zscore"],concentration_uM=r["concentration_uM"]); G.add_edge(center,n,relation="concentration_response",modality="MEA_zscore",zscore=r["zscore"],concentration_uM=r["concentration_uM"])
    for i,r in enumerate(p.get("toxicity_endpoints",[])):
        endpoint=str(r.get("endpoint",f"endpoint_{i}")); n=f"toxendpoint::{i}::{endpoint}"; label=endpoint+("\n"+str(r["label"]) if r.get("label") else ""); attrs={k:str(v) for k,v in r.items() if k not in {"endpoint","label"} and v is not None}; add_node(G,n,"toxicity_endpoint",label,**attrs); G.add_edge(center,n,relation="toxicity_or_injury_endpoint",modality="downstream_endpoint")
    for r in p.get("foundation_embedding_neighbors",[]): n="embedding_neighbor::"+r["DTXSID"]; add_node(G,n,"embedding_neighbor",r["DTXSID"]); G.add_edge(center,n,relation="foundation_embedding_neighbor",modality="foundation_embedding",cosine_distance=r["cosine_distance"])
    return G


def save_png(G,path):
    """Render the query graph to a static PNG with a marker shape per node type."""
    plt.figure(figsize=(16,12)); pos=nx.spring_layout(G,seed=42,k=1.1)
    shapes={"chemical":"o","structure":"s","super_class":"h","class":"h","literature_target":"^","predicted_target":"v","toxcast_tox21_assay":"8","hazard_context":"d","exposure_context":"d","mea_metric":"D","zscore_response":"P","toxicity_endpoint":"*","embedding_neighbor":"X"}
    for typ,shape in shapes.items():
        nodes=[n for n,a in G.nodes(data=True) if a.get("node_type")==typ]
        if nodes: nx.draw_networkx_nodes(G,pos,nodelist=nodes,node_shape=shape,node_size=[1500 if typ=="chemical" else 650]*len(nodes))
    nx.draw_networkx_edges(G,pos,alpha=.4); nx.draw_networkx_labels(G,pos,labels={n:a.get("label",n) for n,a in G.nodes(data=True)},font_size=7)
    plt.title("Queryable multimodal toxicology knowledge graph"); plt.axis("off"); plt.tight_layout(); path.parent.mkdir(parents=True,exist_ok=True); plt.savefig(path,dpi=220,bbox_inches="tight"); plt.close()


def save_interactive_html(G,path):
    """Render the query graph to an interactive pyvis HTML page (hover
    tooltips per node/edge). Returns False (no error) if pyvis isn't
    installed, since this output is a nice-to-have, not a hard requirement."""
    try: from pyvis.network import Network
    except ImportError: return False
    net=Network(height="850px",width="100%",bgcolor="#ffffff",font_color="#222222"); net.barnes_hut()
    for node,attrs in G.nodes(data=True):
        title=[f"<b>{attrs.get('label',node)}</b>",f"Type: {attrs.get('node_type','')}"]+[f"{k}: {v}" for k,v in attrs.items() if k not in {"label","node_type"} and str(v)]; net.add_node(node,label=attrs.get("label",node),title="<br>".join(title))
    for u,v,attrs in G.edges(data=True): net.add_edge(u,v,title="<br>".join(f"{k}: {val}" for k,val in attrs.items()),label=str(attrs.get("relation","")))
    path.parent.mkdir(parents=True,exist_ok=True); net.write_html(str(path),open_browser=False); return True


def main():
    """CLI entry point: resolve the query, build+print the profile, and (if
    found) save the JSON profile plus graph outputs under results/."""
    parser=argparse.ArgumentParser(); parser.add_argument("--chemical",help="Chemical name or DTXSID; prompt if omitted"); parser.add_argument("--top-targets",type=int,default=12); parser.add_argument("--top-metrics",type=int,default=12); parser.add_argument("--top-toxcast",type=int,default=25); parser.add_argument("--no-pubchem",action="store_true"); args=parser.parse_args(); chemical=args.chemical or input("Enter chemical name or DTXSID: ").strip()
    p=profile_from_tables(chemical,load_tables(),args.top_targets,args.top_metrics,args.top_toxcast,not args.no_pubchem); print_profile(p)
    if p.get("status")!="ok": return
    OUTPUTS.mkdir(exist_ok=True); slug=slugify(p["preferred_name"]); json_path=OUTPUTS/f"{slug}_multimodal_profile.json"; json_path.write_text(json.dumps(p,indent=2,default=str),encoding="utf-8")
    G=build_query_graph(p); graphml=OUTPUTS/f"{slug}_knowledge_graph.graphml"; write_graphml_unique_edge_ids(G,graphml); png=OUTPUTS/f"{slug}_knowledge_graph.png"; save_png(G,png); html=OUTPUTS/f"{slug}_knowledge_graph.html"; html_ok=save_interactive_html(G,html)
    print("\n[OUTPUT FILES]"); print(" ",json_path); print(" ",graphml); print(" ",png); print(" ",html if html_ok else "Interactive HTML skipped (install requirements.txt for pyvis).")


if __name__=="__main__": main()
