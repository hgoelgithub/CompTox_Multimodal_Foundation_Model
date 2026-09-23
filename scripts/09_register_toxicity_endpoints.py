"""STEP 9 -- Append real, measured toxicity/injury endpoint results (hERG,
DILI, hepatotoxicity, etc.) into a single registry CSV, validating required
columns and identifiers first. Deliberately kept separate from *predicted*
targets (ToxProfiler) elsewhere in the project: a predicted KCNH2 target hit
must never be silently treated as a measured hERG result. This registry is
what 07_query_chemical.py and 08_build_knowledge_graph.py read as
"[TOXICITY / INJURY ENDPOINTS]"."""
import argparse
import pandas as pd
from core.paths import ENDPOINT_DATA

OUT=ENDPOINT_DATA/"endpoint_results.csv"; REQUIRED={"endpoint","result_type"}; IDENTIFIERS={"chemical_name","compound_id","DTXSID"}
ALL=["chemical_name","compound_id","DTXSID","endpoint","endpoint_group","result_type","task_type","value","label","probability","unit","model","source","notes"]


def main():
    """CLI entry point: validate `--input`'s columns/identifiers, then append
    it to the registry (deduplicated) -- or with --check, just report the
    registry's current size."""
    p=argparse.ArgumentParser(); p.add_argument("--input"); p.add_argument("--check",action="store_true"); a=p.parse_args()
    if a.check:
        print("Endpoint registry:",OUT); print("Rows:",len(pd.read_csv(OUT)) if OUT.exists() else 0); return
    if not a.input: p.error("--input is required unless --check is used")
    new=pd.read_csv(a.input); missing=REQUIRED-set(new.columns)
    if missing: raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not (IDENTIFIERS & set(new.columns)): raise ValueError("Provide at least one identifier: chemical_name, compound_id, or DTXSID")
    for col in ALL:
        if col not in new: new[col]=""
    new=new[ALL]; current=pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=ALL)
    merged=new.copy() if current.empty else pd.concat([current,new],ignore_index=True)
    merged=merged.drop_duplicates(); OUT.parent.mkdir(parents=True,exist_ok=True); merged.to_csv(OUT,index=False)
    print("Registered input rows:",len(new)); print("Unique endpoint rows:",len(merged)); print("Saved:",OUT)


if __name__=="__main__": main()
