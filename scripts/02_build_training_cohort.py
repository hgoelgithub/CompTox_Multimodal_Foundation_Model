"""STEP 2 -- Build the training cohort the foundation model actually trains on.

Joins chemicals.csv (dsstox/01b), bioactivity.csv (toxcast/01a), hazard.csv
(toxval/01), and exposure.csv (cpdat/01) by DTXSID, computes 16 RDKit
molecular descriptors per compound, scores each compound by how much
multimodal evidence it has, and keeps the top `max_compounds` (config.yaml)
by that score -- not simply the compounds with the most SMILES. Reads the
four normalized tables through core/duckdb_io.py's Parquet mirror rather than
re-parsing the raw CSVs. Output: data/processed/comptox_v3.parquet.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem import Descriptors,Crippen,Lipinski,rdMolDescriptors
from core.paths import RAW_COMPTox, PROCESSED_COMPTox, CONFIG
from core.table_io import write_table
from core import duckdb_io

OUT=PROCESSED_COMPTox/"comptox_v3.parquet"
DESCRIPTOR_NAMES = [
    "mw", "logp", "tpsa", "hbd", "hba", "rotatable_bonds",
    "ring_count", "aromatic_ring_count", "aliphatic_ring_count",
    "saturated_ring_count", "heteroatom_count", "fraction_csp3",
    "heavy_atom_count", "formal_charge", "radical_electron_count",
    "valence_electron_count",
]


def require_columns(df,required,name):
    """Fail fast with a clear message if `df` (named `name` for the error) is
    missing any column the rest of this script assumes exists."""
    missing=set(required)-set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def descriptors(smiles):
    """Compute the 16 RDKit descriptors in DESCRIPTOR_NAMES for one SMILES
    string; returns all-NaN if RDKit can't parse it."""
    mol=Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return [np.nan]*len(DESCRIPTOR_NAMES)
    return [
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol),
        rdMolDescriptors.CalcNumSaturatedRings(mol),
        rdMolDescriptors.CalcNumHeteroatoms(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        rdMolDescriptors.CalcNumHeavyAtoms(mol),
        Chem.GetFormalCharge(mol),
        Descriptors.NumRadicalElectrons(mol),
        Descriptors.NumValenceElectrons(mol),
    ]


def prepare_bioactivity(bio,max_assays,min_measurements):
    """Pivot long-format bioactivity rows (one row per chemical/assay) into a
    wide per-chemical matrix: one biohit__<assay>/bioac50__<assay>/bioeff__<assay>
    column per assay, keeping only the `max_assays` best-measured assays with
    at least `min_measurements` chemicals tested (config.yaml). AC50 is
    log10-transformed since potency spans several orders of magnitude."""
    require_columns(bio,{"DTXSID","program","assay","hitcall"},"bioactivity")
    bio["hitcall"]=pd.to_numeric(bio["hitcall"],errors="coerce")
    coverage=bio.groupby("assay")["hitcall"].count()
    assays=coverage[coverage>=min_measurements].sort_values(ascending=False).head(max_assays).index
    if len(assays)==0:
        raise ValueError(
            f"No assays meet min_assay_measurements={min_measurements}. "
            "Lower the threshold after checking the normalized data."
        )
    bio=bio[bio["assay"].isin(assays)].copy()
    matrices=[]
    for col,prefix in [("hitcall","biohit__"),("ac50_uM","bioac50__"),("efficacy","bioeff__")]:
        if col not in bio.columns:
            continue
        values=pd.to_numeric(bio[col],errors="coerce")
        if col=="ac50_uM":
            values=pd.Series(np.where(values>0,np.log10(values),np.nan),index=bio.index)
        temp=bio[["DTXSID","assay"]].copy(); temp["value"]=values
        wide=temp.pivot_table(index="DTXSID",columns="assay",values="value",aggfunc="median")
        wide.columns=[prefix+str(x) for x in wide.columns]
        matrices.append(wide)
    if not matrices:
        raise ValueError("No usable bioactivity matrices could be created.")
    result=pd.concat(matrices,axis=1).reset_index()
    if "DTXSID" not in result.columns:
        raise ValueError("Bioactivity matrix did not contain any DTXSID rows.")
    return result


def prepare_hazard(h):
    """Keep evidence counts; pool dose values only within compatible contexts.

    Convert oral rat repeated-dose NOAEL/LOAEL to mg/kg/day. Other units and
    contexts remain in the normalized evidence table, not in scalar targets.
    """
    require_columns(h,{"DTXSID","metric","value"},"hazard")
    rows=[]
    for sid,g in h.groupby("DTXSID"):
        result={"DTXSID":sid,"hazard_count":len(g),"noael_log":np.nan,"loael_log":np.nan}
        if {"units","route","species","study_type","duration"}.issubset(g):
            units=g["units"].astype(str).str.lower().str.replace("µ","u").str.replace("μ","u").str.replace(" ","")
            factors=units.map({"mg/kg/day":1.,"mg/kg-d":1.,"mg/kg-bw/day":1.,"ug/kg/day":.001,"g/kg/day":1000.})
            context=g["route"].astype(str).str.lower().eq("oral") & g["species"].astype(str).str.lower().isin(["rat","rats"])
            sub=g[context & factors.notna()].copy()
            sub["dose"]=pd.to_numeric(g["value"],errors="coerce")*factors
            for metric,key in [("NOAEL","noael_log"),("LOAEL","loael_log")]:
                selected=sub[sub["metric"].astype(str).str.upper().str.strip().eq(metric)]
                # Avoid collapsing distinct study types/durations into one target.
                if len(selected) and selected[["study_type","duration"]].notna().all().all() and len(selected[["study_type","duration"]].drop_duplicates())==1:
                    dose=selected.loc[selected["dose"]>0,"dose"]
                    if len(dose): result[key]=np.log10(dose.median())
        rows.append(result)
    return pd.DataFrame(rows,columns=["DTXSID","hazard_count","noael_log","loael_log"])


def prepare_exposure(e):
    """Collapse per-product exposure rows into per-chemical summary counts
    (how many products/use categories a chemical shows up in)."""
    require_columns(e,{"DTXSID","product_id","use_category"},"exposure")
    return e.groupby("DTXSID").agg(
        exposure_count=("DTXSID","size"),product_count=("product_id","nunique"),
        use_category_count=("use_category","nunique")).reset_index()


def preflight():
    """Check the two hard-required inputs exist before doing any real work."""
    required=[RAW_COMPTox/"chemicals.csv",RAW_COMPTox/"bioactivity.csv"]
    missing=[p for p in required if not p.exists()]
    if missing:
        print("CompTox raw data are not installed yet.")
        for p in missing: print("  missing:",p)
        print("Run scripts/01_normalize_epa_data.py first. This is a data-availability issue, not a code failure.")
        return False
    print("CompTox input files found.")
    return True


def main():
    """CLI entry point: load+merge the four normalized modalities, compute
    descriptors, score and cap the cohort by information coverage, and write
    comptox_v3.parquet plus a small compound-identifier index."""
    parser=argparse.ArgumentParser(); parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    if args.check:
        raise SystemExit(0 if preflight() else 1)
    if not preflight():
        raise FileNotFoundError("Required normalized CompTox files are missing. Run with --check for details.")

    cfg=yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["data"]
    # Refresh the Parquet/DuckDB mirrors so the cohort is built from columnar
    # storage instead of re-parsing the full normalized CSVs on every run.
    duckdb_io.refresh()
    chemicals=duckdb_io.query_df("SELECT * FROM chemicals")
    require_columns(chemicals,{"DTXSID","smiles"},"chemicals.csv")
    chemicals=chemicals.drop_duplicates("DTXSID")
    chemicals=chemicals[chemicals["smiles"].notna()].copy()
    if chemicals.empty: raise ValueError("No chemicals with SMILES available.")
    chemicals["smiles"]=chemicals["smiles"].map(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)) if Chem.MolFromSmiles(x) is not None else None)
    chemicals=chemicals.dropna(subset=["smiles"])
    if chemicals.empty: raise ValueError("No valid structures available.")
    d=np.array([descriptors(x) for x in chemicals["smiles"]])
    for i,name in enumerate(DESCRIPTOR_NAMES): chemicals[name]=d[:,i]
    chemicals=chemicals[np.isfinite(chemicals["mw"])].copy()
    bio=prepare_bioactivity(duckdb_io.query_df("SELECT * FROM bioactivity"),cfg["max_assays"],cfg["min_assay_measurements"])
    data=chemicals.merge(bio,on="DTXSID",how="left")
    if duckdb_io.table_available("hazard"): data=data.merge(prepare_hazard(duckdb_io.query_df("SELECT * FROM hazard")),on="DTXSID",how="left")
    if duckdb_io.table_available("exposure"): data=data.merge(prepare_exposure(duckdb_io.query_df("SELECT * FROM exposure")),on="DTXSID",how="left")
    hit=[c for c in data if c.startswith("biohit__")]
    data["bioactivity_coverage"]=data[hit].notna().sum(axis=1)
    data["hazard_available"]=data.get("hazard_count",pd.Series(index=data.index,dtype=float)).notna().astype(int)
    data["exposure_available"]=data.get("exposure_count",pd.Series(index=data.index,dtype=float)).notna().astype(int)
    data["coverage_score"]=3*data["bioactivity_coverage"]+20*data["hazard_available"]+10*data["exposure_available"]
    data=data.sort_values(["coverage_score","bioactivity_coverage"],ascending=False).head(cfg["max_compounds"])
    if len(data)<2: raise ValueError("At least two prepared chemicals are required.")
    saved=write_table(data,OUT)
    print(f"Selected {len(data):,} chemicals")
    print(f"Bioactivity: {(data.bioactivity_coverage>0).sum():,}")
    print(f"Hazard: {data.hazard_available.sum():,}")
    print(f"Exposure: {data.exposure_available.sum():,}")
    print("Saved:",saved)

    # Small identifier index used by the interactive chemical-query script.
    index_cols=[c for c in ["DTXSID","preferred_name","CASRN","smiles"] if c in chemicals.columns]
    chemicals[index_cols].drop_duplicates("DTXSID").to_csv(PROCESSED_COMPTox/"comptox_query_index.csv",index=False)
    print("Saved query index:",PROCESSED_COMPTox/"comptox_query_index.csv")


if __name__=="__main__": main()
