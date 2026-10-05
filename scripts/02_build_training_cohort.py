"""Step 2 - Build the training cohort

Joins the four normalized tables by DTXSID into one table with a row per chemical, which is what the
foundation model is trained on:
  chemicals.csv     structure (SMILES) and names        -> 16 RDKit physicochemical descriptors
  bioactivity.csv   ToxCast/Tox21 assay results         -> wide matrix, one column per assay
                                                            biohit__<assay>   hit call (0-1)
                                                            bioac50__<assay>  log10 AC50 (uM)
                                                            bioeff__<assay>   efficacy (top of the curve)
  hazard.csv        ToxValDB study values               -> hazard_count, noael_log, loael_log
  exposure.csv      CPDat product / use data            -> exposure_count, product_count, use_category_count

Chemicals are ranked by how much evidence they have (bioactivity coverage, plus bonuses for hazard and
exposure data) and the top `max_compounds` (config.yaml) are kept. The normalized CSVs are also mirrored to
Parquet (data/normalized/parquet/), which steps 07 and 08 use for fast per-chemical lookups.

Input : data/normalized/{chemicals,bioactivity,hazard,exposure}.csv   (from step 01 or 01a/01b)
Output: data/processed/comptox_v3.parquet, data/processed/comptox_query_index.csv
Run   : python scripts/02_build_training_cohort.py [--check]
"""

# %%
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

# %% [markdown]
# ## 1. Paths and constants

# %%
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "config.yaml"
NORMALIZED = PROJECT_ROOT / "data" / "normalized"
PARQUET_MIRROR = NORMALIZED / "parquet"
COHORT = PROJECT_ROOT / "data" / "processed" / "comptox_v3.parquet"
QUERY_INDEX = PROJECT_ROOT / "data" / "processed" / "comptox_query_index.csv"

TABLES = ("chemicals", "bioactivity", "hazard", "exposure", "chemical_identifiers")
DESCRIPTOR_NAMES = ["mw", "logp", "tpsa", "hbd", "hba", "rotatable_bonds", "ring_count", "aromatic_ring_count",
                    "aliphatic_ring_count", "saturated_ring_count", "heteroatom_count", "fraction_csp3",
                    "heavy_atom_count", "formal_charge", "radical_electron_count", "valence_electron_count"]

# %% [markdown]
# ## 2. Structures and descriptors
# Each SMILES is parsed and re-written in RDKit's canonical form (so identical structures look identical), and
# the 16 descriptors are computed. A structure RDKit cannot parse is dropped.

# %%
def descriptors(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return [np.nan] * len(DESCRIPTOR_NAMES)
    return [
        Descriptors.MolWt(mol), Crippen.MolLogP(mol), rdMolDescriptors.CalcTPSA(mol),
        Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol), rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol), rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol), rdMolDescriptors.CalcNumSaturatedRings(mol),
        rdMolDescriptors.CalcNumHeteroatoms(mol), rdMolDescriptors.CalcFractionCSP3(mol),
        rdMolDescriptors.CalcNumHeavyAtoms(mol), Chem.GetFormalCharge(mol),
        Descriptors.NumRadicalElectrons(mol), Descriptors.NumValenceElectrons(mol),
    ]


def canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def require_columns(df, required, name):
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")

# %% [markdown]
# ## 3. The three evidence tables
# * **Bioactivity** is pivoted from one row per chemical and assay to one row per chemical, keeping the
#   `max_assays` best-measured assays that have at least `min_assay_measurements` chemicals (both in
#   `config.yaml`). Repeated measurements are combined by their median. AC50 is log10-transformed because
#   potency spans many orders of magnitude.
# * **Hazard**: only oral, rat, repeated-dose NOAEL and LOAEL values with a recognised dose unit are turned
#   into a number (log10 mg/kg/day), and only when all of a chemical's values come from a single study type
#   and duration, so different study designs are never averaged together. Every chemical also gets a count
#   of its hazard records.
# * **Exposure**: per-product rows are reduced to counts of records, distinct products and use categories.

# %%
def prepare_bioactivity(bio, max_assays, min_measurements):
    require_columns(bio, {"DTXSID", "program", "assay", "hitcall"}, "bioactivity")
    bio["hitcall"] = pd.to_numeric(bio["hitcall"], errors="coerce")
    coverage = bio.groupby("assay")["hitcall"].count()
    assays = coverage[coverage >= min_measurements].sort_values(ascending=False).head(max_assays).index
    if len(assays) == 0:
        raise ValueError(f"No assay has at least {min_measurements} measurements; lower min_assay_measurements.")
    bio = bio[bio["assay"].isin(assays)].copy()
    matrices = []
    for column, prefix in [("hitcall", "biohit__"), ("ac50_uM", "bioac50__"), ("efficacy", "bioeff__")]:
        if column not in bio.columns:
            continue
        values = pd.to_numeric(bio[column], errors="coerce")
        if column == "ac50_uM":
            values = pd.Series(np.where(values > 0, np.log10(values), np.nan), index=bio.index)
        long = bio[["DTXSID", "assay"]].copy()
        long["value"] = values
        wide = long.pivot_table(index="DTXSID", columns="assay", values="value", aggfunc="median")
        wide.columns = [prefix + str(a) for a in wide.columns]
        matrices.append(wide)
    if not matrices:
        raise ValueError("No usable bioactivity matrices could be created.")
    return pd.concat(matrices, axis=1).reset_index()


DOSE_TO_MG_PER_KG_DAY = {"mg/kg/day": 1.0, "mg/kg-d": 1.0, "mg/kg-bw/day": 1.0, "ug/kg/day": 0.001, "g/kg/day": 1000.0}


def prepare_hazard(hazard):
    require_columns(hazard, {"DTXSID", "metric", "value"}, "hazard")
    rows = []
    for dtxsid, group in hazard.groupby("DTXSID"):
        result = {"DTXSID": dtxsid, "hazard_count": len(group), "noael_log": np.nan, "loael_log": np.nan}
        if {"units", "route", "species", "study_type", "duration"}.issubset(group):
            units = group["units"].astype(str).str.lower().str.replace("µ", "u").str.replace("μ", "u").str.replace(" ", "")
            factors = units.map(DOSE_TO_MG_PER_KG_DAY)
            oral_rat = group["route"].astype(str).str.lower().eq("oral") & group["species"].astype(str).str.lower().isin(["rat", "rats"])
            usable = group[oral_rat & factors.notna()].copy()
            usable["dose"] = pd.to_numeric(group["value"], errors="coerce") * factors
            for metric, key in [("NOAEL", "noael_log"), ("LOAEL", "loael_log")]:
                selected = usable[usable["metric"].astype(str).str.upper().str.strip().eq(metric)]
                one_design = len(selected) and selected[["study_type", "duration"]].notna().all().all() \
                    and len(selected[["study_type", "duration"]].drop_duplicates()) == 1
                if one_design:
                    dose = selected.loc[selected["dose"] > 0, "dose"]
                    if len(dose):
                        result[key] = np.log10(dose.median())
        rows.append(result)
    return pd.DataFrame(rows, columns=["DTXSID", "hazard_count", "noael_log", "loael_log"])


def prepare_exposure(exposure):
    require_columns(exposure, {"DTXSID", "product_id", "use_category"}, "exposure")
    return exposure.groupby("DTXSID").agg(exposure_count=("DTXSID", "size"), product_count=("product_id", "nunique"),
                                          use_category_count=("use_category", "nunique")).reset_index()

# %% [markdown]
# ## 4. Build the cohort
# Steps inside `build_cohort()`:
# 1. Mirror the normalized CSVs to Parquet.
# 2. Read chemicals, keep one row per DTXSID with a valid structure, and compute descriptors.
# 3. Merge bioactivity, hazard and exposure evidence onto them (chemicals lacking a modality keep missing
#    values, which the model handles through its masks).
# 4. Score each chemical: `3 x (#assays measured) + 20 (has hazard) + 10 (has exposure)`, keep the top
#    `max_compounds`.
# 5. Save the cohort and a small name / CAS / SMILES index used by the chemical-query step.

# %%
def check_inputs():
    missing = [NORMALIZED / f"{name}.csv" for name in ("chemicals", "bioactivity") if not (NORMALIZED / f"{name}.csv").exists()]
    for path in missing:
        print("missing:", path)
    if missing:
        print("Run step 01 first (or 01a and 01b). This is a data-availability issue, not a code failure.")
    else:
        print("Normalized input files found.")
    return not missing


def mirror_to_parquet():
    """Write data/normalized/parquet/<table>.parquet for every normalized CSV that exists."""
    PARQUET_MIRROR.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        source = NORMALIZED / f"{name}.csv"
        if source.exists():
            pd.read_csv(source, low_memory=False).to_parquet(PARQUET_MIRROR / f"{name}.parquet", index=False)


def read_normalized(name):
    path = PARQUET_MIRROR / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else None


def build_cohort(output=COHORT, index_output=QUERY_INDEX):
    if not check_inputs():
        raise FileNotFoundError("Required normalized tables are missing.")
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["data"]
    mirror_to_parquet()

    chemicals = read_normalized("chemicals")
    require_columns(chemicals, {"DTXSID", "smiles"}, "chemicals.csv")
    chemicals = chemicals.drop_duplicates("DTXSID")
    chemicals = chemicals[chemicals["smiles"].notna()].copy()
    chemicals["smiles"] = chemicals["smiles"].map(canonical_smiles)
    chemicals = chemicals.dropna(subset=["smiles"])
    if chemicals.empty:
        raise ValueError("No chemicals with a valid structure.")
    values = np.array([descriptors(s) for s in chemicals["smiles"]])
    for i, name in enumerate(DESCRIPTOR_NAMES):
        chemicals[name] = values[:, i]
    chemicals = chemicals[np.isfinite(chemicals["mw"])].copy()

    bioactivity = prepare_bioactivity(read_normalized("bioactivity"), cfg["max_assays"], cfg["min_assay_measurements"])
    data = chemicals.merge(bioactivity, on="DTXSID", how="left")
    hazard, exposure = read_normalized("hazard"), read_normalized("exposure")
    if hazard is not None:
        data = data.merge(prepare_hazard(hazard), on="DTXSID", how="left")
    if exposure is not None:
        data = data.merge(prepare_exposure(exposure), on="DTXSID", how="left")

    hit_columns = [c for c in data if c.startswith("biohit__")]
    data["bioactivity_coverage"] = data[hit_columns].notna().sum(axis=1)
    data["hazard_available"] = data.get("hazard_count", pd.Series(index=data.index, dtype=float)).notna().astype(int)
    data["exposure_available"] = data.get("exposure_count", pd.Series(index=data.index, dtype=float)).notna().astype(int)
    data["coverage_score"] = 3 * data["bioactivity_coverage"] + 20 * data["hazard_available"] + 10 * data["exposure_available"]
    data = data.sort_values(["coverage_score", "bioactivity_coverage"], ascending=False).head(cfg["max_compounds"])
    if len(data) < 2:
        raise ValueError("At least two prepared chemicals are required.")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output, index=False)
    print(f"Selected {len(data):,} chemicals")
    print(f"  with bioactivity: {(data.bioactivity_coverage > 0).sum():,}")
    print(f"  with hazard:      {data.hazard_available.sum():,}")
    print(f"  with exposure:    {data.exposure_available.sum():,}")
    print("Saved:", output)

    index_columns = [c for c in ["DTXSID", "preferred_name", "CASRN", "smiles"] if c in chemicals.columns]
    chemicals[index_columns].drop_duplicates("DTXSID").to_csv(index_output, index=False)
    print("Saved query index:", index_output)
    return data

# %% Command line
def main():
    parser = argparse.ArgumentParser(description="Build the multimodal training cohort.")
    parser.add_argument("--check", action="store_true", help="only verify that the inputs exist")
    args = parser.parse_args()
    if args.check:
        raise SystemExit(0 if check_inputs() else 1)
    build_cohort()


if __name__ == "__main__":
    main()
