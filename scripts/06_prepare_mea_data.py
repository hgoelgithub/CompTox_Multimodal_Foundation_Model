"""Step 6 - Prepare the MEA neurotoxicity workbook

Parses `data/mea/MEA_T_DATA.xlsx`, an in-house microelectrode-array (MEA) neurotoxicity screen, into tidy CSV
tables that later steps use as an independent downstream endpoint and as knowledge-graph evidence.

Workbook sheets (one row per compound):
  logAC50                          potency (log AC50) for each of 19 MEA network metrics
  HIT                              hit code per metric: 0 = no effect, 1 = increase, 2 = decrease
  z-score                          z-score of each metric at 6 concentrations (0.1 - 30 uM)
  Literature drug-target interact  literature-reported molecular targets
  molecule class                   chemical super-class and class
  ToxProfiler Prediction           *predicted* target-activity scores (predictions, not measurements)

Input : data/mea/MEA_T_DATA.xlsx
Output: data/mea_processed/*.csv and MEA_SUMMARY.md, notably
          mea_compound_summary.csv                one row per compound, incl. `n_mea_metric_hits` (the transfer label)
          mea_to_comptox_mapping_template.csv     compound -> DTXSID/SMILES (your curated entries are preserved on re-runs)
Run   : python scripts/06_prepare_mea_data.py
"""

# %%
import argparse
import re
from pathlib import Path

import pandas as pd

# %% [markdown]
# ## 1. Paths and constants

# %%
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = PROJECT_ROOT / "data" / "mea" / "MEA_T_DATA.xlsx"
OUT_DIR = PROJECT_ROOT / "data" / "mea_processed"

CONCENTRATIONS_UM = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]      # the six tested concentrations, in column order
HIT_DIRECTION = {0: "no_hit", 1: "increase", 2: "decrease"}
TOXPROFILER_POSITIVE = 1.0                                  # predicted score at or above this counts as a positive

# %% [markdown]
# ## 2. Helpers
# Compound identifiers are kept exactly as they appear in the workbook. `base_name` additionally strips a
# trailing plate-well suffix such as `_Y1P7`, if one is present, so replicate wells could be grouped.

# %%
def base_name(name):
    return re.sub(r"_Y\d+P\d+$", "", str(name).strip())


def split_terms(value):
    """Split a free-text target cell on ';', '|' or ' + ' and drop empty / 'nan' entries."""
    if pd.isna(value) or not str(value).strip():
        return []
    pieces = re.split(r"\s*[;|]\s*|\s+\+\s+", str(value).strip())
    return [p.strip() for p in pieces if p.strip() and p.strip().lower() not in {"nan", "none"}]


def read_sheet(sheet):
    """One workbook sheet with its first column renamed `compound_id`, plus a `base_compound_name` column."""
    df = pd.read_excel(WORKBOOK, sheet_name=sheet)
    df = df.rename(columns={df.columns[0]: "compound_id"})
    df["base_compound_name"] = df["compound_id"].map(base_name)
    return df


def melt_metrics(df, value_name):
    """Wide (one column per metric) -> long (one row per compound and metric)."""
    return df.melt(id_vars=["compound_id", "base_compound_name"], var_name="metric", value_name=value_name)

# %% [markdown]
# ## 3. Parse each sheet
# Each function writes a wide CSV (as in the workbook) and/or a long tidy CSV, and returns the long table.
# * **logAC50 / HIT**: per compound and metric. HIT also gets a readable `direction` column.
# * **Literature targets**: every annotation column after the compound id is split into individual targets and
#   merged into one deduplicated list per compound.
# * **ToxProfiler**: all columns except the identifiers are predicted target scores; scores >= 1.0 are also
#   written to a separate "positive predictions" file. These are predictions and are kept clearly separate from
#   measured MEA effects everywhere in the project.
# * **z-score**: each metric appears in six consecutive columns, one per concentration; they are grouped back
#   together and melted into (compound, metric, concentration, z-score) rows, which preserves the
#   concentration-response shape.

# %%
def prepare_logac50():
    df = read_sheet("logAC50")
    df.to_csv(OUT_DIR / "mea_logac50_wide.csv", index=False)
    long = melt_metrics(df, "logac50")
    long.to_csv(OUT_DIR / "mea_logac50_long.csv", index=False)
    return long


def prepare_hit():
    df = read_sheet("HIT")
    df.to_csv(OUT_DIR / "mea_hit_wide.csv", index=False)
    long = melt_metrics(df, "hit_code")
    long["hit_code"] = pd.to_numeric(long["hit_code"], errors="coerce")
    long["direction"] = long["hit_code"].map(HIT_DIRECTION)
    long.to_csv(OUT_DIR / "mea_hit_long.csv", index=False)
    return long


def prepare_literature_targets():
    df = read_sheet("Literature drug-target interact")
    annotation_columns = [c for c in df.columns if c not in {"compound_id", "base_compound_name"}]
    rows = []
    for _, row in df.iterrows():
        targets = []
        for column in annotation_columns:
            targets.extend(split_terms(row[column]))
        for target in dict.fromkeys(targets):  # deduplicate, keep order
            rows.append({"compound_id": row["compound_id"], "base_compound_name": row["base_compound_name"],
                         "target": target, "source": "literature", "evidence_text": target})
    long = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "mea_literature_targets_wide.csv", index=False)
    long.to_csv(OUT_DIR / "mea_literature_targets_long.csv", index=False)
    return long


def prepare_molecule_class():
    df = read_sheet("molecule class")
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
    df.to_csv(OUT_DIR / "mea_molecule_class.csv", index=False)
    return df


def prepare_toxprofiler():
    df = read_sheet("ToxProfiler Prediction")
    df.columns = [str(c).strip() for c in df.columns]
    id_columns = ["compound_id", "base_compound_name"]
    score_columns = [c for c in df.columns if c not in id_columns]
    long = df.melt(id_vars=id_columns, value_vars=score_columns, var_name="target", value_name="score")
    long["score"] = pd.to_numeric(long["score"], errors="coerce")
    long = long.dropna(subset=["score"])
    long.to_csv(OUT_DIR / "mea_toxprofiler_long.csv", index=False)
    df.to_csv(OUT_DIR / "mea_toxprofiler_wide.csv", index=False)
    long[long["score"] >= TOXPROFILER_POSITIVE].to_csv(OUT_DIR / "mea_toxprofiler_positive_predictions.csv", index=False)
    return long


def prepare_zscore():
    df = read_sheet("z-score")
    metric_columns = {}  # metric name -> its six concentration columns, in order
    for column in df.columns[1:]:
        if str(column).startswith("Unnamed") or column == "base_compound_name":
            continue
        metric = re.sub(r"\.\d+$", "", str(column)).strip()   # pandas renames repeats "x", "x.1", "x.2", ...
        metric_columns.setdefault(metric, []).append(column)
    rows = []
    for _, row in df.iterrows():
        for metric, columns in metric_columns.items():
            if len(columns) != len(CONCENTRATIONS_UM):
                continue
            for concentration, column in zip(CONCENTRATIONS_UM, columns):
                rows.append({"compound_id": row["compound_id"], "base_compound_name": row["base_compound_name"],
                             "concentration_uM": concentration, "metric": metric,
                             "zscore": pd.to_numeric(row[column], errors="coerce")})
    long = pd.DataFrame(rows)
    long.to_csv(OUT_DIR / "mea_zscore_long.csv", index=False)
    df.to_csv(OUT_DIR / "mea_zscore_wide.csv", index=False)
    return long

# %% [markdown]
# ## 4. Combine into summaries
# * `mea_metric_activity_summary.csv`: hit code, direction and logAC50 for every compound and metric.
# * `mea_compound_summary.csv`: one row per compound. **`n_mea_metric_hits`** counts the metrics with any
#   effect (increase or decrease); it is the label used for transfer learning in steps 10-10c. Also counted:
#   increases, decreases, positive ToxProfiler predictions and literature targets, plus the chemical class.
# * `mea_to_comptox_mapping_template.csv`: the table linking each MEA compound to a CompTox DTXSID and SMILES.
#   It is created empty and filled by step 01b (exact name matches) or by hand. **Existing entries are kept
#   when this step is re-run.**

# %%
def build_summaries(logac50, hit, literature, classes, toxprofiler):
    keys = ["compound_id", "base_compound_name"]
    metric = hit.merge(logac50, on=keys + ["metric"], how="outer")
    metric.to_csv(OUT_DIR / "mea_metric_activity_summary.csv", index=False)

    code = lambda s: pd.Series(s).fillna(0)
    summary = metric.groupby(keys, as_index=False).agg(
        n_mea_metric_hits=("hit_code", lambda s: int((code(s) > 0).sum())),
        n_hit_code_1=("hit_code", lambda s: int((code(s) == 1).sum())),
        n_hit_code_2=("hit_code", lambda s: int((code(s) == 2).sum())),
        n_increase=("hit_code", lambda s: int((code(s) == 1).sum())),
        n_decrease=("hit_code", lambda s: int((code(s) == 2).sum())),
    )
    if not toxprofiler.empty:
        tox = toxprofiler.groupby(keys, as_index=False).agg(
            n_toxprofiler_positive_targets=("score", lambda s: int((pd.Series(s) >= TOXPROFILER_POSITIVE).sum())),
            max_toxprofiler_score=("score", "max"))
        summary = summary.merge(tox, on=keys, how="left")
    if not literature.empty:
        summary = summary.merge(literature.groupby(keys, as_index=False).agg(n_literature_targets=("target", "nunique")),
                                on=keys, how="left")
    class_columns = [c for c in ["compounds_super_class", "compounds_class"] if c in classes.columns]
    summary = summary.merge(classes[keys + class_columns], on=keys, how="left")
    summary.to_csv(OUT_DIR / "mea_compound_summary.csv", index=False)

    mapping_path = OUT_DIR / "mea_to_comptox_mapping_template.csv"
    mapping = summary[keys].drop_duplicates().assign(DTXSID="", SMILES="")
    if mapping_path.exists():  # keep DTXSID / SMILES entries that already exist
        try:
            existing = pd.read_csv(mapping_path, dtype=str).fillna("")
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame()
        if {"compound_id", "DTXSID", "SMILES"}.issubset(existing.columns):
            existing = existing[["compound_id", "DTXSID", "SMILES"]].drop_duplicates("compound_id", keep="last")
            mapping = mapping.drop(columns=["DTXSID", "SMILES"]).merge(existing, on="compound_id", how="left")
            mapping[["DTXSID", "SMILES"]] = mapping[["DTXSID", "SMILES"]].fillna("")
    mapping.to_csv(mapping_path, index=False)


def write_summary_report():
    """MEA_SUMMARY.md: sheet sizes and parsed row counts, for a quick sanity check."""
    lines = ["# MEA workbook summary", "", f"Workbook: `{WORKBOOK.name}`", "", "## Sheets"]
    for sheet in pd.ExcelFile(WORKBOOK).sheet_names:
        shape = pd.read_excel(WORKBOOK, sheet_name=sheet).shape
        lines.append(f"- **{sheet}**: {shape[0]} rows × {shape[1]} columns")
    summary = pd.read_csv(OUT_DIR / "mea_compound_summary.csv")
    hit = pd.read_csv(OUT_DIR / "mea_hit_long.csv")
    lines += ["", "## Parsed summary",
              f"- Unique compound identifiers: {summary['compound_id'].nunique()}",
              f"- Unique base compound names: {summary['base_compound_name'].nunique()}",
              f"- Literature target edges: {len(pd.read_csv(OUT_DIR / 'mea_literature_targets_long.csv'))}",
              f"- Positive ToxProfiler prediction rows (score ≥ {TOXPROFILER_POSITIVE}): "
              f"{len(pd.read_csv(OUT_DIR / 'mea_toxprofiler_positive_predictions.csv'))}",
              f"- Non-zero MEA hit rows: {int(hit['hit_code'].fillna(0).gt(0).sum())}",
              f"- Z-score long rows: {len(pd.read_csv(OUT_DIR / 'mea_zscore_long.csv'))}"]
    (OUT_DIR / "MEA_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

# %% [markdown]
# ## 5. Run

# %%
def prepare_mea_data():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logac50 = prepare_logac50()
    hit = prepare_hit()
    literature = prepare_literature_targets()
    classes = prepare_molecule_class()
    toxprofiler = prepare_toxprofiler()
    prepare_zscore()
    build_summaries(logac50, hit, literature, classes, toxprofiler)
    write_summary_report()
    print("Prepared MEA tables in:", OUT_DIR)

# %% Command line
def main():
    argparse.ArgumentParser(description="Parse the MEA workbook into tidy CSV tables.").parse_args()
    prepare_mea_data()


if __name__ == "__main__":
    main()
