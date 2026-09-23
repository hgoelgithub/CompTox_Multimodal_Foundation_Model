"""STEP 6 -- Parse the uploaded MEA (microelectrode array) neurotoxicity
workbook into clean, analysis-ready CSVs. This is the downstream evidence
source used to validate the foundation model against a real, independent
toxicity readout (see 08_build_knowledge_graph.py and 10_transfer_learning.py).

Workbook sheets expected:
- logAC50
- HIT
- Literature drug-target interact
- molecule class
- ToxProfiler Prediction
- z-score

Important:
- compound identifiers are kept exactly as given
- a base chemical name is also derived by removing a trailing well suffix like _Y1P7
- HIT values are preserved as raw codes and direction is added (0=no hit, 1=increase, 2=decrease)
- z-score data are converted from a repeated 6-concentration wide table into tidy long format
"""
from pathlib import Path
from core.paths import MEA_RAW, MEA_PROCESSED
import re
import pandas as pd
import numpy as np

RAW = MEA_RAW / "MEA_T_DATA.xlsx"
OUT = MEA_PROCESSED
CONCENTRATIONS = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
HIT_DIRECTION = {0: "no_hit", 1: "increase", 2: "decrease"}


def base_name(name: str) -> str:
    """Strip a trailing well-plate suffix like '_Y1P7' from a compound id, so
    multiple wells of the same compound can be grouped under one base name."""
    text = str(name).strip()
    return re.sub(r"_Y\d+P\d+$", "", text)


def split_terms(value):
    """Split target strings conservatively on common separators."""
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    pieces = re.split(r"\s*[;|]\s*|\s+\+\s+", text)
    cleaned = []
    for piece in pieces:
        piece = piece.strip()
        if piece and piece.lower() not in {"nan", "none"}:
            cleaned.append(piece)
    return cleaned


def normalize_first_column(df, new_name):
    """Rename an Excel sheet's unlabeled first column (the compound id) to `new_name`."""
    df = df.copy()
    df = df.rename(columns={df.columns[0]: new_name})
    return df


def prepare_logac50():
    """Parse the 'logAC50' sheet (one column per MEA metric) into a wide CSV
    and a tidy long CSV (one row per compound/metric)."""
    df = pd.read_excel(RAW, sheet_name="logAC50")
    df = normalize_first_column(df, "compound_id")
    df["base_compound_name"] = df["compound_id"].map(base_name)
    df.to_csv(OUT / "mea_logac50_wide.csv", index=False)

    long_df = df.melt(
        id_vars=["compound_id", "base_compound_name"],
        var_name="metric",
        value_name="logac50"
    )
    long_df.to_csv(OUT / "mea_logac50_long.csv", index=False)
    return df, long_df


def prepare_hit():
    """Parse the 'HIT' sheet into wide + long CSVs, adding a human-readable
    `direction` column (no_hit/increase/decrease) alongside the raw hit code."""
    df = pd.read_excel(RAW, sheet_name="HIT")
    df = normalize_first_column(df, "compound_id")
    df["base_compound_name"] = df["compound_id"].map(base_name)
    df.to_csv(OUT / "mea_hit_wide.csv", index=False)

    long_df = df.melt(
        id_vars=["compound_id", "base_compound_name"],
        var_name="metric",
        value_name="hit_code"
    )
    long_df["hit_code"] = pd.to_numeric(long_df["hit_code"], errors="coerce")
    long_df["direction"] = long_df["hit_code"].map(HIT_DIRECTION)
    long_df.to_csv(OUT / "mea_hit_long.csv", index=False)
    return df, long_df


def prepare_literature_targets():
    """Parse the literature drug-target sheet: merge its two free-text target
    columns into one deduplicated long table (one row per compound/target)."""
    df = pd.read_excel(RAW, sheet_name="Literature drug-target interact")
    df = normalize_first_column(df, "compound_id")
    df["base_compound_name"] = df["compound_id"].map(base_name)

    # Merge the two literature-target columns into one deduplicated list.
    col1 = "Putative target"
    col2 = "Himanshu Addition"

    rows = []
    for _, row in df.iterrows():
        merged = []
        merged.extend(split_terms(row.get(col1)))
        merged.extend(split_terms(row.get(col2)))
        merged = list(dict.fromkeys(merged))
        for target in merged:
            rows.append({
                "compound_id": row["compound_id"],
                "base_compound_name": row["base_compound_name"],
                "target": target,
                "source": "literature",
                "evidence_text": target
            })

    long_df = pd.DataFrame(rows)
    df.to_csv(OUT / "mea_literature_targets_wide.csv", index=False)
    long_df.to_csv(OUT / "mea_literature_targets_long.csv", index=False)
    return df, long_df


def prepare_molecule_class():
    """Parse the 'molecule class' sheet (chemical super-class/class labels)."""
    df = pd.read_excel(RAW, sheet_name="molecule class")
    df = normalize_first_column(df, "compound_id")
    df["base_compound_name"] = df["compound_id"].map(base_name)
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
    df.to_csv(OUT / "mea_molecule_class.csv", index=False)
    return df


def prepare_toxprofiler():
    """Parse the ToxProfiler predicted-target sheet into a long table of
    (compound, target, score), plus a separate CSV of just the strong
    positive predictions (score >= 1.0) for downstream graphing."""
    df = pd.read_excel(RAW, sheet_name="ToxProfiler Prediction")
    df = normalize_first_column(df, "compound_id")
    df["base_compound_name"] = df["compound_id"].map(base_name)

    # Preserve BHSAI summary columns if present.
    renamed = {}
    for col in df.columns:
        if isinstance(col, str):
            stripped = col.strip()
            if stripped == "BHSAI 10  number":
                renamed[col] = "BHSAI_10_count"
            elif stripped == "BHSAI 20  number":
                renamed[col] = "BHSAI_20_count"
    df = df.rename(columns=renamed)

    protected = {"compound_id", "base_compound_name", "BHSAI_10_count", "BHSAI_20_count"}
    score_cols = [c for c in df.columns if c not in protected]

    # Clean target names.
    clean_map = {c: str(c).strip() for c in score_cols}
    df = df.rename(columns=clean_map)
    score_cols = [clean_map[c] for c in score_cols]

    long_df = df.melt(
        id_vars=[c for c in ["compound_id", "base_compound_name", "BHSAI_10_count", "BHSAI_20_count"] if c in df.columns],
        value_vars=score_cols,
        var_name="target",
        value_name="score"
    )
    long_df["score"] = pd.to_numeric(long_df["score"], errors="coerce")
    long_df = long_df.dropna(subset=["score"])
    long_df.to_csv(OUT / "mea_toxprofiler_long.csv", index=False)
    df.to_csv(OUT / "mea_toxprofiler_wide.csv", index=False)

    # Also save only stronger positive predictions for downstream graphing.
    positive_df = long_df[long_df["score"] >= 1.0].copy()
    positive_df.to_csv(OUT / "mea_toxprofiler_positive_predictions.csv", index=False)
    return df, long_df


def prepare_zscore():
    """Parse the 'z-score' sheet: it repeats each metric 6 times (one column
    per tested concentration), so this groups those 6 columns back together
    per metric and melts them into a tidy (compound, metric, concentration,
    zscore) long table -- the concentration-response curve shape."""
    df = pd.read_excel(RAW, sheet_name="z-score")
    df = normalize_first_column(df, "compound_id")
    df["base_compound_name"] = df["compound_id"].map(base_name)

    cols = list(df.columns)
    metric_groups = {}
    ordered_bases = []

    for col in cols[1:]:
        if str(col).startswith("Unnamed"):
            continue
        if col == "base_compound_name":
            continue
        base = re.sub(r"\.\d+$", "", str(col)).strip()
        if base not in metric_groups:
            metric_groups[base] = []
            ordered_bases.append(base)
        metric_groups[base].append(col)

    rows = []
    for _, row in df.iterrows():
        for base in ordered_bases:
            columns = metric_groups[base]
            if len(columns) != 6:
                continue
            for conc, col in zip(CONCENTRATIONS, columns):
                rows.append({
                    "compound_id": row["compound_id"],
                    "base_compound_name": row["base_compound_name"],
                    "concentration_uM": conc,
                    "metric": base,
                    "zscore": pd.to_numeric(row[col], errors="coerce"),
                })

    long_df = pd.DataFrame(rows)
    long_df.to_csv(OUT / "mea_zscore_long.csv", index=False)
    df.to_csv(OUT / "mea_zscore_wide.csv", index=False)
    return df, long_df


def build_summary_tables(logac50_long, hit_long, lit_long, class_df, tox_long):
    """Combine all the per-sheet long tables into two summaries: one row per
    (compound, metric) and one row per compound. Also (re)writes the
    MEA-to-CompTox DTXSID/SMILES mapping template, preserving any DTXSID a
    user already curated by hand rather than blanking it out on re-run."""
    # Merge hit + logAC50 at the metric level.
    merged_metric = hit_long.merge(
        logac50_long,
        on=["compound_id", "base_compound_name", "metric"],
        how="outer"
    )
    if "direction" not in merged_metric.columns:
        merged_metric["direction"] = pd.to_numeric(
            merged_metric["hit_code"], errors="coerce"
        ).map(HIT_DIRECTION)
    merged_metric.to_csv(OUT / "mea_metric_activity_summary.csv", index=False)

    # Per-compound counts.
    compound_summary = merged_metric.groupby(
        ["compound_id", "base_compound_name"], as_index=False
    ).agg(
        n_mea_metric_hits=("hit_code", lambda s: int(pd.Series(s).fillna(0).gt(0).sum())),
        n_hit_code_1=("hit_code", lambda s: int((pd.Series(s).fillna(0) == 1).sum())),
        n_hit_code_2=("hit_code", lambda s: int((pd.Series(s).fillna(0) == 2).sum())),
        n_increase=("hit_code", lambda s: int((pd.Series(s).fillna(0) == 1).sum())),
        n_decrease=("hit_code", lambda s: int((pd.Series(s).fillna(0) == 2).sum())),
    )

    if not tox_long.empty:
        tox_summary = tox_long.groupby(
            ["compound_id", "base_compound_name"], as_index=False
        ).agg(
            n_toxprofiler_positive_targets=("score", lambda s: int((pd.Series(s) >= 1.0).sum())),
            max_toxprofiler_score=("score", "max"),
        )
        compound_summary = compound_summary.merge(
            tox_summary, on=["compound_id", "base_compound_name"], how="left"
        )

    if not lit_long.empty:
        lit_summary = lit_long.groupby(
            ["compound_id", "base_compound_name"], as_index=False
        ).agg(n_literature_targets=("target", "nunique"))
        compound_summary = compound_summary.merge(
            lit_summary, on=["compound_id", "base_compound_name"], how="left"
        )

    if not class_df.empty:
        class_cols = [c for c in ["compounds_super_class", "compounds_class"] if c in class_df.columns]
        compound_summary = compound_summary.merge(
            class_df[["compound_id", "base_compound_name"] + class_cols],
            on=["compound_id", "base_compound_name"],
            how="left"
        )

    compound_summary.to_csv(OUT / "mea_compound_summary.csv", index=False)

    # Mapping table for CompTox transfer learning. Preserve any mappings already
    # curated by the user instead of overwriting them when this script is rerun.
    mapping_path = OUT / "mea_to_comptox_mapping_template.csv"
    mapping = compound_summary[["compound_id", "base_compound_name"]].drop_duplicates().copy()
    mapping["DTXSID"] = ""
    mapping["SMILES"] = ""

    if mapping_path.exists():
        try:
            existing = pd.read_csv(mapping_path, dtype=str).fillna("")
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame()
        keep_cols = [c for c in ["compound_id", "DTXSID", "SMILES"] if c in existing.columns]
        if {"compound_id", "DTXSID", "SMILES"}.issubset(keep_cols):
            existing = existing[keep_cols].drop_duplicates("compound_id", keep="last")
            mapping = mapping.drop(columns=["DTXSID", "SMILES"]).merge(
                existing, on="compound_id", how="left"
            )
            mapping[["DTXSID", "SMILES"]] = mapping[["DTXSID", "SMILES"]].fillna("")

    mapping.to_csv(mapping_path, index=False)


def build_summary_report():
    """Write MEA_SUMMARY.md: a human-readable recap of sheet sizes and parsed row counts."""
    xls = pd.ExcelFile(RAW)
    lines = []
    lines.append("# MEA workbook summary")
    lines.append("")
    lines.append(f"Workbook: `{RAW.name}`")
    lines.append("")
    lines.append("## Sheets")
    for sheet in xls.sheet_names:
        df = pd.read_excel(RAW, sheet_name=sheet)
        lines.append(f"- **{sheet}**: {df.shape[0]} rows × {df.shape[1]} columns")

    compound_summary = pd.read_csv(OUT / "mea_compound_summary.csv")
    lines.append("")
    lines.append("## Parsed summary")
    lines.append(f"- Unique compound identifiers: {compound_summary['compound_id'].nunique()}")
    lines.append(f"- Unique base compound names: {compound_summary['base_compound_name'].nunique()}")

    lit = pd.read_csv(OUT / "mea_literature_targets_long.csv")
    tox = pd.read_csv(OUT / "mea_toxprofiler_positive_predictions.csv")
    hit = pd.read_csv(OUT / "mea_hit_long.csv")
    z = pd.read_csv(OUT / "mea_zscore_long.csv")

    lines.append(f"- Literature target edges: {len(lit)}")
    lines.append(f"- Positive ToxProfiler prediction rows (score ≥ 1.0): {len(tox)}")
    lines.append(f"- Non-zero MEA hit rows: {int(hit['hit_code'].fillna(0).gt(0).sum())}")
    lines.append(f"- Z-score long rows: {len(z)}")

    with open(OUT / "MEA_SUMMARY.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    """CLI entry point: parse every sheet of the MEA workbook, build the
    cross-sheet summaries, and write the human-readable report."""
    OUT.mkdir(parents=True, exist_ok=True)

    logac50_wide, logac50_long = prepare_logac50()
    hit_wide, hit_long = prepare_hit()
    lit_wide, lit_long = prepare_literature_targets()
    class_df = prepare_molecule_class()
    tox_wide, tox_long = prepare_toxprofiler()
    z_wide, z_long = prepare_zscore()

    build_summary_tables(logac50_long, hit_long, lit_long, class_df, tox_long)
    build_summary_report()

    print("Prepared MEA tables in:", OUT)
    print("Key outputs:")
    for name in [
        "mea_compound_summary.csv",
        "mea_metric_activity_summary.csv",
        "mea_literature_targets_long.csv",
        "mea_toxprofiler_long.csv",
        "mea_zscore_long.csv",
        "mea_to_comptox_mapping_template.csv",
        "MEA_SUMMARY.md",
    ]:
        print(" -", name)


if __name__ == "__main__":
    main()
