"""Step 9 - Register measured toxicity endpoints

Collects *measured* toxicity / injury results (hERG, DILI, hepatotoxicity, ...) into one registry file,
`data/endpoints/endpoint_results.csv`. Steps 07 and 08 read this registry and show its rows as toxicity
endpoints, and it is the natural source of labels for further transfer-learning tasks.

Only real experimental results belong here. *Predicted* targets, such as ToxProfiler scores, are kept in a
separate part of the project: a predicted KCNH2 hit must never be mistaken for a measured hERG result.

An input CSV needs an `endpoint` and a `result_type` column and at least one identifier column
(`DTXSID`, `compound_id` or `chemical_name`). Optional columns: endpoint_group, task_type, value, label,
probability, unit, model, source, notes. Example row:
    DTXSID,endpoint,result_type,task_type,label,source
    DTXSID8022292,hERG,measured,classification,0,my_lab_2024

Input : any CSV of endpoint results (--input)
Output: data/endpoints/endpoint_results.csv   (rows are appended; exact duplicates are dropped)
Run   : python scripts/09_register_toxicity_endpoints.py --check
        python scripts/09_register_toxicity_endpoints.py --input path/to/my_endpoints.csv
"""

# %%
import argparse
from pathlib import Path

import pandas as pd

# %% [markdown]
# ## 1. Paths and required columns

# %%
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = PROJECT_ROOT / "data" / "endpoints" / "endpoint_results.csv"

REQUIRED = {"endpoint", "result_type"}
IDENTIFIERS = {"chemical_name", "compound_id", "DTXSID"}
COLUMNS = ["chemical_name", "compound_id", "DTXSID", "endpoint", "endpoint_group", "result_type", "task_type",
           "value", "label", "probability", "unit", "model", "source", "notes"]

# %% [markdown]
# ## 2. Register a file
# The input is validated (required columns present, at least one identifier), padded to the registry's
# column layout, appended to the existing registry and deduplicated.

# %%
def registry_size():
    return len(pd.read_csv(REGISTRY)) if REGISTRY.exists() else 0


def register(input_path):
    new = pd.read_csv(input_path)
    missing = REQUIRED - set(new.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not IDENTIFIERS & set(new.columns):
        raise ValueError("Provide at least one identifier column: chemical_name, compound_id or DTXSID")
    for column in COLUMNS:
        if column not in new:
            new[column] = ""
    new = new[COLUMNS]
    current = pd.read_csv(REGISTRY) if REGISTRY.exists() else pd.DataFrame(columns=COLUMNS)
    merged = new.copy() if current.empty else pd.concat([current, new], ignore_index=True)
    merged = merged.drop_duplicates()
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(REGISTRY, index=False)
    print("Registered input rows:", len(new))
    print("Rows in registry:", len(merged))
    print("Saved:", REGISTRY)

# %% Command line
def main():
    parser = argparse.ArgumentParser(description="Register measured toxicity endpoint results.")
    parser.add_argument("--input", type=Path, help="CSV of endpoint results to add")
    parser.add_argument("--check", action="store_true", help="only report how many rows are registered")
    args = parser.parse_args()
    if args.check:
        print("Registry:", REGISTRY)
        print("Rows:", registry_size())
        return
    if not args.input:
        parser.error("--input is required unless --check is used")
    register(args.input)


if __name__ == "__main__":
    main()
