"""Step 10c - Would more physchem descriptors help? (baselines only)

The foundation model sees just 16 physchem descriptors, while RDKit can compute about 200. Before deciding
whether to widen the model's physchem input (which means recomputing the cohort and retraining step 03),
test the cheap question first: do ordinary models improve when given the full RDKit descriptor list?

Setup is identical to steps 10 and 10b: same labels, fixed scaffold split, nested training subsets, seeds and
metrics. Feature sets:
  physchem16         the 16 descriptors the foundation model uses (reference)
  rdkit_all          every RDKit descriptor (~200), computed from the SMILES
  rdkit_all+morgan   the full descriptor list plus a 2048-bit Morgan fingerprint

How to read it: if the random forest on `rdkit_all` is clearly better than on `physchem16`, descriptors are a
strong lever for this endpoint (the baselines improve too, so the foundation model would have to improve
*more* to justify retraining). If little changes, a wider input will not rescue the model on this task.

Input : data/processed/comptox_v3.parquet, data/mea_processed/*.csv
Output: results/extended_descriptors_mea_metrics.csv
Run   : python scripts/10c_extended_descriptors.py --seeds 5
"""

# %%
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# %% [markdown]
# ## 1. Paths and settings

# %%
PROJECT_ROOT = Path(__file__).resolve().parents[1]
COHORT = PROJECT_ROOT / "data" / "processed" / "comptox_v3.parquet"
MEA_PROCESSED = PROJECT_ROOT / "data" / "mea_processed"
RESULTS = PROJECT_ROOT / "results"

FRACTIONS = [0.10, 0.25, 0.50, 1.00]
FEATURE_SETS = ["physchem16", "rdkit_all", "rdkit_all+morgan"]
MODELS = ["ridge", "random_forest"]
PHYSCHEM16 = ["mw", "logp", "tpsa", "hbd", "hba", "rotatable_bonds", "ring_count", "aromatic_ring_count",
              "aliphatic_ring_count", "saturated_ring_count", "heteroatom_count", "fraction_csp3",
              "heavy_atom_count", "formal_charge", "radical_electron_count", "valence_electron_count"]

# %% [markdown]
# ## 2. The MEA task

# %% [markdown]
# ### The MEA task: labels, a fixed scaffold split, nested training subsets
# * **Label** (`target`). Three choices, all built from the MEA tables of step 06 and joined to the cohort
#   through the compound-to-DTXSID mapping:
#   * `n_hits` (default): the number of the 19 MEA network metrics on which a compound has an effect
#     (increase or decrease);
#   * `min_hit_logac50`: log10 AC50 (uM) of the compound's *most potent* effect, the smallest logAC50 over its
#     quality-checked hits (lower = more potent);
#   * `median_hit_logac50`: the median logAC50 over its quality-checked hits.
#   **Potency labels use only reliable AC50 values.** A logAC50 is meaningful only for a real hit: the ~2,400
#   no-hit entries hold placeholders (about 60% are exactly 1.176 = 15 uM, 10% exactly -2.0 = 0.01 uM). So an entry
#   counts only if (1) its hit code is exactly 1 or 2, (2) a logAC50 is present, (3) it is not a placeholder value
#   (these two values also occur among hits), and (4) it lies inside the tested range of 0.1-30 uM. Compounds
#   with no qualifying entry get no potency label and are left out of the task. The hit count `n_hits` is unchanged.
# * **Split.** One fixed, deterministic *scaffold* split. Compounds sharing a Bemis-Murcko scaffold
#   (their core ring system) always land on the same side, so test compounds never have a
#   training compound with the same scaffold. The largest scaffold groups go to training first; the
#   remaining rarer scaffolds form the test set (about 25% of compounds).
# * **Nested subsets.** To study low-data behaviour, a seed shuffles the training pool and takes the
#   first 10%, 25%, 50% or 100% of it, so smaller subsets are contained in larger ones.

# %%
# Potency quality rules (MEA concentrations were 0.1-30 uM)
TESTED_RANGE_LOG10 = (-1.0, float(np.log10(30.0)))   # a logAC50 outside this range was not actually measured
PLACEHOLDER_LOGAC50 = (-2.0, 1.176)                   # values the workbook uses to fill in "no real AC50"


def quality_hits(metrics):
    """Metric entries usable as AC50 (potency) labels, plus the number left after each check.

    1. hit code exactly 1 (increase) or 2 (decrease): no-hit entries (0) and odd fractional codes are excluded;
    2. a logAC50 is present;
    3. it is not one of the placeholder values (exactly -2.0 = 0.01 uM or 1.176 = 15 uM);
    4. it lies inside the tested concentration range, 0.1-30 uM (log10 -1 to 1.477)."""
    steps = [("all metric entries", len(metrics))]
    kept = metrics[metrics["hit_code"].isin([1, 2])]
    steps.append(("hit code exactly 1 or 2", len(kept)))
    kept = kept[np.isfinite(kept["logac50"].astype(float))]
    steps.append(("logAC50 present", len(kept)))
    kept = kept[~kept["logac50"].round(3).isin([round(v, 3) for v in PLACEHOLDER_LOGAC50])]
    steps.append(("not a placeholder value", len(kept)))
    low, high = TESTED_RANGE_LOG10
    kept = kept[(kept["logac50"] >= low - 1e-9) & (kept["logac50"] <= high + 1e-9)]
    steps.append(("inside the tested range 0.1-30 uM", len(kept)))
    return kept, steps


MEA_TARGETS = ["n_hits", "min_hit_logac50", "median_hit_logac50"]


def mea_compound_labels(target="n_hits"):
    """One row per MEA compound with its label: columns compound_id, mea_label."""
    if target not in MEA_TARGETS:
        raise ValueError(f"target must be one of {MEA_TARGETS}")
    metrics = pd.read_csv(MEA_PROCESSED / "mea_metric_activity_summary.csv")
    if target == "n_hits":
        label = metrics[metrics["hit_code"].fillna(0) > 0].groupby("compound_id").size()
    else:
        potency = quality_hits(metrics)[0].groupby("compound_id")["logac50"]
        label = potency.min() if target == "min_hit_logac50" else potency.median()
    return label.rename("mea_label").reset_index()


def load_mea_labels(target="n_hits"):
    """One row per DTXSID with its MEA label (averaged if several MEA ids map to one DTXSID)."""
    if target != "n_hits":
        steps = quality_hits(pd.read_csv(MEA_PROCESSED / "mea_metric_activity_summary.csv"))[1]
        print("AC50 quality checks (metric entries left): " + " -> ".join(f"{name} {n}" for name, n in steps))
    mapping = pd.read_csv(MEA_PROCESSED / "mea_to_comptox_mapping_template.csv")[["compound_id", "DTXSID"]]
    labels = mea_compound_labels(target).merge(mapping, on="compound_id").dropna(subset=["DTXSID"])
    labels = labels[labels["DTXSID"].astype(str).str.strip() != ""]
    return labels.groupby("DTXSID", as_index=False)["mea_label"].mean()


def scaffold_of(smiles):
    """Bemis-Murcko scaffold SMILES ('' for acyclic molecules, None if unparseable)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(str(smiles))
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol is not None else ""


def scaffold_holdout(scaffolds, test_fraction=0.25, split_seed=0):
    """Deterministic scaffold split -> (train_pool_indices, test_indices)."""
    groups = {}
    for i, scaffold in enumerate(scaffolds):
        groups.setdefault(scaffold, []).append(i)
    keys = sorted(groups)
    np.random.RandomState(split_seed).shuffle(keys)      # split_seed only breaks ties between equal-sized groups
    keys.sort(key=lambda k: -len(groups[k]))              # stable sort: biggest scaffold groups first
    n_train = len(scaffolds) - int(round(test_fraction * len(scaffolds)))
    train, test = [], []
    for key in keys:
        (train if len(train) + len(groups[key]) <= n_train else test).extend(groups[key])
    return np.array(sorted(train)), np.array(sorted(test))


def prepare_mea_task(cohort_path, test_fraction=0.25, split_seed=0, target="n_hits"):
    """MEA-labelled cohort rows, their labels, and the fixed (train_pool, test) index split."""
    labels = load_mea_labels(target)
    cohort = pd.read_parquet(cohort_path)
    frame = cohort.merge(labels, on="DTXSID", how="inner").drop_duplicates("DTXSID").reset_index(drop=True)
    y = frame["mea_label"].to_numpy(float)
    scaffolds = [scaffold_of(s) for s in frame["smiles"]]
    train_pool, test_idx = scaffold_holdout(scaffolds, test_fraction, split_seed)
    print(f"target={target}: {len(frame)} MEA compounds with labels (of {len(labels)} mapped) | "
          f"fixed split: {len(train_pool)} train pool / {len(test_idx)} test")
    return frame, y, train_pool, test_idx


def training_subset(train_pool, seed, fraction):
    """Nested random subset of the training pool: the first `fraction` of a seed-specific shuffle."""
    order = np.random.RandomState(1000 + seed).permutation(train_pool)
    return np.sort(order[:max(8, int(round(fraction * len(order))))])


def regression_metrics(y_true, y_pred):
    """RMSE, Spearman rank correlation and R^2."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = spearmanr(y_true, y_pred).statistic if np.std(y_pred) > 0 else np.nan
    return {"rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))), "spearman": float(rho),
            "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan}

# %% [markdown]
# ## 3. Models

# %% [markdown]
# ### Baseline models
# Two standard regressors, both with median imputation for missing values:
# * **ridge**: linear, standardised inputs clipped at +-5 standard deviations (a few extreme values
#   would otherwise let it extrapolate wildly), penalty chosen by leave-one-out CV on the *training
#   subset only*;
# * **random forest**: 300 trees, minimum leaf size 2, one third of the features per split.
# Neither is tuned on the test set. Two more are available: **gbm** (histogram gradient boosting) and **svr** (support-vector regression with an RBF kernel on standardised, clipped inputs and a
# standardised target).

# %%
def make_model(name, seed):
    if name == "gbm":
        from sklearn.ensemble import HistGradientBoostingRegressor
        # median imputation also drops columns that are entirely missing in the training rows, which would break the binning
        return make_pipeline(SimpleImputer(strategy="median"),
                             HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3,
                                                           min_samples_leaf=5, random_state=seed))
    if name == "svr":
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.svm import SVR
        inner = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              FunctionTransformer(np.clip, kw_args={"a_min": -5.0, "a_max": 5.0}), SVR(C=1.0))
        return TransformedTargetRegressor(regressor=inner, transformer=StandardScaler())
    if name == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             FunctionTransformer(np.clip, kw_args={"a_min": -5.0, "a_max": 5.0}),
                             RidgeCV(alphas=np.logspace(-1, 5, 25)))
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestRegressor(n_estimators=300, min_samples_leaf=2, max_features=0.33,
                                               n_jobs=4, random_state=seed))


def morgan_matrix(smiles, radius=2, n_bits=2048):
    """ECFP4-style Morgan fingerprint bits (unparseable SMILES give all zeros)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(s))
        if mol is not None:
            out[i] = generator.GetFingerprintAsNumPy(mol)
    return out

# %% [markdown]
# ## 4. Feature sets
# RDKit's descriptor list is computed for each SMILES (failures and infinite values become missing). Columns
# that are entirely missing or constant are dropped.

# %%
def rdkit_descriptor_matrix(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    RDLogger.DisableLog("rdApp.*")
    names = [name for name, _ in Descriptors._descList]
    out = np.full((len(smiles), len(names)), np.nan)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(s))
        if mol is None:
            continue
        for j, (_, function) in enumerate(Descriptors._descList):
            try:
                out[i, j] = function(mol)
            except Exception:
                pass
    out[~np.isfinite(out)] = np.nan
    return out, names


def build_feature_sets(frame):
    physchem = frame[[c for c in PHYSCHEM16 if c in frame]].to_numpy(float)
    full, names = rdkit_descriptor_matrix(frame["smiles"].tolist())
    keep = np.isfinite(full).any(axis=0) & (np.nanstd(full, axis=0) > 0)
    full, names = full[:, keep], [n for n, k in zip(names, keep) if k]
    fingerprint = morgan_matrix(frame["smiles"].tolist())
    return {"physchem16": physchem, "rdkit_all": full, "rdkit_all+morgan": np.hstack([full, fingerprint])}, names


def top_importances(X, y, names, train_idx, n=10):
    """Which descriptors drive a random-forest fit on the full training pool."""
    forest = make_model("random_forest", 0).fit(X[train_idx], y[train_idx])
    importances = forest[-1].feature_importances_
    order = np.argsort(importances)[::-1][:n]
    return [(names[i], float(importances[i])) for i in order]

# %% [markdown]
# ## 5. Run
# Same loop as steps 10 and 10b. Results are saved after every fit (`resume=True` continues an interrupted run).
# At the end the top random-forest descriptors are printed, showing what the models rely on.

# %%
def run_extended(seeds=5, fractions=FRACTIONS, features=FEATURE_SETS, models=MODELS, test_fraction=0.25,
                 split_seed=0, output=None, resume=False, target="n_hits"):
    warnings.filterwarnings("ignore", message="(?s).*At least one non-missing value.*")
    frame, y, train_pool, test_idx = prepare_mea_task(COHORT, test_fraction, split_seed, target)
    matrices, names = build_feature_sets(frame)
    print("Feature sets:", {name: matrices[name].shape[1] for name in features})

    suffix = "" if target == "n_hits" else f"_{target}"
    output = Path(output) if output else RESULTS / f"extended_descriptors_mea_metrics{suffix}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows, done = [], set()
    if resume and output.exists():
        rows = pd.read_csv(output).to_dict("records")
        done = {(int(r["seed"]), r["model"], r["features"], float(r["fraction"])) for r in rows}
        print(f"Resuming: {len(done)} finished runs found in {output.name}")

    for seed in range(seeds):
        if (seed, "mean_baseline", "none", 1.0) not in done:
            baseline = np.full(len(test_idx), y[train_pool].mean())
            rows.append({"seed": seed, "model": "mean_baseline", "features": "none", "fraction": 1.0,
                         "n_train": len(train_pool), "n_test": len(test_idx), **regression_metrics(y[test_idx], baseline)})
        for fraction in fractions:
            train_idx = training_subset(train_pool, seed, fraction)
            for feature_name in features:
                X = matrices[feature_name]
                for model_name in models:
                    if (seed, model_name, feature_name, float(fraction)) in done:
                        continue
                    model = make_model(model_name, seed).fit(X[train_idx], y[train_idx])
                    metrics = regression_metrics(y[test_idx], model.predict(X[test_idx]))
                    rows.append({"seed": seed, "model": model_name, "features": feature_name, "fraction": fraction,
                                 "n_train": len(train_idx), "n_test": len(test_idx), **metrics})
                    print(f"seed={seed} fraction={fraction:>4.0%} {model_name:<13} {feature_name:<17} "
                          f"rmse={metrics['rmse']:.3f} spearman={metrics['spearman']:.3f}", flush=True)
                    pd.DataFrame(rows).to_csv(output, index=False)
    result = pd.DataFrame(rows)
    result.to_csv(output, index=False)
    table = result[result["model"] != "mean_baseline"].groupby(["model", "features", "fraction"])[["rmse", "spearman"]].mean()
    print("\nRMSE by feature set (mean over seeds, lower is better):")
    print(table["rmse"].unstack("fraction").round(3).to_string())
    print("\nSpearman by feature set:")
    print(table["spearman"].unstack("fraction").round(3).to_string())
    print("\nTop random-forest descriptors (full training pool, rdkit_all):")
    for name, value in top_importances(matrices["rdkit_all"], y, names, train_pool):
        print(f"  {name:<28} {value:.3f}")
    print("Saved:", output)
    return result

# %% Command line
def main():
    parser = argparse.ArgumentParser(description="Baselines with the full RDKit descriptor list.")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--features", nargs="+", choices=FEATURE_SETS, default=FEATURE_SETS)
    parser.add_argument("--fractions", nargs="+", type=float, default=FRACTIONS)
    parser.add_argument("--target", choices=MEA_TARGETS, default="n_hits", help="what to predict: hit count or AC50-based potency")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--split-seed", type=int, default=0, help="must match step 10's --split-seed")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true", help="skip runs already saved in --output")
    args = parser.parse_args()
    run_extended(args.seeds, args.fractions, args.features, args.models, args.test_fraction,
                 args.split_seed, args.output, args.resume, args.target)


if __name__ == "__main__":
    main()
