"""Step 10b - Baseline models for the MEA task

Does the pretrained embedding beat ordinary models fitted on raw features? Any transfer claim has to clear
that bar, so this step fits ridge regression and random forests on simple feature sets, using exactly the
same labels (`--target`), fixed scaffold split, nested training subsets, seeds and metrics as step 10.

Feature sets (all from the cohort table, or computed from the SMILES):
  physchem          the 16 RDKit descriptors the foundation model receives
  morgan            2048-bit Morgan (ECFP4) fingerprint
  morgan_physchem   fingerprint + physchem
  toxcast_summary   physchem + ToxCast/Tox21 summaries + hazard + exposure
  all_raw           physchem + every hit-call / AC50 / efficacy column + hazard + exposure
  emb_sp, emb_all   frozen foundation-model embedding (SMILES+physchem visible / all modalities visible)
  emb_sp+physchem, emb_all+physchem   the embedding concatenated with the raw physchem columns

Two diagnostics on the embedding:
  * models on `emb_*` vs `physchem`: does the embedding hold the information a simple model can use?
  * `--probe`: how well can each physchem descriptor be recovered from the embedding?

Input : data/processed/comptox_v3.parquet, data/mea_processed/*.csv, checkpoints/comptox_v3_best.pt
Output: results/baseline_mea_metrics.csv, results/embedding_probe.csv
Run   : python scripts/10b_baseline_models.py --seeds 5            # baselines
        python scripts/10b_baseline_models.py --probe             # descriptor-recovery check
"""

# %%
import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from torch.utils.data import Dataset

# %% [markdown]
# ## 1. Paths and settings

# %%
PROJECT_ROOT = Path(__file__).resolve().parents[1]
COHORT = PROJECT_ROOT / "data" / "processed" / "comptox_v3.parquet"
MEA_PROCESSED = PROJECT_ROOT / "data" / "mea_processed"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "comptox_v3_best.pt"
CONFIG = PROJECT_ROOT / "config.yaml"
RESULTS = PROJECT_ROOT / "results"

FRACTIONS = [0.10, 0.25, 0.50, 1.00]
FEATURE_SETS = ["physchem", "morgan", "morgan_physchem", "toxcast_summary", "all_raw",
                "emb_sp", "emb_all", "emb_sp+physchem", "emb_all+physchem"]
MODELS = ["ridge", "random_forest"]
INPUT_SETS = {"smiles_physchem": [True, False, False, False, False, False], "all": [True] * 6}

# %% [markdown]
# ## 2. Model code (only needed to compute the frozen embeddings)

# %% [markdown]
# ### SMILES tokenizer
# A deliberately simple character-level tokenizer. Every character of a SMILES string becomes one
# token. The vocabulary is fit on the **training** compounds only, and four special tokens are
# reserved: `[PAD]` (padding), `[UNK]` (unseen character), `[MASK]` (masked-language-model target)
# and `[CLS]` (start token whose output becomes the SMILES embedding).

# %%
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[MASK]", "[CLS]"]


class SmilesTokenizer:
    def __init__(self, vocab=None):
        self.vocab = vocab or {token: i for i, token in enumerate(SPECIAL_TOKENS)}

    @property
    def pad_id(self):
        return self.vocab["[PAD]"]

    @property
    def unk_id(self):
        return self.vocab["[UNK]"]

    @property
    def mask_id(self):
        return self.vocab["[MASK]"]

    @property
    def cls_id(self):
        return self.vocab["[CLS]"]

    def fit(self, smiles, max_vocab=512):
        """Add the most frequent characters in `smiles` to the vocabulary."""
        counts = Counter(ch for s in smiles for ch in str(s))
        for ch, _ in counts.most_common(max_vocab - len(self.vocab)):
            if ch not in self.vocab:
                self.vocab[ch] = len(self.vocab)
        return self

    def encode(self, smiles, max_length):
        """[CLS] + one id per character, truncated or right-padded to `max_length`."""
        ids = [self.cls_id] + [self.vocab.get(ch, self.unk_id) for ch in str(smiles)]
        ids = ids[:max_length]
        return ids + [self.pad_id] * (max_length - len(ids))

    def save(self, path):
        Path(path).write_text(json.dumps(self.vocab, indent=2), encoding="utf-8")

# %% [markdown]
# ### Dataset: values plus "was it measured?" masks
# Each compound is described by SMILES plus six numeric **modalities**, in this fixed order:
# physchem, hit call, AC50, efficacy, hazard, exposure. Most compounds lack most measurements, so
# every modality is stored as a `(values, mask)` pair, where `mask = 1` means measured and `0`
# means missing. The model and the losses use the mask, so a missing value is never mistaken for a
# real zero. Values are z-scored with statistics fitted on the training split only (hit calls stay
# on their 0-1 scale).

# %%
PROPERTY = ["mw", "logp", "tpsa", "hbd", "hba", "rotatable_bonds",
            "ring_count", "aromatic_ring_count", "aliphatic_ring_count",
            "saturated_ring_count", "heteroatom_count", "fraction_csp3",
            "heavy_atom_count", "formal_charge", "radical_electron_count",
            "valence_electron_count"]
HAZARD = ["hazard_count", "noael_log", "loael_log"]
EXPOSURE = ["exposure_count", "product_count", "use_category_count"]
MODALITY_NAMES = ["physchem", "hitcall", "ac50", "efficacy", "hazard", "exposure"]


def value_mask(row, columns, scaler=None):
    """Values and observed-mask for one modality of one compound (missing -> value 0, mask 0)."""
    values = np.array([row.get(c, np.nan) for c in columns], dtype=np.float32)
    mask = np.isfinite(values).astype(np.float32)
    if scaler is not None and len(columns):
        values = (values - np.asarray(scaler["mean"], dtype=np.float32)) / np.asarray(scaler["std"], dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.tensor(values), torch.tensor(mask)


def fit_group_scalers(df, groups):
    """Per-column mean/std of each modality, ignoring missing values. Hit calls (group 1) are not scaled."""
    scalers = []
    for i, columns in enumerate(groups):
        if i == 1 or not columns:
            scalers.append(None)
            continue
        table = df[columns].to_numpy(dtype=np.float64)
        means, stds = [], []
        for j in range(table.shape[1]):
            finite = table[:, j][np.isfinite(table[:, j])]
            if finite.size == 0:
                means.append(0.0)
                stds.append(1.0)
            else:
                std = float(finite.std())
                means.append(float(finite.mean()))
                stds.append(std if std > 1e-8 else 1.0)
        scalers.append({"mean": means, "std": stds})
    return scalers


class CompToxDataset(Dataset):
    """One item per compound: (token ids, [6 value tensors], [6 mask tensors]).

    Pass `groups` and `scalers` from a checkpoint to reproduce its exact training-time feature layout."""

    def __init__(self, df, tokenizer, max_length, scalers=None, groups=None):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.groups = [
            [c for c in PROPERTY if c in df],
            [c for c in df if c.startswith("biohit__")],
            [c for c in df if c.startswith("bioac50__")],
            [c for c in df if c.startswith("bioeff__")],
            [c for c in HAZARD if c in df],
            [c for c in EXPOSURE if c in df],
        ]
        if groups is not None:
            self.groups = [list(g) for g in groups]
            for column in [c for g in groups for c in g]:
                if column not in self.df:
                    self.df[column] = np.nan
        if not self.groups[1]:
            raise ValueError("No ToxCast/Tox21 hit-call columns (biohit__*) found.")
        self.scalers = scalers if scalers is not None else [None] * 6

    @property
    def hit_columns(self):
        return self.groups[1]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        ids = torch.tensor(self.tokenizer.encode(row["smiles"], self.max_length), dtype=torch.long)
        values, masks = [], []
        for columns, scaler in zip(self.groups, self.scalers):
            v, m = value_mask(row, columns, scaler)
            values.append(v)
            masks.append(m)
        return ids, values, masks

# %% [markdown]
# ### The model
# ```
# SMILES tokens -> Transformer encoder -> [CLS] vector ---\
# physchem   (values, mask) -> NumericEncoder ------------\
# hit call   (values, mask) -> NumericEncoder -------------> concatenate -> fusion MLP -> shared embedding
# AC50       (values, mask) -> NumericEncoder ------------/
# efficacy   (values, mask) -> NumericEncoder -----------/
# hazard     (values, mask) -> NumericEncoder ----------/
# exposure   (values, mask) -> NumericEncoder ---------/
# ```
# The shared embedding is trained to reconstruct masked SMILES characters (a language-model head)
# and masked numeric values (one linear head per modality). The embedding is what later steps use.

# %%
class NumericEncoder(nn.Module):
    """Encodes one modality's (values, mask) pair into a `latent_dim` vector.

    The mask is an input, not just a filter, so the network can tell "measured as zero" from "missing"."""

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = None if input_dim == 0 else nn.Sequential(
            nn.Linear(input_dim * 2, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, values, mask):
        if self.net is None:  # a modality with no columns contributes a zero vector
            return torch.zeros(values.shape[0], self.latent_dim, device=values.device, dtype=values.dtype)
        return self.net(torch.cat([values, mask], dim=-1))


class CompToxFoundationModel(nn.Module):
    def __init__(self, vocab_size, modality_dims, d_model=256, n_heads=8, n_layers=4,
                 feedforward_dim=768, latent_dim=256, dropout=0.1, pad_id=0, max_positions=512):
        super().__init__()
        if len(modality_dims) != 6:
            raise ValueError("modality_dims needs 6 entries: physchem, hitcall, ac50, efficacy, hazard, exposure.")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.pad_id = pad_id
        self.max_positions = max_positions
        self.token = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position = nn.Embedding(max_positions, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=feedforward_dim,
                                           dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.smiles_projection = nn.Linear(d_model, latent_dim)
        self.numeric_encoders = nn.ModuleList([NumericEncoder(n, latent_dim) for n in modality_dims])
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 7, latent_dim * 2), nn.GELU(), nn.LayerNorm(latent_dim * 2),
            nn.Dropout(dropout), nn.Linear(latent_dim * 2, latent_dim),
        )
        self.mlm_head = nn.Linear(d_model, vocab_size)
        self.numeric_heads = nn.ModuleList([nn.Linear(latent_dim, n) if n else nn.Identity() for n in modality_dims])

    def forward(self, input_ids, values, masks):
        """Returns the shared embedding, per-position SMILES logits and one prediction per numeric modality."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence].")
        if len(values) != 6 or len(masks) != 6:
            raise ValueError("values and masks must each contain six modalities.")
        batch, length = input_ids.shape
        if length > self.max_positions:
            raise ValueError(f"Sequence length {length} exceeds max_positions={self.max_positions}.")
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0).expand(batch, length)
        hidden = self.token(input_ids) + self.position(positions)
        hidden = self.transformer(hidden, src_key_padding_mask=input_ids.eq(self.pad_id))
        smiles_z = self.smiles_projection(hidden[:, 0])  # the [CLS] position
        numeric_z = [enc(x, m) for enc, x, m in zip(self.numeric_encoders, values, masks)]
        shared = self.fusion(torch.cat([smiles_z] + numeric_z, dim=-1))
        return {
            "embedding": shared,
            "mlm_logits": self.mlm_head(hidden),
            "numeric_predictions": [head(shared) for head in self.numeric_heads],
        }

# %% [markdown]
# ### Loading a trained checkpoint
# A checkpoint stores everything needed to reproduce training-time inference: the weights, the
# tokenizer vocabulary, the feature columns of each modality, the scalers and the config. Loading
# it through these helpers guarantees that new data is encoded exactly as the model saw it.

# %%
def build_model(checkpoint, tokenizer, device="cpu"):
    """An untrained model with the checkpoint's architecture (used for from-scratch baselines)."""
    cfg = checkpoint["config"]
    m = cfg["model"]
    return CompToxFoundationModel(
        len(tokenizer.vocab), checkpoint["modality_dims"],
        d_model=m["d_model"], n_heads=m["n_heads"], n_layers=m["layers"],
        feedforward_dim=m["feedforward_dim"], latent_dim=m["latent_dim"], dropout=m["dropout"],
        pad_id=tokenizer.pad_id, max_positions=max(512, cfg["data"]["max_smiles_length"]),
    ).to(device)


def load_checkpoint(path, device="cpu"):
    """Returns (model, tokenizer, checkpoint dict) with the trained weights loaded."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not {"groups", "vocab", "config", "scalers"}.issubset(checkpoint):
        raise ValueError("This checkpoint has no feature schema/tokenizer; retrain with step 03.")
    tokenizer = SmilesTokenizer(checkpoint["vocab"])
    model = build_model(checkpoint, tokenizer, device)
    model.load_state_dict(checkpoint["model_state"])
    return model, tokenizer, checkpoint


def dataset_for_checkpoint(frame, tokenizer, checkpoint):
    """Wrap a compound table in the dataset layout the checkpoint was trained with."""
    return CompToxDataset(frame, tokenizer, checkpoint["config"]["data"]["max_smiles_length"],
                          scalers=checkpoint["scalers"], groups=checkpoint["groups"])

# %% [markdown]
# ### Encoding compounds for the frozen model
# `tensors_for` turns a table of compounds into model inputs in one go. Modalities not listed in `keep`
# are zeroed and marked missing, which is how the model is told "this information is not available".
# `embed` runs the model on a subset of rows and returns their 256-d embeddings.

# %%
def tensors_for(frame, tokenizer, checkpoint, keep):
    dataset = dataset_for_checkpoint(frame, tokenizer, checkpoint)
    items = [dataset[i] for i in range(len(dataset))]
    ids = torch.stack([item[0] for item in items])
    values = [torch.stack([item[1][j] for item in items]) for j in range(6)]
    masks = [torch.stack([item[2][j] for item in items]) for j in range(6)]
    for j in range(6):
        if not keep[j]:
            values[j] = torch.zeros_like(values[j])
            masks[j] = torch.zeros_like(masks[j])
    return ids, values, masks


def embed(model, data, idx):
    ids, values, masks = data
    return model(ids[idx], [v[idx] for v in values], [m[idx] for m in masks])["embedding"]

# %% [markdown]
# ## 3. The MEA task

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
# ## 4. Models

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
# ## 5. Feature sets
# `toxcast_summary` compresses the ~1,500 bioactivity columns into five numbers per compound: how many
# assays were measured, how many were active (hit call >= 0.9), the active fraction, and the median log10
# AC50 and efficacy. The `emb_*` sets are the frozen foundation-model embeddings, computed once per
# compound with either only SMILES+physchem visible or all modalities visible (the same masking as step 10).

# %%
def toxcast_summary(frame, threshold):
    hit = frame[[c for c in frame if c.startswith("biohit__")]].to_numpy(float)
    ac50 = frame[[c for c in frame if c.startswith("bioac50__")]].to_numpy(float)
    efficacy = frame[[c for c in frame if c.startswith("bioeff__")]].to_numpy(float)
    measured = np.isfinite(hit).sum(axis=1)
    active = np.nansum(np.where(np.isfinite(hit), hit >= threshold, 0), axis=1)
    with np.errstate(all="ignore"):
        fraction = np.where(measured > 0, active / np.maximum(measured, 1), np.nan)

    def row_median(a):
        return np.array([np.nanmedian(r) if np.isfinite(r).any() else np.nan for r in a])

    return np.column_stack([measured, active, fraction, row_median(ac50), row_median(efficacy)])


def foundation_embeddings(frame, checkpoint_path):
    """Frozen embeddings with SMILES+physchem visible (emb_sp) and with all modalities visible (emb_all)."""
    model, tokenizer, checkpoint = load_checkpoint(checkpoint_path)
    model.eval()
    everyone = np.arange(len(frame))
    out = {}
    with torch.no_grad():
        for name, key in (("emb_sp", "smiles_physchem"), ("emb_all", "all")):
            data = tensors_for(frame, tokenizer, checkpoint, INPUT_SETS[key])
            out[name] = embed(model, data, everyone).numpy()
    return out


def build_features(frame, threshold, embeddings=None):
    physchem = frame[[c for c in PROPERTY if c in frame]].to_numpy(float)
    context = frame[[c for c in HAZARD + EXPOSURE if c in frame]].to_numpy(float)
    bioactivity = frame[[c for c in frame if c.startswith(("biohit__", "bioac50__", "bioeff__"))]].to_numpy(float)
    fingerprint = morgan_matrix(frame["smiles"].tolist())
    features = {
        "physchem": physchem,
        "morgan": fingerprint,
        "morgan_physchem": np.hstack([fingerprint, physchem]),
        "toxcast_summary": np.hstack([physchem, toxcast_summary(frame, threshold), context]),
        "all_raw": np.hstack([physchem, bioactivity, context]),
    }
    for name, z in (embeddings or {}).items():
        features[name] = z
        features[name + "+physchem"] = np.hstack([z, physchem])
    return features

# %% [markdown]
# ## 6. Run the baselines
# For every seed, fraction, feature set and model: fit on the nested training subset, score on the fixed test
# set. Results are saved after every fit, so an interrupted run keeps its work (`resume=True` continues it).
# A "predict the training mean" row is included as the floor any real model must beat.

# %%
def run_baselines(seeds=5, fractions=FRACTIONS, features=FEATURE_SETS, models=MODELS, test_fraction=0.25,
                  split_seed=0, checkpoint_path=CHECKPOINT, output=None, resume=False, target="n_hits"):
    warnings.filterwarnings("ignore", message="(?s).*At least one non-missing value.*")  # empty columns are dropped
    threshold = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["data"]["toxcast_active_threshold"]
    frame, y, train_pool, test_idx = prepare_mea_task(COHORT, test_fraction, split_seed, target)
    embeddings = foundation_embeddings(frame, checkpoint_path) if any(f.startswith("emb_") for f in features) else None
    matrices = build_features(frame, threshold, embeddings)
    print("Feature sets:", {name: matrices[name].shape[1] for name in features})

    suffix = "" if target == "n_hits" else f"_{target}"
    output = Path(output) if output else RESULTS / f"baseline_mea_metrics{suffix}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows, done = [], set()
    if resume and output.exists():
        rows = pd.read_csv(output).to_dict("records")
        done = {(int(r["seed"]), r["model"], r["features"], float(r["fraction"])) for r in rows}
        print(f"Resuming: {len(done)} finished runs found in {output.name}")
    elif output.exists():
        print(f"WARNING: {output.name} exists and will be overwritten (use resume=True to continue it).")

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
                    print(f"seed={seed} fraction={fraction:>4.0%} {model_name:<13} {feature_name:<16} "
                          f"rmse={metrics['rmse']:.3f} spearman={metrics['spearman']:.3f}", flush=True)
                    pd.DataFrame(rows).to_csv(output, index=False)
    result = pd.DataFrame(rows)
    result.to_csv(output, index=False)
    summary = result.groupby(["model", "features", "fraction"])[["rmse", "spearman", "r2"]].agg(["mean", "std"]).round(3)
    print("\nMean / std over seeds (test = fixed scaffold-held-out compounds):")
    print(summary.to_string())
    print("Saved:", output)
    return result

# %% [markdown]
# ## 7. Diagnostic: can the physchem descriptors be recovered from the embedding?
# If the embedding kept the physchem information, a simple model should be able to predict each descriptor
# back from the embedding alone. Here ridge and a random forest are fitted on the checkpoint's *training*
# compounds (a 3,000-compound sample) and scored on its held-out *validation* compounds. R^2 close to 1 means
# the information is retained; near 0 means it is lost. (Formal charge and radical-electron count are almost
# always zero in the cohort, so they carry nothing to recover.)

# %%
def embed_batched(model, data, batch=256):
    ids, values, masks = data
    chunks = []
    with torch.no_grad():
        for start in range(0, len(ids), batch):
            s = slice(start, start + batch)
            chunks.append(model(ids[s], [v[s] for v in values], [m[s] for m in masks])["embedding"].numpy())
    return np.vstack(chunks)


def run_probe(models=MODELS, n_train=3000, checkpoint_path=CHECKPOINT, output=None):
    model, tokenizer, checkpoint = load_checkpoint(checkpoint_path)
    model.eval()
    cohort = pd.read_parquet(COHORT)
    ids = cohort["DTXSID"].astype(str)
    rng = np.random.RandomState(0)
    train_ids = rng.choice(np.array(checkpoint["train_ids"]), size=min(n_train, len(checkpoint["train_ids"])), replace=False)
    train = cohort[ids.isin(set(train_ids))].reset_index(drop=True)
    valid = cohort[ids.isin(set(checkpoint["validation_ids"]))].reset_index(drop=True)
    print(f"Probe: fit on {len(train)} training compounds, score on {len(valid)} held-out validation compounds")

    rows = []
    for label, key in (("emb_sp", "smiles_physchem"), ("emb_all", "all")):
        z_train = embed_batched(model, tensors_for(train, tokenizer, checkpoint, INPUT_SETS[key]))
        z_valid = embed_batched(model, tensors_for(valid, tokenizer, checkpoint, INPUT_SETS[key]))
        for column in [c for c in PROPERTY if c in cohort]:
            y_train, y_valid = train[column].to_numpy(float), valid[column].to_numpy(float)
            fit_ok, test_ok = np.isfinite(y_train), np.isfinite(y_valid)
            if fit_ok.sum() < 50 or test_ok.sum() < 20 or np.std(y_valid[test_ok]) == 0:
                continue
            for model_name in models:
                estimator = make_model(model_name, 0).fit(z_train[fit_ok], y_train[fit_ok])
                pred = estimator.predict(z_valid[test_ok])
                ss_res = float(((y_valid[test_ok] - pred) ** 2).sum())
                ss_tot = float(((y_valid[test_ok] - y_valid[test_ok].mean()) ** 2).sum())
                rows.append({"embedding": label, "model": model_name, "descriptor": column,
                             "r2": 1 - ss_res / ss_tot, "n_valid": int(test_ok.sum())})
        print("finished", label, flush=True)
    result = pd.DataFrame(rows)
    output = Path(output) if output else RESULTS / "embedding_probe.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print("\nHeld-out R^2 for recovering each descriptor from the embedding (1.0 = perfect):")
    print(result.pivot_table(index="descriptor", columns=["embedding", "model"], values="r2").round(3).to_string())
    print("Saved:", output)
    return result

# %% Command line
def main():
    parser = argparse.ArgumentParser(description="Baseline models for the MEA task, and an embedding diagnostic.")
    parser.add_argument("--probe", action="store_true", help="run the descriptor-recovery diagnostic and exit")
    parser.add_argument("--probe-train", type=int, default=3000, help="training compounds used to fit the probe")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--features", nargs="+", choices=FEATURE_SETS, default=FEATURE_SETS)
    parser.add_argument("--fractions", nargs="+", type=float, default=FRACTIONS)
    parser.add_argument("--target", choices=MEA_TARGETS, default="n_hits", help="what to predict: hit count or AC50-based potency")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--split-seed", type=int, default=0, help="must match step 10's --split-seed")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true", help="skip runs already saved in --output")
    args = parser.parse_args()
    if args.probe:
        run_probe(args.models, args.probe_train, args.checkpoint, args.output)
    else:
        run_baselines(args.seeds, args.fractions, args.features, args.models, args.test_fraction,
                      args.split_seed, args.checkpoint, args.output, args.resume, args.target)


if __name__ == "__main__":
    main()
