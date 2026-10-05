"""Step 10d - Predicting ToxCast/Tox21 potency (log AC50) from structure

AC50 is the concentration at which a chemical reaches half of its maximal effect in an assay: a *potency*.
This step asks whether the chemical information the foundation model receives (SMILES and physchem) predicts
that potency, and whether the frozen pretrained embedding helps compared with raw features.

AC50 versus log AC50 -- read this first
  * The normalized table `bioactivity.csv` stores the **linear** AC50 in micromolar (`ac50_uM`).
  * The training cohort (step 02) stores **log10(AC50 in uM)** in the columns `bioac50__<assay>`.
    Those values are already logarithms: never take a log of them again, and never average them as if they
    were concentrations. (log10 AC50 = -2 is 0.01 uM, 0 is 1 uM, +2 is 100 uM.)
  * Models here are trained and scored on **log10 AC50 (uM)**, because potency spans many orders of magnitude.
    Errors are therefore in log10 units: RMSE_log10 = 1.0 means a typical error of one order of magnitude.
    `fold_error` = 10 ** MAE_log10 restates the mean absolute error as a multiplicative factor.
  * An AC50 exists **only for active calls** (hit call >= 0.9); for inactive assays it is missing. So each task
    is conditional: "how potent, given that the chemical is active in this assay".

Targets
  most_potent      the smallest log10 AC50 over all assays in which the chemical is active (its strongest effect).
                   Note it also depends on how many assays a chemical was active in.
  assay:<name>     log10 AC50 in one assay, for chemicals active in it. By default the 5 assays with the most
                   AC50 values in the training compounds are used.

No leakage from the assay itself: features are structure only (physchem, fingerprints, and the embedding computed
with *only SMILES + physchem visible*), so no hit call or AC50 of any assay is ever an input.

Split: the checkpoint's own split. Models are fitted on the pretraining-train compounds and scored on its
held-out validation compounds, which the foundation model never saw. (The pretrained model did see the training
compounds' AC50 values during pretraining, which the baselines cannot use; if anything this favours the
embedding.) Training subsets are nested (10% inside 25% inside 50% inside 100%) and change with the seed.

Second part (`--mea-predict`): predict AC50 for the in-house MEA screen's test compounds. Train on the MEA training
compounds and predict one AC50 per test compound (default: the median over its quality-checked hits), using the
same fixed scaffold split as steps 10-10c. See section 7.

Input : data/processed/comptox_v3.parquet, checkpoints/comptox_v3_best.pt, data/mea_processed/*.csv (for --mea-predict)
Output: results/potency_metrics.csv, results/mea_test_ac50_predictions_<target>.csv,
        data/processed/comptox_embeddings_smiles_physchem.parquet (cache)
Run   : python scripts/10d_toxcast_potency.py --seeds 3 --features physchem morgan_physchem emb_sp
        python scripts/10d_toxcast_potency.py --mea-predict --target median_hit_logac50
"""

# %%
import argparse
import json
import warnings
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from torch.utils.data import DataLoader, Dataset

# %% [markdown]
# ## 1. Paths and settings

# %%
PROJECT_ROOT = Path(__file__).resolve().parents[1]
COHORT = PROJECT_ROOT / "data" / "processed" / "comptox_v3.parquet"
EMBEDDINGS_ALL = PROJECT_ROOT / "data" / "processed" / "comptox_embeddings.parquet"   # step 04: all modalities visible
EMBEDDING_CACHE = PROJECT_ROOT / "data" / "processed" / "comptox_embeddings_smiles_physchem.parquet"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "comptox_v3_best.pt"
RESULTS = PROJECT_ROOT / "results"

AC50_PREFIX = "bioac50__"        # cohort columns hold log10(AC50 in micromolar), NOT linear AC50
FRACTIONS = [0.10, 0.25, 0.50, 1.00]
FEATURE_SETS = ["physchem", "morgan_physchem", "emb_sp"]
MODELS = ["ridge", "random_forest"]
MEA_PROCESSED = PROJECT_ROOT / "data" / "mea_processed"
MEA_POTENCY_TARGETS = ["min_hit_logac50", "median_hit_logac50"]
KEEP_SMILES_PHYSCHEM = [True, False, False, False, False, False]   # visible modalities: only physchem (+ SMILES)

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
# ### A neural head on the pretrained model
# `TaskHead` is a small MLP that maps the 256-d embedding to one number. `configure_finetuning` decides what is
# trained: `head_only` trains just the head on a frozen model, `partial` also unfreezes the last two transformer
# layers and the fusion block, `full` unfreezes everything, and `scratch` (in `fit_and_predict`) trains the same
# architecture from random weights. The target is standardised with the training subset's mean and standard
# deviation and predictions are converted back before they are returned.

# %%
class TaskHead(nn.Module):
    def __init__(self, latent_dim=256, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 128), nn.GELU(), nn.Dropout(0.20), nn.Linear(128, output_dim))

    def forward(self, embedding):
        return self.net(embedding)


def configure_finetuning(model, head, strategy, last_n_layers=2):
    for parameter in model.parameters():
        parameter.requires_grad = False
    if strategy == "partial":
        for layer in model.transformer.layers[-last_n_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        for parameter in model.fusion.parameters():
            parameter.requires_grad = True
    elif strategy == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True
    elif strategy != "head_only":
        raise ValueError("strategy must be head_only, partial or full")
    for parameter in head.parameters():
        parameter.requires_grad = True


def fit_and_predict(strategy, pretrained, checkpoint, tokenizer, data, y, train_idx, eval_idx, epochs=40, seed=0, batch_size=32):
    """Train `strategy` on `train_idx` and return predictions (original units) for `eval_idx`."""
    torch.manual_seed(seed)
    y_mean, y_std = float(y[train_idx].mean()), float(y[train_idx].std() or 1.0)
    target = torch.tensor((y - y_mean) / y_std, dtype=torch.float32)
    model = build_model(checkpoint, tokenizer) if strategy == "scratch" else deepcopy(pretrained)
    head = TaskHead(checkpoint["config"]["model"]["latent_dim"])
    configure_finetuning(model, head, "full" if strategy == "scratch" else strategy)
    if strategy == "head_only":  # frozen backbone: embed once, train only the head
        model.eval()
        with torch.no_grad():
            z_train, z_eval = embed(model, data, train_idx), embed(model, data, eval_idx)
        groups = [{"params": head.parameters(), "lr": 1e-3}]
    else:
        backbone = [p for p in model.parameters() if p.requires_grad]
        groups = [{"params": head.parameters(), "lr": 1e-3}, {"params": backbone, "lr": 1e-3 if strategy == "scratch" else 1e-4}]
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    loss_fn = nn.MSELoss()
    n = len(train_idx)
    for epoch in range(epochs):
        head.train()
        order = np.random.RandomState(seed * 1000 + epoch).permutation(n)
        for start in range(0, n, batch_size):
            batch = order[start:start + batch_size]
            if len(batch) < 2:
                continue
            optimizer.zero_grad()
            if strategy == "head_only":
                pred = head(z_train[batch]).squeeze(-1)
            else:
                model.train()
                pred = head(embed(model, data, train_idx[batch])).squeeze(-1)
            loss_fn(pred, target[train_idx[batch]]).backward()
            optimizer.step()
    head.eval()
    model.eval()
    with torch.no_grad():
        z = z_eval if strategy == "head_only" else embed(model, data, eval_idx)
        return head(z).squeeze(-1).numpy() * y_std + y_mean


def fit_head_on_features(X, y, train_idx, eval_idx, epochs=100, seed=0, batch_size=32, patience=10):
    """A neural head (`TaskHead`, an MLP) trained on precomputed feature vectors, e.g. frozen embeddings.

    * Inputs are centred and scaled with the training rows' statistics. A column's scale is never smaller than the
      median column scale, so low-variance embedding dimensions are not blown up, while raw physchem columns with
      large ranges are brought to a comparable size. The target is standardised as in `fit_and_predict`.
    * With so few labelled compounds a head overfits quickly, so 20% of the training rows are held out internally
      for **early stopping**: training stops when their loss has not improved for `patience` epochs and the best
      weights are restored. The test compounds are never used."""
    torch.manual_seed(seed)
    mean, std = np.nanmean(X[train_idx], axis=0), np.nanstd(X[train_idx], axis=0)
    scale = np.maximum(std, np.median(std[std > 1e-8]) if (std > 1e-8).any() else 1.0)
    Z = torch.tensor(np.nan_to_num((X - mean) / scale), dtype=torch.float32)
    y_mean, y_std = float(y[train_idx].mean()), float(y[train_idx].std() or 1.0)
    target = torch.tensor((y - y_mean) / y_std, dtype=torch.float32)

    shuffled = np.random.RandomState(seed).permutation(train_idx)
    n_val = max(1, int(round(0.2 * len(shuffled))))
    val_rows, fit_rows = shuffled[:n_val], shuffled[n_val:]
    head = TaskHead(X.shape[1])
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.05)
    loss_fn = nn.MSELoss()
    best_loss, best_state, stale = float("inf"), deepcopy(head.state_dict()), 0
    for epoch in range(epochs):
        head.train()
        order = np.random.RandomState(seed * 1000 + epoch).permutation(len(fit_rows))
        for start in range(0, len(fit_rows), batch_size):
            batch = fit_rows[order[start:start + batch_size]]
            if len(batch) < 2:
                continue
            optimizer.zero_grad()
            loss_fn(head(Z[batch]).squeeze(-1), target[batch]).backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(head(Z[val_rows]).squeeze(-1), target[val_rows]))
        if val_loss < best_loss - 1e-6:
            best_loss, best_state, stale = val_loss, deepcopy(head.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        return head(Z[eval_idx]).squeeze(-1).numpy() * y_std + y_mean

# %% [markdown]
# ## 3. Baseline models

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
# ## 4. Units: log AC50 versus AC50
# `micromolar` converts a log10 AC50 back to a concentration for display only. All modelling and all reported
# error metrics stay in log10 units.

# %%
def micromolar(log10_ac50):
    """Linear AC50 in uM from a log10 AC50 (display only)."""
    return 10.0 ** np.asarray(log10_ac50, dtype=float)


def regression_metrics(y_true, y_pred):
    """Errors in log10 AC50 units; `fold_error` restates the mean absolute error as a multiplicative factor."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = spearmanr(y_true, y_pred).statistic if np.std(y_pred) > 0 else np.nan
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"rmse_log10": float(np.sqrt(np.mean((y_true - y_pred) ** 2))), "mae_log10": mae,
            "fold_error": 10.0 ** mae, "spearman": float(rho), "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan}


def training_subset(train_pool, seed, fraction):
    """Nested random subset of the training pool: the first `fraction` of a seed-specific shuffle."""
    order = np.random.RandomState(1000 + seed).permutation(train_pool)
    return np.sort(order[:max(8, int(round(fraction * len(order))))])

# %% [markdown]
# ## 5. Building the tasks
# * `load_split` selects the checkpoint's training and validation compounds from the cohort.
# * `potency_targets` extracts the log10 AC50 labels. `most_potent` is the row-wise minimum over the assays in
#   which the chemical is active (missing if it has no AC50 at all). `top_assays` lists the assays with the most
#   AC50 values in the training compounds.
# * `smiles_physchem_embeddings` computes the frozen embedding with only SMILES and physchem visible for the
#   compounds that are needed, and caches it, because embedding thousands of compounds takes several minutes.

# %%
def load_split(checkpoint_path=CHECKPOINT):
    model, tokenizer, checkpoint = load_checkpoint(checkpoint_path)
    cohort = pd.read_parquet(COHORT)
    ids = cohort["DTXSID"].astype(str)
    train = cohort[ids.isin(set(checkpoint["train_ids"]))].reset_index(drop=True)
    valid = cohort[ids.isin(set(checkpoint["validation_ids"]))].reset_index(drop=True)
    return model, tokenizer, checkpoint, train, valid


def potency_targets(frame, assays):
    """{target name: log10 AC50 array, NaN where the chemical has no AC50 for that target}."""
    log_ac50 = frame[[c for c in frame if c.startswith(AC50_PREFIX)]].to_numpy(float)
    finite = np.isfinite(log_ac50)
    most_potent = np.where(finite.any(axis=1), np.min(np.where(finite, log_ac50, np.inf), axis=1), np.nan)
    targets = {"most_potent": most_potent}
    for assay in assays:
        targets[f"assay:{assay}"] = frame[AC50_PREFIX + assay].to_numpy(float)
    return targets


def top_assays(train, n_assays):
    counts = train[[c for c in train if c.startswith(AC50_PREFIX)]].notna().sum().sort_values(ascending=False)
    return [c[len(AC50_PREFIX):] for c in counts.head(n_assays).index]


@torch.no_grad()
def embed_frame(model, tokenizer, checkpoint, frame, keep=KEEP_SMILES_PHYSCHEM, batch_size=128):
    """Frozen embeddings with only the modalities flagged in `keep` visible."""
    model.eval()
    loader = DataLoader(dataset_for_checkpoint(frame, tokenizer, checkpoint), batch_size=batch_size)
    chunks = []
    for ids, values, masks in loader:
        values = [v if k else torch.zeros_like(v) for v, k in zip(values, keep)]
        masks = [m if k else torch.zeros_like(m) for m, k in zip(masks, keep)]
        chunks.append(model(ids, values, masks)["embedding"].numpy())
    return np.concatenate(chunks, axis=0)


def smiles_physchem_embeddings(model, tokenizer, checkpoint, frame):
    """Embeddings for `frame` (rows in the same order), reusing/updating the on-disk cache."""
    z_columns = None
    cache = pd.read_parquet(EMBEDDING_CACHE) if EMBEDDING_CACHE.exists() else pd.DataFrame(columns=["DTXSID"])
    have = set(cache["DTXSID"].astype(str))
    missing = frame[~frame["DTXSID"].astype(str).isin(have)]
    if len(missing):
        print(f"Embedding {len(missing):,} compounds with only SMILES + physchem visible (cached afterwards)...", flush=True)
        z = embed_frame(model, tokenizer, checkpoint, missing)
        z_columns = [f"z{i:03d}" for i in range(z.shape[1])]
        new = pd.concat([pd.DataFrame({"DTXSID": missing["DTXSID"].astype(str).values}),
                         pd.DataFrame(z, columns=z_columns)], axis=1)
        cache = pd.concat([cache, new], ignore_index=True) if len(cache) else new
        cache.to_parquet(EMBEDDING_CACHE, index=False)
    z_columns = [c for c in cache if c.startswith("z")]
    lookup = cache.drop_duplicates("DTXSID").set_index(cache["DTXSID"].astype(str).drop_duplicates())[z_columns]
    return lookup.loc[frame["DTXSID"].astype(str)].to_numpy(dtype=np.float32)


def feature_matrices(frame, names, embeddings=None):
    physchem = frame[[c for c in PROPERTY if c in frame]].to_numpy(float)
    matrices = {"physchem": physchem}
    if "morgan_physchem" in names:
        matrices["morgan_physchem"] = np.hstack([morgan_matrix(frame["smiles"].tolist()), physchem])
    if embeddings is not None:
        matrices["emb_sp"] = embeddings
    return matrices

# %% [markdown]
# ## 6. Run
# For every target, seed, training fraction, feature set and model: fit on the nested training subset of the
# labelled training compounds and score on the labelled validation compounds. Reported for each run: RMSE and
# MAE in log10 units, the corresponding `fold_error`, Spearman correlation and R^2. A "predict the training mean
# log AC50" row is included as the floor. Results are saved after every fit (`resume=True` continues a run).

# %%
def run_potency(seeds=3, fractions=FRACTIONS, features=FEATURE_SETS, models=MODELS, n_assays=5,
                assays=None, use_most_potent=True, checkpoint_path=CHECKPOINT, output=None, resume=False):
    warnings.filterwarnings("ignore", message="(?s).*At least one non-missing value.*")
    model, tokenizer, checkpoint, train, valid = load_split(checkpoint_path)
    assays = list(assays) if assays else top_assays(train, n_assays)
    train_targets, valid_targets = potency_targets(train, assays), potency_targets(valid, assays)
    names = [t for t in train_targets if use_most_potent or t != "most_potent"]
    print(f"{len(train):,} training / {len(valid):,} validation compounds (the checkpoint's split)")
    print("Assays:", assays)

    def usable(frame_targets, name):
        return np.isfinite(frame_targets[name])

    # embeddings only for rows that some target uses
    used_train = np.any([usable(train_targets, n) for n in names], axis=0)
    used_valid = np.any([usable(valid_targets, n) for n in names], axis=0)
    if "emb_sp" in features:
        both = pd.concat([train[used_train], valid[used_valid]], ignore_index=True)
        z = smiles_physchem_embeddings(model, tokenizer, checkpoint, both)
        z_train, z_valid = np.full((len(train), z.shape[1]), np.nan, np.float32), np.full((len(valid), z.shape[1]), np.nan, np.float32)
        z_train[used_train], z_valid[used_valid] = z[:used_train.sum()], z[used_train.sum():]
    X_train_all = feature_matrices(train, features, z_train if "emb_sp" in features else None)
    X_valid_all = feature_matrices(valid, features, z_valid if "emb_sp" in features else None)

    output = Path(output) if output else RESULTS / "potency_metrics.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows, done = [], set()
    if resume and output.exists():
        rows = pd.read_csv(output).to_dict("records")
        done = {(int(r["seed"]), r["target"], r["model"], r["features"], float(r["fraction"])) for r in rows}
        print(f"Resuming: {len(done)} finished runs found in {output.name}")

    for target in names:
        tr_rows, te_rows = np.where(usable(train_targets, target))[0], np.where(usable(valid_targets, target))[0]
        y_train, y_test = train_targets[target][tr_rows], valid_targets[target][te_rows]
        print(f"\n[{target}] n_train={len(tr_rows):,} n_test={len(te_rows):,} | log10 AC50 mean {y_train.mean():.2f} "
              f"(median AC50 {micromolar(np.median(y_train)):.3g} uM), sd {y_train.std():.2f}", flush=True)
        for seed in range(seeds):
            if (seed, target, "mean_baseline", "none", 1.0) not in done:
                rows.append({"seed": seed, "target": target, "model": "mean_baseline", "features": "none", "fraction": 1.0,
                             "n_train": len(tr_rows), "n_test": len(te_rows),
                             **regression_metrics(y_test, np.full(len(y_test), y_train.mean()))})
            for fraction in fractions:
                subset = training_subset(np.arange(len(tr_rows)), seed, fraction)
                for feature_name in features:
                    X_tr, X_te = X_train_all[feature_name][tr_rows], X_valid_all[feature_name][te_rows]
                    for model_name in models:
                        if (seed, target, model_name, feature_name, float(fraction)) in done:
                            continue
                        fitted = make_model(model_name, seed).fit(X_tr[subset], y_train[subset])
                        metrics = regression_metrics(y_test, fitted.predict(X_te))
                        rows.append({"seed": seed, "target": target, "model": model_name, "features": feature_name,
                                     "fraction": fraction, "n_train": len(subset), "n_test": len(te_rows), **metrics})
                        print(f"  seed={seed} fraction={fraction:>4.0%} {model_name:<13} {feature_name:<16} "
                              f"rmse_log10={metrics['rmse_log10']:.3f} fold_error={metrics['fold_error']:.2f}x "
                              f"spearman={metrics['spearman']:.3f}", flush=True)
                        pd.DataFrame(rows).to_csv(output, index=False)
    result = pd.DataFrame(rows)
    result.to_csv(output, index=False)
    summary = result[result["model"] != "mean_baseline"].groupby(["target", "model", "features", "fraction"])[
        ["rmse_log10", "fold_error", "spearman"]].mean().round(3)
    print("\nMean over seeds (errors in log10 AC50 units; fold_error = 10**MAE_log10):")
    print(summary.to_string())
    print("\nFloor (predict the training mean log AC50):")
    print(result[result["model"] == "mean_baseline"].groupby("target")[["rmse_log10", "fold_error"]].mean().round(3).to_string())
    print("Saved:", output)
    return result

# %% [markdown]
# ## 7. Predicting AC50 for the MEA test compounds
# **Goal:** train on the MEA training compounds, predict an AC50 for each MEA **test** compound, and compare it with
# the measured value.
#
# * **Which compounds.** The same fixed scaffold split as steps 10-10c (about 25% of the MEA compounds are test
#   compounds, and no test compound shares a scaffold with a training compound).
# * **Which number.** Each MEA compound has up to 19 AC50 values, one per metric. It gets **one** number:
#   `median_hit_logac50`, the median logAC50 over its real hits (or `min_hit_logac50`, its most potent effect).
# * **Which values count ("quality checks").** An entry is used only if (1) its hit code is exactly 1 or 2, (2) a
#   logAC50 is present, (3) it is not a placeholder value (exactly -2.0 = 0.01 uM or 1.176 = 15 uM, which fill in
#   "no real AC50"), and (4) it lies inside the tested range of 0.1-30 uM (log10 -1 to 1.477); outside that range the
#   AC50 was extrapolated, not measured. Compounds with no qualifying entry are left out.
# * **AC50 versus log AC50.** The model is trained and scored on **log10 AC50 (uM)**. The table shows both
#   the log10 value and, for reading, the same number as a concentration (`AC50_uM = 10**log10`). `fold_error` =
#   10**|actual - predicted| in log10 units, i.e. how many times too high or too low the prediction is.

# %%
TESTED_RANGE_LOG10 = (-1.0, float(np.log10(30.0)))   # a logAC50 outside this range was not actually measured
PLACEHOLDER_LOGAC50 = (-2.0, 1.176)                   # values the workbook uses to fill in "no real AC50"


def quality_hits(metrics):
    """Metric entries usable as AC50 labels, and the number left after each check (see the text above)."""
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


def mea_potency_labels(target):
    """One row per MEA compound: its log10 AC50 (uM), from the quality-checked hits."""
    if target not in MEA_POTENCY_TARGETS:
        raise ValueError(f"target must be one of {MEA_POTENCY_TARGETS}")
    kept, steps = quality_hits(pd.read_csv(MEA_PROCESSED / "mea_metric_activity_summary.csv"))
    print("AC50 quality checks (entries left): " + " -> ".join(f"{name} {n}" for name, n in steps))
    hits = kept.groupby("compound_id")["logac50"]
    label = hits.min() if target == "min_hit_logac50" else hits.median()
    return label.rename("log10_ac50").reset_index()


def scaffold_of(smiles):
    """Bemis-Murcko scaffold ('' for acyclic molecules)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(str(smiles))
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol is not None else ""


def scaffold_holdout(scaffolds, test_fraction=0.25, split_seed=0):
    """Deterministic scaffold split -> (train_indices, test_indices): whole scaffolds go to one side, the largest
    scaffold groups to training first, and the remaining rarer scaffolds form the test set."""
    groups = {}
    for i, scaffold in enumerate(scaffolds):
        groups.setdefault(scaffold, []).append(i)
    keys = sorted(groups)
    np.random.RandomState(split_seed).shuffle(keys)
    keys.sort(key=lambda k: -len(groups[k]))
    n_train = len(scaffolds) - int(round(test_fraction * len(scaffolds)))
    train, test = [], []
    for key in keys:
        (train if len(train) + len(groups[key]) <= n_train else test).extend(groups[key])
    return np.array(sorted(train)), np.array(sorted(test))


def mea_task(target, test_fraction=0.25, split_seed=0):
    """The MEA compounds with a usable AC50 label, and the fixed scaffold split shared with steps 10-10c."""
    mapping = pd.read_csv(MEA_PROCESSED / "mea_to_comptox_mapping_template.csv")[["compound_id", "DTXSID"]]
    names = pd.read_csv(MEA_PROCESSED / "mea_compound_summary.csv")[["compound_id", "base_compound_name"]]
    mapping = mapping[mapping["DTXSID"].astype(str).str.strip() != ""].dropna(subset=["DTXSID"]).merge(names, on="compound_id")
    cohort = pd.read_parquet(COHORT)
    frame = cohort[cohort["DTXSID"].isin(mapping["DTXSID"])].drop_duplicates("DTXSID").reset_index(drop=True)
    scaffolds = np.array([scaffold_of(s) for s in frame["smiles"]])
    train_pool, test_pool = scaffold_holdout(scaffolds, test_fraction, split_seed)   # the split of steps 10-10c
    labels = mea_potency_labels(target).merge(mapping, on="compound_id").groupby("DTXSID", as_index=False).agg(
        log10_ac50=("log10_ac50", "mean"), compound=("base_compound_name", "first"))
    frame = frame.merge(labels, on="DTXSID", how="left")
    y = frame["log10_ac50"].to_numpy(float)
    train_idx, test_idx = train_pool[np.isfinite(y[train_pool])], test_pool[np.isfinite(y[test_pool])]
    reference = RESULTS / "transfer_fixed_smiles_physchem_split.csv"
    if reference.exists():   # confirm the test compounds are those of step 10 (minus any without a usable AC50)
        earlier = set(pd.read_csv(reference).query("set == 'test'")["DTXSID"])
        print("Test compounds all belong to step 10's test set:", set(frame["DTXSID"].values[test_idx]) <= earlier)
    print(f"{len(frame)} MEA compounds -> {len(train_idx)} training and {len(test_idx)} test compounds with a usable AC50 "
          f"(the fixed split has {len(train_pool)} training and {len(test_pool)} test compounds)")
    print(f"Training AC50: median {micromolar(np.median(y[train_idx])):.3g} uM (log10 {np.median(y[train_idx]):.2f}), "
          f"sd {y[train_idx].std():.2f} log10 units")
    return frame, y, scaffolds, train_idx, test_idx


def bootstrap_rmse_interval(y_true, y_pred, n=500, seed=0):
    """95% bootstrap interval of the RMSE over test compounds (a small test set has a wide interval)."""
    rng = np.random.RandomState(seed)
    values = [np.sqrt(np.mean((y_true[i] - y_pred[i]) ** 2)) for i in (rng.randint(0, len(y_true), len(y_true)) for _ in range(n))]
    return f"{np.percentile(values, 2.5):.2f}-{np.percentile(values, 97.5):.2f}"

# %% [markdown]
# ### Comparing many methods with 5-fold cross-validation
# `run_mea_compare` scores every combination of **input** and **model** the same two ways:
#
# * **Inputs (features).** `physchem` (the 16 descriptors), `morgan` (a SMILES fingerprint), `morgan_physchem`,
#   `emb_sp` (the frozen foundation-model embedding, SMILES + physchem visible), `emb_sp+physchem`, and `all_raw`
#   (physchem plus every ToxCast/Tox21 column, hazard and exposure).
# * **Models.** ridge, random forest, gradient boosting (`gbm`), SVR (`svr`), and predicting the training mean
#   (the floor). In addition, **neural heads**:
#   * `nn_head|<embedding>`: a small MLP trained on top of a frozen embedding: `emb_sp` (SMILES + physchem visible),
#     `emb_sp+physchem` (that embedding plus the raw physchem values) and `emb_all` (step 04's embedding with all
#     modalities visible);
#   * `nn_head_only|smiles+physchem`: the same idea, but computed inside the pretrained model (no saved embedding);
#   * `nn_partial|smiles+physchem`: also fine-tunes the last two transformer layers and the fusion block
#     (optionally `nn_full` / `nn_scratch`).
# * **5-fold CV (`cv_*` columns).** The *training* compounds are split into 5 folds grouped by scaffold; each method
#   is trained on 4 folds and predicts the 5th, so every training compound gets an out-of-fold prediction. This
#   compares methods without touching the test set.
# * **Test (`test_*` columns).** Each method is then trained on the training compounds and scored on the fixed
#   test compounds, at 10%, 25%, 50% and 100% of the training compounds (nested random subsets, averaged over
#   several seeds; the table below shows the 100% result and the learning curve shows all four). `test_rmse_ci` is a 95% bootstrap interval: with about 50 test compounds it is wide, so small
#   differences between methods are not meaningful.
# * The method with the lowest CV RMSE is reported as "selected by CV", together with its test score: the honest way
#   to pick a method and then evaluate it.
#
# All errors are in **log10 AC50 (uM)**; `fold_error` = 10**MAE. Nothing here is tuned on the test set.

# %%
def run_mea_compare(target="median_hit_logac50", n_folds=5, fractions=(0.10, 0.25, 0.50, 1.0), seeds=3,
                    nn_strategies=("head_only", "partial"), nn_epochs=40,
                    head_features=("emb_sp", "emb_sp+physchem", "emb_all"),
                    features=("physchem", "morgan", "morgan_physchem", "emb_sp", "emb_sp+physchem", "all_raw"),
                    models=("ridge", "random_forest", "gbm", "svr"), test_fraction=0.25, split_seed=0,
                    checkpoint_path=CHECKPOINT, output=None):
    warnings.filterwarnings("ignore", message="(?s).*At least one non-missing value.*")
    torch.set_num_threads(4)
    frame, y, scaffolds, train_idx, test_idx = mea_task(target, test_fraction, split_seed)

    # 1. feature matrices for all MEA compounds
    physchem = frame[[c for c in PROPERTY if c in frame]].to_numpy(float)
    matrices = {"physchem": physchem}
    if any(f.startswith("morgan") for f in features):
        fingerprint = morgan_matrix(frame["smiles"].tolist())
        matrices["morgan"], matrices["morgan_physchem"] = fingerprint, np.hstack([fingerprint, physchem])
    if "all_raw" in features:
        bioactivity = frame[[c for c in frame if c.startswith(("biohit__", "bioac50__", "bioeff__"))]].to_numpy(float)
        context = frame[[c for c in HAZARD + EXPOSURE if c in frame]].to_numpy(float)
        matrices["all_raw"] = np.hstack([physchem, bioactivity, context])
    pretrained = tokenizer = checkpoint = data = None
    if any(f.startswith("emb_") for f in list(features) + list(head_features)) or nn_strategies:
        pretrained, tokenizer, checkpoint = load_checkpoint(checkpoint_path)
        matrices["emb_sp"] = smiles_physchem_embeddings(pretrained, tokenizer, checkpoint, frame)
        matrices["emb_sp+physchem"] = np.hstack([matrices["emb_sp"], physchem])
        data = tensors_for(frame, tokenizer, checkpoint, KEEP_SMILES_PHYSCHEM) if nn_strategies else None
    if "emb_all" in list(features) + list(head_features):   # step 04's embedding: ALL modalities visible (allowed for the MEA task)
        table = pd.read_parquet(EMBEDDINGS_ALL).drop_duplicates("DTXSID").set_index("DTXSID")
        matrices["emb_all"] = table[[c for c in table if c.startswith("z") and c[1:].isdigit()]].loc[frame["DTXSID"]].to_numpy(np.float32)

    # 2. every method as a function (fit indices, evaluate indices, seed) -> predictions
    methods = {"mean_baseline": lambda fit, ev, seed: np.full(len(ev), y[fit].mean())}
    for feature_name in features:
        for model_name in models:
            methods[f"{model_name}|{feature_name}"] = (
                lambda fit, ev, seed, X=matrices[feature_name], m=model_name: make_model(m, seed).fit(X[fit], y[fit]).predict(X[ev]))
    for feature_name in head_features:   # a neural head (MLP) on top of a frozen embedding
        methods[f"nn_head|{feature_name}"] = (
            lambda fit, ev, seed, X=matrices[feature_name]: fit_head_on_features(X, y, fit, ev, seed=seed))
    for strategy in nn_strategies:
        methods[f"nn_{strategy}|smiles+physchem"] = (
            lambda fit, ev, seed, st=strategy: fit_and_predict(st, pretrained, checkpoint, tokenizer, data, y, fit, ev, nn_epochs, seed))

    # 3. 5-fold CV on all training compounds (folds grouped by scaffold), and the learning curve on the test set
    folds = list(GroupKFold(n_splits=n_folds).split(train_idx, groups=scaffolds[train_idx]))
    cv_rows, curve_rows, seed0_predictions = [], [], {}
    for name, predict in methods.items():
        oof = np.full(len(train_idx), np.nan)
        for fit_pos, val_pos in folds:
            oof[val_pos] = predict(train_idx[fit_pos], train_idx[val_pos], 0)
        cv = regression_metrics(y[train_idx], oof)
        cv_rows.append({"method": name, "cv_rmse_log10": cv["rmse_log10"], "cv_spearman": cv["spearman"]})
        for fraction in fractions:
            for seed in range(seeds):
                subset = training_subset(train_idx, seed, fraction)   # nested: 10% inside 25% inside 50% inside 100%
                predictions = predict(subset, test_idx, seed)
                if fraction == 1.0 and seed == 0:
                    seed0_predictions[name] = predictions
                curve_rows.append({"method": name, "fraction": fraction, "seed": seed, "n_train": len(subset),
                                   **regression_metrics(y[test_idx], predictions)})
        at_full = [r for r in curve_rows if r["method"] == name and r["fraction"] == max(fractions)]
        print(f"  {name:<34} cv_rmse={cv['rmse_log10']:.3f}  test_rmse@{max(fractions):.0%}={np.mean([r['rmse_log10'] for r in at_full]):.3f}", flush=True)
    curve = pd.DataFrame(curve_rows)
    by_fraction = curve.groupby(["method", "fraction"], as_index=False)[["rmse_log10", "fold_error", "spearman", "r2"]].mean()

    # 4. summary at 100% of the training data: CV score, test score and a bootstrap interval for the test RMSE
    full = max(fractions)
    summary = pd.DataFrame(cv_rows)
    tests = by_fraction[by_fraction["fraction"] == full].set_index("method")
    summary["test_rmse_log10"] = summary["method"].map(tests["rmse_log10"])
    summary["test_rmse_ci"] = summary["method"].map(lambda m: bootstrap_rmse_interval(y[test_idx], seed0_predictions[m]))
    summary["test_fold_error"] = summary["method"].map(tests["fold_error"])
    summary["test_spearman"] = summary["method"].map(tests["spearman"])
    summary["test_r2"] = summary["method"].map(tests["r2"])
    summary = summary.sort_values("cv_rmse_log10").reset_index(drop=True)
    chosen = summary[summary["method"] != "mean_baseline"].iloc[0]["method"]

    # 5. save the tables and actual vs predicted AC50 for each test compound (100% of training, seed 0)
    table = pd.DataFrame({"DTXSID": frame["DTXSID"].values[test_idx], "compound": frame["compound"].values[test_idx],
                          "actual_log10_ac50": y[test_idx], "actual_AC50_uM": micromolar(y[test_idx])})
    for name, values in seed0_predictions.items():
        table[f"pred_log10_ac50|{name}"] = values
    for name in ("mean_baseline", chosen):
        table[f"pred_AC50_uM|{name}"] = micromolar(seed0_predictions[name])
    table["fold_error_selected"] = 10.0 ** np.abs(table["actual_log10_ac50"] - table[f"pred_log10_ac50|{chosen}"])
    output = Path(output) if output else RESULTS / f"mea_test_ac50_predictions_{target}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    summary.to_csv(output.with_name(output.stem + "_methods.csv"), index=False)
    curve.to_csv(output.with_name(output.stem + "_learning_curve_runs.csv"), index=False)
    by_fraction.to_csv(output.with_name(output.stem + "_learning_curve.csv"), index=False)

    print(f"\nAll methods at 100% of the training data ({len(train_idx)} training, {len(test_idx)} test compounds), "
          "best 5-fold CV first (errors in log10 AC50 units):")
    print(summary.round(3).to_string(index=False))
    print("\nLearning curve: test RMSE (log10 AC50) by fraction of the training compounds, mean over "
          f"{seeds} seeds; rows sorted by the 100% score:")
    curve_table = by_fraction.pivot(index="method", columns="fraction", values="rmse_log10").round(3)
    print(curve_table.sort_values(full).to_string())
    row = summary[summary["method"] == chosen].iloc[0]
    floor = summary[summary["method"] == "mean_baseline"].iloc[0]
    print(f"\nSelected by 5-fold CV: {chosen} -> test RMSE {row['test_rmse_log10']:.3f} (95% CI {row['test_rmse_ci']}) "
          f"against {floor['test_rmse_log10']:.3f} for predicting the mean; median miss "
          f"{np.median(table['fold_error_selected']):.1f}-fold per compound.")
    print("Saved:", output)
    return table, summary, by_fraction

# %% Command line
def main():
    parser = argparse.ArgumentParser(description="Predict ToxCast/Tox21 log AC50 from structure.")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--features", nargs="+", choices=FEATURE_SETS, default=FEATURE_SETS)
    parser.add_argument("--fractions", nargs="+", type=float, default=FRACTIONS)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--n-assays", type=int, default=5, help="single-assay targets: the assays with most AC50 values")
    parser.add_argument("--assays", nargs="+", help="explicit assay names (without the bioac50__ prefix)")
    parser.add_argument("--no-most-potent", action="store_true", help="skip the most-potent-effect target")
    parser.add_argument("--mea-predict", action="store_true", help="compare methods for predicting AC50 of the MEA test compounds, then exit")
    parser.add_argument("--nn-strategies", nargs="*", default=["head_only", "partial"], choices=["head_only", "partial", "full", "scratch"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--only-heads", action="store_true", help="--mea-predict: run only the neural heads on embeddings (fast)")
    parser.add_argument("--mea-seeds", type=int, default=3, help="seeds for the learning curve of --mea-predict")
    parser.add_argument("--target", choices=MEA_POTENCY_TARGETS, default="median_hit_logac50", help="MEA potency label for --mea-predict")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true", help="skip runs already saved in --output")
    args = parser.parse_args()
    if args.mea_predict:
        only_heads = dict(features=(), models=(), nn_strategies=()) if args.only_heads else {}
        run_mea_compare(args.target, args.folds, tuple(args.fractions), args.mea_seeds,
                        **({"nn_strategies": tuple(args.nn_strategies)} | only_heads),
                        checkpoint_path=args.checkpoint, output=args.output)
        return
    run_potency(args.seeds, args.fractions, args.features, args.models, args.n_assays, args.assays,
                not args.no_most_potent, args.checkpoint, args.output, args.resume)


if __name__ == "__main__":
    main()
