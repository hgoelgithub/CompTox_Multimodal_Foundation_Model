"""Step 3 - Pretrain the multimodal foundation model

Trains one shared molecular embedding from SMILES plus six numeric modalities (physchem, ToxCast/Tox21
hit call, AC50, efficacy, hazard, exposure) with two self-supervised objectives:
  1. masked-language modelling on SMILES characters, and
  2. masked reconstruction of observed numeric values in every modality,
plus whole-modality dropout, so the model cannot lean on one always-present input.

Input : data/processed/comptox_v3.parquet   (from step 02)
Output: checkpoints/comptox_v3_best.pt      (best validation loss; weights, tokenizer, scalers, config)
        checkpoints/tokenizer_v3.json, checkpoints/training_history.csv
Run   : python scripts/03_train_foundation_model.py [--epochs N] [--max-compounds N] [--device cpu|cuda|mps]
        python scripts/03_train_foundation_model.py --check      # only verify the input exists
"""

# %%
import argparse
import json
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset

# %% [markdown]
# ## 1. Paths and configuration
# Hyper-parameters (model size, learning rate, masking probabilities, loss weights) live in
# `config.yaml` so an experiment can be changed without touching code.

# %%
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "config.yaml"
COHORT = PROJECT_ROOT / "data" / "processed" / "comptox_v3.parquet"
CHECKPOINTS = PROJECT_ROOT / "checkpoints"

# %% [markdown]
# ## 2. Building blocks
# The tokenizer, dataset and model are defined here. They are the same building blocks that steps 04,
# 05 and 10b use to load the trained checkpoint.

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
# ## 3. Masking, losses and one training epoch
# Self-supervised training hides part of the input and asks the model to restore it:
# * **SMILES masking**: about 15% of characters are replaced by `[MASK]` (at least one per molecule).
# * **Numeric masking**: about 15% of the *observed* values in each modality are zeroed out and become
#   reconstruction targets. Missing values are never targets.
# * **Modality dropout** (training only): with probability 0.2 an entire modality is hidden for a
#   compound and all its observed values must be reconstructed from the other modalities.
# * **Losses** are averaged only over target positions. Hit calls use a class-balanced binary
#   cross-entropy because active assays are rare; everything else uses mean squared error.

# %%
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device():
    """CUDA if available, then Apple-silicon MPS, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mask_smiles(ids, tokenizer, probability):
    """Replace random SMILES tokens by [MASK]; returns (masked ids, labels with -100 where not masked)."""
    masked, labels = ids.clone(), ids.clone()
    eligible = ids.ne(tokenizer.pad_id) & ids.ne(tokenizer.cls_id)
    selected = (torch.rand_like(ids.float()) < probability) & eligible
    if probability > 0:
        for i in range(ids.shape[0]):  # guarantee at least one masked token per molecule
            if eligible[i].any() and not selected[i].any():
                selected[i, torch.where(eligible[i])[0][0]] = True
    masked[selected] = tokenizer.mask_id
    labels[~selected] = -100
    return masked, labels


def mask_numeric(values, observed, probability):
    """Zero a random fraction of the *observed* values. Returns (corrupted values, new input mask, targets)."""
    selected = (torch.rand_like(values) < probability) & observed.bool()
    corrupted, input_mask = values.clone(), observed.clone()
    corrupted[selected] = 0
    input_mask[selected] = 0
    return corrupted, input_mask, selected.float()


def apply_modality_dropout(values, masks, targets, observed, probability):
    """Hide whole modalities for random compounds; their observed values become reconstruction targets."""
    values = [x.clone() for x in values]
    masks = [m.clone() for m in masks]
    targets = [t.clone() for t in targets]
    batch = values[0].shape[0]
    for i in range(len(values)):
        if values[i].shape[1] == 0:
            continue
        drop = torch.rand(batch, device=values[i].device) < probability
        if drop.any():
            values[i][drop] = 0
            masks[i][drop] = 0
            targets[i][drop] = observed[i][drop]
    return values, masks, targets


def masked_mse(pred, target, mask):
    """Mean squared error over the positions where mask = 1."""
    if target.numel() == 0:
        return pred.new_tensor(0.0)
    return (((pred - target) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)


def masked_balanced_bce(logits, target, mask, pos_weight):
    """Binary cross-entropy with per-assay positive weights, over the positions where mask = 1."""
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pos_weight)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def positive_weights(df, hit_columns, device, active_threshold=0.90):
    """Per-assay weight = (#inactive / #active), clipped to [1, 50], to counter class imbalance."""
    hits = df[hit_columns].to_numpy(dtype=np.float32)
    observed = np.isfinite(hits)
    positive = np.sum(observed & (hits >= active_threshold), axis=0)
    negative = np.sum(observed & (hits < active_threshold), axis=0)
    return torch.tensor(np.clip(negative / np.maximum(positive, 1), 1, 50), dtype=torch.float32, device=device)


def run_epoch(model, loader, tokenizer, cfg, pos_weight, device, optimizer=None):
    """One pass over `loader`. Trains if an optimizer is given, otherwise evaluates. Returns the mean loss."""
    training = optimizer is not None
    model.train(training)
    tc, weights = cfg["training"], cfg["loss_weights"]
    total = 0.0
    for ids, values, masks in loader:
        ids = ids.to(device)
        values = [x.to(device) for x in values]
        masks = [m.to(device) for m in masks]
        original, observed = [x.clone() for x in values], [m.clone() for m in masks]

        ids, labels = mask_smiles(ids, tokenizer, tc["smiles_mask_probability"])
        corrupted, input_masks, targets = [], [], []
        for x, m in zip(values, masks):
            c, im, t = mask_numeric(x, m, tc["numeric_mask_probability"])
            corrupted.append(c)
            input_masks.append(im)
            targets.append(t)
        if training:
            corrupted, input_masks, targets = apply_modality_dropout(
                corrupted, input_masks, targets, observed, tc["modality_dropout_probability"])

        out = model(ids, corrupted, input_masks)
        pred = out["numeric_predictions"]
        selected = labels != -100
        mlm = F.cross_entropy(out["mlm_logits"][selected], labels[selected]) if selected.any() \
            else out["mlm_logits"].new_tensor(0.0)
        loss = weights["smiles"] * mlm
        loss += weights["properties"] * masked_mse(pred[0], original[0], targets[0])
        loss += weights["hitcall"] * masked_balanced_bce(pred[1], original[1], targets[1], pos_weight)
        for i, key in [(2, "ac50"), (3, "efficacy"), (4, "hazard"), (5, "exposure")]:
            loss += weights[key] * masked_mse(pred[i], original[i], targets[i])
        if not torch.isfinite(loss):
            raise FloatingPointError("Training loss became NaN/Inf; check input scaling and the normalized data.")
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["gradient_clip"])
            optimizer.step()
        total += float(loss.detach().cpu())
    return total / max(len(loader), 1)

# %% [markdown]
# ## 4. Train
# Steps inside `train()`:
# 1. Load the cohort and split it into training and validation compounds. Rows are grouped by SMILES
#    so the same structure can never appear on both sides.
# 2. Fit the tokenizer and the per-column scalers on the **training** split only.
# 3. Build the model from `config.yaml`.
# 4. Each epoch: one training pass, then one validation pass. The validation pass uses a fixed random
#    seed, so the same values are masked every epoch and the validation loss is comparable across epochs.
# 5. Keep the checkpoint with the lowest validation loss, and stop after `patience` epochs without
#    improvement.

# %%
def check_inputs():
    """True if the training cohort exists."""
    if not COHORT.exists():
        print("Training cohort not found. Run scripts/02_build_training_cohort.py first.")
        return False
    print("Training cohort found:", COHORT)
    return True


def train(config_path=CONFIG, output_dir=CHECKPOINTS, epochs=None, max_compounds=None, device=None):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if epochs:
        cfg["training"]["epochs"] = epochs
    seed_all(cfg["seed"])
    device = torch.device(device) if device else choose_device()
    torch.set_num_threads(cfg["training"].get("cpu_threads", 2))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Device:", device)

    # 1. data and split
    df = pd.read_parquet(COHORT)
    if max_compounds:
        df = df.sample(min(max_compounds, len(df)), random_state=cfg["seed"])
    if len(df) < 4:
        raise ValueError("Need at least four compounds to make a train/validation split.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=cfg["data"]["validation_fraction"], random_state=cfg["seed"])
    train_idx, valid_idx = next(splitter.split(df, groups=df["smiles"]))
    train_df, valid_df = df.iloc[train_idx], df.iloc[valid_idx]

    # 2. tokenizer and scalers (training split only)
    max_length = cfg["data"]["max_smiles_length"]
    tokenizer = SmilesTokenizer().fit(train_df["smiles"].astype(str), cfg["model"]["vocab_size"])
    tokenizer.save(output_dir / "tokenizer_v3.json")
    groups = CompToxDataset(train_df, tokenizer, max_length).groups
    scalers = fit_group_scalers(train_df, groups)
    train_ds = CompToxDataset(train_df, tokenizer, max_length, scalers=scalers)
    valid_ds = CompToxDataset(valid_df, tokenizer, max_length, scalers=scalers)
    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    # 3. model and optimizer
    dims = [len(g) for g in train_ds.groups]
    m = cfg["model"]
    model = CompToxFoundationModel(
        len(tokenizer.vocab), dims, d_model=m["d_model"], n_heads=m["n_heads"], n_layers=m["layers"],
        feedforward_dim=m["feedforward_dim"], latent_dim=m["latent_dim"], dropout=m["dropout"],
        pad_id=tokenizer.pad_id, max_positions=max(512, max_length)).to(device)
    pos_weight = positive_weights(train_df, train_ds.hit_columns, device, cfg["data"]["toxcast_active_threshold"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                                  weight_decay=cfg["training"]["weight_decay"])

    # 4-5. training loop with early stopping
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_loss = run_epoch(model, train_loader, tokenizer, cfg, pos_weight, device, optimizer)
        with torch.random.fork_rng(), torch.no_grad():  # fixed masks, without disturbing the training RNG stream
            seed_all(cfg["seed"] + 1)
            valid_loss = run_epoch(model, valid_loader, tokenizer, cfg, pos_weight, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": valid_loss})
        print(f"Epoch {epoch:02d} | train={train_loss:.4f} | val={valid_loss:.4f}")
        if valid_loss < best:
            best, stale = valid_loss, 0
            torch.save({"model_state": deepcopy(model.state_dict()), "config": cfg, "modality_dims": dims,
                        "best_validation_loss": best, "scalers": scalers, "groups": train_ds.groups,
                        "vocab": tokenizer.vocab, "train_ids": train_df["DTXSID"].astype(str).tolist(),
                        "validation_ids": valid_df["DTXSID"].astype(str).tolist()},
                       output_dir / "comptox_v3_best.pt")
        else:
            stale += 1
        if stale >= cfg["training"]["patience"]:
            print("Early stopping.")
            break
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    print("Best validation loss:", best)
    print("Saved:", output_dir / "comptox_v3_best.pt")

# %% Command line
def main():
    parser = argparse.ArgumentParser(description="Pretrain the CompTox multimodal foundation model.")
    parser.add_argument("--check", action="store_true", help="only verify that the input exists")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-dir", type=Path, default=CHECKPOINTS)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-compounds", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()
    if args.check:
        raise SystemExit(0 if check_inputs() else 1)
    if not check_inputs():
        raise SystemExit(1)
    train(args.config, args.output_dir, args.epochs, args.max_compounds, args.device)


if __name__ == "__main__":
    main()
