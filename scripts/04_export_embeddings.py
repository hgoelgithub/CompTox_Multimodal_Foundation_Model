"""Step 4 - Export embeddings

Runs every compound in the cohort through the trained foundation model (no masking, everything the
model can see is provided) and saves its 256-dimensional shared embedding, plus a 2-D PCA projection
for quick plots.

Input : data/processed/comptox_v3.parquet, checkpoints/comptox_v3_best.pt
Output: data/processed/comptox_embeddings.parquet   (DTXSID, PC1, PC2, z000 ... z255)
Run   : python scripts/04_export_embeddings.py [--check]
"""

# %%
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset

# %% [markdown]
# ## 1. Paths

# %%
PROJECT_ROOT = Path(__file__).resolve().parents[1]
COHORT = PROJECT_ROOT / "data" / "processed" / "comptox_v3.parquet"
EMBEDDINGS = PROJECT_ROOT / "data" / "processed" / "comptox_embeddings.parquet"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "comptox_v3_best.pt"

# %% [markdown]
# ## 2. Model code
# The model classes must match the ones used in training (step 03) so the saved weights load correctly.

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
# ## 3. Compute the embeddings
# The trained model is put in evaluation mode (no dropout, no masking). Each compound's embedding is
# the fused vector that combines its SMILES and all numeric modalities. PCA reduces the embeddings to
# two dimensions purely for visualisation; retrieval and transfer use the full vectors.

# %%
def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def check_inputs():
    missing = [p for p in (COHORT, CHECKPOINT) if not p.exists()]
    for p in missing:
        print("missing:", p)
    if not missing:
        print("Inputs found.")
    return not missing


@torch.no_grad()
def embed_all(model, loader, device):
    model.eval()
    chunks = []
    for ids, values, masks in loader:
        out = model(ids.to(device), [v.to(device) for v in values], [m.to(device) for m in masks])
        chunks.append(out["embedding"].cpu().numpy())
    return np.concatenate(chunks, axis=0)


def export_embeddings():
    if not check_inputs():
        raise FileNotFoundError("Run steps 02 and 03 first.")
    device = choose_device()
    model, tokenizer, checkpoint = load_checkpoint(CHECKPOINT, device)
    torch.set_num_threads(checkpoint["config"]["training"].get("cpu_threads", 2))

    cohort = pd.read_parquet(COHORT)
    dataset = dataset_for_checkpoint(cohort, tokenizer, checkpoint)
    loader = DataLoader(dataset, batch_size=checkpoint["config"]["training"]["batch_size"], shuffle=False)
    z = embed_all(model, loader, device)
    pcs = PCA(n_components=2).fit_transform(z)

    table = pd.concat([
        pd.DataFrame({"DTXSID": cohort["DTXSID"].astype(str), "PC1": pcs[:, 0], "PC2": pcs[:, 1]}),
        pd.DataFrame(z, columns=[f"z{i:03d}" for i in range(z.shape[1])], index=cohort.index),
    ], axis=1)
    table.to_parquet(EMBEDDINGS, index=False)
    print("Saved embeddings:", table.shape, "->", EMBEDDINGS)

# %% Command line
def main():
    parser = argparse.ArgumentParser(description="Export foundation-model embeddings for every cohort compound.")
    parser.add_argument("--check", action="store_true", help="only verify that the inputs exist")
    args = parser.parse_args()
    if args.check:
        raise SystemExit(0 if check_inputs() else 1)
    export_embeddings()


if __name__ == "__main__":
    main()
