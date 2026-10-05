"""Step 5 - Evaluate the pretrained model

Four checks of what the foundation model has learned, all on compounds it was *not* trained on
(the checkpoint's validation split):

  retrieval    nearest neighbours of one chemical in embedding space (a sanity check)
  ablation     reconstruction accuracy when only some input modalities are visible
  cross-modal  reconstruct each modality from only the *other* five
  novelty      reconstruction accuracy for validation compounds whose scaffold does / does not
               appear in training (a leakage check that needs no retraining)

Input : checkpoints/comptox_v3_best.pt, data/processed/comptox_v3.parquet, comptox_embeddings.parquet
Output: results/ablation_metrics.csv, cross_modal_metrics.csv, novelty_metrics.csv
Run   : python scripts/05_evaluate_foundation_model.py retrieval --dtxsid DTXSID8022292
        python scripts/05_evaluate_foundation_model.py ablation | cross-modal | novelty
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
from sklearn.metrics import (average_precision_score, f1_score, matthews_corrcoef,
                             pairwise_distances, roc_auc_score)
from torch.utils.data import DataLoader, Dataset

# %% [markdown]
# ## 1. Paths

# %%
PROJECT_ROOT = Path(__file__).resolve().parents[1]
COHORT = PROJECT_ROOT / "data" / "processed" / "comptox_v3.parquet"
EMBEDDINGS = PROJECT_ROOT / "data" / "processed" / "comptox_embeddings.parquet"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "comptox_v3_best.pt"
RESULTS = PROJECT_ROOT / "results"

# %% [markdown]
# ## 2. Model code
# These definitions must match training (step 03) so the checkpoint loads.

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
# ## 3. Retrieval: which chemicals sit next to each other?
# Every compound is a point in the 256-dimensional embedding space. Cosine distance between two
# points is small when the model considers the chemicals similar. Looking at the nearest neighbours of
# a familiar chemical is a quick way to see whether that notion of similarity is chemically sensible.

# %%
def nearest_neighbors(embeddings, ids, query_index, k=5):
    """The k nearest compounds (cosine distance) to the compound at `query_index`."""
    embeddings = np.asarray(embeddings)
    if len(ids) != len(embeddings):
        raise ValueError("ids and embeddings must have the same number of rows.")
    if not 0 <= query_index < len(ids):
        raise IndexError("query_index is out of range.")
    distances = pairwise_distances(embeddings[query_index:query_index + 1], embeddings, metric="cosine")[0]
    order = [i for i in np.argsort(distances) if i != query_index][:max(0, k)]
    return pd.DataFrame({"DTXSID": [ids[i] for i in order], "cosine_distance": [float(distances[i]) for i in order]})


def retrieval(dtxsid, k=5):
    if not EMBEDDINGS.exists():
        raise FileNotFoundError("Run step 04 first.")
    table = pd.read_parquet(EMBEDDINGS)
    ids = table["DTXSID"].astype(str).tolist()
    if str(dtxsid) not in ids:
        raise ValueError(f"{dtxsid} is not in the embedding table.")
    z_columns = [c for c in table if c.startswith("z") and c[1:].isdigit()]
    result = nearest_neighbors(table[z_columns].to_numpy(), ids, ids.index(str(dtxsid)), k)
    print(result.to_string(index=False))
    return result

# %% [markdown]
# ## 4. Reconstruction score
# How well can the model restore values it cannot see?
# * A **fixed** random 15% of each modality's observed values are chosen as targets. The choice depends
#   only on a seed and the batch order, never on which modalities are visible, so every experiment is
#   scored on exactly the same entries and the numbers are directly comparable.
# * `keep[i] = False` hides the *whole* modality `i` from the model's input (an ablation) without
#   changing which entries are scored.
# * Every modality gets MSE, MAE, median absolute error and a winsorised MSE (robust to a few extreme
#   values). For hit calls, which are rare-positive, there is also AUROC, average precision, MCC, F1,
#   sensitivity, specificity and a per-assay macro average.

# %%
def binary_metrics(y, p, threshold):
    """Hit-call classification metrics (positive = hit call >= threshold; predictions thresholded at 0.5)."""
    truth = (y >= threshold).astype(int)
    row = {"positive_rate": float(truth.mean())}
    if len(np.unique(truth)) == 2:
        row["auroc"] = float(roc_auc_score(truth, p))
        row["average_precision"] = float(average_precision_score(truth, p))
    pred = (p >= 0.5).astype(int)
    tp = int(((pred == 1) & (truth == 1)).sum())
    tn = int(((pred == 0) & (truth == 0)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    row["mcc"] = float(matthews_corrcoef(truth, pred))
    row["f1"] = float(f1_score(truth, pred, zero_division=0))
    row["sensitivity"] = tp / (tp + fn) if tp + fn else np.nan
    row["specificity"] = tn / (tn + fp) if tn + fp else np.nan
    return row


def macro_per_assay(y, p, assay_index, threshold, min_each=5):
    """AUROC / average precision averaged over assays with at least `min_each` hits and non-hits."""
    aucs, aps = [], []
    for assay in np.unique(assay_index):
        rows = assay_index == assay
        truth = (y[rows] >= threshold).astype(int)
        if truth.sum() >= min_each and (len(truth) - truth.sum()) >= min_each:
            aucs.append(roc_auc_score(truth, p[rows]))
            aps.append(average_precision_score(truth, p[rows]))
    return {"macro_auroc_per_assay": float(np.mean(aucs)) if aucs else np.nan,
            "macro_ap_per_assay": float(np.mean(aps)) if aps else np.nan, "n_assays_scored": len(aucs)}


@torch.no_grad()
def reconstruction(model, loader, keep=None, seed=43, mask_probability=0.15, threshold=0.9, device="cpu"):
    """Reconstruction metrics per modality on a fixed set of masked target entries."""
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    truth = [[] for _ in MODALITY_NAMES]
    predictions = [[] for _ in MODALITY_NAMES]
    columns = [[] for _ in MODALITY_NAMES]
    for ids, values, masks in loader:
        ids = ids.to(device)
        values = [v.to(device) for v in values]
        masks = [m.to(device) for m in masks]
        inputs, input_masks, targets = [], [], []
        for i, (v, m) in enumerate(zip(values, masks)):
            target = (torch.rand(v.shape, generator=generator).to(device) < mask_probability) & m.bool()
            hidden = m.bool() if keep is not None and not keep[i] else target
            x, im = v.clone(), m.clone()
            x[hidden] = 0
            im[hidden] = 0
            inputs.append(x)
            input_masks.append(im)
            targets.append(target)
        outputs = model(ids, inputs, input_masks)["numeric_predictions"]
        for i, target in enumerate(targets):
            if target.any():
                pred = outputs[i].sigmoid() if i == 1 else outputs[i]
                truth[i].extend(values[i][target].cpu().tolist())
                predictions[i].extend(pred[target].cpu().tolist())
                columns[i].extend(target.nonzero()[:, 1].cpu().tolist())
    rows = []
    for i, name in enumerate(MODALITY_NAMES):
        y, p, c = np.asarray(truth[i]), np.asarray(predictions[i]), np.asarray(columns[i])
        row = {"modality": name, "n_observed_targets": len(y)}
        if len(y):
            row["mse"] = float(np.mean((y - p) ** 2))
            row["mae"] = float(np.mean(np.abs(y - p)))
            row["median_ae"] = float(np.median(np.abs(y - p)))
            lo, hi = np.percentile(y, [1, 99])
            row["mse_winsorized"] = float(np.mean((np.clip(y, lo, hi) - np.clip(p, lo, hi)) ** 2))
            if i == 1:
                row.update(binary_metrics(y, p, threshold))
                row.update(macro_per_assay(y, p, c, threshold))
        rows.append(row)
    return rows

# %% [markdown]
# ## 5. The three evaluation experiments
# All three score the checkpoint's held-out **validation** compounds.
#
# * **Ablation.** Hide some modalities entirely and see how much reconstruction degrades. The first four
#   settings add modalities step by step; the `full_minus_*` settings remove one modality from the full
#   input to measure its *unique* contribution.
#   **Shortcut warning for hit calls.** An AC50 (and its efficacy) exists only for *active* calls. If a hit call
#   is hidden but the same assay's AC50 is still visible, the model can read "active" off the AC50's mere
#   presence. The settings that show AC50/efficacy therefore over-state how well hit calls are predicted.
#   `physchem_hitcall` (hit calls visible, AC50/efficacy hidden) is the honest measure of predicting a hit call
#   from a chemical's other hit calls; `physchem_ac50_efficacy` (hit calls hidden) exposes the shortcut.
# * **Cross-modal.** For each modality in turn, hide it completely and reconstruct it from the other five.
# * **Novelty.** Split the validation compounds by whether any training compound shares their Bemis-Murcko
#   scaffold, and score each group separately. If scaffold-novel compounds score about as well as the rest,
#   results are not inflated by near-duplicates of training compounds.

# %%
# Which modalities stay visible, in the order: physchem, hitcall, ac50, efficacy, hazard, exposure.
ABLATIONS = {
    "smiles": [False, False, False, False, False, False],
    "smiles_physchem": [True, False, False, False, False, False],
    "physchem_hitcall": [True, True, False, False, False, False],            # hit calls only: no AC50 / efficacy shortcut
    "physchem_ac50_efficacy": [True, False, True, True, False, False],       # AC50 + efficacy only: exposes the shortcut
    "smiles_physchem_bioactivity": [True, True, True, True, False, False],
    "all_modalities": [True, True, True, True, True, True],
    "full_minus_physchem": [False, True, True, True, True, True],
    "full_minus_bioactivity": [True, False, False, False, True, True],   # bioactivity = hitcall + ac50 + efficacy
    "full_minus_hazard": [True, True, True, True, False, True],
    "full_minus_exposure": [True, True, True, True, True, False],
}


def setup_validation(checkpoint_path=CHECKPOINT):
    """Load the checkpoint and select its held-out validation compounds from the cohort."""
    model, tokenizer, checkpoint = load_checkpoint(checkpoint_path)
    torch.set_num_threads(2)
    cohort = pd.read_parquet(COHORT)
    frame = cohort[cohort["DTXSID"].astype(str).isin(checkpoint["validation_ids"])]
    if frame.empty:
        raise ValueError("None of the checkpoint's validation compounds are in the cohort.")
    return model, tokenizer, checkpoint, frame


def make_loader(frame, tokenizer, checkpoint):
    return DataLoader(dataset_for_checkpoint(frame, tokenizer, checkpoint), batch_size=64)


def save_table(rows, name):
    RESULTS.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    path = RESULTS / name
    table.to_csv(path, index=False)
    print(table.to_string(index=False))
    print("Saved:", path)
    return table


def ablation(checkpoint_path=CHECKPOINT):
    model, tokenizer, checkpoint, frame = setup_validation(checkpoint_path)
    loader = make_loader(frame, tokenizer, checkpoint)
    threshold = checkpoint["config"]["data"]["toxcast_active_threshold"]
    rows = []
    for name, keep in ABLATIONS.items():
        rows += [{"experiment": name, **r} for r in reconstruction(model, loader, keep, threshold=threshold)]
    table = pd.DataFrame(rows)
    spread = table.groupby("modality")["n_observed_targets"].nunique()  # every experiment must score identical targets
    if (spread > 1).any():
        raise RuntimeError(f"Target sets differ across experiments: {spread[spread > 1].index.tolist()}")
    return save_table(rows, "ablation_metrics.csv")


def cross_modal(checkpoint_path=CHECKPOINT):
    model, tokenizer, checkpoint, frame = setup_validation(checkpoint_path)
    loader = make_loader(frame, tokenizer, checkpoint)
    threshold = checkpoint["config"]["data"]["toxcast_active_threshold"]
    rows = []
    for i, name in enumerate(MODALITY_NAMES):
        keep = [j != i for j in range(6)]  # hide modality i, keep the other five
        rows += [{"experiment": name, **r} for r in reconstruction(model, loader, keep, threshold=threshold)
                 if r["modality"] == name]
    return save_table(rows, "cross_modal_metrics.csv")


def novelty(checkpoint_path=CHECKPOINT):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")

    def scaffold(smiles):
        mol = Chem.MolFromSmiles(str(smiles))
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol is not None else None

    model, tokenizer, checkpoint, frame = setup_validation(checkpoint_path)
    cohort = pd.read_parquet(COHORT)
    train_smiles = cohort[cohort["DTXSID"].astype(str).isin(checkpoint["train_ids"])]["smiles"].unique()
    train_scaffolds = {scaffold(s) for s in train_smiles} - {None}
    is_novel = np.array([scaffold(s) not in train_scaffolds for s in frame["smiles"]])
    threshold = checkpoint["config"]["data"]["toxcast_active_threshold"]
    rows = []
    for label, part in (("scaffold_novel", frame[is_novel]), ("scaffold_seen", frame[~is_novel])):
        if part.empty:
            continue
        loader = make_loader(part, tokenizer, checkpoint)
        rows += [{"group": label, "n_compounds": len(part), **r}
                 for r in reconstruction(model, loader, threshold=threshold)]
    return save_table(rows, "novelty_metrics.csv")

# %% Command line
def main():
    parser = argparse.ArgumentParser(description="Evaluate the pretrained foundation model.")
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("retrieval", help="nearest neighbours of one chemical")
    r.add_argument("--dtxsid", required=True)
    r.add_argument("--k", type=int, default=5)
    for name in ("ablation", "cross-modal", "novelty"):
        sub.add_parser(name).add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args()
    if args.command == "retrieval":
        retrieval(args.dtxsid, args.k)
    else:
        {"ablation": ablation, "cross-modal": cross_modal, "novelty": novelty}[args.command](args.checkpoint)


if __name__ == "__main__":
    main()
