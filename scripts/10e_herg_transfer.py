"""Step 10e - Transfer to an independent endpoint: hERG channel blockade

The MEA transfer test (steps 10-10d) uses compounds mostly already inside the pretraining cohort, and its label
is small and coarse. This step is a cleaner test: an independent, published toxicity endpoint, with SMILES and a
binary label, drawn from **Therapeutics Data Commons** (TDC) rather than anything in this project's own data.

hERG (the cardiac potassium channel) blockade is a standard early safety screen: a chemical that blocks hERG risks
cardiac arrhythmia. The label is binary (blocker / non-blocker). Most of these 644 compounds are drugs and are
**not** in this project's 9,746-chemical pretraining cohort (only 77 are, of which 63 were in the pretraining
*training* split -- reported below, but never used as inputs here). So, unlike the MEA task, this is mostly a true
test of the embedding on chemistry the foundation model has never seen at all, computed from nothing but SMILES
and the 16 physchem descriptors -- the only inputs available for a genuinely new compound.

Source: Therapeutics Data Commons (https://tdcommons.ai), a curated version of Karim et al.'s hERG blocker data
(threshold: 10 uM). Downloaded once and cached; see download_herg().

Input : (downloads on first run) -> data/herg/herg_tdc.tab
Output: results/herg_transfer_metrics.csv (+ _methods.csv, _predictions.csv, _learning_curve.csv)
Run   : python scripts/10e_herg_transfer.py --seeds 5
"""

# %%
import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from torch.utils.data import Dataset

# %% [markdown]
# ## 1. Paths and settings

# %%
PROJECT_ROOT = Path(__file__).resolve().parents[1]
COHORT = PROJECT_ROOT / "data" / "processed" / "comptox_v3.parquet"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "comptox_v3_best.pt"
RESULTS = PROJECT_ROOT / "results"
HERG_DIR = PROJECT_ROOT / "data" / "herg"
HERG_RAW = HERG_DIR / "herg_tdc.tab"
HERG_URL = "https://dataverse.harvard.edu/api/access/datafile/4259588"   # TDC's hERG (Karim threshold) dataset

FEATURE_SETS = ["physchem", "morgan", "morgan_physchem", "emb_sp"]
MODELS = ["logistic", "random_forest", "gbm", "svc"]
PHYSCHEM_NAMES = ["mw", "logp", "tpsa", "hbd", "hba", "rotatable_bonds", "ring_count", "aromatic_ring_count",
                  "aliphatic_ring_count", "saturated_ring_count", "heteroatom_count", "fraction_csp3",
                  "heavy_atom_count", "formal_charge", "radical_electron_count", "valence_electron_count"]
KEEP_SMILES_PHYSCHEM = [True, False, False, False, False, False]

# %% [markdown]
# ## 2. Downloading the data
# A one-time download (cached to `data/herg/herg_tdc.tab`) of TDC's hERG dataset: one row per compound, its SMILES
# and a 0/1 blocker label. A handful of compounds appear more than once (different salts/sources of the same
# structure); duplicates that agree are merged, and the few that disagree on the label are dropped rather than
# guessed at.

# %%
def download_herg():
    HERG_DIR.mkdir(parents=True, exist_ok=True)
    if not HERG_RAW.exists():
        response = requests.get(HERG_URL, timeout=(30, 120))
        response.raise_for_status()
        HERG_RAW.write_bytes(response.content)
        print("Downloaded:", HERG_RAW)
    else:
        print("Already downloaded:", HERG_RAW)


def load_herg():
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    download_herg()
    raw = pd.read_csv(HERG_RAW, sep="\t", quotechar='"')
    raw = raw.rename(columns={"Drug_ID": "name", "Drug": "smiles", "Y": "label"})
    mol = raw["smiles"].map(lambda s: Chem.MolFromSmiles(str(s)))
    raw = raw[mol.notna()].copy()
    raw["smiles"] = [Chem.MolToSmiles(m) for m in mol[mol.notna()]]
    raw["inchikey"] = raw["smiles"].map(lambda s: Chem.MolToInchiKey(Chem.MolFromSmiles(s)))
    agreement = raw.groupby("inchikey")["label"].nunique()
    conflicting = agreement[agreement > 1].index
    if len(conflicting):
        print(f"Dropping {len(conflicting)} structures with conflicting labels across duplicate entries.")
    raw = raw[~raw["inchikey"].isin(conflicting)].drop_duplicates("inchikey").reset_index(drop=True)
    raw["label"] = raw["label"].astype(int)
    print(f"hERG: {len(raw)} unique compounds | blockers: {raw['label'].mean():.1%}")
    return raw

# %% [markdown]
# ## 3. Model code (only needed to compute the frozen embedding)

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
# ## 4. Physchem descriptors for an arbitrary new compound
# Unlike the MEA task, most hERG compounds are **not** in the pretraining cohort, so their physchem descriptors
# and embedding have to be computed from scratch here, with the exact same 16 RDKit descriptors, in the exact same
# order, that the foundation model was trained on (step 02). Every other modality (ToxCast, hazard, exposure) is
# simply left missing -- the model's masks handle that the same way they do inside the cohort.

# %%
def physchem_descriptors(smiles):
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return [np.nan] * len(PHYSCHEM_NAMES)
    return [
        Descriptors.MolWt(mol), Crippen.MolLogP(mol), rdMolDescriptors.CalcTPSA(mol),
        Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol), rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol), rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol), rdMolDescriptors.CalcNumSaturatedRings(mol),
        rdMolDescriptors.CalcNumHeteroatoms(mol), rdMolDescriptors.CalcFractionCSP3(mol),
        rdMolDescriptors.CalcNumHeavyAtoms(mol), Chem.GetFormalCharge(mol),
        Descriptors.NumRadicalElectrons(mol), Descriptors.NumValenceElectrons(mol),
    ]


def embed_new_compounds(frame, checkpoint_path=CHECKPOINT):
    """The frozen foundation-model embedding for each row of `frame` (any compounds, cohort or not), computed
    with only SMILES + physchem visible -- the only inputs available for a chemical the model was never trained on."""
    model, tokenizer, checkpoint = load_checkpoint(checkpoint_path)
    # Pre-create every modality column the checkpoint expects (all missing, since these compounds have no ToxCast/
    # hazard/exposure data) in one pass, rather than letting CompToxDataset add them one at a time -- with ~1,500
    # such columns, doing that column-by-column badly fragments the DataFrame and is slow.
    missing = [c for group in checkpoint["groups"] for c in group if c not in frame.columns]
    frame = pd.concat([frame, pd.DataFrame(np.nan, index=frame.index, columns=missing)], axis=1)
    data = tensors_for(frame, tokenizer, checkpoint, KEEP_SMILES_PHYSCHEM)
    model.eval()
    with torch.no_grad():
        return embed(model, data, np.arange(len(frame))).numpy()

# %% [markdown]
# ## 5. Checking overlap with the pretraining cohort
# Flags (does not exclude) any hERG compound that was in the foundation model's own training or validation split,
# so a reader can see how much of this test set the model may have encountered before, via its ToxCast/hazard/
# exposure data during pretraining -- never through this hERG label, which the model has no way to have seen.

# %%
def report_cohort_overlap(frame, checkpoint_path=CHECKPOINT):
    if not COHORT.exists():
        return
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    cohort = pd.read_parquet(COHORT, columns=["DTXSID", "smiles"])
    cohort["inchikey"] = cohort["smiles"].map(lambda s: Chem.MolToInchiKey(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else None)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_ids, valid_ids = set(checkpoint["train_ids"]), set(checkpoint["validation_ids"])
    merged = frame.merge(cohort, on="inchikey", how="left")
    print(f"hERG compounds also in the pretraining cohort: {merged['DTXSID'].notna().sum()} of {len(frame)} "
          f"({merged['DTXSID'].isin(train_ids).sum()} in its training split, "
          f"{merged['DTXSID'].isin(valid_ids).sum()} in its validation split). "
          "Their ToxCast/hazard/exposure data (not hERG) may have been seen in pretraining; "
          "this task never gives the model that data as input.")

# %% [markdown]
# ## 6. Scaffold split and metrics

# %%
def scaffold_of(smiles):
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


def training_subset(train_pool, seed, fraction):
    order = np.random.RandomState(1000 + seed).permutation(train_pool)
    return np.sort(order[:max(8, int(round(fraction * len(order))))])


def classification_metrics(y_true, p_pred):
    """AUROC, average precision, MCC and accuracy (predictions thresholded at 0.5)."""
    y_true = np.asarray(y_true, int)
    p_pred = np.asarray(p_pred, float)
    row = {}
    if len(np.unique(y_true)) == 2:
        row["auroc"] = float(roc_auc_score(y_true, p_pred))
        row["average_precision"] = float(average_precision_score(y_true, p_pred))
    else:
        row["auroc"] = row["average_precision"] = np.nan
    predicted = (p_pred >= 0.5).astype(int)
    row["accuracy"] = float((predicted == y_true).mean())
    row["mcc"] = float(matthews_corrcoef(y_true, predicted)) if len(np.unique(predicted)) > 1 else 0.0
    return row


def bootstrap_auroc_interval(y_true, p_pred, n=500, seed=0):
    rng = np.random.RandomState(seed)
    values = []
    for _ in range(n):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) == 2:
            values.append(roc_auc_score(y_true[idx], p_pred[idx]))
    if not values:
        return "n/a"
    return f"{np.percentile(values, 2.5):.2f}-{np.percentile(values, 97.5):.2f}"

# %% [markdown]
# ## 7. Models
# Four standard classifiers, all class-weighted (hERG blockers are the majority class, 69%, so this keeps a model
# from just predicting "blocker" for everyone) and none tuned on the test set:
# * **logistic**: L2-penalised logistic regression, penalty chosen by a fast internal 3-fold search over 8 values
#   of C, standardised and clipped inputs;
# * **random_forest**: 300 trees;
# * **gbm**: histogram gradient boosting;
# * **svc**: RBF-kernel support-vector classifier. Scored by its decision function rather than a calibrated
#   probability (`probability=True` would fit its own internal 5-fold model just for calibration); AUROC and
#   average precision only need a valid ranking, which the decision function already gives.
# Plus a neural head (small MLP, binary cross-entropy, early-stopped on a held-out 20% of the training fold) on
# top of the frozen embedding, the same design used in step 10d.

# %%
def make_classifier(name, seed):
    if name == "logistic":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             FunctionTransformer(np.clip, kw_args={"a_min": -5.0, "a_max": 5.0}),
                             LogisticRegressionCV(Cs=8, cv=3, class_weight="balanced", max_iter=2000, scoring="roc_auc"))
    if name == "random_forest":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features=0.33,
                                                    class_weight="balanced", n_jobs=4, random_state=seed))
    if name == "gbm":
        return make_pipeline(SimpleImputer(strategy="median"),
                             HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3,
                                                            min_samples_leaf=5, class_weight="balanced", random_state=seed))
    if name == "svc":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             FunctionTransformer(np.clip, kw_args={"a_min": -5.0, "a_max": 5.0}),
                             SVC(C=1.0, class_weight="balanced", random_state=seed))
    raise ValueError(name)


def classifier_score(fitted, X):
    """A score in [0, 1] for AUROC/AP/accuracy: predict_proba where available, else a sigmoid of the SVC decision
    function. The sigmoid is monotonic (AUROC/AP, which only need a ranking, are unaffected) and maps the decision
    function's actual class boundary (0) to exactly 0.5, so thresholding at 0.5 for accuracy/MCC stays correct."""
    if hasattr(fitted[-1], "predict_proba"):
        return fitted.predict_proba(X)[:, 1]
    return 1.0 / (1.0 + np.exp(-fitted.decision_function(X)))


def morgan_matrix(smiles, radius=2, n_bits=2048):
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


def fit_head_classifier(X, y, train_idx, eval_idx, epochs=100, seed=0, batch_size=32, patience=10):
    """A small MLP head trained on a precomputed feature matrix with binary cross-entropy, early-stopped on 20% of
    the training rows held out internally (the test set is never touched). See step 10d's regression version."""
    torch.manual_seed(seed)
    mean, std = np.nanmean(X[train_idx], axis=0), np.nanstd(X[train_idx], axis=0)
    scale = np.maximum(std, np.median(std[std > 1e-8]) if (std > 1e-8).any() else 1.0)
    Z = torch.tensor(np.nan_to_num((X - mean) / scale), dtype=torch.float32)
    target = torch.tensor(y, dtype=torch.float32)
    shuffled = np.random.RandomState(seed).permutation(train_idx)
    n_val = max(1, int(round(0.2 * len(shuffled))))
    val_rows, fit_rows = shuffled[:n_val], shuffled[n_val:]
    head = nn.Sequential(nn.Linear(X.shape[1], 128), nn.GELU(), nn.Dropout(0.20), nn.Linear(128, 1))
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.05)
    loss_fn = nn.BCEWithLogitsLoss()
    best_loss, best_state, stale = float("inf"), None, 0
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
            best_loss, best_state, stale = val_loss, {k: v.clone() for k, v in head.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= patience:
                break
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        return torch.sigmoid(head(Z[eval_idx]).squeeze(-1)).numpy()

# %% [markdown]
# ## 8. Run the comparison
# Same honest protocol as step 10d: a **fixed scaffold split** (no test compound shares a scaffold with a training
# compound), **5-fold CV grouped by scaffold** on the training compounds to compare methods and pick one without
# touching the test set, then a **learning curve** (10/25/50/100% of training compounds, several seeds) evaluated
# once on the fixed test set. AUROC is the primary metric (a 500-resample bootstrap gives its 95% interval).

# %%
def run_herg_compare(n_folds=5, fractions=(0.10, 0.25, 0.50, 1.0), seeds=5, features=FEATURE_SETS, models=MODELS,
                     include_nn_head=True, test_fraction=0.25, split_seed=0, checkpoint_path=CHECKPOINT, output=None):
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    frame = load_herg()
    report_cohort_overlap(frame, checkpoint_path)
    scaffolds = np.array([scaffold_of(s) for s in frame["smiles"]])
    train_idx, test_idx = scaffold_holdout(scaffolds, test_fraction, split_seed)
    y = frame["label"].to_numpy(int)
    print(f"Fixed scaffold split: {len(train_idx)} training / {len(test_idx)} test compounds "
          f"(test blocker rate {y[test_idx].mean():.1%}, training {y[train_idx].mean():.1%})")

    physchem = np.array([physchem_descriptors(s) for s in frame["smiles"]])
    matrices = {"physchem": physchem}
    if any(f.startswith("morgan") for f in features):
        fingerprint = morgan_matrix(frame["smiles"].tolist())
        matrices["morgan"], matrices["morgan_physchem"] = fingerprint, np.hstack([fingerprint, physchem])
    if "emb_sp" in features:
        embed_frame = frame.copy()
        for name, values in zip(PHYSCHEM_NAMES, physchem.T):
            embed_frame[name] = values
        matrices["emb_sp"] = embed_new_compounds(embed_frame, checkpoint_path)

    methods = {"majority_baseline": lambda fit, ev, seed: np.full(len(ev), y[fit].mean())}
    for feature_name in features:
        for model_name in models:
            methods[f"{model_name}|{feature_name}"] = (
                lambda fit, ev, seed, X=matrices[feature_name], m=model_name:
                    classifier_score(make_classifier(m, seed).fit(X[fit], y[fit]), X[ev]))
    if include_nn_head:
        for feature_name in ("physchem", "emb_sp"):
            if feature_name in matrices:
                methods[f"nn_head|{feature_name}"] = (
                    lambda fit, ev, seed, X=matrices[feature_name]: fit_head_classifier(X, y, fit, ev, seed=seed))

    folds = list(GroupKFold(n_splits=n_folds).split(train_idx, groups=scaffolds[train_idx]))
    cv_rows, curve_rows, seed0_predictions = [], [], {}
    for name, predict in methods.items():
        oof = np.full(len(train_idx), np.nan)
        for fit_pos, val_pos in folds:
            oof[val_pos] = predict(train_idx[fit_pos], train_idx[val_pos], 0)
        cv = classification_metrics(y[train_idx], oof)
        cv_rows.append({"method": name, "cv_auroc": cv["auroc"], "cv_mcc": cv["mcc"]})
        for fraction in fractions:
            for seed in range(seeds):
                subset = training_subset(train_idx, seed, fraction)
                predictions = predict(subset, test_idx, seed)
                if fraction == max(fractions) and seed == 0:
                    seed0_predictions[name] = predictions
                curve_rows.append({"method": name, "fraction": fraction, "seed": seed, "n_train": len(subset),
                                   **classification_metrics(y[test_idx], predictions)})
        at_full = np.mean([r["auroc"] for r in curve_rows if r["method"] == name and r["fraction"] == max(fractions)])
        print(f"  {name:<24} cv_auroc={cv['auroc']:.3f}  test_auroc@{max(fractions):.0%}={at_full:.3f}", flush=True)

    curve = pd.DataFrame(curve_rows)
    by_fraction = curve.groupby(["method", "fraction"], as_index=False)[["auroc", "average_precision", "mcc", "accuracy"]].mean()
    full = max(fractions)
    summary = pd.DataFrame(cv_rows).merge(by_fraction[by_fraction["fraction"] == full], on="method").drop(columns="fraction")
    summary["test_auroc_ci"] = summary["method"].map(lambda m: bootstrap_auroc_interval(y[test_idx], seed0_predictions[m]))
    summary = summary.sort_values("cv_auroc", ascending=False).reset_index(drop=True)
    chosen = summary[summary["method"] != "majority_baseline"].iloc[0]["method"]

    predictions_table = pd.DataFrame({"name": frame["name"].values[test_idx], "smiles": frame["smiles"].values[test_idx],
                                      "actual_blocker": y[test_idx]})
    for name, values in seed0_predictions.items():
        predictions_table[f"predicted_prob|{name}"] = values
    output = Path(output) if output else RESULTS / "herg_transfer_metrics.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    predictions_table.to_csv(output.with_name(output.stem + "_predictions.csv"), index=False)
    curve.to_csv(output.with_name(output.stem + "_learning_curve_runs.csv"), index=False)
    by_fraction.to_csv(output.with_name(output.stem + "_learning_curve.csv"), index=False)

    print(f"\nAll methods at 100% of the training data ({len(train_idx)} training, {len(test_idx)} test compounds), "
          "best 5-fold CV first:")
    print(summary.round(3).to_string(index=False))
    print("\nLearning curve: test AUROC by fraction of the training compounds, mean over "
          f"{seeds} seeds; rows sorted by the 100% score:")
    print(by_fraction.pivot(index="method", columns="fraction", values="auroc").round(3).sort_values(full, ascending=False).to_string())
    row = summary[summary["method"] == chosen].iloc[0]
    floor = summary[summary["method"] == "majority_baseline"].iloc[0]
    print(f"\nSelected by 5-fold CV: {chosen} -> test AUROC {row['auroc']:.3f} (95% CI {row['test_auroc_ci']}) "
          f"against {floor['auroc']:.3f} for the majority-class baseline.")
    print("Saved:", output)
    return summary, predictions_table, by_fraction

# %% Command line
def main():
    parser = argparse.ArgumentParser(description="Transfer test on an independent endpoint: hERG blockade (TDC).")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.10, 0.25, 0.50, 1.0])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--features", nargs="+", choices=FEATURE_SETS, default=FEATURE_SETS)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--no-nn-head", action="store_true")
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_herg_compare(args.folds, tuple(args.fractions), args.seeds, args.features, args.models,
                     not args.no_nn_head, args.test_fraction, args.split_seed, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
