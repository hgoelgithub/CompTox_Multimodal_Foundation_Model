"""STEP 10 -- Transfer learning: does the pretrained foundation model help
predict a real downstream endpoint, compared with training from scratch?

Endpoint: MEA neurotoxicity -- `n_mea_metric_hits` (how many of the MEA
network metrics a compound perturbs), regressed from structure. The 220 MEA
compounds are matched to the CompTox cohort through the DTXSID mapping.

Strategies (see configure_finetuning): scratch (random init, everything
trained), head_only (frozen pretrained model + new head), partial (head + last
transformer layers + fusion), full (everything, pretrained init). Each is run
at several fractions of the labelled training data and over several random
seeds; a scaffold-grouped hold-out set is shared by all strategies within a
seed. Model inputs are limited to SMILES (+ physchem, which is computed from
SMILES) by default, because a new compound would not have ToxCast data.

Commands:
  --show-plan   print the experiment matrix and exit
  (default)     run the matrix and write results/transfer_mea_metrics.csv

Caveat: ~200 labelled compounds, so results are noisy; read the mean and std
across seeds, not a single number."""
import argparse
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr

from core.checkpoint import build_model, dataset_for_checkpoint, load_checkpoint
from core.paths import CHECKPOINTS, MEA_PROCESSED, PROCESSED_COMPTox, RESULTS
from core.table_io import read_table

FRACTIONS = [0.10, 0.25, 0.50, 1.00]
STRATEGIES = ["scratch", "head_only", "partial", "full"]
# Which input modalities the model may see: physchem, hitcall, ac50, efficacy, hazard, exposure.
INPUT_SETS = {
    "smiles": [False] * 6,
    "smiles_physchem": [True, False, False, False, False, False],
    "all": [True] * 6,
}


class TaskHead(nn.Module):
    """Small MLP bolted onto the pretrained model's shared embedding to predict
    one downstream endpoint (default: a single scalar/logit output)."""

    def __init__(self, latent_dim=256, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, output_dim),
        )

    def forward(self, embedding):
        return self.net(embedding)


def configure_finetuning(model, task_head, strategy="head_only", last_n_layers=2):
    """Freeze/unfreeze `model`'s parameters in place according to `strategy`:
    head_only trains just the new TaskHead; partial also unfreezes the last
    `last_n_layers` transformer layers + the fusion block; full unfreezes
    everything. The task head's own parameters are always trainable."""
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
        raise ValueError("strategy must be head_only, partial, or full")

    for parameter in task_head.parameters():
        parameter.requires_grad = True


def load_mea_labels():
    """MEA compounds mapped to a DTXSID, with the endpoint `n_mea_metric_hits`."""
    summary = pd.read_csv(MEA_PROCESSED / "mea_compound_summary.csv")[["compound_id", "n_mea_metric_hits"]]
    mapping = pd.read_csv(MEA_PROCESSED / "mea_to_comptox_mapping_template.csv")[["compound_id", "DTXSID"]]
    labels = summary.merge(mapping, on="compound_id").dropna(subset=["DTXSID"])
    labels = labels[labels.DTXSID.astype(str).str.strip() != ""]
    # A DTXSID can map to several MEA compound ids; average their labels.
    return labels.groupby("DTXSID", as_index=False)["n_mea_metric_hits"].mean()


def scaffold_of(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(str(smiles))
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol is not None else ""


def scaffold_holdout(scaffolds, test_fraction, split_seed=0):
    """One fixed, deterministic scaffold split. Whole scaffolds go to one side,
    so no test compound shares a scaffold with a training compound. The largest
    scaffold groups are placed in train first (so test size lands close to
    `test_fraction` instead of overshooting when a big group is drawn); the
    remaining smaller/rarer scaffolds form the test set. `split_seed` only breaks
    ties between equal-sized groups. Returns (train_idx, test_idx)."""
    groups = {}
    for i, s in enumerate(scaffolds):
        groups.setdefault(s, []).append(i)
    keys = sorted(groups)
    np.random.RandomState(split_seed).shuffle(keys)
    keys.sort(key=lambda k: -len(groups[k]))  # stable: ties keep the shuffled order
    n_train_target = len(scaffolds) - int(round(test_fraction * len(scaffolds)))
    train, test = [], []
    for k in keys:
        (train if len(train) + len(groups[k]) <= n_train_target else test).extend(groups[k])
    return np.array(sorted(train)), np.array(sorted(test))


def tensors_for(frame, tok, checkpoint, keep):
    """Encode the whole frame once; modalities not in `keep` are hidden."""
    ds = dataset_for_checkpoint(frame, tok, checkpoint)
    items = [ds[i] for i in range(len(ds))]
    ids = torch.stack([x[0] for x in items])
    values = [torch.stack([x[1][j] for x in items]) for j in range(6)]
    masks = [torch.stack([x[2][j] for x in items]) for j in range(6)]
    for j in range(6):
        if not keep[j]:
            values[j] = torch.zeros_like(values[j])
            masks[j] = torch.zeros_like(masks[j])
    return ids, values, masks


def embed(model, data, idx):
    ids, values, masks = data
    return model(ids[idx], [v[idx] for v in values], [m[idx] for m in masks])["embedding"]


def regression_metrics(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    rho = spearmanr(y, p).statistic if np.std(p) > 0 else np.nan
    return {"rmse": float(np.sqrt(np.mean((y - p) ** 2))), "spearman": float(rho),
            "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan}


def fit_and_score(strategy, base_model, checkpoint, tok, data, y, train_idx, test_idx, epochs, seed, batch_size=32):
    """Fine-tune under `strategy` on train_idx, return metrics on test_idx (original units)."""
    torch.manual_seed(seed)
    y_mean, y_std = float(y[train_idx].mean()), float(y[train_idx].std() or 1.0)
    target = torch.tensor((y - y_mean) / y_std, dtype=torch.float32)
    if strategy == "scratch":
        model = build_model(checkpoint, tok)
    else:
        model = deepcopy(base_model)
    latent = checkpoint["config"]["model"]["latent_dim"]
    head = TaskHead(latent)
    configure_finetuning(model, head, "full" if strategy == "scratch" else strategy)

    if strategy == "head_only":  # frozen backbone: embed once, train the head only
        model.eval()
        with torch.no_grad():
            z_train, z_test = embed(model, data, train_idx), embed(model, data, test_idx)
        params = [{"params": head.parameters(), "lr": 1e-3}]
    else:
        backbone = [p for p in model.parameters() if p.requires_grad]
        lr_backbone = 1e-3 if strategy == "scratch" else 1e-4
        params = [{"params": head.parameters(), "lr": 1e-3}, {"params": backbone, "lr": lr_backbone}]
    optimizer = torch.optim.AdamW(params, weight_decay=0.01)
    loss_fn = nn.MSELoss()
    n = len(train_idx)
    losses = []  # mean training loss per epoch (standardized target units)
    for epoch in range(epochs):
        head.train()
        epoch_loss, batches = 0.0, 0
        order = np.random.RandomState(seed * 1000 + epoch).permutation(n)
        for start in range(0, n, batch_size):
            b = order[start:start + batch_size]
            if len(b) < 2:
                continue
            optimizer.zero_grad()
            if strategy == "head_only":
                pred = head(z_train[b]).squeeze(-1)
            else:
                model.train()
                pred = head(embed(model, data, train_idx[b])).squeeze(-1)
            loss = loss_fn(pred, target[train_idx[b]])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss)
            batches += 1
        losses.append(epoch_loss / max(batches, 1))
    head.eval()
    model.eval()
    with torch.no_grad():
        z = z_test if strategy == "head_only" else embed(model, data, test_idx)
        pred = head(z).squeeze(-1).numpy() * y_std + y_mean
    return regression_metrics(y[test_idx], pred), losses


def run_matrix(args):
    model, tok, checkpoint = load_checkpoint(args.checkpoint or CHECKPOINTS / "comptox_v3_best.pt")
    labels = load_mea_labels()
    cohort = read_table(PROCESSED_COMPTox / "comptox_v3.parquet")
    frame = cohort.merge(labels, on="DTXSID", how="inner").drop_duplicates("DTXSID").reset_index(drop=True)
    print(f"MEA compounds with labels in cohort: {len(frame)} (of {len(labels)} mapped)")
    if len(frame) < 30:
        raise ValueError("Too few labelled compounds for a meaningful transfer test.")
    y = frame["n_mea_metric_hits"].to_numpy(float)
    data = tensors_for(frame, tok, checkpoint, INPUT_SETS[args.inputs])
    scaffolds = [scaffold_of(s) for s in frame["smiles"]]
    rows = []
    curves = []
    # One fixed split shared by every seed, strategy and fraction. Seeds vary only
    # the training subsets (nested: 10% inside 25% inside 50% inside 100%) and the
    # model's random start.
    train_pool, test_idx = scaffold_holdout(scaffolds, args.test_fraction, args.split_seed)
    print(f"Fixed split: {len(train_pool)} train pool / {len(test_idx)} test compounds")
    output = args.output or RESULTS / "transfer_mea_metrics.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    split_path = output.with_name(output.stem + "_split.csv")
    pd.DataFrame({"DTXSID": frame["DTXSID"], "set": np.where(np.isin(np.arange(len(frame)), test_idx), "test", "train_pool")}
                 ).to_csv(split_path, index=False)
    for seed in range(args.seeds):
        rng = np.random.RandomState(1000 + seed)
        order = rng.permutation(train_pool)
        rows.append({"seed": seed, "strategy": "mean_baseline", "fraction": 1.0, "n_train": len(train_pool),
                     "n_test": len(test_idx), **regression_metrics(y[test_idx], np.full(len(test_idx), y[train_pool].mean()))})
        for fraction in args.fractions:
            train_idx = np.sort(order[:max(8, int(round(fraction * len(order))))])
            for strategy in args.strategies:
                m, losses = fit_and_score(strategy, model, checkpoint, tok, data, y, train_idx, test_idx, args.epochs, seed)
                rows.append({"seed": seed, "strategy": strategy, "fraction": fraction, "n_train": len(train_idx),
                             "n_test": len(test_idx), **m})
                curves.extend({"seed": seed, "strategy": strategy, "fraction": fraction, "epoch": e + 1, "train_loss": l}
                              for e, l in enumerate(losses))
                # Loss at 50% / 100% of training: if the last stretch is still falling, add epochs.
                mid, last = losses[len(losses) // 2 - 1], losses[-1]
                tail = losses[-max(1, len(losses) // 5):]
                slope = (tail[-1] - tail[0]) / max(len(tail) - 1, 1)
                print(f"seed={seed} fraction={fraction:>4.0%} {strategy:<9} rmse={m['rmse']:.3f} spearman={m['spearman']:.3f} "
                      f"| train_loss mid={mid:.3f} end={last:.3f} last-20%-slope/epoch={slope:+.4f}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(output, index=False)
    curves_path = output.with_name(output.stem + "_loss_curves.csv")
    pd.DataFrame(curves).to_csv(curves_path, index=False)
    summary = result.groupby(["strategy", "fraction"])[["rmse", "spearman", "r2"]].agg(["mean", "std"]).round(3)
    print("\nMean/std across seeds (test = scaffold-held-out compounds):")
    print(summary.to_string())
    print("Saved:", output)
    print("Saved loss curves:", curves_path)
    print("Saved split (test compounds):", split_path)


def main():
    """CLI entry point: print the plan (--show-plan) or run the transfer matrix."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-plan", action="store_true")
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=STRATEGIES)
    parser.add_argument("--fractions", nargs="+", type=float, default=FRACTIONS)
    parser.add_argument("--inputs", choices=list(INPUT_SETS), default="smiles_physchem")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--split-seed", type=int, default=0, help="tie-break seed for the fixed train/test split")
    parser.add_argument("--checkpoint", type=__import__("pathlib").Path)
    parser.add_argument("--output", type=__import__("pathlib").Path)
    args = parser.parse_args()
    if args.show_plan:
        print("Low-data transfer-learning experiment matrix")
        for fraction in args.fractions:
            for strategy in args.strategies:
                print(f"fraction={fraction:>4.0%}  strategy={strategy}")
        return
    torch.set_num_threads(4)
    run_matrix(args)


if __name__ == "__main__":
    main()
