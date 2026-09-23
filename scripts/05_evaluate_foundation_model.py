"""STEP 5 -- Evaluate what the pretrained foundation model actually learned.

Commands:
  retrieval    nearest neighbors of one DTXSID in learned embedding space
               (uses step 04's embeddings)
  plan         print the recommended low-data transfer-learning experiment
               matrix (fractions x fine-tuning strategies) -- a checklist,
               doesn't run any experiments itself
  ablation     re-run reconstruction on the held-out validation set with only
               some input modalities visible, to see which modalities the
               model actually leans on
  novelty      score the full model separately on validation compounds whose
               scaffold does/doesn't appear in training (leakage check, no retraining)
  cross-modal  reconstruct each modality from just the *other* modalities, to
               see how much cross-modal information the shared embedding
               actually captures
"""
import argparse
import re

from core.evaluation import nearest_neighbors
from core.paths import PROCESSED_COMPTox
from core.table_io import read_table, table_exists

EMBEDDINGS = PROCESSED_COMPTox / "comptox_embeddings.parquet"
FRACTIONS = [0.10, 0.25, 0.50, 1.00]
STRATEGIES = ["scratch", "head_only", "partial", "full"]
ABLATIONS = {
    "smiles": [False, False, False, False, False, False],
    "smiles_physchem": [True, False, False, False, False, False],
    "smiles_physchem_bioactivity": [True, True, True, True, False, False],
    "all_modalities": [True, True, True, True, True, True],
    # Leave-one-out from the full model: unique contribution of each modality.
    # Order: physchem, hitcall, ac50, efficacy, hazard, exposure. "bioactivity"
    # = hitcall + ac50 + efficacy (the three ToxCast/Tox21 assay modalities).
    "full_minus_physchem": [False, True, True, True, True, True],
    "full_minus_bioactivity": [True, False, False, False, True, True],
    "full_minus_hazard": [True, True, True, True, False, True],
    "full_minus_exposure": [True, True, True, True, True, False],
}
CROSS_MODAL_TARGETS = ["physchem", "hitcall", "ac50", "efficacy", "hazard", "exposure"]


def retrieval(dtxsid, k):
    """Print the k chemicals whose embedding is closest (cosine distance) to
    the given DTXSID's -- a sanity check that "similar in embedding space"
    lines up with chemical/toxicological intuition."""
    if not table_exists(EMBEDDINGS):
        raise FileNotFoundError("Run scripts/04_export_embeddings.py first.")
    df = read_table(EMBEDDINGS)
    if str(dtxsid) not in set(df["DTXSID"].astype(str)):
        raise ValueError(f"DTXSID {dtxsid} is not in the embedding table.")
    zcols = [c for c in df if re.match(r"^z\d+$", c)]
    if not zcols:
        raise ValueError("Embedding columns z000... were not found.")
    i = df.index[df["DTXSID"].astype(str).eq(str(dtxsid))][0]
    result = nearest_neighbors(df[zcols].to_numpy(), df["DTXSID"].astype(str).tolist(), i, k)
    print(result.to_string(index=False))


def main():
    """CLI entry point dispatching to the four commands described in the
    module docstring."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("retrieval")
    r.add_argument("--dtxsid", required=True)
    r.add_argument("--k", type=int, default=5)
    sub.add_parser("plan")
    for name in ['ablation','cross-modal','novelty']:
        command=sub.add_parser(name)
        command.add_argument('--checkpoint',type=__import__('pathlib').Path)
        command.add_argument('--output',type=__import__('pathlib').Path)
    args = parser.parse_args()

    if args.command == "retrieval":
        retrieval(args.dtxsid, args.k)
    elif args.command == "plan":
        print("Low-data transfer-learning experiment matrix")
        for fraction in FRACTIONS:
            for strategy in STRATEGIES:
                print(f"fraction={fraction:>4.0%}  strategy={strategy}")
    elif args.command in {'ablation','cross-modal','novelty'}:
        import pandas as pd
        import torch
        from torch.utils.data import DataLoader
        from core.paths import CHECKPOINTS,RESULTS
        from core.checkpoint import load_checkpoint,dataset_for_checkpoint
        from core.validation import reconstruction,NAMES
        model,tok,checkpoint=load_checkpoint(args.checkpoint or CHECKPOINTS/'comptox_v3_best.pt')
        torch.set_num_threads(2)
        frame=read_table(PROCESSED_COMPTox/'comptox_v3.parquet')
        frame=frame[frame.DTXSID.astype(str).isin(checkpoint['validation_ids'])]
        if frame.empty: raise ValueError('No checkpoint validation compounds in cohort')
        if args.command=='novelty':
            # Split validation compounds by whether any training compound shares
            # their Bemis-Murcko scaffold, then score the full model on each group.
            from rdkit import Chem, RDLogger
            from rdkit.Chem.Scaffolds import MurckoScaffold
            RDLogger.DisableLog('rdApp.*')
            def scaffold(smiles):
                mol=Chem.MolFromSmiles(str(smiles))
                return MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol is not None else None
            everything=read_table(PROCESSED_COMPTox/'comptox_v3.parquet')
            train_smiles=everything[everything.DTXSID.astype(str).isin(checkpoint['train_ids'])].smiles
            train_scaffolds={scaffold(x) for x in train_smiles.unique()}-{None}
            frame=frame.assign(_novel=[scaffold(x) not in train_scaffolds for x in frame.smiles])
            rows=[]
            for label,part in (('scaffold_novel',frame[frame._novel]),('scaffold_seen',frame[~frame._novel])):
                if part.empty: continue
                part_loader=DataLoader(dataset_for_checkpoint(part.drop(columns='_novel'),tok,checkpoint),batch_size=64)
                metrics=reconstruction(model,part_loader,'cpu',tok,threshold=checkpoint['config']['data']['toxcast_active_threshold'])
                rows.extend({'group':label,'n_compounds':len(part),**r} for r in metrics)
            output=args.output or RESULTS/'novelty_metrics.csv'; output.parent.mkdir(parents=True,exist_ok=True)
            result=pd.DataFrame(rows); result.to_csv(output,index=False); print(result.to_string(index=False)); print('Saved:',output)
            return
        loader=DataLoader(dataset_for_checkpoint(frame,tok,checkpoint),batch_size=64)
        cases=ABLATIONS if args.command=='ablation' else {name:[j!=i for j in range(6)] for i,name in enumerate(NAMES)}
        rows=[]
        for name,keep in cases.items():
            metrics=reconstruction(model,loader,'cpu',tok,keep=keep,threshold=checkpoint['config']['data']['toxcast_active_threshold'])
            if args.command=='cross-modal': metrics=[r for r in metrics if r['modality']==name]
            rows.extend({'experiment':name,**r} for r in metrics)
        output=args.output or RESULTS/(args.command+'_metrics.csv'); output.parent.mkdir(parents=True,exist_ok=True)
        result=pd.DataFrame(rows)
        # Every experiment must be scored on the identical target entries.
        spread=result.groupby('modality')['n_observed_targets'].nunique()
        if (spread>1).any(): raise RuntimeError(f'Target sets differ across experiments: {spread[spread>1].index.tolist()}')
        result.to_csv(output,index=False); print(result.to_string(index=False)); print('Saved:',output)



if __name__ == "__main__":
    main()
