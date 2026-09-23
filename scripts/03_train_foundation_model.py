"""STEP 3 -- Pretrain the multimodal foundation model on comptox_v3.parquet
(from step 02). Learns a shared embedding by masked SMILES language modeling
plus masked numeric reconstruction across the six modalities (physchem,
hitcall, AC50, efficacy, hazard, exposure), with whole-modality dropout so
the model doesn't just learn to rely on one always-present input. Trains
with early stopping and saves the best checkpoint to checkpoints/comptox_v3_best.pt.
"""
from copy import deepcopy
import argparse, random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from core.data import CompToxDataset,fit_group_scalers
from core.losses import masked_mse,masked_balanced_bce
from core.model import CompToxFoundationModel
from core.tokenizer import SmilesTokenizer
from core.paths import PROCESSED_COMPTox,CHECKPOINTS,CONFIG
from core.table_io import read_table,table_exists

DATA=PROCESSED_COMPTox/"comptox_v3.parquet"


def seed_all(seed):
    """Seed every RNG this training run touches, for reproducible splits/masking."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def choose_device():
    """Prefer CUDA, then Apple Silicon MPS, else fall back to CPU."""
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends,"mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def mask_smiles(ids,tok,p):
    """Mask tokens and guarantee at least one selected token per non-empty row."""
    x=ids.clone(); labels=ids.clone()
    eligible=ids.ne(tok.pad_id)&ids.ne(tok.cls_id)
    selected=(torch.rand_like(ids.float())<p)&eligible
    if p>0:
        for i in range(ids.shape[0]):
            if eligible[i].any() and not selected[i].any():
                first=torch.where(eligible[i])[0][0]
                selected[i,first]=True
    x[selected]=tok.mask_id; labels[~selected]=-100
    return x,labels


def mask_numeric(x,observed,p):
    """Randomly zero out a fraction `p` of *observed* values in one numeric
    modality (masking missing values would be a no-op, so only observed ones
    are eligible). Returns the corrupted input, the mask that now also hides
    the corrupted entries, and which entries were corrupted (the reconstruction
    target)."""
    selected=(torch.rand_like(x)<p)&observed.bool()
    corrupted=x.clone(); input_mask=observed.clone()
    corrupted[selected]=0; input_mask[selected]=0
    return corrupted,input_mask,selected.float()


def apply_modality_dropout(values,masks,reconstruct_masks,observed_masks,p):
    """Hide entire modalities and ask the model to reconstruct all observed dropped values."""
    values=[x.clone() for x in values]; masks=[m.clone() for m in masks]
    reconstruct_masks=[m.clone() for m in reconstruct_masks]
    batch=values[0].shape[0]
    for i in range(len(values)):
        if values[i].shape[1]==0: continue
        drop=torch.rand(batch,device=values[i].device)<p
        if drop.any():
            values[i][drop]=0; masks[i][drop]=0
            reconstruct_masks[i][drop]=observed_masks[i][drop]
    return values,masks,reconstruct_masks


def positive_weights(df,cols,dev,active_threshold=0.90):
    """Class-balance weights for continuous invitrodb v4+ hit calls."""
    a=df[cols].to_numpy(dtype=np.float32)
    observed=np.isfinite(a)
    pos=np.sum(observed & (a>=active_threshold),axis=0)
    neg=np.sum(observed & (a<active_threshold),axis=0)
    return torch.tensor(np.clip(neg/np.maximum(pos,1),1,50),dtype=torch.float32,device=dev)


def run_epoch(model,loader,tok,cfg,pos_weight,dev,optimizer=None):
    """Run one pass over `loader`: mask SMILES + each numeric modality, apply
    modality dropout (training only), compute the weighted multi-task loss
    (masked-language-model + per-modality reconstruction), and step the
    optimizer if one is given. `optimizer=None` means eval mode (no dropout
    applied, no gradient step) -- used for both validation and --check-style
    inference."""
    training=optimizer is not None; model.train(training)
    tc=cfg["training"]; lw=cfg["loss_weights"]; total=0.0
    for ids,values,masks in loader:
        ids=ids.to(dev); values=[x.to(dev) for x in values]; masks=[x.to(dev) for x in masks]
        original=[x.clone() for x in values]; observed=[m.clone() for m in masks]
        ids,labels=mask_smiles(ids,tok,tc["smiles_mask_probability"])
        corrupted=[]; input_masks=[]; reconstruct=[]
        for x,m in zip(values,masks):
            a,b,c=mask_numeric(x,m,tc["numeric_mask_probability"])
            corrupted.append(a); input_masks.append(b); reconstruct.append(c)
        if training:
            corrupted,input_masks,reconstruct=apply_modality_dropout(
                corrupted,input_masks,reconstruct,observed,tc["modality_dropout_probability"])
        out=model(ids,corrupted,input_masks); pred=out["numeric_predictions"]
        selected=(labels!=-100)
        if selected.any():
            mlm=F.cross_entropy(out["mlm_logits"][selected],labels[selected])
        else:
            mlm=out["mlm_logits"].new_tensor(0.0)
        loss=lw["smiles"]*mlm
        loss+=lw["properties"]*masked_mse(pred[0],original[0],reconstruct[0])
        loss+=lw["hitcall"]*masked_balanced_bce(pred[1],original[1],reconstruct[1],pos_weight)
        for i,key in [(2,"ac50"),(3,"efficacy"),(4,"hazard"),(5,"exposure")]:
            loss+=lw[key]*masked_mse(pred[i],original[i],reconstruct[i])
        if not torch.isfinite(loss):
            raise FloatingPointError("Training loss became NaN/Inf; inspect input scaling and normalized values.")
        if training:
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),tc["gradient_clip"]); optimizer.step()
        total+=float(loss.detach().cpu())
    return total/max(len(loader),1)


def preflight():
    """Check the training cohort (step 02's output) exists before doing anything else."""
    if not table_exists(DATA):
        print("Prepared CompTox table is missing. Run scripts/02_build_training_cohort.py first.")
        return False
    print("Prepared CompTox table found.")
    return True


def main():
    """CLI entry point: load the cohort, split train/val by scaffold-agnostic
    grouping on SMILES (so near-duplicate rows can't leak across the split),
    fit the tokenizer and per-modality scalers on the training split only,
    then train with early stopping, saving the best checkpoint + training
    history."""
    parser=argparse.ArgumentParser()
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--config",type=__import__('pathlib').Path,default=CONFIG)
    parser.add_argument("--output-dir",type=__import__('pathlib').Path,default=CHECKPOINTS)
    parser.add_argument("--epochs",type=int)
    parser.add_argument("--max-compounds",type=int)
    parser.add_argument("--device",choices=['cpu','cuda','mps'])
    args=parser.parse_args()
    if args.check: raise SystemExit(0 if preflight() else 1)
    if not preflight(): raise FileNotFoundError("Prepared CompTox data are missing.")
    cfg=yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.epochs: cfg['training']['epochs']=args.epochs
    seed_all(cfg["seed"]); dev=torch.device(args.device) if args.device else choose_device()
    torch.set_num_threads(cfg['training'].get('cpu_threads',2))
    output_dir=args.output_dir; output_dir.mkdir(parents=True,exist_ok=True)
    print("Device:",dev)
    df=read_table(DATA)
    if args.max_compounds: df=df.sample(min(args.max_compounds,len(df)),random_state=cfg["seed"])
    if len(df)<4: raise ValueError("Need at least four compounds for a train/validation demonstration.")
    split=GroupShuffleSplit(n_splits=1,test_size=cfg['data']['validation_fraction'],random_state=cfg['seed'])
    train_idx,val_idx=next(split.split(df,groups=df['smiles']))
    train,val=df.iloc[train_idx],df.iloc[val_idx]
    if len(val)==0: raise ValueError("Validation split is empty; increase dataset size or validation_fraction.")
    tok=SmilesTokenizer().fit(train["smiles"].astype(str),cfg["model"]["vocab_size"])
    tok.save(output_dir/"tokenizer_v3.json")
    probe=CompToxDataset(train,tok,cfg["data"]["max_smiles_length"])
    scalers=fit_group_scalers(train,probe.groups)
    train_ds=CompToxDataset(train,tok,cfg["data"]["max_smiles_length"],scalers=scalers)
    val_ds=CompToxDataset(val,tok,cfg["data"]["max_smiles_length"],scalers=scalers)
    train_dl=DataLoader(train_ds,batch_size=cfg["training"]["batch_size"],shuffle=True,num_workers=0)
    val_dl=DataLoader(val_ds,batch_size=cfg["training"]["batch_size"],shuffle=False,num_workers=0)
    dims=[len(x) for x in train_ds.groups]; m=cfg["model"]
    model=CompToxFoundationModel(
        vocab_size=len(tok.vocab),modality_dims=dims,d_model=m["d_model"],n_heads=m["n_heads"],
        n_layers=m["layers"],feedforward_dim=m["feedforward_dim"],latent_dim=m["latent_dim"],
        dropout=m["dropout"],pad_id=tok.pad_id,max_positions=max(512,cfg["data"]["max_smiles_length"]),
    ).to(dev)
    pos_weight=positive_weights(train,train_ds.hit_columns,dev,cfg["data"]["toxcast_active_threshold"])
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["training"]["learning_rate"],weight_decay=cfg["training"]["weight_decay"])
    best=float("inf"); stale=0; history=[]
    for ep in range(1,cfg["training"]["epochs"]+1):
        tr=run_epoch(model,train_dl,tok,cfg,pos_weight,dev,optimizer)
        # Fixed corruption across validation epochs, without changing training RNG.
        cpu_rng=torch.get_rng_state(); numpy_rng=np.random.get_state(); py_rng=random.getstate()
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        mps_rng=torch.mps.get_rng_state() if dev.type=='mps' else None
        seed_all(cfg['seed']+1)
        with torch.no_grad(): va=run_epoch(model,val_dl,tok,cfg,pos_weight,dev)
        torch.set_rng_state(cpu_rng); np.random.set_state(numpy_rng); random.setstate(py_rng)
        if cuda_rng is not None: torch.cuda.set_rng_state_all(cuda_rng)
        if mps_rng is not None: torch.mps.set_rng_state(mps_rng)
        history.append({"epoch":ep,"train_loss":tr,"validation_loss":va})
        print(f"Epoch {ep:02d} | train={tr:.4f} | val={va:.4f}")
        if va<best:
            best=va; stale=0
            torch.save({"model_state":deepcopy(model.state_dict()),"config":cfg,"modality_dims":dims,
                        "best_validation_loss":best,"scalers":scalers,"groups":train_ds.groups,"vocab":tok.vocab,
                        "train_ids":train['DTXSID'].astype(str).tolist(),"validation_ids":val['DTXSID'].astype(str).tolist()},output_dir/"comptox_v3_best.pt")
        else: stale+=1
        if stale>=cfg["training"]["patience"]:
            print("Early stopping."); break
    pd.DataFrame(history).to_csv(output_dir/"training_history.csv",index=False)
    print("Best validation loss:",best)
    print("Saved:",output_dir/"comptox_v3_best.pt")


if __name__=="__main__": main()
