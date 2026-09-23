"""Dataset utilities with explicit missingness masks and train-fitted scaling.

Every numeric modality is represented as a (value, mask) pair rather than
just a value: `mask=0` means "not measured for this compound", and the model
(core/model.py) and losses (core/losses.py) are built to skip masked-out
positions entirely rather than treating a missing measurement as zero."""
import numpy as np
import torch
from torch.utils.data import Dataset

PROPERTY=["mw","logp","tpsa","hbd","hba","rotatable_bonds",
          "ring_count","aromatic_ring_count","aliphatic_ring_count",
          "saturated_ring_count","heteroatom_count","fraction_csp3",
          "heavy_atom_count","formal_charge","radical_electron_count","valence_electron_count"]
HAZARD=["hazard_count","noael_log","loael_log"]
EXPOSURE=["exposure_count","product_count","use_category_count"]


def value_mask(row,columns,scaler=None):
    """Pull `columns` out of one cohort row as a (values, mask) tensor pair:
    mask is 1 where the value is finite (observed), values are z-scored by
    `scaler` if given, and NaN/inf entries are zeroed (they're masked out
    anyway, so the actual placeholder value never reaches the loss)."""
    values=np.array([row.get(c,np.nan) for c in columns],dtype=np.float32)
    mask=np.isfinite(values).astype(np.float32)
    if scaler is not None and len(columns):
        mean=np.asarray(scaler["mean"],dtype=np.float32); std=np.asarray(scaler["std"],dtype=np.float32)
        values=(values-mean)/std
    values=np.nan_to_num(values,nan=0.0,posinf=0.0,neginf=0.0)
    return torch.tensor(values,dtype=torch.float32),torch.tensor(mask,dtype=torch.float32)


def fit_group_scalers(df,groups):
    """Fit z-score scalers on training data only; hitcall remains binary/unscaled."""
    scalers=[]
    for i,columns in enumerate(groups):
        if i==1 or not columns:  # hitcall or empty modality
            scalers.append(None); continue
        arr=df[columns].to_numpy(dtype=np.float64)
        means=[]; stds=[]
        for j in range(arr.shape[1]):
            finite=arr[:,j][np.isfinite(arr[:,j])]
            if finite.size==0:
                means.append(0.0); stds.append(1.0)
            else:
                means.append(float(finite.mean()))
                sd=float(finite.std()); stds.append(sd if sd>1e-8 else 1.0)
        scalers.append({"mean":means,"std":stds})
    return scalers


class CompToxDataset(Dataset):
    """One row = one compound. __getitem__ returns (token_ids, values, masks)
    where `values`/`masks` are 6-element lists, one per modality group
    (physchem, hitcall, ac50, efficacy, hazard, exposure), matching the model's
    modality_dims order (core/model.py). `groups`/`scalers` from an existing
    checkpoint let inference reproduce the exact training-time feature
    layout even if the current dataframe has different/extra columns."""

    def __init__(self,df,tokenizer,max_length,scalers=None,groups=None):
        self.df=df.reset_index(drop=True); self.tokenizer=tokenizer; self.max_length=max_length
        self.groups=[
            [c for c in PROPERTY if c in df],
            [c for c in df if c.startswith("biohit__")],
            [c for c in df if c.startswith("bioac50__")],
            [c for c in df if c.startswith("bioeff__")],
            [c for c in HAZARD if c in df],
            [c for c in EXPOSURE if c in df],
        ]
        if groups is not None:
            self.groups=[list(g) for g in groups]
            for col in [c for g in groups for c in g]:
                if col not in self.df: self.df[col]=np.nan
        if not self.groups[1]: raise ValueError("No ToxCast/Tox21 hitcall columns found.")
        self.scalers=scalers if scalers is not None else [None]*6
        if len(self.scalers)!=6: raise ValueError("scalers must contain six entries.")

    @property
    def hit_columns(self): return self.groups[1]
    def __len__(self): return len(self.df)

    def __getitem__(self,i):
        row=self.df.iloc[i]
        ids=torch.tensor(self.tokenizer.encode(row["smiles"],self.max_length),dtype=torch.long)
        vals,masks=[],[]
        for cols,scaler in zip(self.groups,self.scalers):
            v,m=value_mask(row,cols,scaler); vals.append(v); masks.append(m)
        return ids,vals,masks
