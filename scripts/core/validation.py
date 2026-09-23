"""Reusable reconstruction evaluation on an explicitly held-out cohort."""
import numpy as np
import torch
from sklearn.metrics import (average_precision_score,f1_score,matthews_corrcoef,roc_auc_score)

NAMES=['properties','hitcall','ac50','efficacy','hazard','exposure']


def _binary_metrics(y,p,threshold):
    """Hitcall metrics on rare positives: AUROC alone is misleading, so also
    report AP, MCC, F1, sensitivity, specificity (at p>=0.5) and prevalence."""
    binary=(y>=threshold).astype(int); row={'positive_rate':float(binary.mean())}
    if len(np.unique(binary))==2:
        row['auroc']=float(roc_auc_score(binary,p)); row['average_precision']=float(average_precision_score(binary,p))
    pred=(p>=.5).astype(int)
    tp=int(((pred==1)&(binary==1)).sum()); tn=int(((pred==0)&(binary==0)).sum())
    fp=int(((pred==1)&(binary==0)).sum()); fn=int(((pred==0)&(binary==1)).sum())
    row['mcc']=float(matthews_corrcoef(binary,pred)); row['f1']=float(f1_score(binary,pred,zero_division=0))
    row['sensitivity']=tp/(tp+fn) if tp+fn else np.nan; row['specificity']=tn/(tn+fp) if tn+fp else np.nan
    return row


def _macro_per_assay(y,p,cols,threshold,min_pos=5,min_neg=5):
    """Macro-average AUROC/AP over assays (columns) that have enough of both classes."""
    aucs=[]; aps=[]
    for c in np.unique(cols):
        s=cols==c; b=(y[s]>=threshold).astype(int)
        if b.sum()>=min_pos and (len(b)-b.sum())>=min_neg:
            aucs.append(roc_auc_score(b,p[s])); aps.append(average_precision_score(b,p[s]))
    return {'macro_auroc_per_assay':float(np.mean(aucs)) if aucs else np.nan,
            'macro_ap_per_assay':float(np.mean(aps)) if aps else np.nan,'n_assays_scored':len(aucs)}


@torch.no_grad()
def reconstruction(model,loader,device,tok,keep=None,seed=43,mask_probability=.15,threshold=.9):
    """Score reconstruction of a FIXED set of held-out target entries.

    The target entries (a `mask_probability` random subset of each modality's
    observed values) depend only on `seed` and batch order -- never on `keep` --
    so every ablation is scored on exactly the same compound/column entries and
    the metrics are directly comparable. `keep[i]=False` additionally hides the
    whole modality i from the model's *input* (ablation), without changing
    which entries are scored. Reports MSE/MAE (+ median AE and a winsorized MSE
    that is robust to extreme residuals) per modality, and for hitcall AUROC,
    AP, MCC, F1, sensitivity, specificity and per-assay macro metrics."""
    model.eval()
    gen=torch.Generator().manual_seed(seed)
    truth=[[] for _ in NAMES]; predictions=[[] for _ in NAMES]; columns=[[] for _ in NAMES]
    for ids,values,masks in loader:
        ids=ids.to(device); values=[v.to(device) for v in values]; masks=[v.to(device) for v in masks]
        inputs=[]; imasks=[]; targets=[]
        for i,(v,m) in enumerate(zip(values,masks)):
            target=(torch.rand(v.shape,generator=gen).to(device)<mask_probability)&m.bool()
            hidden=m.bool() if keep is not None and not keep[i] else target
            x=v.clone(); im=m.clone(); x[hidden]=0; im[hidden]=0
            inputs.append(x); imasks.append(im); targets.append(target)
        out=model(ids,inputs,imasks)['numeric_predictions']
        for i,target in enumerate(targets):
            if target.any():
                pred=out[i].sigmoid() if i==1 else out[i]
                truth[i].extend(values[i][target].cpu().tolist()); predictions[i].extend(pred[target].cpu().tolist())
                columns[i].extend(target.nonzero()[:,1].cpu().tolist())
    rows=[]
    for i,name in enumerate(NAMES):
        y=np.asarray(truth[i]); p=np.asarray(predictions[i]); c=np.asarray(columns[i]); row={'modality':name,'n_observed_targets':len(y)}
        if len(y):
            row['mse']=float(np.mean((y-p)**2)); row['mae']=float(np.mean(np.abs(y-p)))
            row['median_ae']=float(np.median(np.abs(y-p)))
            lo,hi=np.percentile(y,[1,99]); row['mse_winsorized']=float(np.mean((np.clip(y,lo,hi)-np.clip(p,lo,hi))**2))
            if i==1:
                row.update(_binary_metrics(y,p,threshold)); row.update(_macro_per_assay(y,p,c,threshold))
        rows.append(row)
    return rows
