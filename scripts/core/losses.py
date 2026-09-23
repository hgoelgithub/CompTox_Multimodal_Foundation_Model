"""Masked losses: genuinely missing measurements never contribute.

`mask` is 1 for positions that should count toward the loss (i.e. the value
was both observed and selected for reconstruction) and 0 elsewhere; dividing
by mask.sum() rather than by the tensor's full size turns this into a proper
mean over only the positions that matter, so sparsely-measured modalities
aren't diluted by all their missing entries."""
import torch.nn.functional as F

def masked_mse(pred,target,mask):
    """Mean squared error, counting only masked (observed+selected) positions."""
    if target.numel()==0: return pred.new_tensor(0.0)
    return (((pred-target)**2)*mask).sum()/mask.sum().clamp_min(1.0)

def masked_balanced_bce(logits,target,mask,pos_weight):
    """Binary cross-entropy with a per-column pos_weight (for class-imbalanced
    hit calls -- see positive_weights() in 03_train_foundation_model.py),
    counting only masked positions."""
    loss=F.binary_cross_entropy_with_logits(
        logits,target,reduction="none",pos_weight=pos_weight)
    return (loss*mask).sum()/mask.sum().clamp_min(1.0)
