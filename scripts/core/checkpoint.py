"""Load model, tokenizer, feature order, and scaling as one inference contract."""
import torch
from core.model import CompToxFoundationModel
from core.tokenizer import SmilesTokenizer
from core.data import CompToxDataset


def load_checkpoint(path, device='cpu'):
    checkpoint=torch.load(path,map_location=device,weights_only=False)
    if not {'groups','vocab','config','scalers'}.issubset(checkpoint):
        raise ValueError('Legacy checkpoint lacks feature schema/tokenizer. Retrain with step 03.')
    cfg=checkpoint['config']; m=cfg['model']; tok=SmilesTokenizer(checkpoint['vocab'])
    model=CompToxFoundationModel(len(tok.vocab),checkpoint['modality_dims'],
        d_model=m['d_model'],n_heads=m['n_heads'],n_layers=m['layers'],
        feedforward_dim=m['feedforward_dim'],latent_dim=m['latent_dim'],dropout=m['dropout'],
        pad_id=tok.pad_id,max_positions=max(512,cfg['data']['max_smiles_length'])).to(device)
    model.load_state_dict(checkpoint['model_state'])
    return model,tok,checkpoint


def dataset_for_checkpoint(frame,tok,checkpoint):
    return CompToxDataset(frame,tok,checkpoint['config']['data']['max_smiles_length'],
        scalers=checkpoint['scalers'],groups=checkpoint['groups'])
