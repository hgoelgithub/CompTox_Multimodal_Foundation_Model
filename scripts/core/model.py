"""Multimodal architecture: SMILES + six numeric modalities -> shared embedding.

  SMILES tokens --> Transformer encoder --> [CLS] embedding --\
  physchem (value, mask) --> NumericEncoder ---------------------\
  hitcall  (value, mask) --> NumericEncoder ----------------------> concat -> fusion MLP -> shared embedding
  AC50     (value, mask) --> NumericEncoder ---------------------/
  efficacy (value, mask) --> NumericEncoder --------------------/
  hazard   (value, mask) --> NumericEncoder -------------------/
  exposure (value, mask) --> NumericEncoder ------------------/

The shared embedding then feeds back out to a masked-language-model head
(next-token/character prediction over masked SMILES positions) and one
regression/classification head per numeric modality, for the pretraining
objectives in 03_train_foundation_model.py.
"""
import torch
import torch.nn as nn


class NumericEncoder(nn.Module):
    """Encodes one modality's (values, mask) pair into a latent_dim vector.
    Feeding the mask in alongside the values (not just zeroing missing
    entries) lets the network tell "observed and zero" apart from "missing".
    A modality with zero columns (e.g. a checkpoint trained without hazard
    data) becomes a no-op that always returns a zero vector."""

    def __init__(self,input_dim,latent_dim):
        super().__init__()
        self.input_dim=input_dim
        self.latent_dim=latent_dim
        self.net=None if input_dim==0 else nn.Sequential(
            nn.Linear(input_dim*2,latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim,latent_dim),
        )

    def forward(self,values,mask):
        """Encode (values, mask) -> a latent_dim vector, or a zero vector if
        this modality has no columns at all."""
        if self.net is None:
            return torch.zeros(values.shape[0],self.latent_dim,device=values.device,dtype=values.dtype)
        return self.net(torch.cat([values,mask],dim=-1))


class CompToxFoundationModel(nn.Module):
    """The full multimodal model: a SMILES transformer encoder plus one
    NumericEncoder per modality, fused into one shared embedding that is then
    read back out for masked-language-model + per-modality reconstruction
    losses (see 03_train_foundation_model.py's run_epoch)."""

    def __init__(self,vocab_size,modality_dims,d_model=256,n_heads=8,n_layers=4,
                 feedforward_dim=768,latent_dim=256,dropout=0.1,pad_id=0,
                 max_positions=512):
        super().__init__()
        if len(modality_dims) != 6:
            raise ValueError("modality_dims must contain 6 values: PhysChem, hitcall, AC50, efficacy, hazard, exposure.")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.pad_id=pad_id
        self.max_positions=max_positions
        self.token=nn.Embedding(vocab_size,d_model,padding_idx=pad_id)
        self.position=nn.Embedding(max_positions,d_model)
        layer=nn.TransformerEncoderLayer(
            d_model=d_model,nhead=n_heads,dim_feedforward=feedforward_dim,
            dropout=dropout,activation="gelu",batch_first=True,norm_first=True,
        )
        self.transformer=nn.TransformerEncoder(layer,num_layers=n_layers,enable_nested_tensor=False)
        self.smiles_projection=nn.Linear(d_model,latent_dim)
        self.numeric_encoders=nn.ModuleList([NumericEncoder(n,latent_dim) for n in modality_dims])
        self.fusion=nn.Sequential(
            nn.Linear(latent_dim*7,latent_dim*2),nn.GELU(),nn.LayerNorm(latent_dim*2),
            nn.Dropout(dropout),nn.Linear(latent_dim*2,latent_dim),
        )
        self.mlm_head=nn.Linear(d_model,vocab_size)
        self.numeric_heads=nn.ModuleList([
            nn.Linear(latent_dim,n) if n else nn.Identity() for n in modality_dims
        ])

    def forward(self,input_ids,values,masks):
        """One forward pass: encode SMILES tokens, encode each of the 6
        numeric modalities, concatenate all 7 latent vectors and fuse them
        into the shared embedding, then read that embedding back out through
        the MLM head (per SMILES position) and each numeric head."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence].")
        if len(values) != 6 or len(masks) != 6:
            raise ValueError("values and masks must each contain six modalities.")
        batch,length=input_ids.shape
        if length > self.max_positions:
            raise ValueError(f"Sequence length {length} exceeds max_positions={self.max_positions}.")
        positions=torch.arange(length,device=input_ids.device).unsqueeze(0).expand(batch,length)
        hidden=self.token(input_ids)+self.position(positions)
        hidden=self.transformer(hidden,src_key_padding_mask=input_ids.eq(self.pad_id))
        smiles_z=self.smiles_projection(hidden[:,0])
        numeric_z=[enc(x,m) for enc,x,m in zip(self.numeric_encoders,values,masks)]
        shared=self.fusion(torch.cat([smiles_z]+numeric_z,dim=-1))
        return {
            "embedding":shared,
            "mlm_logits":self.mlm_head(hidden),
            "numeric_predictions":[head(shared) for head in self.numeric_heads],
        }
