import torch
from core.tokenizer import SmilesTokenizer
from core.model import CompToxFoundationModel
from core.evaluation import pca_projection,nearest_neighbors


def test_tokenizer_length():
    t=SmilesTokenizer().fit(["CCO","c1ccccc1"]); assert len(t.encode("CCO",16))==16


def test_model_forward():
    dims=[5,10,10,10,3,3]
    m=CompToxFoundationModel(vocab_size=32,modality_dims=dims,d_model=32,n_heads=4,n_layers=2,feedforward_dim=64,latent_dim=16,pad_id=0)
    ids=torch.randint(1,31,(4,20)); values=[torch.randn(4,n) for n in dims]; masks=[torch.ones(4,n) for n in dims]
    out=m(ids,values,masks); assert out["embedding"].shape==(4,16); assert len(out["numeric_predictions"])==6


def test_pca_small_input_and_retrieval():
    x=torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]).numpy(); p=pca_projection(x); assert p.shape==(3,2)
    nn=nearest_neighbors(x,["A","B","C"],0,2); assert len(nn)==2
