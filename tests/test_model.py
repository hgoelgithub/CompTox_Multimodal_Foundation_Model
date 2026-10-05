import numpy as np
import torch


def test_tokenizer_encodes_to_fixed_length(train):
    tokenizer = train.SmilesTokenizer().fit(["CCO", "c1ccccc1"])
    ids = tokenizer.encode("CCO", 16)
    assert len(ids) == 16
    assert ids[0] == tokenizer.cls_id and ids[-1] == tokenizer.pad_id


def test_unseen_character_maps_to_unk(train):
    tokenizer = train.SmilesTokenizer().fit(["CC"])
    assert tokenizer.encode("N", 4)[1] == tokenizer.unk_id


def test_model_forward_shapes(train):
    dims = [5, 10, 10, 10, 3, 3]
    model = train.CompToxFoundationModel(vocab_size=32, modality_dims=dims, d_model=32, n_heads=4, n_layers=2,
                                         feedforward_dim=64, latent_dim=16, pad_id=0)
    ids = torch.randint(1, 31, (4, 20))
    values = [torch.randn(4, n) for n in dims]
    masks = [torch.ones(4, n) for n in dims]
    out = model(ids, values, masks)
    assert out["embedding"].shape == (4, 16)
    assert out["mlm_logits"].shape == (4, 20, 32)
    assert len(out["numeric_predictions"]) == 6


def test_smiles_masking_always_masks_at_least_one_token(train):
    tokenizer = train.SmilesTokenizer().fit(["CCO", "c1ccccc1"])
    ids = torch.tensor([tokenizer.encode("CCO", 8)])
    masked, labels = train.mask_smiles(ids, tokenizer, probability=0.0001)
    assert (labels != -100).sum() >= 1
    assert (masked == tokenizer.mask_id).sum() == (labels != -100).sum()


def test_numeric_masking_only_targets_observed_values(train):
    values = torch.ones(2, 4)
    observed = torch.tensor([[1., 1., 0., 0.], [1., 0., 1., 0.]])
    _, input_mask, targets = train.mask_numeric(values, observed, probability=1.0)
    assert torch.equal(targets, observed)          # everything observed is a target, nothing missing is
    assert input_mask.sum() == 0                   # and the model no longer sees those values


def test_masked_mse_ignores_unmasked_positions(train):
    pred, target = torch.tensor([[1.0, 5.0]]), torch.tensor([[0.0, 0.0]])
    assert float(train.masked_mse(pred, target, torch.tensor([[1.0, 0.0]]))) == 1.0


def test_scalers_ignore_missing_values(train):
    import pandas as pd
    df = pd.DataFrame({"a": [1.0, 3.0, np.nan]})
    scaler = train.fit_group_scalers(df, [["a"], [], [], [], [], []])[0]
    assert scaler["mean"] == [2.0]
