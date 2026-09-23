"""Reusable evaluation utilities for demonstrating foundation-model behavior."""
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    all_z=[]
    for ids, values, masks in loader:
        ids=ids.to(device)
        values=[x.to(device) for x in values]
        masks=[x.to(device) for x in masks]
        all_z.append(model(ids,values,masks)["embedding"].cpu().numpy())
    if not all_z:
        return np.empty((0,0),dtype=np.float32)
    return np.concatenate(all_z,axis=0)


def pca_projection(embeddings, n_components=2):
    embeddings=np.asarray(embeddings)
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("Embeddings must be a non-empty 2-D array.")
    actual=min(n_components, embeddings.shape[0], embeddings.shape[1])
    pcs=PCA(n_components=actual).fit_transform(embeddings)
    if actual < n_components:
        pcs=np.pad(pcs,((0,0),(0,n_components-actual)),constant_values=0.0)
    return pcs


def nearest_neighbors(embeddings, ids, query_index, k=5):
    embeddings=np.asarray(embeddings)
    if len(ids) != len(embeddings):
        raise ValueError("ids and embeddings must have the same number of rows.")
    if not 0 <= query_index < len(ids):
        raise IndexError("query_index is out of range.")
    distances=pairwise_distances(embeddings[query_index:query_index+1],embeddings,metric="cosine")[0]
    order=[i for i in np.argsort(distances) if i != query_index][:max(0,k)]
    return pd.DataFrame({"DTXSID":[ids[i] for i in order],
                         "cosine_distance":[float(distances[i]) for i in order]})
