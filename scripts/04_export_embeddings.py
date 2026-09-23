"""STEP 4 -- Run the trained model (step 03) over every compound in the
cohort in inference mode (no masking) to get its shared embedding, plus a
2-D PCA projection for quick plotting. Output: comptox_embeddings.parquet,
used by 05's retrieval command and 08's embedding-similarity graph edges."""
import argparse,json
import pandas as pd
import torch,yaml
from torch.utils.data import DataLoader
from core.data import CompToxDataset
from core.model import CompToxFoundationModel
from core.tokenizer import SmilesTokenizer
from core.evaluation import extract_embeddings,pca_projection
from core.paths import PROCESSED_COMPTox,CHECKPOINTS,CONFIG
from core.table_io import read_table,write_table,table_exists

DATA=PROCESSED_COMPTox/"comptox_v3.parquet"; OUT=PROCESSED_COMPTox/"comptox_embeddings.parquet"


def choose_device():
    """Prefer CUDA, then Apple Silicon MPS, else fall back to CPU."""
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends,"mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def preflight():
    """Check the cohort and a trained checkpoint both exist before running inference."""
    missing=[]
    if not table_exists(DATA): missing.append(str(DATA))
    for p in [CHECKPOINTS/"tokenizer_v3.json",CHECKPOINTS/"comptox_v3_best.pt"]:
        if not p.exists(): missing.append(str(p))
    if missing:
        print("Embedding export prerequisites are missing:")
        for p in missing: print("  missing:",p)
        return False
    print("Embedding export prerequisites found."); return True


def main():
    """CLI entry point: load the checkpoint's exact feature schema/scalers
    (via core/checkpoint.py, so inference matches training exactly), run the
    whole cohort through the model with no masking, and save embeddings + PCA."""
    p=argparse.ArgumentParser(); p.add_argument("--check",action="store_true"); a=p.parse_args()
    if a.check: raise SystemExit(0 if preflight() else 1)
    if not preflight(): raise FileNotFoundError("Train the foundation model before exporting embeddings.")
    from core.checkpoint import load_checkpoint,dataset_for_checkpoint
    df=read_table(DATA); dev=choose_device()
    model,tok,ckpt=load_checkpoint(CHECKPOINTS/"comptox_v3_best.pt",dev)
    torch.set_num_threads(ckpt['config']['training'].get('cpu_threads',2))
    ds=dataset_for_checkpoint(df,tok,ckpt)
    loader=DataLoader(ds,batch_size=ckpt['config']['training']['batch_size'],shuffle=False,num_workers=0)
    z=extract_embeddings(model,loader,dev); pcs=pca_projection(z)
    # Build all z000..zNNN columns in one concat rather than inserting them
    # one at a time (which fragments the DataFrame and is O(n^2) for a wide
    # embedding -- 256 columns triggers pandas' PerformanceWarning on every
    # insert past the fragmentation threshold).
    z_cols=pd.DataFrame(z,columns=[f"z{i:03d}" for i in range(z.shape[1])],index=df.index)
    out=pd.concat([pd.DataFrame({"DTXSID":df["DTXSID"].astype(str),"PC1":pcs[:,0],"PC2":pcs[:,1]}),z_cols],axis=1)
    saved=write_table(out,OUT); print("Saved embeddings:",out.shape,"->",saved)


if __name__=="__main__": main()
