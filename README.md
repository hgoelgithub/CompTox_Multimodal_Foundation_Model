# CompTox Multimodal Foundation Model

A clean, reproducible prototype for learning a reusable toxicology representation from **chemical structure, physicochemical properties, ToxCast/Tox21 bioactivity, hazard, and exposure**, with downstream use for **MEA neurotoxicity, hERG, DILI, and other toxicity/injury endpoints**.

The repository is intentionally simple:

- all executable Python workflows are in `scripts/`;
- all Jupyter workflows are in `notebooks/`;
- every numbered script has a matching numbered notebook;
- reusable implementation utilities stay under `scripts/core/`;
- tests stay under `scripts/tests/`;
- project documentation is consolidated in this README.

## Repository structure

```text
CompTox_Multimodal_Foundation_Model/
├── README.md
├── config.yaml
├── requirements.txt
├── .gitignore
│
├── scripts/
│   ├── 00_download_epa_data.py
│   ├── 01_normalize_epa_data.py
│   ├── 02_build_training_cohort.py
│   ├── 03_train_foundation_model.py
│   ├── 04_export_embeddings.py
│   ├── 05_evaluate_foundation_model.py
│   ├── 06_prepare_mea_data.py
│   ├── 07_query_chemical.py
│   ├── 08_build_knowledge_graph.py
│   ├── 09_register_toxicity_endpoints.py
│   ├── 10_transfer_learning.py
│   ├── 11_validate_project.py
│   ├── core/
│   └── tests/
│
├── notebooks/
│   ├── 00_download_epa_data.ipynb
│   ├── 01_normalize_epa_data.ipynb
│   ├── 02_build_training_cohort.ipynb
│   ├── 03_train_foundation_model.ipynb
│   ├── 04_export_embeddings.ipynb
│   ├── 05_evaluate_foundation_model.ipynb
│   ├── 06_prepare_mea_data.ipynb
│   ├── 07_query_chemical.ipynb
│   ├── 08_build_knowledge_graph.ipynb
│   ├── 09_register_toxicity_endpoints.ipynb
│   ├── 10_transfer_learning.ipynb
│   └── 11_validate_project.ipynb
│
├── data/
│   ├── source/
│   ├── normalized/
│   ├── processed/
│   ├── mea/MEA_T_DATA.xlsx
│   ├── mea_processed/
│   └── endpoints/endpoint_results.csv
├── checkpoints/
└── results/
```

There are **12 numbered workflow scripts (`00`–`11`)** and therefore **12 matching notebooks**. The earlier three combined/umbrella notebooks were removed because they duplicated several workflow stages and made the repository inconsistent.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### EPA API key

Do **not** put the key in a Python file or notebook.

```bash
export EPA_API_KEY="YOUR_KEY_HERE"
```

On macOS you may place that line in `~/.zshrc` if you want it to persist.

The large pretraining datasets are handled primarily as **bulk downloads**. The API key is intended for targeted chemical lookup/enrichment rather than repeatedly downloading the full training corpus chemical-by-chemical.

## Main workflow

### 00 — Download EPA source data

Safe listing first:

```bash
python scripts/00_download_epa_data.py --list-only
```

Download model-relevant bulk files:

```bash
python scripts/00_download_epa_data.py
```

### 01 — Normalize EPA data

```bash
python scripts/01_normalize_epa_data.py
```

### ToxCast SQL database import

The ToxCast/invitrodb release is a MySQL dump. MySQL 8 is already available on
macOS systems with MacPorts, but the server may require a configured client
login. Create a login path once (do not put the password in this repository):

```bash
mysql_config_editor set --login-path=comptox --host=localhost --user=YOUR_USER --password
python scripts/00a_import_toxcast_sql.py --login-path comptox
```

Use `--check --login-path comptox` to verify the connection. The importer
streams the compressed dump into MySQL and exports `chemicals.csv` and
`bioactivity.csv`; the SQL schema does not contain SMILES, so structure
enrichment from DSSTox/PubChem remains a separate step. If the dump has already
been imported, use `--skip-import`.

The complete local-data sequence is:

```bash
python scripts/00a_import_toxcast_sql.py --login-path comptox
python scripts/01a_export_toxcast_mysql.py
python scripts/00b_download_structures.py
python scripts/01b_prepare_structures.py
python scripts/01_normalize_epa_data.py --no-extract
python scripts/02_build_training_cohort.py
```

`01a` exports measured endpoints from MySQL in chunks. `01b` joins the official
DSSTox structures and updates the MEA mapping audit before cohort construction.

Expected normalized tables:

```text
data/normalized/chemicals.csv
data/normalized/bioactivity.csv
data/normalized/hazard.csv
data/normalized/exposure.csv
```

Core schemas:

```text
chemicals.csv:
DTXSID, smiles, preferred_name, CASRN

bioactivity.csv:
DTXSID, program, assay, hitcall, ac50_uM, efficacy

hazard.csv:
DTXSID, metric, value, ...

exposure.csv:
DTXSID, product_id, use_category, function_category
```

ToxCast/Tox21 **hitcall, AC50, and efficacy are kept as separate channels**. Missing/unmeasured assays remain missing and are handled by masks rather than being interpreted as inactive.

### Parquet/DuckDB mirror of the normalized tables

The normalized CSVs above are the source of truth, but everything downstream
(the cohort builder and the interactive chemical query) reads them through
`scripts/core/duckdb_io.py` instead of re-parsing full CSVs on every run:

```text
data/normalized/parquet/*.parquet   # Parquet mirror of each normalized CSV
data/normalized/comptox.duckdb      # DuckDB views over those Parquet files
```

`duckdb_io.refresh()` rebuilds both from the current CSVs (run automatically
by `02_build_training_cohort.py`); `duckdb_io.query_df(sql, params)` runs a
filtered SQL query and returns a DataFrame. The MySQL server is only touched
by `00a`/`01a`/`01b`; once the Parquet mirror exists, everything else is
CSV/Parquet-only and MySQL can be stopped or have its raw import databases
dropped without affecting training or querying.

### 02 — Build the training cohort

```bash
python scripts/02_build_training_cohort.py
```

The cohort is selected by multimodal information coverage rather than by maximizing the number of SMILES. The default cap is 30,000 chemicals and can be changed in `config.yaml`.

The model uses:

```text
SMILES / molecular structure
Physicochemical descriptors
ToxCast/Tox21 hitcall
ToxCast/Tox21 AC50
ToxCast/Tox21 efficacy
Hazard summaries
Exposure summaries
```

### 03 — Train the foundation-model prototype

```bash
python scripts/03_train_foundation_model.py
```

Architecture:

```text
SMILES ---------------- Transformer ----\
PhysChem -------------- MLP ------------\
Hitcall --------------- MLP -------------\
AC50 ------------------ MLP --------------> multimodal fusion -> shared embedding
Efficacy -------------- MLP -------------/
Hazard ---------------- MLP ------------/
Exposure -------------- MLP -----------/
```

Pretraining includes masked SMILES learning, masked numeric reconstruction, class-balanced hitcall loss, modality dropout, validation, early stopping, and best-checkpoint saving.

### 04 — Export learned embeddings

```bash
python scripts/04_export_embeddings.py
```

### 05 — Evaluate foundation-model behavior

```bash
python scripts/05_evaluate_foundation_model.py plan
python scripts/05_evaluate_foundation_model.py ablation
python scripts/05_evaluate_foundation_model.py cross-modal
```

Nearest-neighbor retrieval after embeddings exist:

```bash
python scripts/05_evaluate_foundation_model.py retrieval --dtxsid DTXSID_HERE --k 5
```

Recommended evidence for foundation-model behavior:

- scratch vs frozen encoder vs partial fine-tuning vs full fine-tuning;
- 10%, 25%, 50%, and 100% labeled-data experiments;
- modality ablations;
- embedding visualization/retrieval;
- cross-modal reconstruction.

## MEA downstream workflow

The supplied workbook contains 220 chemicals with:

- MEA logAC50 values;
- MEA HIT codes (`1 = increase`, `2 = decrease`);
- six-concentration z-scores (`0.1, 0.3, 1, 3, 10, 30 µM`);
- literature drug-target interactions;
- ToxProfiler target predictions;
- molecular class/super-class information.

Prepare it with:

```bash
python scripts/06_prepare_mea_data.py
```

## Chemical query

```bash
python scripts/07_query_chemical.py --chemical "Permethrin"
```

The query can integrate available information from:

- structure and chemical class;
- literature targets;
- ToxProfiler predicted targets;
- MEA hit direction, logAC50, and concentration-response z-scores;
- ToxCast/Tox21 hitcall, AC50, and efficacy for mapped chemicals;
- registered hERG, DILI, or other toxicity endpoints;
- foundation-model embedding neighbors when embeddings/mappings are available.

## Knowledge graph

```bash
python scripts/08_build_knowledge_graph.py
```

The graph can connect:

```text
Chemical
├── molecular structure / chemical class
├── structural similarity
├── literature-supported targets
├── predicted targets
├── ToxCast/Tox21 assays
├── MEA phenotypes
├── hazard context
├── exposure context
├── hERG / DILI / other toxicity endpoints
└── foundation-model embedding similarity
```

Graph outputs are written to `data/mea_processed/` and/or `results/` depending on the workflow.

## Register additional toxicity endpoints

Input CSVs may contain real hERG, DILI, hepatotoxicity, nephrotoxicity, mitochondrial toxicity, genotoxicity, developmental toxicity, or another independent endpoint.

```bash
python scripts/09_register_toxicity_endpoints.py --input my_endpoint_results.csv
```

The registry deliberately keeps target-level predictions separate from endpoint-level toxicity evidence. For example, a KCNH2 target prediction is not automatically labeled as a measured hERG toxicity result.

## Transfer learning

```bash
python scripts/10_transfer_learning.py --show-plan
```

Supported strategies:

```text
head_only
partial
full
```

For small downstream toxicity datasets, compare head-only and partial fine-tuning before relying on full fine-tuning.

## Notebook workflow

Every numbered script has a notebook with the same base name. Each notebook is a
real, standalone implementation -- not a wrapper that calls back into the
script -- so it can be read top to bottom to understand that step. Each
notebook contains:

1. project setup (locates the repo root; independent of every other notebook);
2. the implementation, copied verbatim from its `.py` script, one cell per
   top-level function/class in source order, each preceded by a small
   markdown header carrying that function's docstring;
3. a final run cell.

The implementation cells are generated and kept in sync with the `.py` files
by `scripts/sync_notebooks.py` -- **never hand-edit them**. After changing a
script:

```bash
python scripts/sync_notebooks.py            # regenerate every notebook
python scripts/sync_notebooks.py --check     # verify only, exit 1 if stale
```

`11_validate_project.py` runs the `--check` automatically, so a notebook that
has drifted out of sync with its script fails validation immediately instead
of silently going stale.

The final run cell uses:

```python
RUN_STEP = False
```

by default so `Run All` does not accidentally start a large download, long training job, or overwrite generated outputs. Set it to `True` when you are ready to run that stage. `RUN_ARGS` in the same cell controls command-line options.

## Validation

Run:

```bash
python scripts/11_validate_project.py
```

The validator checks:

- required numbered scripts;
- exact one-to-one notebook mirror;
- Python syntax;
- notebook JSON format;
- automated tests.

## Scientific interpretation

This repository should be described as a **domain-specific multimodal toxicology foundation-model prototype**. The strongest evidence that it behaves as a reusable foundation model will come from independent transfer experiments showing that its pretrained representation supports multiple downstream toxicity tasks, especially under limited labeled data.
