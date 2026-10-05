# CompTox Multimodal Foundation Model

A research prototype that learns one shared molecular representation from **chemical structure, physicochemical
properties, ToxCast/Tox21 bioactivity, hazard and exposure data**, and tests whether that representation helps with
independent toxicity endpoints (here: in-house MEA neurotoxicity data).

The repository covers the whole path: downloading the public EPA data, building a training cohort, pretraining
the model, evaluating it, and comparing it honestly against simple baselines.

## What is here

| Part | What it does |
|---|---|
| **Data pipeline** (steps 00-02) | Downloads EPA ToxCast, ToxValDB and CPDat data, normalizes them into four tables, and builds a 9,746-chemical multimodal training cohort |
| **Foundation model** (steps 03-05) | Pretrains a SMILES transformer plus six numeric-modality encoders by masked reconstruction, exports embeddings, and evaluates them |
| **MEA neurotoxicity data** (steps 06-09) | Parses a 220-compound microelectrode-array screen, answers per-chemical queries, builds a knowledge graph, and registers other measured endpoints |
| **Transfer experiments** (steps 10, 10b, 10c, 10d, 10e) | Compares fine-tuning strategies with baselines, diagnoses the embedding, tests potency (AC50) prediction, and tests an independent published endpoint (hERG) |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Every step exists in two independent forms: a script `scripts/NN_name.py` and a notebook `notebooks/NN_name.ipynb`.
Both are complete on their own (nothing is imported from a shared package, and neither depends on the other), so
you can read or run any single step on its own. Scripts are run from the command line, and each documents its
inputs, outputs and options at the top of the file:

```bash
python scripts/07_query_chemical.py --chemical Permethrin --no-pubchem
```

The notebooks explain each step in Markdown cells and have a short "Run" cell at the end with plain settings
(for example `CHECK_ONLY = True`). Long-running or destructive steps (downloads, training) default to a safe
mode, so **Run All** never starts a multi-hour job by accident.

## The steps

| Step | Script / notebook | Purpose |
|---|---|---|
| 00 | `00_download_epa_data` | Download ToxCast, ToxValDB and CPDat (list first with `--list-only`) |
| 00a | `00a_import_toxcast_sql` | Import the ToxCast MySQL dump into a local server (needs MySQL) |
| 00b | `00b_download_structures` | Download the DSSTox structure file (DTXSID to SMILES) |
| 01 | `01_normalize_epa_data` | Normalize the raw files into `chemicals`, `bioactivity`, `hazard`, `exposure` tables |
| 01a | `01a_export_toxcast_mysql` | Export ToxCast/Tox21 measurements from MySQL |
| 01b | `01b_prepare_structures` | Attach canonical SMILES to chemicals and map MEA compound names to DTXSIDs |
| 02 | `02_build_training_cohort` | Build the multimodal training table (descriptors + assay matrix + hazard + exposure) |
| 03 | `03_train_foundation_model` | Pretrain the model (masked SMILES and masked numeric reconstruction) |
| 04 | `04_export_embeddings` | Compute a 256-d embedding for every cohort chemical |
| 05 | `05_evaluate_foundation_model` | Retrieval, modality ablations, cross-modal reconstruction, scaffold-novelty check |
| 06 | `06_prepare_mea_data` | Parse the MEA workbook into tidy tables |
| 07 | `07_query_chemical` | Everything the project knows about one chemical, plus a small knowledge graph |
| 08 | `08_build_knowledge_graph` | One knowledge graph over all MEA compounds (GraphML for Gephi / Cytoscape) |
| 09 | `09_register_toxicity_endpoints` | Register measured endpoints (hERG, DILI, ...) for use elsewhere |
| 10 | `10_transfer_learning` | MEA transfer test: scratch vs frozen vs partial vs full fine-tuning |
| 10b | `10b_baseline_models` | Ridge / random-forest baselines and embedding diagnostics |
| 10c | `10c_extended_descriptors` | Do ~200 RDKit descriptors help more than the model's 16? |
| 10d | `10d_toxcast_potency` | Predict ToxCast/Tox21 potency (log AC50) from structure; baselines vs the frozen embedding |
| 10e | `10e_herg_transfer` | Transfer to hERG channel blockade (Therapeutics Data Commons), mostly outside the pretraining cohort |

Typical order for a fresh checkout: `00 -> (00a -> 01a -> 00b -> 01b) or 01 -> 02 -> 03 -> 04 -> 05`, then `06 -> 07/08`,
then `10 / 10b / 10c / 10d / 10e`. If the ToxCast MySQL route is not used, `01` builds the tables from the downloaded files.

## How the model works

```text
SMILES tokens -> Transformer encoder -> [CLS] vector ---\
physchem   (values, mask) -> encoder -------------------\
hit call   (values, mask) -> encoder --------------------> concatenate -> fusion MLP -> shared 256-d embedding
AC50       (values, mask) -> encoder -------------------/
efficacy   (values, mask) -> encoder ------------------/
hazard     (values, mask) -> encoder -----------------/
exposure   (values, mask) -> encoder ----------------/
```

* **Missing data is explicit.** Every numeric modality is a `(values, mask)` pair, so "not measured" is never
  confused with "zero" or "inactive".
* **Pretraining** hides about 15% of SMILES characters and 15% of the observed values in each modality, and
  additionally hides whole modalities at random. The model must reconstruct what it cannot see. Hit calls use a
  class-balanced loss, because active assays are rare.
* **Configuration** (model size, learning rate, masking rates, loss weights, cohort size) is in `config.yaml`.
* **Data are split by compound.** Chemicals with the same SMILES never appear on both sides of a split. The
  MEA transfer experiments use a stricter Bemis-Murcko *scaffold* split.

## Data layout

```text
data/source/       raw EPA downloads (large; not tracked by git)
data/normalized/   chemicals.csv, bioactivity.csv, hazard.csv, exposure.csv (+ Parquet mirrors)
data/processed/    comptox_v3.parquet (training cohort), comptox_embeddings.parquet, comptox_query_index.csv
data/mea/          MEA_T_DATA.xlsx (input workbook)
data/mea_processed tidy MEA tables, the compound-to-DTXSID mapping, the knowledge graph
data/endpoints/    endpoint_results.csv (registry of measured endpoints)
data/herg/         herg_tdc.tab (downloaded once by step 10e; an independent published dataset, not this project's own data)
checkpoints/       trained model (not tracked by git)
results/           metrics from steps 05, 10, 10b, 10c, 10d, 10e and example query outputs
tests/             automated tests
```

The four normalized tables use these schemas:

```text
chemicals.csv    DTXSID, smiles, preferred_name, CASRN
bioactivity.csv  DTXSID, program, assay, hitcall, ac50_uM, efficacy, signed_hitcall
hazard.csv       DTXSID, metric, value, units, (study_type, species, route, duration, source)
exposure.csv     DTXSID, product_id, use_category, function_category
```

ToxCast/Tox21 **hit call, AC50 and efficacy are kept as separate channels**, and unmeasured assays stay missing.

## Findings so far

These come from the current 9,746-chemical cohort and the checkpoint in `checkpoints/`.

**Pretraining works, with a caveat.** On held-out validation chemicals, reconstruction is accurate and consistent
across modality ablations (evaluated on identical target entries), and results are similar for compounds whose
scaffold never appears in training (step 05 `novelty`). Hit-call AUROC is about 0.97 with all modalities visible,
but part of that is a shortcut: an AC50 exists only for active calls, so a visible AC50 reveals that its hit call
is active. With AC50 and efficacy hidden (hit calls of the other assays visible) AUROC is about 0.92 on a
400-compound check, and with only physchem visible about 0.77. Use the `physchem_hitcall` ablation as the honest
bioactivity figure.

**Transfer to MEA neurotoxicity is weak.** The label is the number of MEA metrics a compound perturbs
(`n_mea_metric_hits`, 203 labelled compounds); the test set is a fixed set of 51 rare-scaffold compounds.

* Using SMILES and physchem inputs, the frozen pretrained embedding beats an identically sized model trained from
  scratch once at least half of the training labels are used (step 10). With all modalities visible it does not.
* But a random forest on the 16 raw physchem descriptors does better than both (RMSE about 3.05 against 3.42
  for the pretrained model at full data; step 10b). Adding the embedding to the descriptors does not help.
* The embedding keeps the physchem information (each descriptor can be recovered with R^2 of about 0.95 or
  more), but it adds nothing beyond it for this endpoint.
* Using all ~200 RDKit descriptors instead of 16 does not improve the baselines (step 10c).
* **Potency (AC50) tasks.** For the MEA screen, AC50 labels use only quality-checked hits (hit code exactly 1 or 2,
  a real logAC50 that is not a placeholder value and lies inside the tested 0.1-30 uM range), leaving 188 labelled
  compounds. On those, out-of-fold models barely beat predicting the mean (step 10d `--mea-predict`): for the
  median-potency label RMSE 0.49 against 0.51 log10 units, and for the most-potent label no skill at all. Earlier
  runs that included placeholder values looked better, which was an artefact. Predicting ToxCast log AC50 from
  structure (step 10d) is a separate, larger task.

**Transfer to hERG channel blockade (an independent, published endpoint) works, but the embedding still does not
win.** Unlike MEA, most of these 641 compounds are outside the pretraining cohort (77 of 641; only 63 were in its
training split, and none of their hERG labels were ever seen during pretraining), so this is close to a true
test on new chemistry. Fixed scaffold split: 481 training / 160 test compounds (step 10e).

* Every method clears the majority-class floor by a wide margin (AUROC 0.50 -> 0.77-0.87), so structure genuinely
  predicts hERG blockade here -- a real, usable signal, unlike MEA.
* Selected honestly by 5-fold CV on the training compounds: a random forest on a fingerprint + physchem, test
  AUROC 0.807 (95% CI 0.74-0.87). The best pure-embedding method (an SVC on the frozen embedding) reaches 0.799,
  inside that same interval -- competitive, not clearly better.
* On the learning curve, embedding-based methods are slightly ahead with very few labels (10%: about 0.77-0.79)
  but plateau; fingerprint + physchem methods start similar and keep improving to 0.84-0.85 by 100% of the
  training data, overtaking the embedding.

**Interpretation.** Across two independent endpoints (MEA and hERG), the pattern repeats: the pretrained embedding
is a working, never-worse-than-chance featurization, but it does not show a clear, reproducible advantage over
conventional fingerprint/physchem features. hERG shows the representation *can* support real transfer (the task
itself is learnable); MEA shows the embedding specifically doesn't add value there. Dominant MEA predictors are
molecular size and lipophilicity. A supervised or contrastive pretraining signal is the natural next step, since
two honest downstream comparisons now point the same way.

## Notes on the methods

* **Scaffold split.** Compounds sharing a Bemis-Murcko scaffold (their core ring system) always land on the same
  side, so a test compound never has a training compound with the same core. The largest scaffold groups go to
  training, and the rarer ones form the test set.
* **Low-data curves.** A seed shuffles the training pool and takes the first 10%, 25%, 50% or 100%. Smaller
  subsets are contained in larger ones.
* **AC50 vs log AC50.** `bioactivity.csv` stores the linear AC50 in micromolar (`ac50_uM`, present only for active
  calls). The training cohort stores **log10(AC50 in uM)** in `bioac50__<assay>`. Potency models are trained and scored
  in log10 units, so an RMSE of 1.0 means a typical error of one order of magnitude (step 10d spells this out).
* **Interrupted runs.** The experiment steps (10, 10b, 10c, 10d) save results after every finished run and accept
  `--resume` to continue.
* **Measured vs predicted.** ToxProfiler target scores are model predictions and are never mixed with measured
  MEA effects or registered endpoints.
* **Credentials.** Nothing secret is stored. The optional EPA API key is read from `EPA_API_KEY`, and MySQL access
  uses a login path (`mysql_config_editor`) or the `COMPTOX_MYSQL_*` environment variables.

## Testing

```bash
python -m pytest tests
```

Runs the automated tests in `tests/`.

## Data sources

* EPA CompTox Chemicals Dashboard bulk data: ToxCast / invitrodb, ToxValDB and CPDat (public releases).
* EPA DSSTox chemical structures.
* MEA neurotoxicity data: an in-house workbook (`data/mea/MEA_T_DATA.xlsx`), not part of the public releases.
* hERG blocker labels (step 10e): Therapeutics Data Commons (https://tdcommons.ai), a curated version of Karim et
  al.'s hERG dataset, for research use only.
