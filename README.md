# ML Project 1 — MICHD prediction from BRFSS survey data

CS-433 Machine Learning, EPFL, Fall 2026. Binary classification of
myocardial infarction / coronary heart disease (MICHD) from the 2015 BRFSS
survey, using NumPy only.

## Setup

```bash
conda env create -f environment.yml
conda activate ml-project1
```

Versions match the course grading environment (Python 3.9, NumPy 1.23.1,
black 22.6.0). scikit-learn is left out on purpose: it is not allowed in
this project. Keep any environment **outside** the repository, since the
grading tests run black on every `*.py` file in it.

Put the data in `data/dataset/` as described in [data/README.md](data/README.md).

## Repository layout

| Path | Content |
|---|---|
| `implementations.py` | The six mandatory methods and their loss/gradient helpers |
| `helpers.py` | Course-provided loading and submission helpers (unchanged) |
| `feature_types.py` | BRFSS coding rules: which columns are dropped, categorical or continuous, and their missing/"none" codes |
| `docs/feature_types.csv` | The resulting table for all 321 columns (`python feature_types.py` regenerates it) |
| `preprocessing.py` | Cached data loading, `fit_preprocessor` / `transform`, preprocessing variants (`CONFIGS`) |
| `evaluation.py` | Stratified split and k-fold, F1/precision/recall/accuracy, ROC AUC, threshold selection |
| `baselines.py` | Baseline models on an 80/20 stratified split (writes `outputs/baselines.json`) |
| `ablation.py` | Preprocessing ablation study (writes `outputs/ablation.json`) |
| `run.py` | Reproduces the selected AIcrowd submission (`outputs/submission.csv`) |
| `tests/` | Unit tests |
| `notebooks/` | EDA and experiments |
| `report/` | LaTeX report |
| `data/`, `outputs/` | Dataset and generated files (not committed) |

## Conventions shared by all modules

- Labels: `{0, 1}` inside the code; converted to `{-1, 1}` only when writing the submission.
- Shapes: `y` is `(N,)`, `tx` is `(N, D)` and includes the bias column, `w` is `(D,)`. All `float64`.
- Preprocessing is fitted on training rows only: `params = fit_preprocessor(x_train, feature_names, config)`, then `tx = transform(x, params)` for train, validation and test.
- Random seed: `42`.

## Reproducing the submission

```bash
python run.py
```

The first run parses the CSVs (~1 min) and caches them in `data/cache.npz`.
`run.py` fits on 80% of the training rows, picks the F1-maximizing threshold
on the other 20%, refits on all rows and writes `outputs/submission.csv`.

Main metric: F1 of the positive class. AIcrowd does not name its metric, but
leaderboard values (~0.46) are consistent with F1, not accuracy, which is
0.91 for the all-negative predictor.

## Preprocessing

`run.py` uses the `"full"` config:

1. Drop interview/ID columns, survey design weights and raw columns that have
   a cleaned derived version (e.g. `WEIGHT2` → `WTKG3`, `ALCDAY5` → `DROCDY3_`).
2. BRFSS "don't know"/"refused" codes (7/9, 77/99, 777/999, ...) become
   missing; "none" codes (88, 888) become 0. Rules and the codebook-checked
   exceptions are in `feature_types.py`.
3. Categorical columns (≤ 10 distinct values): one-hot, with missing as its
   own level, the most frequent level as reference and levels rarer than 0.1%
   merged into it. Missing values follow the survey's skip logic (e.g. the age
   at diabetes diagnosis is only asked to diabetics), so missingness is kept
   as information instead of being imputed away.
4. Continuous columns: clipped to the 0.5–99.5 training percentiles,
   standardized on observed rows, missing filled with 0 plus a missing indicator.
5. Constant and duplicate columns dropped, then standardization and a bias column.

Result: 638 features.

## Current results (validation, 80/20 stratified split, seed 42)

Baselines (baseline preprocessing, 163 features):

| Model | F1 | Precision | Recall | Accuracy | AUC |
|---|---:|---:|---:|---:|---:|
| Majority (all negative) | 0.000 | – | 0.000 | 0.912 | – |
| Least squares | 0.414 | 0.343 | 0.522 | 0.870 | 0.845 |
| Ridge, λ=1e-5 | 0.415 | 0.344 | 0.521 | 0.870 | 0.845 |
| Logistic regression | 0.418 | 0.360 | 0.499 | 0.877 | 0.851 |

Preprocessing ablation (each row adds one step; `python ablation.py`):

| Config | Features | Ridge F1 | Logistic F1 | Logistic AUC |
|---|---:|---:|---:|---:|
| `baseline` | 164 | 0.415 | 0.418 | 0.851 |
| `clean_codes`: special codes as missing, 88 → 0 | 144 | 0.415 | 0.428 | 0.860 |
| `one_hot`: one-hot, missing as a level | 579 | 0.424 | 0.439 | 0.866 |
| `full`: + missing indicators, clipping (submitted) | 638 | 0.424 | 0.440 | 0.866 |

Logistic regression: 500 GD steps with γ=0.5 (1000 and 2000 steps change
the training loss by < 1e-4). The threshold is chosen on the same
validation split, so scores are slightly optimistic. AIcrowd test F1:
0.411 for the baseline logistic regression.

## Tests

Our unit tests:

```bash
pytest tests/
```

Course public tests, run from the course repository (not copied here):

```bash
cd <ML_course>/projects/project1/grading_tests
pytest --github_link <absolute path to this repo> .
```

`test_github_link_format` fails with a local path; that is expected. Before
submitting, rerun with the GitHub URL of the final commit
(`https://github.com/<org>/<repo>/tree/<commit-hash>`).

## Team

| Member | Area |
|---|---|
| A | `implementations.py`, tests |
| B | EDA, preprocessing, feature engineering |
| C | Evaluation, model selection, report |
