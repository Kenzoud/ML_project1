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
| `run.py` | Reproduces the selected AIcrowd submission |
| `tests/` | Our unit tests for `implementations.py` |
| `notebooks/` | EDA and experiments |
| `report/` | LaTeX report |
| `data/`, `outputs/` | Dataset and generated files (not committed) |

## Conventions shared by all modules

- Labels: `{0, 1}` inside the code; converted to `{-1, 1}` only when writing the submission.
- Shapes: `y` is `(N,)`, `tx` is `(N, D)` and includes the bias column, `w` is `(D,)`. All `float64`.
- Preprocessing is fitted on training rows only: `params = fit_preprocessor(x_train, feature_names)`, then `tx = transform(x, params)` for train, validation and test.
- Random seed: `42`.

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
