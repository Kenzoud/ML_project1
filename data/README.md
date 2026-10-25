# Data

The dataset is not committed. Download `dataset_to_release.zip` from the
[AIcrowd challenge](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1)
and unzip it here:

```bash
unzip dataset_to_release.zip -d data/
```

This creates `data/dataset/` containing `x_train.csv`, `y_train.csv`,
`x_test.csv` and `sample_submission.csv`. Load it with
`helpers.load_csv_data("data/dataset/")`.

| File | Rows | Columns |
|---|---:|---|
| `x_train.csv` | 328,135 | `Id` + 321 features |
| `y_train.csv` | 328,135 | `Id`, `_MICHD` in {-1, 1} (28,975 positives, 8.83%) |
| `x_test.csv` | 109,379 | `Id` + 321 features |

Source: BRFSS 2015 survey, CDC
([codebook](https://www.cdc.gov/brfss/annual_data/annual_2015.html)).
Some columns are already rescaled relative to the codebook (e.g. `_BMI5`
ranges 12.02–97.65), so check actual value ranges before applying codebook rules.
