"""Reproduce the selected AIcrowd submission.

Usage (from the repository root, with data in data/dataset/):
    python run.py

The full pipeline (preprocessing, training, prediction) is added once the
first baseline model is in place. For now this only checks that the data loads.
"""

import numpy as np

from helpers import load_csv_data

DATA_PATH = "data/dataset/"
SEED = 42


def main():
    """Load the data and print its shapes."""
    np.random.seed(SEED)
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(DATA_PATH)
    print("x_train", x_train.shape, "x_test", x_test.shape, "y_train", y_train.shape)


if __name__ == "__main__":
    main()
