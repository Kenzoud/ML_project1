"""Unit tests for implementations.py, beyond the course's public tests.

Run from the repository root with: pytest tests/
"""

import numpy as np
import pytest

import implementations as impl

ITERATIVE = {
    "gd": lambda y, tx, w, it, g: impl.mean_squared_error_gd(y, tx, w, it, g),
    "sgd": lambda y, tx, w, it, g: impl.mean_squared_error_sgd(y, tx, w, it, g),
    "logistic": lambda y, tx, w, it, g: impl.logistic_regression(y, tx, w, it, g),
    "reg_logistic": lambda y, tx, w, it, g: impl.reg_logistic_regression(
        y, tx, 0.1, w, it, g
    ),
}


@pytest.fixture()
def data():
    """Well-conditioned regression data with a bias column and {0, 1} labels."""
    rng = np.random.default_rng(0)
    n, d = 200, 4
    tx = np.c_[np.ones(n), rng.normal(size=(n, d - 1))]
    w_true = np.array([0.5, -1.0, 2.0, 0.3])
    y = tx @ w_true + 0.1 * rng.normal(size=n)
    y01 = (rng.random(n) < impl.sigmoid(tx @ w_true)).astype(float)
    return y, y01, tx


def numerical_gradient(f, w, eps=1e-6):
    """Central finite-difference gradient of scalar function f at w."""
    grad = np.zeros_like(w)
    for i in range(len(w)):
        step = np.zeros_like(w)
        step[i] = eps
        grad[i] = (f(w + step) - f(w - step)) / (2 * eps)
    return grad


@pytest.mark.parametrize("name", ITERATIVE)
def test_initial_w_not_modified(name, data):
    y, y01, tx = data
    labels = y if name in ("gd", "sgd") else y01
    initial_w = np.zeros(tx.shape[1])
    ITERATIVE[name](labels, tx, initial_w, 5, 0.1)
    np.testing.assert_array_equal(initial_w, 0.0)


@pytest.mark.parametrize("name", ITERATIVE)
def test_zero_iterations_return_copy(name, data):
    y, y01, tx = data
    labels = y if name in ("gd", "sgd") else y01
    initial_w = np.arange(tx.shape[1], dtype=float)
    w, loss = ITERATIVE[name](labels, tx, initial_w, 0, 0.1)
    np.testing.assert_array_equal(w, initial_w)
    assert w is not initial_w
    assert w.shape == (tx.shape[1],)
    assert np.ndim(loss) == 0


@pytest.mark.parametrize("name", ITERATIVE)
def test_integer_initial_w(name, data):
    """Integer weights (as in the labs) must not truncate updates."""
    y, y01, tx = data
    labels = y if name in ("gd", "sgd") else y01
    w, _ = ITERATIVE[name](labels, tx, np.zeros(tx.shape[1], dtype=int), 3, 0.1)
    assert w.dtype == float
    assert np.any(w != 0)


def test_mse_gradient_matches_finite_differences(data):
    y, _, tx = data
    w = np.array([0.1, 0.2, -0.3, 0.4])
    expected = numerical_gradient(lambda v: impl.compute_mse(y, tx, v), w)
    np.testing.assert_allclose(impl.compute_mse_gradient(y, tx, w), expected, rtol=1e-6)


def test_logistic_gradient_matches_finite_differences(data):
    _, y01, tx = data
    w = np.array([0.1, 0.2, -0.3, 0.4])
    expected = numerical_gradient(lambda v: impl.compute_logistic_loss(y01, tx, v), w)
    np.testing.assert_allclose(
        impl.compute_logistic_gradient(y01, tx, w), expected, rtol=1e-6
    )


def test_gd_converges_to_least_squares(data):
    y, _, tx = data
    w_ls, loss_ls = impl.least_squares(y, tx)
    w_gd, loss_gd = impl.mean_squared_error_gd(y, tx, np.zeros(4), 1000, 0.5)
    np.testing.assert_allclose(w_gd, w_ls, atol=1e-8)
    np.testing.assert_allclose(loss_gd, loss_ls, rtol=1e-8)


def test_sgd_is_reproducible_and_approaches_least_squares(data):
    y, _, tx = data
    np.random.seed(1)
    w1, loss1 = impl.mean_squared_error_sgd(y, tx, np.zeros(4), 5000, 0.01)
    np.random.seed(1)
    w2, loss2 = impl.mean_squared_error_sgd(y, tx, np.zeros(4), 5000, 0.01)
    np.testing.assert_array_equal(w1, w2)
    w_ls, loss_ls = impl.least_squares(y, tx)
    np.testing.assert_allclose(w1, w_ls, atol=0.05)
    # Returned loss is the full-dataset MSE at the final weights.
    np.testing.assert_allclose(loss1, impl.compute_mse(y, tx, w1))


def test_least_squares_matches_lstsq(data):
    """lstsq is only used here as an independent reference, never in the solution."""
    y, _, tx = data
    w, loss = impl.least_squares(y, tx)
    np.testing.assert_allclose(w, np.linalg.lstsq(tx, y, rcond=None)[0], rtol=1e-10)
    assert np.ndim(loss) == 0


def test_ridge_lambda_zero_equals_least_squares(data):
    y, _, tx = data
    w_r, loss_r = impl.ridge_regression(y, tx, 0.0)
    w_ls, loss_ls = impl.least_squares(y, tx)
    np.testing.assert_allclose(w_r, w_ls)
    np.testing.assert_allclose(loss_r, loss_ls)


def test_ridge_is_stationary_point_and_returns_unpenalized_loss(data):
    y, _, tx = data
    lambda_ = 0.3
    w, loss = impl.ridge_regression(y, tx, lambda_)
    grad = impl.compute_mse_gradient(y, tx, w) + 2 * lambda_ * w
    np.testing.assert_allclose(grad, 0.0, atol=1e-10)
    np.testing.assert_allclose(loss, impl.compute_mse(y, tx, w))


def test_ridge_shrinks_weights(data):
    y, _, tx = data
    norms = [np.linalg.norm(impl.ridge_regression(y, tx, l)[0]) for l in (0, 0.1, 10)]
    assert norms[0] > norms[1] > norms[2]


def test_reg_logistic_lambda_zero_equals_logistic(data):
    _, y01, tx = data
    w_l, loss_l = impl.logistic_regression(y01, tx, np.zeros(4), 50, 0.5)
    w_r, loss_r = impl.reg_logistic_regression(y01, tx, 0.0, np.zeros(4), 50, 0.5)
    np.testing.assert_allclose(w_r, w_l)
    np.testing.assert_allclose(loss_r, loss_l)


def test_reg_logistic_returns_unpenalized_loss(data):
    _, y01, tx = data
    w, loss = impl.reg_logistic_regression(y01, tx, 1.0, np.ones(4), 3, 0.1)
    np.testing.assert_allclose(loss, impl.compute_logistic_loss(y01, tx, w))


def test_logistic_loss_decreases(data):
    _, y01, tx = data
    losses = [
        impl.logistic_regression(y01, tx, np.zeros(4), it, 0.5)[1]
        for it in (0, 10, 100)
    ]
    assert losses[0] > losses[1] > losses[2]


def test_logistic_is_numerically_stable_for_large_scores():
    tx = np.array([[1.0, 1000.0], [1.0, -1000.0]])
    y = np.array([0.0, 1.0])
    w = np.array([0.0, 1.0])
    loss = impl.compute_logistic_loss(y, tx, w)
    np.testing.assert_allclose(loss, 1000.0)
    assert np.all(np.isfinite(impl.compute_logistic_gradient(y, tx, w)))
    with np.errstate(over="raise", invalid="raise"):
        np.testing.assert_allclose(
            impl.sigmoid(np.array([-800.0, 0.0, 800.0])), [0, 0.5, 1]
        )
