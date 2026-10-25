"""The six mandatory ML methods of Project 1 (CS-433, EPFL).

Conventions (checked against the public grading tests):
    * y has shape (N,), tx has shape (N, D), w has shape (D,).
    * MSE loss is L(w) = 1/(2N) * ||y - tx w||^2.
    * Logistic regression expects labels y in {0, 1}.
    * Every function returns (w, loss): the last weight vector and the
      loss evaluated at that vector. Regularized methods return the loss
      without the penalty term.
    * Iterative methods run exactly max_iters update steps; max_iters = 0
      returns initial_w (as a copy) and its loss.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Loss and gradient helpers
# ---------------------------------------------------------------------------


def compute_mse(y, tx, w):
    """Compute the MSE loss 1/(2N) * ||y - tx w||^2.

    Args:
        y: numpy array of shape (N,).
        tx: numpy array of shape (N, D).
        w: numpy array of shape (D,).

    Returns:
        Scalar numpy float.
    """
    e = y - tx @ w
    return 0.5 * np.mean(e**2)


def compute_mse_gradient(y, tx, w):
    """Compute the gradient of the MSE loss at w: -1/N * tx^T (y - tx w).

    Args:
        y: numpy array of shape (N,).
        tx: numpy array of shape (N, D).
        w: numpy array of shape (D,).

    Returns:
        numpy array of shape (D,).
    """
    e = y - tx @ w
    return -tx.T @ e / len(y)


def sigmoid(t):
    """Apply the logistic function 1 / (1 + exp(-t)) elementwise.

    Written as exp(-log(1 + exp(-t))) so that large |t| does not overflow.

    Args:
        t: scalar or numpy array.

    Returns:
        Scalar or numpy array of the same shape, with values in [0, 1].
    """
    return np.exp(-np.logaddexp(0, -t))


def compute_logistic_loss(y, tx, w):
    """Compute the mean negative log-likelihood of logistic regression.

    Uses the identity -[y log s(z) + (1 - y) log(1 - s(z))] = log(1 + e^z) - y z
    with z = tx w, evaluated with logaddexp for numerical stability.

    Args:
        y: numpy array of shape (N,), labels in {0, 1}.
        tx: numpy array of shape (N, D).
        w: numpy array of shape (D,).

    Returns:
        Scalar numpy float.
    """
    z = tx @ w
    return np.mean(np.logaddexp(0, z) - y * z)


def compute_logistic_gradient(y, tx, w):
    """Compute the gradient of the logistic loss at w: 1/N * tx^T (s(tx w) - y).

    Args:
        y: numpy array of shape (N,), labels in {0, 1}.
        tx: numpy array of shape (N, D).
        w: numpy array of shape (D,).

    Returns:
        numpy array of shape (D,).
    """
    return tx.T @ (sigmoid(tx @ w) - y) / len(y)


# ---------------------------------------------------------------------------
# Mandatory methods
# ---------------------------------------------------------------------------


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression with full gradient descent on the MSE loss.

    Args:
        y: numpy array of shape (N,).
        tx: numpy array of shape (N, D).
        initial_w: numpy array of shape (D,), not modified.
        max_iters: number of gradient steps.
        gamma: step size.

    Returns:
        w: numpy array of shape (D,), weights after max_iters steps.
        loss: scalar, MSE loss at w.
    """
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        w = w - gamma * compute_mse_gradient(y, tx, w)
    return w, compute_mse(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression with stochastic gradient descent on the MSE loss.

    Each step uses one datapoint drawn uniformly at random (mini-batch size 1)
    with np.random, so call np.random.seed beforehand for reproducible runs.

    Args:
        y: numpy array of shape (N,).
        tx: numpy array of shape (N, D).
        initial_w: numpy array of shape (D,), not modified.
        max_iters: number of SGD steps.
        gamma: step size.

    Returns:
        w: numpy array of shape (D,), weights after max_iters steps.
        loss: scalar, MSE loss at w on the full dataset (y, tx).
    """
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        n = np.random.randint(len(y))
        w = w - gamma * compute_mse_gradient(y[n : n + 1], tx[n : n + 1], w)
    return w, compute_mse(y, tx, w)


def least_squares(y, tx):
    """Least squares regression by solving the normal equations tx^T tx w = tx^T y.

    Args:
        y: numpy array of shape (N,).
        tx: numpy array of shape (N, D), with tx^T tx invertible.

    Returns:
        w: numpy array of shape (D,), optimal weights.
        loss: scalar, MSE loss at w.
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w, compute_mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Ridge regression by solving (tx^T tx + 2 N lambda_ I) w = tx^T y.

    This minimizes 1/(2N) ||y - tx w||^2 + lambda_ ||w||^2. All weights,
    including a bias column if present, are penalized.

    Args:
        y: numpy array of shape (N,).
        tx: numpy array of shape (N, D).
        lambda_: regularization strength, >= 0.

    Returns:
        w: numpy array of shape (D,), optimal weights.
        loss: scalar, MSE loss at w without the penalty term.
    """
    n, d = tx.shape
    a = tx.T @ tx + 2 * n * lambda_ * np.eye(d)
    w = np.linalg.solve(a, tx.T @ y)
    return w, compute_mse(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression with full gradient descent.

    Args:
        y: numpy array of shape (N,), labels in {0, 1}.
        tx: numpy array of shape (N, D).
        initial_w: numpy array of shape (D,), not modified.
        max_iters: number of gradient steps.
        gamma: step size.

    Returns:
        w: numpy array of shape (D,), weights after max_iters steps.
        loss: scalar, mean negative log-likelihood at w.
    """
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        w = w - gamma * compute_logistic_gradient(y, tx, w)
    return w, compute_logistic_loss(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """L2-regularized logistic regression with full gradient descent.

    Minimizes NLL(w) + lambda_ ||w||^2, so the penalty adds 2 lambda_ w to the
    gradient. All weights, including a bias column if present, are penalized.

    Args:
        y: numpy array of shape (N,), labels in {0, 1}.
        tx: numpy array of shape (N, D).
        lambda_: regularization strength, >= 0.
        initial_w: numpy array of shape (D,), not modified.
        max_iters: number of gradient steps.
        gamma: step size.

    Returns:
        w: numpy array of shape (D,), weights after max_iters steps.
        loss: scalar, mean negative log-likelihood at w without the penalty term.
    """
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad
    return w, compute_logistic_loss(y, tx, w)
