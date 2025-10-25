"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, _ = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n, K))
    ll = 0.0

    for i in range(n):
        # 1. Calculate joint probability P(x_i, j) = p_j * N(x_i | mu_j, var_j)
        for j in range(K):
            likelihood = gaussian(X[i], mixture.mu[j], mixture.var[j])
            post[i, j] = mixture.p[j] * likelihood

        # 2. Calculate P(x_i) = sum_j P(x_i, j)
        total_prob_xi = np.sum(post[i, :])

        # 3. Calculate log-likelihood
        if total_prob_xi > 0:
            ll += np.log(total_prob_xi)

        # 4. Normalize to get posterior P(j | x_i)
        if total_prob_xi > 0:
            post[i, :] = post[i, :] / total_prob_xi
        else:
            # Handle case where all probabilities are zero
            post[i, :] = 1.0 / K  # Assign uniform probability

    return post, ll


def gaussian(x: np.ndarray, mean: np.ndarray, var: float) -> float:
    """
    Computes the probability of vector x under a spherical Gaussian
    N(x | mean, var*I)

    Args:
        x: (d,) array holding the vector's coordinates
        mean: (d,) mean of the gaussian
        var: variance of the gaussian

    Returns:
        float: the probability p(x)
    """
    d = x.shape[0]
    log_prob = -d / 2.0 * np.log(2 * np.pi * var)
    log_prob -= 0.5 * np.sum((x - mean) ** 2) / var

    return np.exp(log_prob)


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape

    # 1. Calculate soft counts n_hat_j
    n_hat = post.sum(axis=0)

    # 2. Update mixing coefficients (p_j)
    p = n_hat / n

    # Initialize new means and variances
    mu = np.zeros((K, d))
    var = np.zeros(K)

    for j in range(K):
        # 3. Update mean mu_j
        mu[j, :] = (X * post[:, j, None]).sum(axis=0) / n_hat[j]

        # 4. Update variance var_j
        sse = ((mu[j] - X) ** 2).sum(axis=1) @ post[:, j]
        var[j] = sse / (d * n_hat[j])

    return GaussianMixture(mu, var, p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    prev_ll = None
    ll = None
    first_iter = True

    # Critério de convergência do projeto:
    # new_log-likelihood - old_log-likelihood <= 10^-6 * |new_log-likelihood|
    convergence_threshold = 1e-6

    while (prev_ll is None or (ll - prev_ll > convergence_threshold * np.abs(ll))):
        prev_ll = ll

        # E-step
        post, ll = estep(X, mixture)

        # Imprime o log-likelihood após a primeira iteração para o benchmark
        if first_iter:
            print(f"[EM Run] Log-likelihood after 1st E-step: {ll:.4f}")
            first_iter = False

        # M-step (chama a nossa versão com 2 argumentos)
        mixture = mstep(X, post)

    return mixture, post, ll
    raise NotImplementedError
