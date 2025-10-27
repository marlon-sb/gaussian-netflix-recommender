"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def log_gaussian(x: np.ndarray, mean: np.ndarray, var: float) -> float:
    """Computes the log probablity of vector x under a normal distribution

    Args:
        x: (d, ) array holding the vector's coordinates
        mean: (d, ) mean of the gaussian
        var: variance of the gaussian

    Returns:
        float: the log probability
    """
    # Ensure variance is not too small
    if var < 1e-10:
        var = 1e-10

    d = len(x)
    log_prob = -d / 2.0 * np.log(2 * np.pi * var)
    log_prob -= 0.5 * ((x - mean) ** 2).sum() / var

    return log_prob


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, _ = X.shape
    K, _ = mixture.mu.shape

    # log_post will be in log-space, post will be in linear-space
    log_post = np.zeros((n, K))
    ll = 0.0

    for i in range(n):
        # 1. Find observed ratings for user i
        mask = (X[i, :] != 0)

        # 2. Calculate log-joint prob log P(x_i, j) for each cluster j
        for j in range(K):
            # Calculate log P(x_i | j) using only observed dimensions
            log_likelihood_j = log_gaussian(X[i, mask], mixture.mu[j, mask],
                                            mixture.var[j])

            # log P(x_i, j) = log(p_j) + log P(x_i | j)
            # Use + 1e-16 for stability as per hint
            log_post[i, j] = np.log(mixture.p[j] + 1e-16) + log_likelihood_j

        # 3. Calculate log-likelihood for user i: log P(x_i)
        total_ll_i = logsumexp(log_post[i, :])

        # 4. Sum to get total log-likelihood
        ll += total_ll_i

        # 5. Normalize to get log-posterior log P(j | x_i)
        # log P(j | x_i) = log P(x_i, j) - log P(x_i)
        log_post[i, :] = log_post[i, :] - total_ll_i

    # Return posterior in linear space (for m-step) and log-likelihood
    return np.exp(log_post), ll



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape

    # Update p (mixing coefficients)
    n_hat = post.sum(axis=0)
    p = n_hat / n

    # Update mu (means)
    mu = mixture.mu.copy()
    var = np.zeros(K)

    for j in range(K):
        sse = 0.0
        weight_sum = 0.0

        for l in range(d):
            mask = (X[:, l] != 0)
            n_sum = post[mask, j].sum()

            if n_sum >= 1:
                mu[j, l] = (X[mask, l] @ post[mask, j]) / n_sum

            sse += ((mu[j, l] - X[mask, l]) ** 2) @ post[mask, j]
            weight_sum += n_sum

        # Update variance (with minimum)
        var[j] = sse / weight_sum
        if var[j] < min_variance:
            var[j] = min_variance

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
    convergence_threshold = 1e-6

    while (prev_ll is None or (ll - prev_ll > convergence_threshold * np.abs(ll))):
        prev_ll = ll

        # E-step
        post, ll = estep(X, mixture)

        # M-step
        mixture = mstep(X, post, mixture)

    return mixture, post, ll


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
