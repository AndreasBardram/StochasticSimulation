import numpy as np
from scipy.stats import norm   # only for the z-quantile

def stratified_exp_integral(n_total=100, n_strata=10, seed=None, alpha=0.05):
    """
    Estimate ∫₀¹ eˣ dx by stratified sampling with equal–width strata.

    Parameters
    ----------
    n_total : int
        Total number of function evaluations (default 100).
    n_strata : int
        Number of strata (default 10 → 10 samples per stratum).
        Must divide n_total exactly.
    seed : int or None
        RNG seed for reproducibility.
    alpha : float
        Significance level (default 0.05 → 95 % CI).

    Returns
    -------
    mu_hat : float
        Point estimate of the integral.
    ci : tuple (lower, upper)
        Two-sided (1–α) confidence interval (normal approximation).
    """
    if n_total % n_strata:
        raise ValueError("n_total must be divisible by n_strata for equal allocation.")

    rng = np.random.default_rng(seed)
    m = n_strata
    n_j = n_total // m                 # samples per stratum
    w = 1.0 / m                        # all strata have width 1/m

    mu_hat = 0.0                       # stratified mean accumulator
    var_est = 0.0                      # variance estimator accumulator

    for j in range(m):
        a, b = j / m, (j + 1) / m                       # stratum interval
        u = rng.uniform(a, b, size=n_j)                 # sample inside stratum
        y = np.exp(u)                                   # f(U)
        mean_j = y.mean()
        var_j  = y.var(ddof=1)

        mu_hat += w * mean_j
        var_est += w**2 * var_j / n_j

    se = np.sqrt(var_est)
    z  = norm.ppf(1.0 - alpha / 2.0)
    ci = (mu_hat - z * se, mu_hat + z * se)
    return mu_hat, ci


if __name__ == "__main__":
    estimate, (lo, hi) = stratified_exp_integral(n_total=100, n_strata=10, seed=42)
    print(f"Stratified estimate: {estimate:.6f}")
    print(f"95% confidence interval: ({lo:.6f}, {hi:.6f})")
