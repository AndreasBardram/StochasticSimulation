import numpy as np
from scipy.stats import norm   # only used for the z-quantile

def antithetic_exp_integral(n_pairs=50, seed=None, alpha=0.05):
    """
    Estimate ∫₀¹ eˣ dx using antithetic variates.

    Each pair (U, 1−U) counts as two function evaluations, so with
    n_pairs=50 we match the 100 evaluations used in the crude estimator.

    Parameters
    ----------
    n_pairs : int
        Number of antithetic pairs (total evaluations = 2*n_pairs).
    seed : int or None
        Random seed for reproducibility.
    alpha : float
        Significance level for the confidence interval (default 0.05 → 95 %).

    Returns
    -------
    mu_hat : float
        Point estimate of the integral.
    ci : tuple
        Two-sided (1–α) confidence interval for the estimate.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, size=n_pairs)   # primary uniforms
    y1 = np.exp(u)           # f(U)
    y2 = np.exp(1.0 - u)     # f(1−U)   ← antithetic partner
    g  = 0.5 * (y1 + y2)     # pair-wise average ⇒ single RV per pair

    mu_hat = g.mean()
    se     = g.std(ddof=1) / np.sqrt(n_pairs)   # SE of the mean of pairs
    z      = norm.ppf(1.0 - alpha / 2.0)
    ci     = (mu_hat - z * se, mu_hat + z * se)
    return mu_hat, ci


if __name__ == "__main__":
    estimate, (lo, hi) = antithetic_exp_integral(n_pairs=50, seed=42)
    print(f"Antithetic Monte Carlo estimate: {estimate:.6f}")
    print(f"95% confidence interval: ({lo:.6f}, {hi:.6f})")
