import numpy as np
from scipy.stats import norm   # only used to get the 1–α/2 quantile

def monte_carlo_exp_integral(n=100, seed=None, alpha=0.05):
    """
    Estimate ∫₀¹ eˣ dx via crude Monte Carlo.

    Parameters
    ----------
    n : int
        Number of uniform samples on [0, 1].
    seed : int or None
        Random-seed for reproducibility.
    alpha : float
        Significance level for the confidence interval (default 0.05 → 95 %).

    Returns
    -------
    mu_hat : float
        Point estimate of the integral.
    ci : tuple (lower, upper)
        Two-sided (1–α) confidence interval.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    y = np.exp(x)

    mu_hat = y.mean()
    se = y.std(ddof=1) / np.sqrt(n)          # classic SE for the mean
    z = norm.ppf(1 - alpha / 2)              # 1.96 for a 95 % CI
    ci = (mu_hat - z * se, mu_hat + z * se)
    return mu_hat, ci

if __name__ == "__main__":
    estimate, (lower, upper) = monte_carlo_exp_integral(n=100, seed=42)
    print(f"Monte Carlo estimate: {estimate:.6f}")
    print(f"95% confidence interval: ({lower:.6f}, {upper:.6f})")