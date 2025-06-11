import numpy as np
from scipy.stats import norm   # only for the z-quantile

def control_variate_exp_integral(n=100, seed=None, alpha=0.05):
    """
    Estimate ∫₀¹ eˣ dx using a control variate.

    Control variate chosen: h(U)=U with E[h]=½, where U∼Unif(0,1).

    Parameters
    ----------
    n : int
        Number of uniform samples (matches the 100 evaluations
        used in the crude and antithetic estimators).
    seed : int or None
        RNG seed for reproducibility.
    alpha : float
        Significance level (default 0.05 → 95 % CI).

    Returns
    -------
    mu_hat : float
        Point estimate of the integral.
    ci : tuple (lower, upper)
        Two-sided (1–α) confidence interval.
    beta_hat : float
        Estimated optimal regression coefficient for the control variate.
    """
    rng = np.random.default_rng(seed)
    u  = rng.uniform(0.0, 1.0, size=n)   # U_i
    y  = np.exp(u)                       # f(U_i) = e^{U_i}
    h  = u                               # control variate h(U)=U
    mu_h = 0.5                           # E[U] for Uniform(0,1)

    # Estimate optimal β = Cov(f,h)/Var(h)
    cov_yh = np.cov(y, h, ddof=1)[0, 1]
    var_h  = h.var(ddof=1)
    beta_hat = cov_yh / var_h

    # Control-variate estimator g_i = f(U_i) − β̂ (h(U_i) − μ_h)
    g = y - beta_hat * (h - mu_h)

    mu_hat = g.mean()
    se     = g.std(ddof=1) / np.sqrt(n)  # SE of the mean of g_i
    z      = norm.ppf(1.0 - alpha / 2.0)
    ci     = (mu_hat - z * se, mu_hat + z * se)
    return mu_hat, ci, beta_hat


if __name__ == "__main__":
    estimate, (lo, hi), beta = control_variate_exp_integral(n=100, seed=42)
    print(f"Control-variate estimate: {estimate:.6f}")
    print(f"95% confidence interval: ({lo:.6f}, {hi:.6f})")
    print(f"Estimated β: {beta:.4f}")
