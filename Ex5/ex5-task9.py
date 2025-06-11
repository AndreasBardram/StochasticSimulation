import numpy as np

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def sample_pareto(alpha, size, xm=1.0, rng=None):
    """Draw from Pareto(α, x_min = xm) via inverse CDF."""
    if rng is None:
        rng = np.random.default_rng()
    u = rng.random(size)
    return xm / (1 - u) ** (1 / alpha)

def pareto_mean(alpha, xm=1.0):
    if alpha <= 1:
        raise ValueError("Mean is infinite when α ≤ 1")
    return alpha * xm / (alpha - 1)

# ----------------------------------------------------------------------
# size-biased proposal  g(x) ∝ x f(x)  →  Pareto with shape α−1
# ----------------------------------------------------------------------
def sample_size_biased_pareto(alpha, size, xm=1.0, rng=None):
    return sample_pareto(alpha - 1, size, xm, rng)

def is_estimator_mean(alpha, n=10_000, xm=1.0, seed=0):
    """
    Importance-sampling estimator of the mean using the size-biased proposal.
    Returns (estimate_array, variance_empirical).
    """
    rng = np.random.default_rng(seed)
    # 1) sample X from g   (shape α−1)
    x_g = sample_size_biased_pareto(alpha, n, xm, rng)

    # 2) IS weight  w(x) = f(x) / g(x) = μ / x
    mu = pareto_mean(alpha, xm)
    w   = mu / x_g

    # 3) IS estimate of mean:  Y_i = x_g * w(x_g)  → constant μ
    y = x_g * w
    return y, y.var(ddof=1)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    α  = 2.5
    xm = 1.0
    n  = 1000

    y, var_emp = is_estimator_mean(α, n, xm, seed=42)
    print(f"First few estimates: {y[:5]}")
    print(f"Empirical variance over {n} samples: {var_emp:.3e}")
    print(f"True mean μ = {pareto_mean(α,xm):.4f}")
