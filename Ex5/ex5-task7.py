"""
Relative-efficiency plots for crude Monte Carlo vs. importance sampling
tail-probability estimators  P(Z>a),  Z~N(0,1).

Saves one PNG per ‘a’ value, e.g.  rel_eff_a2.png .
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# ───────────────────────────────────────────────────────────────────────
# 1.  Building blocks: estimators
# ───────────────────────────────────────────────────────────────────────
def crude_mc_tail(a, n, rng):
    """Crude Monte-Carlo estimate and variance of P[Z>a]."""
    z = rng.standard_normal(n)
    w = (z > a).astype(float)
    return w.mean(), w.var(ddof=1) / n


def is_tail(a, n, sigma2, rng):
    """Importance-sampling estimate and variance with G=N(a,σ²)."""
    z = rng.normal(loc=a, scale=np.sqrt(sigma2), size=n)

    # log φ − log g
    log_phi = -0.5 * z**2 - 0.5 * np.log(2 * np.pi)
    log_g   = -0.5 * ((z - a) ** 2) / sigma2 - 0.5 * np.log(2 * np.pi * sigma2)
    w = np.exp(log_phi - log_g)

    g = (z > a) * w
    return g.mean(), g.var(ddof=1) / n


# ───────────────────────────────────────────────────────────────────────
# 2.  Run experiments and collect RE = Var_CMC / Var_IS
# ───────────────────────────────────────────────────────────────────────
def gather_re(a_vals, n_vals, sigma2_list, seed=42):
    rng_master = np.random.default_rng(seed)
    out = {}
    for a in a_vals:
        for s2 in sigma2_list:
            re_vec = []
            for n in n_vals:
                rng_cmc = np.random.default_rng(rng_master.integers(1 << 63))
                rng_is  = np.random.default_rng(rng_master.integers(1 << 63))
                _, var_cmc = crude_mc_tail(a, n, rng_cmc)
                _, var_is  = is_tail(a, n, s2, rng_is)
                re_vec.append(var_cmc / var_is)
            out[(a, s2)] = np.array(re_vec)
    return out


# ───────────────────────────────────────────────────────────────────────
# 3.  Plotting helper
# ───────────────────────────────────────────────────────────────────────
def plot_re(results, a_vals, n_vals, sigma2_list, prefix="rel_eff_a"):
    for a in a_vals:
        plt.figure(figsize=(7, 5))
        for s2 in sigma2_list:
            plt.plot(
                n_vals,
                results[(a, s2)],
                marker="o",
                label=fr"$\sigma^2={s2}$"
            )
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Sample size  $n$")
        plt.ylabel(r"Relative efficiency  $\mathrm{Var}_{\mathrm{CMC}}/\mathrm{Var}_{\mathrm{IS}}$")
        plt.title(fr"Importance-sampling gain for $a={a}$")
        plt.grid(True, which="both", ls=":")
        plt.legend()
        fname = f"{prefix}{a}.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f"Saved {fname}")


# ───────────────────────────────────────────────────────────────────────
# 4.  Main: tweak settings here if desired
# ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    a_values      = (2, 4)
    n_values      = (10**3, 10**4, 10**5)
    sigma2_values = (0.5, 1.0, 2.0)

    re_dict = gather_re(a_values, n_values, sigma2_values)
    plot_re(re_dict, a_values, n_values, sigma2_values)
