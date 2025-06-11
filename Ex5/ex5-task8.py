import numpy as np, math
from scipy.optimize import minimize_scalar

# analytic variance of the IS weight Y = wλ(X)
def var_is(lmbda):
    num  = 1 - math.exp(-lmbda)
    EY2  = num / lmbda * (math.exp(lmbda + 2) - 1) / (lmbda + 2)
    return EY2 - (math.e - 1)**2          # subtract true mean²

res = minimize_scalar(var_is, bounds=(1e-4, 20), method='bounded')
lam_star = res.x                            # ≈ 1.0e-3
var_star = res.fun

# Monte-Carlo check at λ*
rng = np.random.default_rng(42)
n   = 100_000
u   = rng.random(n)
x   = -np.log(1 - u * (1 - np.exp(-lam_star))) / lam_star   # inverse-cdf
weights = (1 - np.exp(-lam_star)) / lam_star * np.exp((lam_star + 1) * x)
mc_mean, mc_var = weights.mean(), weights.var(ddof=1)

print(lam_star, var_star, mc_mean, mc_var)
