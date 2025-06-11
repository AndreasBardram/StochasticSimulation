import numpy as np, heapq
from scipy import stats

# ──────────────────────────────────────────────────────────────────────────────
# Generic simulator that records both block-indicators Y_i and inter-arrival
# times T_i  (control variate with known mean 1).
# ──────────────────────────────────────────────────────────────────────────────
def simulate_blocking_cv(
    N_customers: int,
    m_servers: int,
    rng: np.random.Generator,
    arrival_sampler,          # fn(rng, size) → array of inter-arrival times
    service_sampler           # fn(rng) → one service time
):
    inter = arrival_sampler(rng, size=N_customers)
    arrivals = np.cumsum(inter)

    busy_pq = []                 # min-heap of departure times
    Y = np.empty(N_customers, dtype=int)   # blocked?  1=yes 0=no

    for k, t in enumerate(arrivals):
        while busy_pq and busy_pq[0] <= t:        # release finished servers
            heapq.heappop(busy_pq)
        if len(busy_pq) < m_servers:
            s = service_sampler(rng)              # accept
            heapq.heappush(busy_pq, t + s)
            Y[k] = 0
        else:
            Y[k] = 1                              # blocked
    return Y, inter


# ──────────────────────────────────────────────────────────────────────────────
# Control-variate estimator for the blocking probability θ = E[Y]
# Uses h(T)=T with known mean μ_h = 1 (exponential(1) inter-arrival times).
# g_i = Y_i − β̂ (T_i − μ_h)
# ──────────────────────────────────────────────────────────────────────────────
def cv_estimate(Y, inter, mu_h=1.0):
    Z = inter - mu_h
    beta_hat = np.cov(Y, Z, ddof=1)[0, 1] / np.var(Z, ddof=1)
    g = Y - beta_hat * Z
    theta_hat = g.mean()
    var_hat   = g.var(ddof=1) / len(g)            # Var(ȳ_g)
    return theta_hat, var_hat, beta_hat


# ──────────────────────────────────────────────────────────────────────────────
# Convenience samplers and a driver that replicates the experiment 10×
# (10 × 10 000 customers) exactly as in Exercise 4, but with the CV estimator.
# ──────────────────────────────────────────────────────────────────────────────
def exp_arrivals(rng, size, rate=1.0):
    return rng.exponential(1 / rate, size=size)

def exp_service(rng, mean=8.0):
    return rng.exponential(mean)

def poisson_exp_control_variates(
    reps=10,               # 10 independent replications
    N_customers=10_000,
    m_servers=10,
    seed0=42
):
    rng_master = np.random.default_rng(seed0)
    est_cv  = []           # control-variate point estimates
    beta    = []           # β̂ values (diagnostic)
    for r in range(reps):
        rng = np.random.default_rng(rng_master.integers(1 << 63))
        Y, inter = simulate_blocking_cv(
            N_customers, m_servers, rng,
            arrival_sampler=lambda R, size: exp_arrivals(R, size, rate=1.0),
            service_sampler=lambda R: exp_service(R, mean=8.0)
        )
        theta_hat, var_hat, beta_hat = cv_estimate(Y, inter)
        est_cv.append(theta_hat)
        beta.append(beta_hat)

    est_cv  = np.array(est_cv)
    beta    = np.array(beta)

    mean_hat = est_cv.mean()
    half_width = stats.t.ppf(0.975, df=reps-1) * est_cv.std(ddof=1) / np.sqrt(reps)
    ci = (mean_hat - half_width, mean_hat + half_width)

    print("Poisson arrivals (control variate):")
    print(f"  Mean blocked fraction = {mean_hat:.5f}")
    print(f"  95% CI = [{ci[0]:.5f}, {ci[1]:.5f}]")
    print(f"  β̂ (replications): {beta.mean():.4f} (avg)")

    return est_cv, beta, ci


if __name__ == "__main__":
    poisson_exp_control_variates()
