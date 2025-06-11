import numpy as np, heapq
from scipy import stats

# ──────────────────────────────────────────────────────────────────────────────
#  1.  Core blocking simulator that takes a *pre-computed* inter-arrival array.
#      (Allows us to feed the *same* base uniforms to both models when we want
#      common random numbers.)
# ──────────────────────────────────────────────────────────────────────────────
def simulate_blocking_given_interarrivals(inter, m_servers, service_sampler, rng):
    arrivals = np.cumsum(inter)
    busy = []                             # min-heap of departure times
    blocked = 0

    for t in arrivals:
        while busy and busy[0] <= t:      # free finished servers
            heapq.heappop(busy)
        if len(busy) < m_servers:         # accept
            s = service_sampler(rng)
            heapq.heappush(busy, t + s)
        else:                             # block
            blocked += 1
    return blocked / len(inter)

# Exponential service times (mean = 8)
def exp_service(rng, mean=8.0):
    return rng.exponential(mean)

# ──────────────────────────────────────────────────────────────────────────────
#  2.  Driver to compare CRN vs independent replications
# ──────────────────────────────────────────────────────────────────────────────
def compare_crn_vs_independent(
    reps=10, N_customers=10_000, m_servers=10,
    mean_service=8.0, seed0=12345
):
    p1, λ1, λ2 = 0.8, 0.8333, 5.0              # hyper-exponential params
    rng_master = np.random.default_rng(seed0)

    diff_crn, diff_ind = [], []

    for r in range(reps):
        # -- master RNG for this replication
        rng = np.random.default_rng(rng_master.integers(1 << 63))

        # ====== Common-Random-Numbers experiment ===========================
        U  = rng.random(N_customers)           # base uniforms  U ~ Unif(0,1)
        Uc = rng.random(N_customers)           # for mixture selection

        inter_poiss = -np.log(U)               # Exp(1)  for Poisson model
        rate        = np.where(Uc < p1, λ1, λ2)
        inter_hyp   = -np.log(U) / rate        # same U   for HyperExp model

        rng_s1 = np.random.default_rng(rng.integers(1 << 63))
        rng_s2 = np.random.default_rng(rng.integers(1 << 63))

        frac_poiss_crn = simulate_blocking_given_interarrivals(
            inter_poiss, m_servers,
            lambda R: exp_service(R, mean_service), rng_s1)

        frac_hyp_crn = simulate_blocking_given_interarrivals(
            inter_hyp, m_servers,
            lambda R: exp_service(R, mean_service), rng_s2)

        diff_crn.append(frac_hyp_crn - frac_poiss_crn)

        # ====== Independent-streams experiment ============================
        rngA = np.random.default_rng(rng.integers(1 << 63))
        rngB = np.random.default_rng(rng.integers(1 << 63))

        inter_poiss_ind = rngA.exponential(1.0, N_customers)

        Usel = rngB.random(N_customers)
        Uexp = rngB.random(N_customers)
        rate_ind = np.where(Usel < p1, λ1, λ2)
        inter_hyp_ind = -np.log(Uexp) / rate_ind

        rng_s3 = np.random.default_rng(rng.integers(1 << 63))
        rng_s4 = np.random.default_rng(rng.integers(1 << 63))

        frac_poiss_ind = simulate_blocking_given_interarrivals(
            inter_poiss_ind, m_servers,
            lambda R: exp_service(R, mean_service), rng_s3)

        frac_hyp_ind = simulate_blocking_given_interarrivals(
            inter_hyp_ind, m_servers,
            lambda R: exp_service(R, mean_service), rng_s4)

        diff_ind.append(frac_hyp_ind - frac_poiss_ind)

    diff_crn = np.array(diff_crn)
    diff_ind = np.array(diff_ind)

    def t_ci(arr):
        m  = arr.mean()
        se = arr.std(ddof=1) / np.sqrt(len(arr))
        hw = stats.t.ppf(0.975, len(arr)-1) * se
        return m, (m - hw, m + hw), se**2

    m_crn, ci_crn, var_crn = t_ci(diff_crn)
    m_ind, ci_ind, var_ind = t_ci(diff_ind)

    print("----- Difference  Δ = P_block(HyperExp) − P_block(Poisson) -----")
    print(f"Common random numbers:  Δ̂ = {m_crn:.5f},  95% CI = {ci_crn}")
    print(f"Independent streams :   Δ̂ = {m_ind:.5f},  95% CI = {ci_ind}")
    print(f"Variance ratio  Var_ind / Var_crn = {var_ind/var_crn:.2f}")

    return (diff_crn, diff_ind, (m_crn, ci_crn), (m_ind, ci_ind))

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    compare_crn_vs_independent()
