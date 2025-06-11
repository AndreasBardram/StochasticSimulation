import numpy as np
import matplotlib.pyplot as plt

# Linear Congruential Generator

def lcg(a, b, M, seed, n):
    """
    Generate `n` uniform(0,1) pseudorandoms using integer LCG:
      X_{k+1} = (a * X_k + b) mod M
    Returns an array of length `n` with values X_k / M in [0, 1).
    """
    x = seed
    result = []
    for _ in range(n):
        x = (a * x + b) % M
        result.append(x / M)
    return np.array(result)

a, b, M = 16807, 0, 2**31 - 1
seed, n = 1, 10000
u = lcg(a, b, M, seed, n)
print(u)

# Histograms with 10 bins
# Define 10 bins on [0,1]:
bins = np.linspace(0, 1, 11)  # edges at 0.0, 0.1, 0.2, …, 1.0
counts = np.zeros(10, dtype=int)

# Assign each u[i] to a bin index 0..9
for val in u:
    idx = min(int(val * 10), 9)
    counts[idx] += 1

plt.figure(figsize=(5,4))
plt.bar(
    (bins[:-1] + bins[1:]) / 2,
    counts,
    width=0.1,
    edgecolor='black'
)
plt.title("Histogram of 10 000 LCG Numbers (10 Bins)")
plt.xlabel("Interval")
plt.ylabel("Count")
plt.xticks(
    np.linspace(0.05, 0.95, 10),
    [f"{i/10:.1f}–{(i+1)/10:.1f}" for i in range(10)],
    rotation=45,
    fontsize=8
)
plt.tight_layout()
plt.show()

# Scatter plot
plt.figure(figsize=(5,5))
plt.scatter(u[:-1], u[1:], s=1)
plt.title("Scatter Plot: U[i] vs U[i+1]")
plt.xlabel("U[i]")
plt.ylabel("U[i+1]")
plt.tight_layout()
plt.show()

