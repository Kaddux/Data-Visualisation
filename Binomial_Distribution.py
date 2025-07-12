import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

def fit_binomial_distribution(n, p):
    x = np.arange(n + 1)
    y = binom.pmf(x, n, p)

    plt.bar(x, y, alpha=0.6, label=f'Histogram (n={n}, p={p})')
    plt.plot(x, y, 'o-', label=f'PMF (n={n}, p={p})')

plt.figure(figsize=(10, 6))
plt.xlabel('Number of Successes')
plt.ylabel('Probability')
plt.title('Binomial Distribution for Different (n, p) Values')

for n, p in [(10, 0.5), (20, 0.5), (10, 0.8), (20, 0.8)]:
    fit_binomial_distribution(n, p)

plt.legend()
plt.grid(True)
plt.show()
