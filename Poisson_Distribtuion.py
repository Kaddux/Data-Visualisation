import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

lambda_val = 17

k = np.arange(poisson.ppf(0.000, lambda_val), poisson.ppf(0.999, lambda_val))

probabilities = poisson.pmf(k, lambda_val)

plt.figure(figsize=(8, 6))
plt.bar(k, probabilities, color='blue')
plt.plot(k, probabilities, 'ro-', label=f'Poisson Distribution (λ = {lambda_val})')
plt.xlabel("Number of occurrences (k)")
plt.ylabel("Probability")
plt.title(f"Poisson Distribution with λ = {lambda_val}")
plt.legend()
plt.grid(True)
plt.show()
