import numpy as np
import csv
import sys

n = sys.argv[1]

np.random.seed(42)  # per avere sempre gli stessi dati

x = np.linspace(-3, 3, int(n))
#y = 1 / (x**2 + 1)
y = 2*x + 1

noise = np.random.normal(0, 0.05, size=x.shape)
y_noisy = y + noise

with open("dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x", "y"])
    for xi, yi in zip(x, y_noisy):
        writer.writerow([xi, yi])