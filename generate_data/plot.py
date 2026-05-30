import matplotlib.pyplot as plt
import numpy as np

# Carica file
data = np.loadtxt("data.txt", delimiter=",")
result = np.loadtxt("result.txt", delimiter=",")

# Colonne
x_data = data[:, 0]
y_target = data[:, 1]

x_result = result[:, 0]
y_pred = result[:, 1]

# Grafico
plt.figure(figsize=(10, 6))

# Target in rosso
plt.scatter(x_data, y_target, s=10, color='red', label='y_target')

# Predizione rete
plt.plot(x_result, y_pred, linewidth=2, label='y_pred')

# Dettagli
plt.xlabel("x")
plt.ylabel("y")
plt.title("Confronto y_target vs y_pred")
plt.legend()
plt.grid(True)

plt.show()