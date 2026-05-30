import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# LETTURA DATASET
# formato:
# x,y
# =========================================================

data = np.loadtxt("shuffled_data.txt", delimiter=",")

x_train = data[:, 0]
y_train = data[:, 1]

# =========================================================
# FUNZIONI DI ATTIVAZIONE
# =========================================================

def tanh(x):
    return np.tanh(x)

def tanh_derivative(output):
    return 1 - output**2


# =========================================================
# INIZIALIZZAZIONE PESI
# =========================================================

rng = np.random.default_rng(42)

#w1 = rng.uniform(-0.5, 0.5)
w1 = -0.12546
#print("valore inizialie w1: ", w1)
b1 = 0.0

w2 = 0.296543
#print("valore inizialie w2: ", w2)
b2 = 0.0

w3 = 0.450714
#print("valore inizialie w3: ", w3)
b3 = 0.0

# =========================================================
# IPERPARAMETRI
# =========================================================

learning_rate = 5e-3
epochs = 2000

# =========================================================
# TRAINING
# =========================================================

loss_history = []

for epoch in range(epochs):

    # gradienti accumulati
    dw1 = 0.0
    db1 = 0.0

    dw2 = 0.0
    db2 = 0.0

    dw3 = 0.0
    db3 = 0.0

    loss = 0.0

    # =====================================================
    # FULL BATCH
    # =====================================================

    for x, y_target in zip(x_train, y_train):

        # -------------------------------------------------
        # FORWARD
        # -------------------------------------------------

        z1 = x * w1 + b1
        #print("z1: ", z1)
        a1 = tanh(z1)
        #print("tanh(z1): ", a1)

        z2 = a1 * w2 + b2
        #print("z2: ", z2)
        a2 = tanh(z2)
        #print("a2: ", a2)

        z3 = a2 * w3 + b3
        #print("z3: ", z3)
        y_pred = z3

        # -------------------------------------------------
        # LOSS
        # -------------------------------------------------

        error = y_pred - y_target
        loss += error**2

        # -------------------------------------------------
        # BACKPROPAGATION
        # -------------------------------------------------

        dL_dy = 2 * error

        # output
        dL_dw3 = dL_dy * a2
        dL_db3 = dL_dy

        # hidden 2
        dL_da2 = dL_dy * w3
        dL_dz2 = dL_da2 * tanh_derivative(a2)

        dL_dw2 = dL_dz2 * a1
        dL_db2 = dL_dz2

        # hidden 1
        dL_da1 = dL_dz2 * w2
        dL_dz1 = dL_da1 * tanh_derivative(a1)

        dL_dw1 = dL_dz1 * x
        dL_db1 = dL_dz1

        # -------------------------------------------------
        # ACCUMULO GRADIENTI
        # -------------------------------------------------

        dw1 += dL_dw1
        db1 += dL_db1

        dw2 += dL_dw2
        db2 += dL_db2

        dw3 += dL_dw3
        db3 += dL_db3

    # =====================================================
    # MEDIA GRADIENTI
    # =====================================================

    n = len(x_train)

    dw1 /= n
    db1 /= n

    dw2 /= n
    db2 /= n

    dw3 /= n
    db3 /= n

    # =====================================================
    # UPDATE PESI
    # =====================================================

    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1

    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2

    w3 -= learning_rate * dw3
    b3 -= learning_rate * db3

    # =====================================================
    # LOSS
    # =====================================================

    loss /= n
    loss_history.append(loss)

    
    print(f"Epoch {epoch} - Loss: {loss:.8f}")

# =========================================================
# INFERENZA FINALE
# =========================================================

y_pred_final = []

for x in x_train:

    z1 = x * w1 + b1
    a1 = tanh(z1)

    z2 = a1 * w2 + b2
    a2 = tanh(z2)

    z3 = a2 * w3 + b3

    y_pred_final.append(z3)

y_pred_final = np.array(y_pred_final)

# =========================================================
# RIORDINO
# =========================================================

indices = np.argsort(x_train)

x_sorted = x_train[indices]
y_target_sorted = y_train[indices]
y_pred_sorted = y_pred_final[indices]

# =========================================================
# SALVATAGGIO RESULT
# =========================================================

result = np.column_stack((x_sorted, y_pred_sorted))

np.savetxt(
    "result.txt",
    result,
    delimiter=",",
    fmt="%.6f"
)

# =========================================================
# GRAFICO
# =========================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    x_sorted,
    y_target_sorted,
    color='red',
    s=10,
    label='Target'
)

plt.plot(
    x_sorted,
    y_pred_sorted,
    linewidth=2,
    label='Prediction'
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Neural Network Regression")

plt.legend()
plt.grid(True)

plt.show()