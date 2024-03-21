import numpy as np
import matplotlib.pyplot as plt


def discretisation(T, N, m, X_0=4.0, Y_0=1.0):
    # construct base arrays for the X and Y values
    X = np.zeros(shape=(m, N))
    X[:, 0] = X_0
    Y = np.zeros(shape=(m, N))
    Y[:, 0] = Y_0
    M = np.ones(shape=(1, N))

    r = 0.0
    dt = T / N
    sqrt_dt = np.sqrt(dt)
    W = np.random.normal(0, 1, size=(m, N))
    # if m > 1:
    #     W[:, 0] = (W[:, 0] - np.mean(W[:, 0])) / np.var(W[:, 0])

    for i in range(1, N):
        # if m > 1:
        #     W[:, i] = (W[:, i] - np.mean(W[:, i]))/np.var(W[:, i])
        X[:, i] = X[:, i-1] + r*X[:, i-1]*dt + 0.38*X[:, i-1]*sqrt_dt*(W[:, i])
        Y[:, i] = Y[:, i-1] + r*Y[:, i-1]*dt + 0.15*Y[:, i-1]*sqrt_dt*(W[:, i])
        M[:, i] = M[:, i-1] + r*M[:, i-1]*dt

    return X, Y, M[0]


def compute_V(X, Y, M, K):
    return np.mean(np.maximum((X[:, -1] - Y[:, -1]) / 2, K)) / M[-1]


Ks = np.arange(0, 10.05, 0.05)
T = 7
N = T * 100
m = 5000
x, y, m = discretisation(T, N, m, X_0=4.0, Y_0=1.0)
V = np.zeros(len(Ks))
i = 0
for k in Ks:
    V[i] = compute_V(x, y, m, k)
    i += 1
plt.plot(Ks, V)
plt.legend()
plt.show()
