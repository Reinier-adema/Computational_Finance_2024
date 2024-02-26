import numpy as np
import matplotlib.pyplot as plt


def discretisation(T, N, m, X_0=4.0, Y_0=1.0):
    # construct base arrays for the X and Y values
    X = np.ones(shape=(m, N))
    X[:, 0] = X_0
    Y = np.ones(shape=(m, N))
    Y[:, 0] = Y_0
    M = np.ones(shape=(1, N))

    dt = T / N
    sqrt_dt = np.sqrt(dt)
    W = np.random.normal(0, 1, size=(m, N))
    if m > 1:
        W[:, 0] = (W[:, 0] - np.mean(W[:, 0])) / np.var(W[:, 0])

    for i in range(1, N):
        if m > 1:
            W[:, i] = (W[:, i] - np.mean(W[:, i]))/np.var(W[:, i])
        X[:, i] = X[:, i-1] + 0.04*X[:, i-1]*dt + 0.38*X[:, i-1]*sqrt_dt*(W[:, i] - W[:, i-1])
        Y[:, i] = Y[:, i-1] + 0.1*Y[:, i-1]*dt + 0.15*Y[:, i-1]*sqrt_dt*(W[:, i] - W[:, i-1])
        M[:, i] = M[:, i-1] + 0.06*M[:, i-1]*dt

    return X, Y, M[0]


T = 7
N = T * 1000
m = 5000
x, y, m = discretisation(T, N, m)


def compute_V(X, Y, M, K, N):
    V = np.zeros(N)
    for i in range(N):
        V[i] = np.mean(np.maximum((X[:, i] - Y[:, i]) / 2, K)) / M[i]
    return V


for k in [0, 0.05, 10]:
    V = compute_V(x, y, m, k, N)
    print(V[-1])
    plt.plot(np.linspace(0, T, N), V, label="strike: " + str(round(k, 2)))
plt.legend()
plt.show()
