import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


N = norm.cdf


def BS_CALL(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N(d1) - K * np.exp(-r*T) * N(d2)


def BS_PUT(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K*np.exp(-r*T)*N(-d2) - S*N(-d1)


def discretisation(T, n, m, S_0=1.0, X_sigma=0.1, r=0, standardisation=True):
    # construct base arrays for the X and Y values
    X = np.zeros(shape=(m, n))
    X[:, 0] = S_0
    Y = np.zeros(shape=(m, n))
    Y[:, 0] = S_0

    dt = T / n
    sqrt_dt = np.sqrt(dt)
    W = np.random.normal(0, 1, size=(m, n))
    if m > 1 and standardisation:
        W[:, 0] = (W[:, 0] - np.mean(W[:, 0])) / np.var(W[:, 0])

    for i in range(1, n):
        Y[:, i] = Y[:, i-1] + r*Y[:, i-1]*dt + X_sigma*Y[:, i-1]*sqrt_dt*(W[:, i])
        if m > 1 and standardisation:
            W[:, i] = (W[:, i] - np.mean(W[:, i]))/np.var(W[:, i])
        X[:, i] = X[:, i-1] + r*X[:, i-1]*dt + X_sigma*X[:, i-1]*sqrt_dt*(W[:, i])

    return X[:, -1], Y[:, -1], np.exp(r*T)


def compute_V(X, M, K):
    return np.mean(np.maximum(X - K, 0)) / M


T = 7
n = T * 100
ms = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
r = 0
sigma = 0.1
S_0 = 1


standardised = []
non_standardised = []
BS = []
for m in ms:
    np.random.seed(0)
    x, y, disc = discretisation(T, n, m, S_0=S_0, X_sigma=sigma, r=r, standardisation=True)
    # np.random.seed(0)
    # y, disc = discretisation(T, n, m, X_0=S_0, X_sigma=sigma, r=r, standardisation=False)

    k = 1
    standardised.append(compute_V(x, disc, k))
    non_standardised.append(compute_V(y, disc, k))
    BS.append(BS_CALL(S_0, k, T, r, sigma))

plt.plot(ms, standardised, label="standardised")
plt.plot(ms, non_standardised, label="non standardised")
plt.plot(ms, BS, label="BS")
plt.legend()
plt.xscale("log")
plt.show()

plt.plot(ms, np.abs(np.array(standardised) - np.array(BS)), label="standardised")
plt.plot(ms, np.abs(np.array(non_standardised) - np.array(BS)), label="non standardised")
plt.legend()
plt.xscale("log")
plt.show()

