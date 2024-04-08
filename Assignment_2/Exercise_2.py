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


def discretisation(T, n, m, S_0=1.0, sigma=0.1, r=0, standardisation=True):
    # construct base arrays for the X and Y values
    X = np.zeros(shape=(m, n))
    X[:, 0] = S_0
    Y = np.zeros(shape=(m, n))
    Y[:, 0] = S_0

    dt = T / n
    sqrt_dt = np.sqrt(dt)
    W = np.random.normal(0, 1, size=(m, n))
    if m > 1 and standardisation:
        W[:, 0] = (W[:, 0] - np.mean(W[:, 0])) / np.std(W[:, 0])

    for i in range(1, n):
        Y[:, i] = Y[:, i-1] + r*Y[:, i-1]*dt + sigma*Y[:, i-1]*sqrt_dt*(W[:, i])
        if m > 1 and standardisation:
            W[:, i] = (W[:, i] - np.mean(W[:, i]))/np.std(W[:, i])
        X[:, i] = X[:, i-1] + r*X[:, i-1]*dt + sigma*X[:, i-1]*sqrt_dt*(W[:, i])

    return X[:, -1], Y[:, -1], np.exp(r*T)


def compute_V(X, M, K):
    return np.mean(np.maximum(X - K, 0)) / M


T = 7
n = T * 100
ms = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
r = 0
sigma = 0.1
S_0 = 1
k = 1

sample_size = 100
standardised = np.zeros(shape=(sample_size, len(ms)))
non_standardised = np.zeros(shape=(sample_size, len(ms)))
BS = [BS_CALL(S_0, k, T, r, sigma) for _ in range(len(ms))]
for i in range(sample_size):
    seed = np.random.get_state()[1][0]
    for j, m in enumerate(ms):
        np.random.seed(seed)
        x, y, disc = discretisation(T, n, m, S_0=S_0, sigma=sigma, r=r, standardisation=True)

        standardised[i, j] = compute_V(x, disc, k)
        non_standardised[i, j] = compute_V(y, disc, k)

plt.plot(ms, np.mean(np.abs(standardised - np.array(BS)), axis=0), label="standardised")
plt.plot(ms, np.mean(np.abs(non_standardised - np.array(BS)), axis=0), label="non standardised")
plt.legend()
plt.xlabel("Number of paths")
plt.ylabel("Mean error between simulation and Black-Schöles")
plt.xscale("log")
plt.show()

