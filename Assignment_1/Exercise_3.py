import numpy as np
import matplotlib.pyplot as plt


def euler_discritisation(dynamics, n_paths, N, T, S_0):
    delta_t = T / N
    paths = np.ones(shape=(n_paths, N+1)) * S_0
    np.random.seed(0)
    W = np.random.normal(0, delta_t, size=(n_paths, N))

    for i in range(1, N+1):
        paths[:, i] = dynamics(paths[:, i-1], delta_t, W[:, i-1])

    return paths[:, -1]


n_paths = 100
N = 1000
T = 7
r = 0.06
Ks = [0, 0.05, 10]

dX = lambda X, dt, W: X + 0.04 * X * dt + 0.38 * X * W
x_paths = euler_discritisation(dX, n_paths, N, T, 4.0)
dY = lambda Y, dt, W: Y + 0.1 * Y * dt + 0.15 * Y * W
y_paths = euler_discritisation(dY, n_paths, N, T, 1.0)

for K in Ks:
    print("K =", K, ":", np.exp(-r*T) * (K + np.mean(0.5*x_paths - 0.5*y_paths - K > 0)))
