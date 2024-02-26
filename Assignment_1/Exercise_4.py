import numpy as np
import matplotlib.pyplot as plt

def simulate_wiener_process(n_paths, n_steps, T):
    dt = T / n_steps  # Time step
    sqrt_dt = np.sqrt(dt)

    # Generate random increments for each path
    dW = np.random.normal(0, sqrt_dt, size=(n_paths, n_steps))

    # Accumulate increments to get the paths
    W = np.cumsum(dW, axis=1)

    # Add an initial zero for each path
    W = np.concatenate((np.zeros((n_paths, 1)), W), axis=1)

    # Create the time vector
    t = np.linspace(0, T, n_steps + 1)

    return t, W

# Parameters
n_paths = 50000
n_steps = 1000
T = 10.0

# Simulate Wiener process paths
t, W = simulate_wiener_process(n_paths, n_steps, T)

dt = T / n_steps
X = np.zeros(n_steps+1)
A = [t + (t / T)**2 * (T - t) - 2 * t / T * min(t, T-t) for t in np.linspace(0, T, n_steps + 1)]
for i in range(n_steps + 1):
    X[i] = np.var(W[:, i] - dt*i / T * W[:, n_steps - i])

plt.plot(np.linspace(0, T, n_steps + 1), X, label="Observed variance")
plt.plot(np.linspace(0, T, n_steps + 1), A, label="Analytical variance")
plt.xlabel('Time')
plt.ylabel(r'Var($X_t$)')
plt.legend()
plt.show()
