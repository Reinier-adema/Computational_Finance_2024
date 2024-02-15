import matplotlib.pyplot as plt
import numpy as np


def b(n, lower, upper):
    np.random.seed(0)
    dt = (upper - lower) / n
    W = np.cumsum(np.random.normal(0, dt, n))

    # compute \int W_s ds
    W_t_dt_l = 0
    for i in range(n-1):
        W_t_dt_l += W[i] * dt

    # compute \int (t - s) dW_s
    t_s_dW_s = upper * W[0]
    for i in range(n-1):
        t_s_dW_s += (upper - (i + 1)*dt) * (W[i+1] - W[i])

    return [W_t_dt_l, t_s_dW_s, np.abs(t_s_dW_s - W_t_dt_l)]


# compute for different partition sizes
x = [10**i for i in range(1, 9)]
res = []
for i in x:
    print(i)
    res.append(b(i, 0, 5))

# plot the values
print([i[2] for i in res])
plt.plot(x, [i[0] for i in res], label=r"$\int_{0}^{t} W_s ds$")
plt.plot(x, [i[1] for i in res], label=r"$\int_{0}^{t} (t - s) dW_s$")
plt.legend()
plt.xscale("log")
plt.xlabel("partition size")
plt.ylabel("integral value")
plt.show()
