from scipy.misc import derivative
import numpy as np
from scipy.optimize import brentq


# Function definition
def f(x):
    return (np.exp(x) + np.exp(-x)) / 2 - (2 * x)


def combined_root_finder(func, interval, tol=1e-6, only_newton=False):
    if func(interval[0]) * func(interval[1]) > 0:
        print("there is no root in this interval!")
    elif func(interval[0]) * func(interval[1]) < 0:
        k = 1
        x = (interval[0] + interval[1]) / 2.0
        delta = - func(x) / derivative(func, x)
        while abs(delta / x) > tol:
            print(k, round(x, 7))
            x = x + delta
            if not only_newton:
                if np.sum(x < interval) in [0, 2]:
                    if func(interval[0]) * func(x) > 0:
                        interval[0] = x - delta
                    else:
                        interval[1] = x - delta
                        x = (interval[0] + interval[1]) / 2.0
            delta = - func(x) / derivative(func, x)
            k += 1
        return x, k


# Combined root-finding
root_1, k1 = combined_root_finder(f, [2, 3])
root_2, k2 = combined_root_finder(f, [0, 2])

print('From combined root we have root = {0}'.format(root_1))
print('From combined root we have root = {0}'.format(root_2))


# Brent algorithm
def brentq_with_print(f, a, b, target_tol=1e-6, maxiter=100):
    tol = 1
    target_iteration = 1
    while tol > target_tol or target_iteration > maxiter:
        iter = target_iteration + 1
        # find tolerance such that the number of iterations equals the target iteration
        while target_iteration != iter:
            root = brentq(f, a, b, xtol=tol, full_output=True)
            iter = root[1]['iterations']

            if iter < target_iteration:
                tol *= 0.9  # Adjust tolerance
            elif iter > target_iteration:
                tol *= 1/0.9

        # print approximation in iteration target_iteration
        print(target_iteration, round(root[0], 7))
        target_iteration += 1

        if abs(f(root[0])) < target_tol:
            print("From Brent algorithm we have root = ", root[0])
            return root[0]

    print("Brent's method did not converge within the specified maximum number of iterations.")

brentq_with_print(f, 0, 2)
brentq_with_print(f, 2, 3)
