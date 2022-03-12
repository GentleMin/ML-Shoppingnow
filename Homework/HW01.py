import numpy as np

# Data points
xy = [[-3.4, 7.5], [-0.9, 2.3], [2.9, 8.7], [4, -6.9], [4.9, -18.9]]
xy = np.asarray(xy)

# Construct basis and matrix
fit_func = [lambda x: np.ones(x.shape), 
            lambda x: x,
            lambda x: x**2,
            np.sin]
Phi = np.ones((xy.shape[0], len(fit_func)))
for i in range(len(fit_func)):
    Phi[:, i] = fit_func[i](xy[:, 0])

# Solve normal eqn.
coeffs = np.linalg.solve(Phi.T @ Phi, Phi.T @ xy[:, 1])
print(coeffs)                   # coefficients
print(Phi @ coeffs)             # recovery
print(xy[:, 1] - Phi @ coeffs)  # residual
