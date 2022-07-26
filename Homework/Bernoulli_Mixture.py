# Mixture Model of Bernoulli distributions

import numpy as np

# obs and latent var.
x_obs = np.array([1, 2, 3])
nx_obs = np.array([30, 20, 60])
z = np.array([1, 2])
nz = np.array([[0, 0], [0, 0]])

# params.
alpha = 1/2
beta = 1/2
gamma = 1/2

for t in range(10):
    # E-step
    nz[0, 0] = nx_obs[0]
    nz[1, 1] = nx_obs[2]
    if gamma/(1 - gamma) <= (1 - beta)/(1 - alpha):
        nz[0, 1] = nx_obs[1]
    else:
        nz[1, 0] = nx_obs[1]

    # M-step
    gamma = np.sum(nz[0])/np.sum(nz)
    alpha = nz[0, 0]/(nz[0, 0] + nz[0, 1])
    beta = nz[1, 1]/(nz[1, 0] + nz[1, 1])

    print("{:.3f}  {:.3f}  {:.3f}".format(alpha, beta, gamma))


pxz = np.array([[gamma*alpha, gamma*(1 - alpha), 0], 
                [0, (1 - gamma)*(1 - beta), (1 - gamma)*beta]])
print("Joint Distribution")
print(pxz)
print(np.prod(np.power(pxz[0, :2], nz[0])*np.power(pxz[1, 1:], nz[1])))
