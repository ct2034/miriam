# source: https://github.com/sagarvegad/Adam-optimizer
import math

import matplotlib.pyplot as plt
import numpy as np

alpha = 0.01
beta_1 = 0.9
beta_2 = 0.999  # initialize the values of the parameters
epsilon = 1


def func(x):
    return np.sqrt(np.sum(np.square(x)))


def grad_func(x):  # calculates the gradient
    root = np.sqrt(np.sum(np.square(x)))
    return x / root + 5 * (np.random.rand(2) - np.array([0.5, 0.5]))


theta_0 = np.random.rand(2) - np.array([0.5, 0.5])  # initialize the vector
m_t = np.array([0, 0])
v_t = np.array([0, 0])
t = 0

thetas = []
ts = []

while t < 1000:  # till it gets converged
    t += 1
    # computes the gradient of the stochastic function
    g_t = grad_func(theta_0)
    # updates the moving averages of the gradient
    m_t = beta_1 * m_t + (1 - beta_1) * g_t
    # updates the moving averages of the squared gradient
    v_t = beta_2 * v_t + (1 - beta_2) * (g_t * g_t)
    m_cap = m_t / (1 - (beta_1**t))  # calculates the bias-corrected estimates
    v_cap = v_t / (1 - (beta_2**t))  # calculates the bias-corrected estimates
    theta_0_prev = theta_0
    # updates the parameters
    theta_0 = theta_0 - np.divide((alpha * m_cap), (np.sqrt(v_cap) + epsilon))
    thetas.append(theta_0)
    ts.append(t)
    # if(t%1000 == 0): #every sth
    # 	print("theta_0: " + str(theta_0))

print("theta_0: " + str(theta_0))
print("t: " + str(t))
thetas = np.array(thetas)

plt.axis("equal")
plt.plot(thetas[:, 0], thetas[:, 1], "k")
plt.plot(theta_0[0], theta_0[1], "xr")
plt.show()
