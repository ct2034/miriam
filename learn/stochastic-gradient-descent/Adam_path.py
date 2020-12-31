# source: https://github.com/sagarvegad/Adam-optimizer
import math
import numpy as np
import matplotlib.pyplot as plt

alpha = 0.01
beta_1 = 0.8
beta_2 = 0.8  # initialize the values of the parameters
epsilon = 1


def func(x):
    return np.sqrt(np.sum(np.square(x)))


def grad_func(x, start, goal):  # calculates the gradient
    p = path(start, goal, x)
    out = np.zeros(shape=[x.shape[0], x.shape[1]])
    le = x.shape[0]
    for i in range(l):
        ip = i+1
        for j in [0, 1]:
            out[i, j] = (
                (p[ip, j] - p[ip-1, j]) /
                math.sqrt((p[ip, 0] - p[ip-1, 0])**2
                          + (p[ip, 1] - p[ip-1, 1])**2) +
                (p[ip, j] - p[ip+1, j]) /
                math.sqrt((p[ip, 0] -
                           p[ip+1, 0])**2 + (p[ip, 1] - p[ip+1, 1])**2)
            )
    return out


def path(start, goal, theta):
    p = np.zeros(shape=[theta.shape[0] + 2, theta.shape[1]])
    p[0, :] = start
    p[1:(1+theta.shape[0]), :] = theta
    p[(1+theta.shape[0]), :] = goal
    return p


start = np.random.rand(2)
goal = np.random.rand(2)
_l = 6
theta_0 = np.random.rand(l, 2)
m_t = np.zeros([l, 2])
v_t = np.zeros([l, 2])
t = 0

thetas = []
ts = []

art = False
fig = plt.figure()
if art:
    plt.axis('off')
    fig.set_facecolor('black')
    fig.set_size_inches(35.5555, 20)
else:
    plt.axis('equal')

tmax = 1000
while (t < tmax):  # till it gets converged
    t += 1
    # computes the gradient of the stochastic function
    g_t = grad_func(theta_0, start, goal)
    # updates the moving averages of the gradient
    m_t = beta_1*m_t + (1-beta_1)*g_t
    # updates the moving averages of the squared gradient
    v_t = beta_2*v_t + (1-beta_2)*(g_t*g_t)
    m_cap = m_t/(1-(beta_1**t))  # calculates the bias-corrected estimates
    v_cap = v_t/(1-(beta_2**t))  # calculates the bias-corrected estimates
    # updates the parameters
    theta_0 = theta_0 - np.divide((alpha * m_cap), (np.sqrt(v_cap) + epsilon))
    if art:
        n = 1
    else:
        n = 10
    if t % n == 0:
        p = path(start, goal, theta_0)
        plt.plot(p[:, 0], p[:, 1], 'r', alpha=float(t)/tmax/2+.5, lw=1)
if not art:
    plt.plot(start[0], start[1], 'xr')
    plt.plot(goal[0], goal[1], 'xg')
    plt.show()
else:
    fig.savefig('adam.png', facecolor='k')
