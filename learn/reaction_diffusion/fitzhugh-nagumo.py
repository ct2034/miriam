import matplotlib.pyplot as plt
import numpy as np

# Parameters
L = 100  # Length of the spatial domain
T = 200  # Total time
dx = 0.5  # Spatial step size
dt = 0.01  # Time step size
D_V = 1.0  # Diffusion coefficient for V
D_W = 0.1  # Diffusion coefficient for W
epsilon = 0.08
a = 0.7
b = 0.8
I_ext = 0.5

# Discretized spatial and temporal grids
x = np.arange(0, L, dx)
t = np.arange(0, T, dt)
V = np.zeros((len(t), len(x)))
W = np.zeros((len(t), len(x)))

# Initial conditions: a localized perturbation
V[0, int(len(x) / 2) - 5 : int(len(x) / 2) + 5] = 1.0
W[0, :] = 0.0

# Finite difference method for the PDEs
for n in range(len(t) - 1):
    V[n + 1, 1:-1] = V[n, 1:-1] + dt * (
        D_V * (V[n, 2:] - 2 * V[n, 1:-1] + V[n, :-2]) / dx**2
        + V[n, 1:-1]
        - V[n, 1:-1] ** 3 / 3
        - W[n, 1:-1]
        + I_ext
    )
    W[n + 1, 1:-1] = W[n, 1:-1] + dt * (
        D_W * (W[n, 2:] - 2 * W[n, 1:-1] + W[n, :-2]) / dx**2
        + epsilon * (V[n, 1:-1] + a - b * W[n, 1:-1])
    )
    # Boundary conditions (Neumann)
    V[n + 1, 0] = V[n + 1, 1]
    V[n + 1, -1] = V[n + 1, -2]
    W[n + 1, 0] = W[n + 1, 1]
    W[n + 1, -1] = W[n + 1, -2]

# Plotting the results
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.imshow(V, aspect="auto", extent=[0, L, T, 0], cmap="hot")
plt.colorbar(label="V (Membrane Potential)")
plt.title("V (Membrane Potential) over Time and Space")
plt.ylabel("Time")

plt.subplot(2, 1, 2)
plt.imshow(W, aspect="auto", extent=[0, L, T, 0], cmap="hot")
plt.colorbar(label="W (Recovery Variable)")
plt.title("W (Recovery Variable) over Time and Space")
plt.xlabel("Space")
plt.ylabel("Time")

plt.tight_layout()
plt.show()
