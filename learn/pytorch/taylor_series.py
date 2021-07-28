#!/usr/bin/env python3
import math

import matplotlib.pyplot as plt
import torch

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# Create random input and output data
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# Randomly initialize weights
a = torch.randn((), device=device, dtype=dtype) * .01
b = torch.randn((), device=device, dtype=dtype) * .01
c = torch.randn((), device=device, dtype=dtype) * .01
d = torch.randn((), device=device, dtype=dtype) * .01
e = torch.randn((), device=device, dtype=dtype) * .01

learning_rate = 1e-7
y_preds = []
for t in range(int(3e4)):
    # Forward pass: compute predicted y
    y_pred = a + b * x + c * x ** 2 + d * x ** 3 + e * x ** 4

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)
    if t % 1000 == 999:
        y_preds.append(y_pred)

        # Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()
    grad_e = (grad_y_pred * x ** 4).sum()

    # Update weights using gradient descent
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d
    e -= learning_rate * grad_e


print(
    f'Result: y={a.item()} + {b.item()} x +' +
    f'{c.item()} x ^ 2 + {d.item()} x ^ 3 + {e.item()} x ^ 4')

plt.plot(x, y, 'r', linewidth=1)
n = len(y_preds)
d_col = .4
col = .4
for y_pred in y_preds:
    col += d_col / n
    plt.plot(x, y_pred, color=(0, col, .4), linewidth=.5)
plt.show()
