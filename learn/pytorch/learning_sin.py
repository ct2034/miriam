#!/usr/bin/env python3
import math

import matplotlib.pyplot as plt
import torch

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# Create random input and output data
t = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(t)

# Randomly initialize weights
x = torch.tensor([0, -0.3, 0, 0, 0], device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-7
y_preds = []
for i in range(int(1e4)):
    # Forward pass: compute predicted y
    y_pred = x[0] + x[1] * t + x[2] * t**2 + x[3] * t**3 + x[4] * t**4

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum()
    if i % 100 == 0:
        print(loss)
    if i % 1000 == 10:
        y_preds.append(y_pred)

    # Backprop to compute gradients of a, b, c, d with respect to loss
    backw = loss.backward(retain_graph=True)
    # print(x)
    # print(x.grad)
    x = x - x.grad * learning_rate
    x = torch.tensor(x, requires_grad=True)
    # print(x)


print(
    f"Result: y={x[0].item()} + {x[1].item()} x +"
    + f"{x[2].item()} x ^ 2 + {x[3].item()} x ^ 3 + {x[4].item()} x ^ 4"
)

plt.plot(t, y, "r", linewidth=1)
n = len(y_preds)
d_col = 0.4
col = 0.4
for y_pred in y_preds:
    col += d_col / n
    plt.plot(t, y_pred.detach().numpy(), color=(0, col, 0.4), linewidth=0.5)
plt.show()
