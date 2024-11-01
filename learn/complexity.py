import numpy as np
import matplotlib.pyplot as plt
from math import factorial

x = np.linspace(0, 2)

legend_str = []
for f, s in [
    (lambda x: x**2, "x^2"),
    (lambda x: x**3, "x^3"),
    (lambda x: 2 ** (x**2), "2^x^2"),
    (lambda x: 2 ** (x**3), "2^x^3"),
    (lambda x: factorial(int(x)), "factorial"),
]:
    plt.plot(x, list(map(f, x)))
    legend_str.append(s)

plt.legend(legend_str)
plt.show()
