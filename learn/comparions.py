#!/usr/bin/env python3
import itertools

# this was for https://stackoverflow.com/questions/65828530/is-there-a-good-framework-or-package-in-python-to-compare-something-over-a-numbe/65829172#65829172


def evaluate(a, b, c):
    return a + b + c


params1 = [0.1, 0.2, 0.3]
params2 = [5, 10, 50]
params3 = range(8)

for (p1, p2, p3) in itertools.product(params1, params2, params3):
    res = evaluate(p1, p2, p3)
    print(res)
