from itertools import permutations
import numpy as np


def permutations_with_uncomplete(n, l=None):
    if l == None:
        l = n
    if l == 1:
        return list(permutations(range(n), 1))
    else:
        return list(permutations(range(n), l)) + \
               list(permutations_with_uncomplete(n, l - 1))


# theoretically possible combinations of tasks per agent
perm = permutations_with_uncomplete(4)
for i in perm:
    print(str(i))
print("length: %d" % len(perm))
lengths = list(map(len, perm))
print("lengths: " + str(
    [lengths.count(l) for l in np.unique(lengths)]
))
