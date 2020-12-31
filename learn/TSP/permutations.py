from itertools import permutations
import numpy as np


def permutations_with_uncomplete(n, _l=None):
    if _l is None:
        _l = n
    if _l == 1:
        return list(permutations(range(n), 1))
    else:
        return list(permutations(range(n), _l)) + \
            list(permutations_with_uncomplete(n, _l - 1))


# theoretically possible combinations of tasks per agent
perm = permutations_with_uncomplete(4)
for i in perm:
    print(str(i))
print("length: %d" % len(perm))
lengths = list(map(len, perm))
print("lengths: " + str(
    [lengths.count(x) for x in np.unique(lengths)]
))
