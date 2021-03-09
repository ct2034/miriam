import time
from datetime import datetime
from multiprocessing import Pool

import numpy as np


def wait(t, x):
    print(f'x:{x}')
    print("Waiting for %.1fs" % t)
    time.sleep(t)
    print("Waited for %.1fs" % t)
    return t


def test_parallelization(par):
    time.sleep(1)
    t = .1
    pool = Pool(5)
    start = datetime.now()
    returns = pool.starmap(wait, np.full((par, 2), (t, 99)))
    print(returns)
    pool.close()
    return (datetime.now() - start).total_seconds()


print(list(map(test_parallelization, sorted([1, 5, 10, 50]))))
