from multiprocessing import Pool
import time
from datetime import datetime
import numpy as np


def wait_rand(t):
    # print("Waiting for %.1fs"%t)
    time.sleep(t)
    # print("Waited for %.1fs"%t)
    return t


def test_parallelization(par):
    time.sleep(1)
    t = 1
    pool = Pool(par)
    start = datetime.now()
    mr = pool.map_async(wait_rand, np.full(par, t))
    mr.get(t * 1.4)
    pool.close()
    return (datetime.now() - start).total_seconds()


print(list(map(test_parallelization, [1, 10, 100, 300, 500, 5, 200])))
