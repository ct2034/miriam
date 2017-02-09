import multiprocessing
from os import getpid
from time import sleep


def worker(procnum):
    print('I am number %d in process %d' % (procnum[0], getpid()))
    sleep(1)
    return getpid(), procnum[1]


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=2)
    print(list(pool.map(worker, [[0, 8], (1, 7), (2, 6), (3, 5)])))
