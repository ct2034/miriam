import numpy as np
from multiprocessing import Pool


def plan(agent_pos, jobs, grid):
    n = len(agent_pos)
    m = len(jobs)
    pool = Pool(n)

    # initial random assignment
    agent_job = []
    available = list(range(m))
    while len(agent_job) < len(agent_pos):
        ij = np.random.choice(available)
        agent_job.append(ij)
        available.remove(ij)

    converged = False
    while not converged:
