#!/usr/bin/python

import logging
import threading
from datetime import datetime
import time
import numpy as np

from planner.mod_cbsextension import Cbsext
from planner.mod_random import Random
from planner.simulation import SimpSim

FORMAT = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)
logging.getLogger("apscheduler").setLevel(logging.WARN)

stations = [[0, 0],
            [9, 9],
            [4, 0],
            [4, 9],
            [0, 9],
            [0, 4],
            [9, 4]]

flow = [[0, 2],
        [1, 3],
        [2, 1],
        [4, 2],
        [3, 3],
        [5, 3],
        [6, 2]
        ]

products_todo = 3
state = np.zeros(products_todo)
t_left = np.zeros(products_todo)

t_step = .1


def run(agv_sim):
    products_finished = 0
    while products_finished < products_todo:
        for p in range(products_todo):
            if state[p] % 1 == .5:  # in transport
                if agv_sim.is_finished(p):
                    state[p] += .5
                    t_left[p] = flow[int(state[p])][1]
            else:  # in station
                if state[p] == len(flow) - 1:  # finished
                    state[p] = len(flow)
                    products_finished += 1
                elif state[p] < len(flow) - 1:  # running
                    if t_left[p] <= 0:
                        agv_sim.new_job(np.array(stations[flow[int(state[p])][0]], dtype=int),
                                        np.array(stations[flow[int(state[p]) + 1][0]], dtype=int),
                                        p)
                        state[p] += .5
                    else:
                        t_left[p] -= t_step
        time.sleep(t_step)


_map = np.zeros([10, 10, 51])


def test_process_Cbsext():
    mod = Cbsext(_map)
    t = run_with_module(mod)
    return t


def test_process_Random():
    mod = Random(_map)
    t = run_with_module(mod)
    return t


def run_with_module(mod):
    agv_sim = SimpSim(False, mod)
    agv_sim.start()
    agv_sim.start_sim(20, 20, 2)
    start_object = run(agv_sim)
    n = start_object.start(3, agv_sim)
    agv_sim.stop()
    return n


if __name__ == "__main__":
    t_cbsext = test_process_Cbsext()
    t_random = test_process_Random()
    print("Random:", str(t_random),
          "\nCbsExt:", str(t_cbsext))
