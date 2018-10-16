#!/usr/bin/python

import logging
import random
from datetime import datetime
import unittest2 as unittest

import numpy as np
from simple_simulation.mod_cbsextension import Cbsext
from simple_simulation.mod_nearest import Nearest

from simple_simulation.mod_random import Random
from simple_simulation.simulation import SimpSim
from tools import is_travis

FORMAT = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)
logging.getLogger("apscheduler.scheduler").setLevel(logging.WARN)

t_step = .1

if is_travis():
    flow_lenght = 7  # full
    products_todo = random.randint(10, 20)
else:
    flow_lenght = 2
    products_todo = 3
logging.info("We will use products_todo = %d" % products_todo)

def run(agv_sim, stations, flow, products_todo):
    print("START")
    state = np.zeros(products_todo)
    t_left = np.zeros(products_todo)
    blocked = np.zeros(len(stations)) - 1
    t_start = datetime.now()
    products_finished = 0
    while products_finished < products_todo:
        for p in range(products_todo):
            if state[p] % 1 == .5:  # in transport
                if agv_sim.is_finished(hash(p + 100 * (state[p] - .5))):
                    state[p] += .5
                    t_left[p] = flow[int(state[p])][1]
                    print("PRODUCT %d in state %1.0f" % (p + 1, state[p]))
            else:  # in station
                if state[p] == len(flow) - 1:  # finished
                    state[p] = len(flow)
                    products_finished += 1
                    print("PRODUCT %d FINISHED" % (p + 1))
                elif state[p] < len(flow) - 1:  # running
                    if (t_left[p] <= 0) & (blocked[int(state[p])] == p):
                        agv_sim.new_job(tuple(stations[flow[int(state[p])][0]]),
                                        tuple(stations[flow[int(state[p]) + 1][0]]),
                                        hash(p + 100 * state[p]))
                        blocked[int(state[p])] = -1
                        state[p] += .5
                        print("PRODUCT %d in state %1.1f" % (p + 1, state[p]))
                    elif blocked[int(state[p])] == p:
                        t_left[p] -= t_step
                    elif blocked[int(state[p])] == -1:  # free
                        blocked[int(state[p])] = p
    logging.debug("run END")
    return (datetime.now() - t_start).total_seconds()

x_res = 10
y_res = 10
_map = np.zeros([x_res, y_res, 51])


@unittest.skip("Runs literally forever")
def test_process_cbsext():
    mod = Cbsext(_map)
    t = run_with_sim(SimpSim(False, mod),
                     products_todo=products_todo,
                     n_agv=4,
                     flow_lenght=flow_lenght)
    return t


@unittest.skip("Runs literally forever")
def test_process_random():
    mod = Random(_map)
    t = run_with_sim(SimpSim(False, mod),
                     products_todo=products_todo,
                     n_agv=4,
                     flow_lenght=flow_lenght)
    return t


@unittest.skip("Runs literally forever")
def test_process_nearest():
    mod = Nearest(_map)
    t = run_with_sim(SimpSim(False, mod),
                     products_todo=products_todo,
                     n_agv=4,
                     flow_lenght=flow_lenght)
    return t


@unittest.skip("Runs literally forever")
def test_benchmark():
    modules = [Random(_map), Nearest(_map), Cbsext(_map)]
    durations = np.zeros(len(modules))
    for i_mod in range(len(modules)):
        try:
            durations[i_mod] = run_with_sim(SimpSim(False, modules[i_mod]),
                                            products_todo=3,
                                            n_agv=2,
                                            flow_lenght=flow_lenght)
        except Exception as e:
            logging.error("Exception on simulation level\n" + str(e))
            raise e

    print("RESULT:\n for ..")
    print("modules: " + str(modules))
    print(durations)


def run_with_sim(agv_sim, products_todo=3, n_agv=2, flow_lenght=7):
    agv_sim.start_sim(x_res, y_res, n_agv)
    idle_goals = [((0, 0), (15, 3)),
                  ((4, 0), (15, 3),),
                  ((9, 0), (15, 3),),
                  ((9, 4), (15, 3),),
                  ((9, 9), (15, 3),),
                  ((4, 9), (15, 3),),
                  ((0, 9), (15, 3),),
                  ((0, 5), (15, 3),)]  # TODO: we have to learn these!
    id = 1000
    for ig in idle_goals:
        agv_sim.new_idle_goal(ig[0], ig[1], id)
        id += 1
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
    assert len(flow) >= flow_lenght, "Can only select max lenght of flow %d" % len(flow)
    flow = flow[:(flow_lenght)]
    n = run(agv_sim, stations, flow, products_todo)
    agv_sim.stop_sim()
    return n


if __name__ == "__main__":
    test_benchmark()  # :)