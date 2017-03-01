#!/usr/bin/python

import logging
from datetime import datetime
import time
import numpy as np

from planner.mod_cbsextension import Cbsext
from planner.mod_nearest import Nearest
from planner.mod_random import Random
from planner.simulation import SimpSim

FORMAT = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)
logging.getLogger("apscheduler").setLevel(logging.WARN)

t_step = .1


def run(agv_sim, stations, flow, products_todo):
    print("START")
    state = np.zeros(products_todo)
    t_left = np.zeros(products_todo)
    blocked = np.zeros(len(stations)) - 1
    t_start = datetime.now()
    products_finished = 0
    while products_finished < products_todo:
        products_finished += products_transport_or_process(agv_sim, blocked, flow, products_todo,
                                                           state, stations, t_left)

        time.sleep(t_step)

    t_finished = datetime.now()
    print("ALL PRODUCTS FINISHED")
    t_dur = (t_finished - t_start).total_seconds()
    print("Took %.3f s" % t_dur)
    return t_dur


def products_transport_or_process(agv_sim, blocked, flow, products_todo, state, stations, t_left):
    products_finished = 0
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
                prodcuts_eval_running_step(agv_sim, blocked, flow, p, state, stations, t_left)
    return products_finished


def prodcuts_eval_running_step(agv_sim, blocked, flow, p, state, stations, t_left):
    if (t_left[p] <= 0) & (blocked[int(state[p])] == p):
        agv_sim.new_job(np.array(stations[flow[int(state[p])][0]], dtype=int),
                        np.array(stations[flow[int(state[p]) + 1][0]], dtype=int),
                        hash(p + 100 * state[p]))
        blocked[int(state[p])] = -1
        state[p] += .5
        print("PRODUCT %d in state %1.1f" % (p + 1, state[p]))
    elif blocked[int(state[p])] == p:
        t_left[p] -= t_step
    elif blocked[int(state[p])] == -1:  # free
        blocked[int(state[p])] = p


x_res = 10
y_res = 10
_map = np.zeros([x_res, y_res, 51])


# def test_process_cbsext():
#     mod = Cbsext(_map)
#     t = run_with_module(mod)
#     return t
#
#
# def test_process_random():
#     mod = Random(_map)
#     t = run_with_module(mod)
#     return t
#
#
# def test_process_nearest():
#     mod = Nearest(_map)
#     t = run_with_module(mod)
#     return t


def test_benchmark():
    ns_agvs = range(1, 5)
    durations = np.zeros([len(ns_agvs), 3])
    for in_agvs in range(len(ns_agvs)):
        modules = [Random(_map), Nearest(_map), Cbsext(_map)]
        for i_mod in range(len(modules)):
            try:
                durations[in_agvs, i_mod] = run_with_module(modules[i_mod], products_todo=3, n_agv=ns_agvs[in_agvs])
            except Exception as e:
                logging.error("Exception on simulation level\n" + str(e))

    print("RESULT:\n for ..")
    print("ns_agvs: " + str(ns_agvs))
    print("modules: " + str(modules))
    print(durations)



def run_with_module(mod, products_todo=3, n_agv=2):
    agv_sim = SimpSim(False, mod)
    agv_sim.start()
    agv_sim.start_sim(x_res, y_res, n_agv)
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
    n = run(agv_sim, stations, flow, products_todo)
    agv_sim.stop_sim()
    return n


if __name__ == "__main__":
    t_nearest = test_process_nearest()
    t_cbsext = test_process_cbsext()
    t_random = test_process_random()
    print("Random:", str(t_random),
          "\nNearest:", str(t_nearest),
          "\nCbsExt:", str(t_cbsext))
