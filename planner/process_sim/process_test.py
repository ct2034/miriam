#!/usr/bin/python

import logging
from datetime import datetime
import time
import csv
import numpy as np

from planner.mod_cbsextension import Cbsext
from planner.mod_nearest import Nearest
from planner.mod_random import Random
from planner.simulation import SimpSim

FORMAT = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)
logging.getLogger("apscheduler").setLevel(logging.WARN)

### global variables
t_step = .1 # sim stepsize
x_res = 10 # size of map in x
y_res = 10 # size of map in y
_map = np.zeros([x_res, y_res, 51]) # create map of size x*y filled with zeroes (why 51 ? )
### end global variables

def run(agv_sim, stations, flow, products_todo):
    print("START")
    state = np.zeros(products_todo)
    t_left = np.zeros(products_todo)
    blocked = np.zeros(len(stations)) - 1
    t_start = datetime.now()
    products_finished = 0
    while products_finished < products_todo:
        for p in range(products_todo):
            if state[p] % 1 == .5:  # in transport state
                if agv_sim.is_finished(hash(p + 100 * (state[p] - .5))): # unique transport hash
                    state[p] += .5
                    t_left[p] = flow[int(state[p])][1]
                    print("PRODUCT %d in state %1.0f" % (p + 1, state[p]))
            else:  # in station
                if state[p] == len(flow) - 1:  # is finished at station and waits for  next step
                    state[p] = len(flow)
                    products_finished += 1
                    print("PRODUCT %d FINISHED" % (p + 1))
                elif state[p] < len(flow) - 1:  # process at station is still running
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
    return (datetime.now() - t_start).total_seconds()

###Test_modules {different algorithms per module}
def test_process_cbsext():
    mod = Cbsext(_map)
    t = run_with_module(mod)
    return t

def test_process_random():
    mod = Random(_map)
    t = run_with_module(mod)
    return t

def test_process_nearest():
    mod = Nearest(_map) # load module nearest and run sim with it
    t = run_with_module(mod)
    return t
###END

def run_with_module(mod, products_todo=3, n_agv=2):
    agv_sim = SimpSim(False, mod) # search for collision problems in simulation 08.03.17
    agv_sim.start()
    agv_sim.start_sim(x_res, y_res, n_agv)

    stations = __read_stations();
    flow = __read_flow(); #

    n = run(agv_sim, stations, flow, products_todo) # here starts the real magic
    agv_sim.stop_sim()
    return n

# UNUSED CODE
# def test_benchmark(): # init which tells, number of products and amount of agvs
#     durations = np.zeros(2)
#     modules = [Nearest(_map), Cbsext(_map)]
#     for i_mod in range(len(modules)):
#         try:
#             durations[i_mod] = run_with_module(modules[i_mod], products_todo=3, n_agv=2)
#         except Exception as e:
#             logging.error("Exception on simulation level\n" + str(e))
#
#     print("RESULT:\n for ..")
#     print("modules: " + str(modules))
#     print(durations)

def __read_flow(): # private function for importing product flow
    #Eingefuegt 08.03.2017 jm
    flow = list()

    with open('miriam_flow.csv', 'r') as flow_file:
        rdr = csv.reader(flow_file, delimiter=',', quotechar='|')
        for row in rdr:
            entry = list()
            for items in row:
                if ((  items.startswith( '#') )): # ignore comments
                    break;
                else:
                    entry.append(int(items));
            if entry: # save entry to flow, if value is not null
                flow.append(entry);
    print (flow)
    return flow;

def __read_stations(): # private function for importing stations
    #Eingefuegt 08.03.2017 jm
    stations = list()

    with open('miriam_stations.csv', 'r') as stations_file:
        rdr = csv.reader(stations_file, delimiter=',', quotechar='|')
        for row in rdr:
            entry = list()
            for items in row:
                if ((  items.startswith( '#') )): # ignore comments
                    break;
                elif (( items.isdigit() )): # sorts out station names
                    entry.append(int(items));
                #else : # adds station names into flow list, for future
                 #   entry.append(items);

            if entry: # ensures that no empty string will be saved in flow
                stations.append(entry);
    print (stations)
    return stations;

if __name__ == "__main__":
    #Start different algorithms
    #t_nearest = test_process_nearest() # works but has collisions
    t_cbsext = test_process_cbsext() # cant find pkl file
    t_random = test_process_random()
    # print results of each testrun
    print("Random:", str(t_random),
          "\nNearest:", str(t_nearest),
          "\nCbsExt:", str(t_cbsext))
