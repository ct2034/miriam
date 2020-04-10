from sim_process.process_test import run_with_sim
from simple_simulation.mod_cbsextension import Cbsext
from simple_simulation.simulation import SimpSim


class RobotSim(SimpSim):
    def __init__(self, module):
        print('init')


if __name__ == "__main__":
    n_agvs = 2
    module = Cbsext()
    run_with_sim(RobotSim(module),
                 products_todo=3,
                 n_agv=n_agvs,
                 flow_lenght=4)
