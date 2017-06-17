from planner.simulation import SimpSim
from process_sim.process_test import run_with_sim


class RobotSim(SimpSim):
    def __init__(self, module):
        print('init')


if __name__ == "__main__":
    n_agvs = 2
    module = CbsExt()
    run_with_sim(RobotSim(module),
                 products_todo=3,
                 n_agv=n_agvs,
                 flow_lenght=4)
