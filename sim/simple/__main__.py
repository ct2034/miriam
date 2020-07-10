import logging
import random
import sys
import threading
import time

import numpy as np
from PyQt5 import QtGui

from sim.simple.mod_random import Random
from sim.simple.simulation import SimpSim
from sim.simple.vis import Vis

FORMAT = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)
logging.getLogger("apscheduler").setLevel(logging.WARN)


def testing(thread: SimpSim):
    width = 20
    height = 20

    time.sleep(.5)
    gridmap = np.zeros([width, width])
    thread.start_sim(gridmap, 3)

    time.sleep(.5)

    for i in range(4):
        thread.new_job(
            np.array([random.randint(0, width), random.randint(0, height)]),
            np.array([random.randint(0, width), random.randint(0, height)]),
            random.randint(0, 1000)
        )
        time.sleep(.1)

    time.sleep(20)
    thread.stop_sim()


if __name__ == '__main__':
    logging.info("__main__.py ...")

    # init switches
    test = False
    vis = False

    n_agvs = 5
    width = 10
    gridmap = np.zeros([width, width])

    # module
    mod = Random(gridmap)

    # sim
    sim_thread = SimpSim(mod)
    sim_thread.start_sim(gridmap, n_agvs)

    # test
    test = True
    if test:
        threading.Thread(target=testing, args=(sim_thread,)).start()

    # vis
    vis = True
    if vis:
        app = QtGui.QApplication(sys.argv)
        window = Vis(sim_thread)
        window.show()
        sys.exit(app.exec_())
