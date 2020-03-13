import logging
import threading
import time

from PyQt4 import QtGui
from numpy import *

from simple_simulation.mod_random import Random
from simple_simulation.simulation import SimpSim
from simple_simulation.vis import Vis

FORMAT = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)
logging.getLogger("apscheduler").setLevel(logging.WARN)


def testing(thread: SimpSim):
    width = 20
    height = 20

    time.sleep(.5)
    thread.start_sim(width, height, 3)

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

    # module
    mod = Random()

    # sim
    simThread = SimpSim(mod)
    simThread.start()

    # test
    test = True
    if test:
        threading.Thread(target=testing, args=(simThread,)).start()

    # vis
    vis = True
    if vis:
        app = QtGui.QApplication(sys.argv)
        window = Vis(sim_thread=simThread)
        window.show()
        sys.exit(app.exec_())
