import logging
import sys

FORMAT = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)
logging.getLogger("apscheduler").setLevel(logging.WARN)
import threading
import time

from PyQt4 import QtGui
from numpy import *

from smartleitstand.simulation import SimpSim
from smartleitstand.vis import Vis


def testing(thread: SimpSim):
    width = 100
    height = 100

    time.sleep(.5)
    thread.start_sim(width, height, 3)

    time.sleep(.5)

    for i in range(4):
        thread.new_job(
            array([random.randint(0, width), random.randint(0, height)]),
            array([random.randint(0, width), random.randint(0, height)]),
            random.randint(0, 1000)
        )
        time.sleep(.1)

    time.sleep(1)
    thread.stop()

    time.sleep(1)

    thread.start_sim(width, height, 3)

    time.sleep(.5)

    for i in range(4):
        thread.new_job(
            array([random.randint(0, width), random.randint(0, height)]),
            array([random.randint(0, width), random.randint(0, height)]),
            random.randint(0, 1000)
        )
        time.sleep(.1)

    time.sleep(1)
    thread.stop()


if __name__ == '__main__':
    logging.info("__main__.py ...")

    # init switches
    msb = False
    test = False
    vis = False
    robotarium = False

    # sim
    msb = True
    simThread = SimpSim(msb)
    simThread.start()

    # test
    # test = True
    if test:
        threading.Thread(target=testing, args=(simThread,)).start()

    # vis
    # vis = True
    if vis:
        app = QtGui.QApplication(sys.argv)
        window = Vis(simThread=simThread)
        window.show()
        sys.exit(app.exec_())
