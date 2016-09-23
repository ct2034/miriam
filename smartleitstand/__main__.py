import time, threading, sys
from numpy import *
from PyQt4 import QtGui, QtCore

from simulation import SimpSim
from vis import Vis

def testing(thread: SimpSim):
    width = 100
    height = 100

    time.sleep(.5)
    thread.start_sim(width, height, 5)

    time.sleep(2)

    for i in range(10):
        thread.new_job(
            array([random.randint(0, width), random.randint(0, height)]),
            array([random.randint(0, width), random.randint(0, height)])
        )
        time.sleep(.5)

    time.sleep(10)

if __name__ == '__main__':
    print("__main__.py ...")

    # init switches
    msb = False
    test = False
    vis = False

    # sim
    msb = True
    simThread = SimpSim(msb)
    simThread.start()

    # test
    test = True
    if test:
        threading.Thread(target=testing, args=(simThread,)).start()

    # vis
    vis = True
    if vis:
        app = QtGui.QApplication(sys.argv)
        window = Vis(simThread=simThread)
        window.show()
        sys.exit(app.exec_())