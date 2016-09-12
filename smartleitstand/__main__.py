import time, threading, sys
from numpy import *
from PyQt4 import QtGui, QtCore

from simulation import SimpSim
from vis import Vis

if __name__ == '__main__':
    print("__main__.py ...")

    simThread = SimpSim()
    simThread.start()

    app = QtGui.QApplication(sys.argv)
    window = Vis(simThread=simThread)
    window.show()
    sys.exit(app.exec_())