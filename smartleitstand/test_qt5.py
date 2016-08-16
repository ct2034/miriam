
#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
ZetCode PyQt5 tutorial

In the example, we draw randomly 1000 red points
on the window.

author: Jan Bodnar
website: zetcode.com
last edited: January 2015
"""

import sys, random, time, threading
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class Example(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()


    def initUI(self):

        self.setGeometry(0, 0, 1000, 1000)
        self.setWindowTitle('Points')
        self.show()




    def paintEvent(self, e):
        print("Paint event ..")
        qp = QPainter()
        qp.begin(self)
        self.drawPoints(qp)
        qp.end()

    def drawPoints(self, qp):

        qp.setPen(Qt.darkBlue)
        size = self.size()

        for i in range(1000):
            x = random.randint(1, size.width()-1)
            y = random.randint(1, size.height()-1)
            qp.drawEllipse(QPointF(x,y), 3, 3)


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Example()

    t = threading.Timer(20 * 60, ex.paintEvent(ex, QEvent()))
    t.start()

    sys.exit(app.exec_())