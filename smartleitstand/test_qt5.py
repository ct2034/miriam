# simple code by Krystian Samp - krychu (samp[dot]krystian[monkey]gmail.com), November 2006

import sys
from PyQt4 import QtGui, QtCore

class MyView(QtGui.QGraphicsView):
    def __init__(self):
        QtGui.QGraphicsView.__init__(self)

        self.scene = QtGui.QGraphicsScene(self)
        self.item = QtGui.QGraphicsEllipseItem(1000, 0, 10, 10)
        color = QtGui.QColor(10, 200, 10, 127)
        brush = QtGui.QBrush(color)
        self.item.setBrush(brush)
        self.scene.addItem(self.item)
        self.setScene(self.scene)
        self.resize(130, 130)

        # Remember to hold the references to QTimeLine and QGraphicsItemAnimation instances.
        # They are not kept anywhere, even if you invoke QTimeLine.start().
        self.tl = QtCore.QTimeLine(1000)
        self.tl.setFrameRange(0, 100)
        self.a = QtGui.QGraphicsItemAnimation()
        self.a.setItem(self.item)
        self.a.setTimeLine(self.tl)

        # Each method determining an animation state (e.g. setPosAt, setRotationAt etc.)
        # takes as a first argument a step which is a value between 0 (the beginning of the
        # animation) and 1 (the end of the animation)
        self.a.setPosAt(0, QtCore.QPointF(0, 0))
        self.a.setPosAt(.25, QtCore.QPointF(0, 100))
        self.a.setPosAt(.5, QtCore.QPointF(100, 100))
        self.a.setPosAt(.75, QtCore.QPointF(100, 0))
        self.a.setPosAt(1, QtCore.QPointF(50, 50))

        self.tl.start()

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    view = MyView()
    view.show()
    sys.exit(app.exec_())