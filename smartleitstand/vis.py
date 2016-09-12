from numpy import *
import time
import threading
import random
import sys
from PyQt4 import QtGui, QtCore


def pointFromPose(pose):
    poseVis = pose * Vis.scale
    print(str(poseVis))
    return QtCore.QPoint(poseVis[0], poseVis[1])


class Vis(QtGui.QWidget):
    """Visualisation of the AGVs and environment"""

    scale = 5
    carCircles = {}
    routeLines = {}
    queueText = False
    dimensions = array([0, 0])
    msbThread = False

    def __init__(self, simThread, parent=None):
        QtGui.QWidget.__init__(self, parent)
        print("init vis")
        print("simThread: " + str(simThread))
        Vis.simThread = simThread
        self.connect(Vis.simThread, QtCore.SIGNAL("open(int, int, PyQt_PyObject)"), self.open)
        self.connect(Vis.simThread, QtCore.SIGNAL("update_car(PyQt_PyObject)"), self.update_car)
        print("connected")

    def open(self, x, y, cars):
        print ("open")
        margin = 20
        width = x * Vis.scale
        height = y * Vis.scale

        view = QtGui.QGraphicsView(self)
        scene = QtGui.QGraphicsScene()

        field = QtGui.QGraphicsRectItem(0, 0, width, height)
        color = QtGui.QColor(10, 10, 10, 127) #dark grey
        brush = QtGui.QBrush(color)
        field.setBrush(brush)
        scene.addItem(field)

        for car in cars:
            Vis.carCircles[car.id] = QtGui.QGraphicsEllipseItem(0, 0, Vis.scale, Vis.scale)
            Vis.carCircles[car.id].setPos(pointFromPose(car.pose))
            scene.addItem(Vis.carCircles[car.id])
            color = QtGui.QColor(10, 200, 10, 127) #green
            brush = QtGui.QBrush(color)
            Vis.carCircles[car.id].setBrush(brush)

        view.setScene(scene)
        view.resize(width+margin, height+margin)
        self.resize(width+margin, height+margin)
        view.show()

    def update_car(self, car):
        if car:
            Vis.carCircles[car.id].setPos(pointFromPose(car.pose))
            print(Vis.carCircles[car.id].pos().x())
            color = QtGui.QColor(200, 10, 10, 127) #red
            brush = QtGui.QBrush(color)
            Vis.carCircles[car.id].setBrush(brush)

    def update_route(self, route):
        print("update_route")
        # if route:
        #     if route.id not in Vis.routeLines.keys():
        #         Vis.routeLines[route.id] = Line(pointFromPose(route.start), pointFromPose(route.goal))
        #         Vis.routeLines[route.id].setFill('red')
        #         Vis.routeLines[route.id].setArrow('last')
        #         Vis.routeLines[route.id].draw(Vis.win)
        #     if route.onRoute:
        #         Vis.routeLines[route.id].setFill('blue')
        #     if route.finished:
        #         Vis.routeLines[route.id].undraw()

    def update_queue(self, queue):
        print("update_queue")
        # if not Vis.queueText:
        #     Vis.queueText = Text(
        #         Point(Vis.scale * 10, Vis.dimensions[1] / 2 + Vis.scale),
        #         ""
        #     )
        #     Vis.queueText.draw(Vis.win)
        #     Vis.queueText.setSize(8)
        # Vis.queueText.setText("\n".join(
        #     [r.to_string() for r in queue]
        # ))