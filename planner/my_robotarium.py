from numpy import *
from robotarium import Robotarium, transformations, graph
import json

def pointFromPose(pose):
    poseVis = pose * Vis.scale
    # print(str(poseVis))
    return QtCore.QPoint(poseVis[0], poseVis[1])


class MyRob:
    """Visualisation of the AGVs and environment"""

    scale = False
    carCircles = {}
    routeLines = {}
    queueText = False
    dimensions = array([0, 0])
    msbThread = False
    scene = False

    def __init__(self, simThread, parent=None):
        print("simThread: " + str(simThread))
        Vis.simThread = simThread
        self.r = Robotarium()

    def open(self, cars):
        x = 1
        y = 1
        assert cars > self.r.get_available_agents(), "not that many agents available"

        self.r.initialize(cars)

        Vis.scale = float(min(
            desktop.geometry().width() / x,
            desktop.geometry().height() / y
        )/1.5)
        if Vis.scale < 1:
            Vis.scale = 1
        print("Vis.scale " + str(Vis.scale))

        print("open")
        margin = 20
        width = x * Vis.scale
        height = y * Vis.scale

        view = QtGui.QGraphicsView(self)
        Vis.scene = QtGui.QGraphicsScene()

        field = QtGui.QGraphicsRectItem(0, 0, width, height)
        brush = QtGui.QBrush(dark_grey)
        field.setBrush(brush)
        Vis.scene.addItem(field)

        for car in cars:
            Vis.carCircles[car.id] = QtGui.QGraphicsEllipseItem(0, 0, Vis.scale, Vis.scale)
            Vis.scene.addItem(Vis.carCircles[car.id])
            self.update_car(car)

        view.setScene(Vis.scene)
        view.resize(width+margin, height+margin)
        self.resize(width+margin, height+margin)
        view.show()

    def update_car(self, car):
        if car:
            Vis.carCircles[car.id].setPos(pointFromPose(car.pose)-QtCore.QPointF(Vis.scale/2, Vis.scale/2))
            # print(Vis.carCircles[car.id].pos().x())
            brush = self.brush_for_car(car)
            Vis.carCircles[car.id].setBrush(brush)

    def update_route(self, route):
        if route:
            if route.id not in Vis.routeLines.keys():
                Vis.routeLines[route.id] = QtGui.QGraphicsLineItem()
                Vis.routeLines[route.id].setLine(QtCore.QLineF(pointFromPose(route.start), pointFromPose(route.goal)))
                Vis.scene.addItem(Vis.routeLines[route.id])
                Vis.routeLines[route.id].setPen(QtGui.QPen(blue))
        #         Vis.routeLines[route.id].setArrow('last')
            elif route.onRoute:
                Vis.routeLines[route.id].setPen(QtGui.QPen(red))
            elif route.finished:
                print("hide")
                Vis.routeLines[route.id].hide()

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