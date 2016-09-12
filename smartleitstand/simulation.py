from numpy import *
import time
import threading
import random
from graphics import *

from vfk_msb_py.msb_ws4py_client import MsbWsClient
from vfk_msb_py.msb_classes import *

import smartleitstand

def iterate():
    SimpSim.running = True
    while SimpSim.running:
        try:
            work_queue()
            # print(".")
            for j in SimpSim.activeRoutes:
                if not j.finished:
                    j.new_step(SimpSim.driveSpeed * SimpSim.simTime)
            time.sleep(SimpSim.simTime)
        except Exception as e:
            print("ERROR:", str(e))
            raise e


def pointFromPose(pose):
    poseVis = pose * Vis.scale
    return Point(poseVis[0], poseVis[1])

# class Vis(object):
#     """Visualisation of the AGVs and environment"""
#
#     scale = 5
#     carCircles = {}
#     routeLines = {}
#     queueText = False
#     dimensions = array([0, 0])
#
#     def open(self, x, y, cars):
#         Vis.dimensions[0] = x * Vis.scale
#         Vis.dimensions[1] = y * Vis.scale
#         Vis.win = GraphWin(
#             'cloudnav',
#             width = Vis.dimensions[0],
#             height = Vis.dimensions[1]
#         )
#         for car in cars:
#             Vis.carCircles[car.id] = Circle(pointFromPose(car.pose), Vis.scale)
#             Vis.carCircles[car.id].draw(Vis.win)
#             Vis.carCircles[car.id].setFill('green')
#
#     def updateCar(self, car):
#         if car:
#             poseVis = car.pose * Vis.scale
#             dx = poseVis[0] - Vis.carCircles[car.id].getCenter().x
#             dy = poseVis[1] - Vis.carCircles[car.id].getCenter().y
#             Vis.carCircles[car.id].move(dx=dx, dy=dy)
#
#     def updateRoute(self, route):
#         if route:
#             if route.id not in Vis.routeLines.keys():
#                 Vis.routeLines[route.id] = Line(pointFromPose(route.start), pointFromPose(route.goal))
#                 Vis.routeLines[route.id].setFill('red')
#                 Vis.routeLines[route.id].setArrow('last')
#                 Vis.routeLines[route.id].draw(Vis.win)
#             if route.onRoute:
#                 Vis.routeLines[route.id].setFill('blue')
#             if route.finished:
#                 Vis.routeLines[route.id].undraw()
#
#     def updateQueue(self, queue):
#         if not Vis.queueText:
#             Vis.queueText = Text(
#                 Point(Vis.scale * 10, Vis.dimensions[1] / 2 + Vis.scale),
#                 ""
#             )
#             Vis.queueText.draw(Vis.win)
#             Vis.queueText.setSize(8)
#         Vis.queueText.setText("\n".join(
#             [r.to_string() for r in queue]
#         ))


def work_queue():
    freeCars = []
    routeTodo = False
    for c in SimpSim.cars:
        if not c.route and len(SimpSim.queue) > 0:
            freeCars.append(c)
            routeTodo = SimpSim.queue.pop()
    if routeTodo:
        routeTodo.assign_car(smartleitstand.which_car(freeCars, routeTodo, []))
        SimpSim.activeRoutes.append(routeTodo)
        # SimpSim.v.updateQueue(SimpSim.queue)

class SimpSim(object):
    """simulation of multiple AGVs"""

    queue = []
    activeRoutes = []
    cars = []
    driveSpeed = 50
    simTime = .01
    running = False
    # v = Vis()

    def __init__(self):
        print("init Simulation")

        self.area = zeros([1])
        self.number_agvs = 1

    def start(self, width, height, number_agvs):
        self.area = zeros([width, height])
        self.number_agvs = number_agvs
        for i in range(self.number_agvs):
            SimpSim.cars.append(Car(self))
        # SimpSim.v.open(width, height, SimpSim.cars)
        threading.Thread(target=iterate).start()

    def stop(self):
        SimpSim.running = False

    def new_job(self, a, b):
        SimpSim.queue.append(Route(a, b, False, self))


def get_distance(a, b):
    assert a.size is 2, "A point needs to have two coordinates"
    assert b.size is 2, "B point needs to have two coordinates"
    return linalg.norm(a - b)

class Route(object):
    """a route to be simulated"""

    nextId = 0

    def __init__(self, start, goal, car, s):
        self.sim = s

        self.id = Route.nextId
        Route.nextId += 1

        assert start.__class__ is ndarray, 'Start needs to be a numpy.ndarray'
        self.start = start
        assert goal.__class__ is ndarray, 'Goal needs to be a numpy.ndarray'
        self.goal = goal

        self.assign_car(car)

        self.onRoute = False

        self.vector = goal - start
        self.distance = linalg.norm(self.vector)
        self.remaining = self.distance

        self.finished = False


        print("Created route with id", str(self.id), "distance:", self.distance)

    def assign_car(self, car):
        self.car = car
        if car:  # if we are setting a car
            assert car.route == False, "car is not on a route"
            car.route = self

            self.preVector = self.start - car.pose
            self.preDistance = linalg.norm(self.preVector)
            self.preRemaining = self.preDistance

    def new_step(self, stepSize):
        if not self.onRoute: # on way to start
            self.car.setPose(
                stepSize * self.preVector /
                self.preDistance +
                self.car.pose
            )

            self.preRemaining -= stepSize

            if self.preRemaining <= 0:
                self.onRoute = True
                self.car.setPose(self.start)
                self.preRemaining = 0
                print(self.to_string(), "reached Start")
        else: # on route
            self.car.setPose(
                stepSize * self.vector /
                self.distance +
                self.car.pose
            )

            self.remaining -= stepSize

            if self.remaining <= 0:
                self.car.route = False
                self.car.setPose(self.goal)
                self.remaining = 0
                self.finished = True
                print(self.to_string(), "reached Goal")

        # SimpSim.v.updateRoute(self)

    def to_string(self):
        return " ".join(("R", str(self.id), ":", str(self.start), "->", str(self.goal)))

class Car(object):
    """an AGV to be simulated"""

    nextId = 0

    def __init__(self, s):
        self.sim = s

        assert s.__class__ is SimpSim, "Pass the simulation object to the new car"
        self.pose = array([
            random.randint(0, s.area.shape[0]),
            random.randint(0, s.area.shape[1])
        ])

        self.route = False

        self.id = Car.nextId
        Car.nextId += 1

        print("New car:", str(self.id), "at", str(self.pose))

    def setPose(self, pose):
        self.pose = pose
        # SimpSim.v.updateCar(self)