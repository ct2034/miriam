from numpy import *
import time
import threading
import random
from PyQt4 import QtGui, QtCore

import smartleitstand
import msb


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
        # SimpSim.v.update_queue(SimpSim.queue)


class SimpSim(QtCore.QThread):
    """simulation of multiple AGVs"""

    queue = []
    activeRoutes = []
    cars = []
    driveSpeed = .5
    simTime = .01
    running = False

    def __init__(self, msb_select: bool, parent=None):
        QtCore.QThread.__init__(self, parent)
        print("init Simulation")

        if msb_select:
            msb.Msb(self)

        self.area = zeros([1])
        self.number_agvs = 1

    def run(self):
        self.iterate()

    def start_sim(self, width, height, number_agvs):
        self.area = zeros([width, height])
        self.number_agvs = number_agvs
        for i in range(self.number_agvs):
            SimpSim.cars.append(Car(self))
        self.emit(QtCore.SIGNAL("open(int, int, PyQt_PyObject)"), width, height, SimpSim.cars)

    def stop(self):
        SimpSim.running = False

    def new_job(self, a, b):
        SimpSim.queue.append(Route(a, b, False, self))

    def iterate(self):
        SimpSim.running = True
        while SimpSim.running:
            try:
                work_queue()
                # print(".")
                for j in SimpSim.activeRoutes:
                    if not j.finished:
                        j.new_step(SimpSim.driveSpeed * SimpSim.simTime)
                self.sleep(SimpSim.simTime)
            except Exception as e:
                print("ERROR:", str(e))
                raise e


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
        if not self.onRoute:  # on way to start
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
        else:  # on route
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

                # SimpSim.v.update_route(self)

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
        self.sim.emit(QtCore.SIGNAL("update_car(PyQt_PyObject)"), self)
