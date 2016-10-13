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

def emit_car(msb, car):
    data = {"id": car.id, "x": float(car.pose[0]), "y": float(car.pose[1])}
    print(data)
    msb.Msb.mwc.emit_event(msb.Msb.application, msb.Msb.ePose, data=data)

class SimpSim(QtCore.QThread):
    """simulation of multiple AGVs"""

    queue = []
    activeRoutes = []
    cars = []
    driveSpeed = 2
    simTime = 1
    running = False

    def __init__(self, msb_select: bool, parent=None):
        QtCore.QThread.__init__(self, parent)
        print("init Simulation")

        self.msb_select = msb_select
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
            c = Car(self)
            SimpSim.cars.append(c)
            if self.msb_select:
                emit_car(msb, c)

        SimpSim.running = True
        self.emit(QtCore.SIGNAL("open(int, int, PyQt_PyObject)"), width, height, SimpSim.cars)

    def stop(self):
        SimpSim.running = False
        self.area = False
        SimpSim.queue = []
        SimpSim.activeRoutes = []
        SimpSim.cars = []
        Car.nextId = 0

    def new_job(self, a, b, id):
        SimpSim.queue.append(Route(a, b, False, id, self))

    def iterate(self):
        while True:
            try:
                if SimpSim.running:
                    work_queue()
                    print(".")
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

    def __init__(self, start, goal, car, id, s):
        self.sim = s

        self.id = id

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

        print(
            "Created route with id",
            str(self.id),
            "distance:",
            self.distance
        )

    def assign_car(self, car):
        self.car = car
        if car:  # if we are setting a car
            assert car.route == False, "car is not on a route"
            car.route = self

            self.preVector = self.start - car.pose
            self.preDistance = linalg.norm(self.preVector)
            self.preRemaining = self.preDistance

            if self.sim.msb_select:
                data = {"agvId": self.car.id, "jobId": self.id}
                msb.Msb.mwc.emit_event(msb.Msb.application, msb.Msb.eAGVAssignment, data=data)

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
                if self.sim.msb_select:
                    msb.Msb.mwc.emit_event(msb.Msb.application, msb.Msb.eReached, data=self.id)

        self.sim.emit(QtCore.SIGNAL("update_route(PyQt_PyObject)"), self)
        if self.sim.msb_select:
            emit_car(msb, self.car)


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
