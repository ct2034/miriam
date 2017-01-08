import logging
import datetime
import random

from PyQt4 import QtCore
from numpy import linalg
from numpy.core.numeric import ndarray, array

from smartleitstand import msb


class Route(object):
    """a route to be simulated"""

    def __init__(self, start, goal, car, id, s):
        self.sim = s

        self.id = id

        assert start.__class__ is ndarray, 'Start needs to be a numpy.ndarray'
        self.start = start
        assert goal.__class__ is ndarray, 'Goal needs to be a numpy.ndarray'
        self.goal = goal

        self.car = None
        self.assign_car(car)

        self.onRoute = False

        self.vector = goal - start
        self.distance = linalg.norm(self.vector)
        self.remaining = self.distance

        self.finished = False

        self.preVector = None
        self.preDistance = None
        self.preRemaining = None

        self.creationTime = datetime.datetime.now()

        logging.info(
            "Created route with id " +
            str(self.id) +
            " distance: " +
            str(self.distance)
        )

    def assign_car(self, _car):
        self.car = _car
        if _car:  # if we are setting a car
            # assert _car.route == self, "car should not be on another route already"
            _car.route = self

            self.preVector = self.start - _car.pose
            self.preDistance = linalg.norm(self.preVector)
            self.preRemaining = self.preDistance

            if self.sim.msb_select:
                data = {"agvId": self.car.id, "jobId": self.id}
                msb.Msb.mwc.emit_event(msb.Msb.application, msb.Msb.eAGVAssignment, data=data)

    def new_step(self, stepSize):
        if self.car is None:
            return
        if not self.onRoute:  # on way to start
            if self.preDistance > 0:
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
                logging.info(str(self) + " reached Start")
                if self.sim.msb_select:
                    data = {"agvId": self.car.id, "jobId": self.id}
                    msb.Msb.mwc.emit_event(msb.Msb.application, msb.Msb.eReachedStart, data=data)
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
                self.sim.queue.remove(self)
                logging.info(str(self) + " reached Goal")
                if self.sim.msb_select:
                    msb.Msb.mwc.emit_event(msb.Msb.application, msb.Msb.eReached, data=self.id)

        self.sim.emit(QtCore.SIGNAL("update_route(PyQt_PyObject)"), self)
        if self.sim.msb_select:
            emit_car(msb, self.car)

    def toJobTuple(self):
        return ((self.start[0], self.start[1]),
                (self.goal[0], self.goal[1]),
                (self.creationTime - datetime.datetime.now()).total_seconds())

    def __str__(self):
        return "R%d: %s -> %s" % (self.id, str(self.start), str(self.goal))


def emit_car(msb, car):
    data = {"id": car.id, "x": float(car.pose[0]), "y": float(car.pose[1])}
    logging.debug(data)
    msb.Msb.mwc.emit_event(msb.Msb.application, msb.Msb.ePose, data=data)


class Car(object):
    """an AGV to be simulated"""

    nextId = 0

    def __init__(self, s):
        self.sim = s

        # assert s.__class__ is SimpSim, "Pass the simulation object to the new car"
        self.pose = array([
            # 10, 15
            random.randint(0, s.area.shape[0]),
            random.randint(0, s.area.shape[1])
        ])

        self.route = False

        self.id = Car.nextId
        Car.nextId += 1

        logging.info("New car:" +
                     str(self.id) +
                     " at "
                     + str(self.pose))

    def setPose(self, pose):
        self.pose = pose
        self.sim.emit(QtCore.SIGNAL("update_car(PyQt_PyObject)"), self)

    def toTuple(self):
        return tuple(self.pose)

    def __str__(self):
        return "C%d: [%.2f %.2f]" % (self.id, self.pose[0], self.pose[1])
