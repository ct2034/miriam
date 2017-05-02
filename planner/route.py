import datetime
import logging
import time
from threading import Lock
from enum import Enum

import numpy as np
from PyQt4 import QtCore
from numpy import linalg
from numpy.core.numeric import ndarray, array

msb = None


class Route(object):
    """a route to be simulated"""

    def __init__(self, start, goal, _id, s, idle_goal_stats=False):
        self.lock = Lock()
        self.sim = s

        self.id = _id
        if idle_goal_stats:
            self.state = RouteState.IDLE_GOAL_QUEUED
            self.idle_goal_stats = idle_goal_stats
            self.goal = goal
        else:
            self.state = RouteState.QUEUED

            assert start.__class__ is tuple, 'Start needs to be a numpy.tuple'
            assert len(start) == 2, 'Start should have 2 coords'
            self.start = start
            assert goal.__class__ is tuple, 'Goal needs to be a numpy.tuple'
            assert len(goal) == 2, 'Goal should have 2 coords'
            self.goal = goal

            self.vector = tuple(np.array(goal) - np.array(start))
            self.distance = linalg.norm(self.vector)

            self.creation_time = datetime.datetime.now()

        self.car = None

        if self.sim.msb_select:
            global msb
            from planner import msb

        logging.debug(str(self))

    def assign_car(self, _car):
        self.lock.acquire()
        logging.debug("Assigning a car to " + str(self))
        if self.car == _car:
            # nothing changed
            self.lock.release()
            return
        if self.state == RouteState.QUEUED:  # starting the route
            self.free_car(_car)
            self.car = _car
            self.state = RouteState.TO_START
            logging.debug(str(self))
            _car.route = self

            if self.sim.msb_select:
                data = {"agvId": self.car.id, "jobId": self.id}
                msb.Msb.mwc.emit_event(msb.Msb.application, msb.Msb.eAGVAssignment, data=data)
        elif self.state == RouteState.TO_START:  # had another car already
            self.free_car(_car)
            assert self.car, "Should have had a car, had: " + str(self.car) + ", should get: " + str(_car)
            self.car = _car
            _car.route = self
        elif self.state == RouteState.IDLE_GOAL_QUEUED:  # is an idle goal
            self.car = _car
            _car.route = self
            self.state = RouteState.IDLE_GOAL_RUNNING
        else:
            assert False, "Can not assign car in state " + str(self.state)
        self.lock.release()

    def free_car(self, _car):
        if _car.route:
            assert _car.route.is_re_assignable(), "This can only have been on the way or on a idle goal"
            if _car.route.state != RouteState.IDLE_GOAL:  # not idle goal
                _car.route.state = RouteState.QUEUED  # Other route is now queued again
            if _car.route.car:  # also idle goals loose their car
                _car.route.car = None  # not on that route any more

    def new_step(self, stepSize):
        self.lock.acquire()
        assert self.car, "Should have a car"
        i_prev = self.car.i
        self.car.i += stepSize
        i_prev_round = int(np.ceil(i_prev))
        i_next_round = int(np.floor(self.car.i))

        # e.g.

        # len() = 4 ..
        # 0    1     2     3
        #                ^   ^
        #           i_prev   car.i
        #              1.7   2.2

        # -> consider pos of t = 3

        assert i_next_round <= len(self.car.paths) + 5, "shooting far over goal"
        i_next_round = min(i_next_round, len(self.car.paths) - 1)  # e.g. 3
        assert not self.is_finished(), "Should not be finished"
        while self.car is None:
            time.sleep(.1)
            logging.warning("Waiting for car to be assigned")
        for _i in range(i_prev_round, i_next_round + 1):  # e.g. [3]
            if not self.is_idle_goal() and \
                    ((self.car.paths[_i][0:2] == tuple(self.start)) or \
                             (tuple(self.car.pose) == tuple(self.start))):
                self.at_start()
            elif ((self.car.paths[_i][0:2] == tuple(self.goal)) & self.is_on_route()) or \
                    (not self.is_idle_goal() and tuple(self.car.pose) == tuple(self.start)):  # @ goal
                self.at_goal()
                break
            # somewhere else
            if self.is_running():
                self.car.set_pose(tuple(self.car.paths[_i][0:2]))

        self.sim.emit(QtCore.SIGNAL("update_route(PyQt_PyObject)"), self)

        if self.sim.msb_select:
            emit_car(msb, self.car)
        self.lock.release()

    def at_goal(self):
        self.car.route = None
        self.car.set_pose(self.goal)
        self.car = None
        if self.is_idle_goal():
            assert self.state == RouteState.IDLE_GOAL_RUNNING, "Must have been running"
            self.state == RouteState.IDLE_GOAL_QUEUED  # queue it again
        else:
            assert self.state == RouteState.ON_ROUTE, "Must have been on route before"
            self.state = RouteState.FINISHED
        logging.info(str(self) + " reached Goal")
        if self.sim.msb_select:
            msb.Msb.mwc.emit_event(msb.Msb.application, msb.Msb.eReached, data=self.id)

    def at_start(self):
        self.car.set_pose(self.start)
        self.state = RouteState.ON_ROUTE
        self.pre_remaining = 0
        logging.info(str(self) + " reached Start")
        if self.sim.msb_select:
            data = {"agvId": self.car.id, "jobId": self.id}
            msb.Msb.mwc.emit_event(msb.Msb.application, msb.Msb.eReachedStart, data=data)

    def is_running(self):
        return (self.state == RouteState.TO_START or
                self.state == RouteState.ON_ROUTE or
                self.state == RouteState.IDLE_GOAL_RUNNING)

    def is_on_route(self):
        return self.state == RouteState.ON_ROUTE

    def is_finished(self):
        return self.state == RouteState.FINISHED

    def is_idle_goal(self):
        return self.state == RouteState.IDLE_GOAL_RUNNING or self.state == RouteState.IDLE_GOAL_QUEUED

    def is_re_assignable(self):
        return self.state == RouteState.TO_START or self.is_idle_goal()

    def to_tuple(self):
        if self.is_idle_goal():
            return tuple([(self.goal[0], self.goal[1]),
                          self.idle_goal_stats])

        else:
            return tuple([(self.start[0], self.start[1]),
                          (self.goal[0], self.goal[1]),
                          (datetime.datetime.now() - self.creation_time).total_seconds()])

    def __str__(self):
        if self.is_idle_goal():
            return "R%d: %s (%s) = %s" % (
                self.id, str(self.goal), str(self.state).split('.')[1], str(self.car))
        else:
            return "R%d: %s -> %s (%s) = %s" % (
                self.id, str(self.start), str(self.goal), str(self.state).split('.')[1], str(self.car))

    def __hash__(self):
        return hash(self.id)


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
        self.pose = tuple(([
            4, 3 + Car.nextId
            # random.randint(0, s.area.shape[0]),
            # random.randint(0, s.area.shape[1])
        ]))

        self.route = False

        self.id = Car.nextId
        Car.nextId += 1

        logging.info("New car:" +
                     str(self.id) +
                     " at "
                     + str(self.pose))

        self.paths = None
        self.lock = Lock()

    def set_pose(self, pose):
        self.lock.acquire()
        self.pose = tuple(pose)
        self.sim.emit(QtCore.SIGNAL("update_car(PyQt_PyObject)"), self)
        logging.info("Car " + str(self.id) + " @ " + str(self.pose))
        self.lock.release()

    def set_paths(self, _paths):
        self.lock.acquire()
        self.i = 0
        self.paths = []
        for path in _paths:
            self.paths += path
        self.lock.release()

    def to_tuple(self):
        assert len(self.pose) == 2, "A cars pose must have 2 coordinates"
        return (int(self.pose[0]),
                int(self.pose[1]))

    def __str__(self):
        return "C%d: [%.2f %.2f]" % (self.id, self.pose[0], self.pose[1])

    def __hash__(self):
        return hash(self.id) + hash(self.pose)


class RouteState(Enum):
    QUEUED = 0
    TO_START = 1
    ON_ROUTE = 2
    FINISHED = 3
    IDLE_GOAL_QUEUED = 4
    IDLE_GOAL_RUNNING = 5
