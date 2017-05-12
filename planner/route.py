import datetime
import logging
import time
from threading import Lock
from enum import Enum

import numpy as np
from numpy import linalg

msb = None



class Route(object):
    """a route to be simulated"""
    lock = Lock()

    def __init__(self, start, goal, _id, s, idle_goal_stats=False):
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

        logging.debug("Init:" + str(self))

    def assign_car(self, _car):
        Route.lock.acquire()
        logging.debug("Assigning " + str(_car) + " to " + str(self))
        if self.car == _car:  # nothing changed
            logging.debug("nothing changed")
        else:
            if _car.get_route() and _car.get_route().is_running():
                free_car(_car)
            if self.state == RouteState.QUEUED:  # starting the route
                self.car = _car
                self.state = RouteState.TO_START
                logging.debug(str(self))

                if self.sim.msb_select:
                    data = {"agvId": self.car.id, "jobId": self.id}
                    msb.Msb.mwc.emit_event(msb.Msb.application, msb.Msb.eAGVAssignment, data=data)
            elif self.state == RouteState.TO_START:  # had another car already
                assert self.car, "Should have had a car, had: " + str(self.car) + ", should get: " + str(_car)
                self.car = _car
            elif self.is_idle_goal():  # is an idle goal
                self.car = _car
                self.state = RouteState.IDLE_GOAL_RUNNING
            else:
                assert False, "Can not assign car in state " + str(self.state)
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            self.sim.print_debug_info()
        Route.lock.release()

    def new_step(self, step_size):
        Route.lock.acquire()
        assert self.car, "Should have a car " + str(self)
        i_prev = self.car.i
        self.car.i += step_size
        i_prev_round = int(np.ceil(i_prev))
        i_next_round = int(np.floor(self.car.i))

        # e.g.

        # len() = 4 ..
        # 0    1     2     3
        #                ^   ^
        #           i_prev   car.i
        #              1.7   2.2

        # -> consider pos of t = 3

        # assert i_next_round <= len(self.car.paths), "shooting far over goal: " + str(self) TODO: can we forget about this?
        i_next_round = min(i_next_round, len(self.car.paths) - 1)  # e.g. 3
        assert not self.is_finished(), "Should not be finished"
        while self.car is None:
            time.sleep(.1)
            logging.warning("Waiting for car to be assigned")
        for _i in range(i_prev_round, i_next_round + 1):  # e.g. [3]
            if not self.is_idle_goal() and \
                    ((self.car.paths[_i][0:2] == tuple(self.start)) or
                         (tuple(self.car.pose) == tuple(self.start))):
                self.at_start()
            elif (self.car.paths[_i][0:2] == tuple(self.goal)) and (
                self.is_on_route() or self.is_idle_goal()):  # @ goal
                self.at_goal()
                break
            # somewhere else
            if self.is_running():
                self.car.set_pose(tuple(self.car.paths[_i][0:2]))

        # self.sim.emit(QtCore.SIGNAL("update_route(PyQt_PyObject)"), self)

        if self.sim.msb_select:
            emit_car(msb, self.car)
        Route.lock.release()

    def at_goal(self):
        self.car.set_pose(self.goal)
        self.car = None
        if self.is_idle_goal():
            assert self.state == RouteState.IDLE_GOAL_RUNNING, "Must have been running"
            self.state = RouteState.IDLE_GOAL_QUEUED  # queue it again
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
        return hash(self.id * 1000)


def emit_car(msb, car):
    data = {"id": car.id, "x": float(car.pose[0]), "y": float(car.pose[1])}
    logging.debug(data)
    msb.Msb.mwc.emit_event(msb.Msb.application, msb.Msb.ePose, data=data)


class Car(object):
    """an AGV to be simulated"""

    next_id = 0

    def __init__(self, s):
        self.sim = s
        self.pose = (4, 3 + Car.next_id)  # TODO: how to init?
        self.id = Car.next_id
        Car.next_id += 1
        self.paths = None
        self.lock = Lock()

        logging.debug("Init:" + str(self))

    def set_pose(self, pose):
        self.lock.acquire()
        self.pose = tuple(pose)
        # self.sim.emit(QtCore.SIGNAL("update_car(PyQt_PyObject)"), self)
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

    def get_route(self):
        """On which route is this car (if any)"""
        for r in self.sim.routes:
            if r.car == self:
                # assert r.is_running(), "The route having this car should be running"
                return r
        return None

    def __str__(self):
        return "C%d: [%.2f %.2f]" % (self.id, self.pose[0], self.pose[1])

    def __hash__(self):
        return hash(100*self.id)


def free_car(_car: Car):
    if _car.get_route():
        assert _car.get_route().is_re_assignable(), "This can only have been on the way or on a idle goal"
        if _car.get_route().is_idle_goal():
            _car.get_route().state = RouteState.IDLE_GOAL_QUEUED  # back to queue
        else: # normal route
            _car.get_route().state = RouteState.QUEUED  # Other route is now queued again

        if _car.get_route().car:  # also routes loose their car
            _car.get_route().car = None


class RouteState(Enum):
    QUEUED = 0
    TO_START = 1
    ON_ROUTE = 2
    FINISHED = 3
    IDLE_GOAL_QUEUED = 4
    IDLE_GOAL_RUNNING = 5
