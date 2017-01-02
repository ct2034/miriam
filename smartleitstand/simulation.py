import logging
import time

logging.basicConfig(level=logging.INFO)

from smartleitstand import msb, which_car
from PyQt4 import QtCore
from apscheduler.schedulers.background import BackgroundScheduler
from numpy import *


def work_queue():
    freeCars = []
    routeTodo = None
    for c in SimpSim.cars:
        if not c.route and len(SimpSim.queue) > 0:
            freeCars.append(c)
            routeTodo = SimpSim.queue.pop()
            break
    if routeTodo:
        routeTodo.assign_car(which_car.which_car(freeCars, routeTodo, []))
        SimpSim.activeRoutes.append(routeTodo)
        # SimpSim.v.update_queue(SimpSim.queue)


def emit_car(msb, car):
    data = {"id": car.id, "x": float(car.pose[0]), "y": float(car.pose[1])}
    logging.debug(data)
    msb.Msb.mwc.emit_event(msb.Msb.application, msb.Msb.ePose, data=data)


def iterate():
    try:
        if SimpSim.running:
            work_queue()
            for j in SimpSim.activeRoutes:
                if not j.finished:
                    j.new_step(
                        SimpSim.driveSpeed *
                        SimpSim.speedMultiplier *
                        SimpSim.simTime
                    )
            SimpSim.i += 1
    except Exception as e:
        logging.error("ERROR:", str(e))
        raise e


class SimpSim(QtCore.QThread):
    """simulation of multiple AGVs"""

    queue = []
    activeRoutes = []
    cars = []
    driveSpeed = 10
    speedMultiplier = 1
    simTime = .1
    running = False
    scheduler = BackgroundScheduler()
    i = 0
    startTime = time.time()

    def __init__(self, msb_select: bool, parent=None):
        QtCore.QThread.__init__(self, parent)
        logging.info("init Simulation")

        self.msb_select = msb_select
        if msb_select:
            msb.Msb(self)

        self.area = zeros([1])
        self.number_agvs = 1

        SimpSim.scheduler.add_job(
            func=iterate,
            trigger='interval',
            id="sim_iterate",
            seconds=SimpSim.simTime,
            max_instances=1,
            replace_existing=True  # for restarting
        )

    def run(self):
        SimpSim.running = True

    def start_sim(self, width, height, number_agvs):
        self.area = zeros([width, height])
        self.number_agvs = number_agvs
        Car.nextId = 0
        SimpSim.cars = []
        for i in range(self.number_agvs):
            c = Car(self)
            SimpSim.cars.append(c)
            if self.msb_select:
                emit_car(msb, c)

        SimpSim.running = True
        if SimpSim.scheduler.running:
            logging.info("Resuming")
            SimpSim.scheduler.resume()
        else:
            logging.info("Resuming")
            SimpSim.scheduler.start()
        self.emit(QtCore.SIGNAL("open(int, int, PyQt_PyObject)"), width, height, SimpSim.cars)

        SimpSim.i = 0
        self.startTime = time.time()

    def stop(self):
        SimpSim.running = False
        self.area = False
        SimpSim.queue = []
        SimpSim.activeRoutes = []
        SimpSim.cars = []
        Car.nextId = 0

        if SimpSim.scheduler.running:
            logging.info("Pause")
            SimpSim.scheduler.pause()

        logging.info('end-start= ' + str(time.time() - self.startTime))
        logging.info('i= ' + str(SimpSim.i))
        logging.info('i*SimTime= ' + str(SimpSim.i * SimpSim.simTime))
        logging.info('missing: ' + str(time.time() - self.startTime - SimpSim.i * SimpSim.simTime) + 's')

    def new_job(self, a, b, job_id):
        SimpSim.queue.append(Route(a, b, False, job_id, self))

    def set_speed_multiplier(self, multiplier):
        SimpSim.speedMultiplier = multiplier


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

        logging.info(
            "Created route with id " +
            str(self.id) +
            " distance: " +
            str(self.distance)
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
                logging.info(str(self) + " reached Goal")
                if self.sim.msb_select:
                    msb.Msb.mwc.emit_event(msb.Msb.application, msb.Msb.eReached, data=self.id)

        self.sim.emit(QtCore.SIGNAL("update_route(PyQt_PyObject)"), self)
        if self.sim.msb_select:
            emit_car(msb, self.car)

    def __str__(self):
        return " ".join(("R", str(self.id), ":", str(self.start), "->", str(self.goal)))


class Car(object):
    """an AGV to be simulated"""

    nextId = 0

    def __init__(self, s):
        self.sim = s

        assert s.__class__ is SimpSim, "Pass the simulation object to the new car"
        self.pose = array([
            50, 10
            # random.randint(0, s.area.shape[0]),
            # random.randint(0, s.area.shape[1])
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

    def __str__(self):
        return "".join(("C", str(self.id), ":", str(self.pose)))
