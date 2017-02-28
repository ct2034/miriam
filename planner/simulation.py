import logging
import time

from PyQt4 import QtCore
from apscheduler.schedulers.background import BackgroundScheduler
from numpy import *

from planner.route import Route, RouteState, Car, emit_car

msb = None


def set_speed_multiplier(multiplier):
    SimpSim.speedMultiplier = multiplier


def get_distance(a, b):
    assert a.size is 2, "A point needs to have two coordinates"
    assert b.size is 2, "B point needs to have two coordinates"
    return linalg.norm(a - b)


def list_hash(l):
    return sum(list(map(hash, l)))


class SimpSim(QtCore.QThread):
    """simulation of multiple AGVs"""
    routes = []
    cars = []
    driveSpeed = 0.9  # m/s
    speedMultiplier = 1
    simTime = 1  # s
    running = False
    scheduler = BackgroundScheduler()
    i = 0
    startTime = time.time()

    def __init__(self, msb_select: bool, _mod, parent=None):
        self.module = _mod

        QtCore.QThread.__init__(self, parent)
        logging.info("init Simulation")

        self.msb_select = msb_select
        if msb_select:
            global msb
            from planner import msb
            msb.Msb(self)

        self.area = zeros([1])
        self.number_agvs = 1

        SimpSim.scheduler.add_job(
            func=self.iterate,
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
        SimpSim.routes = []
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
        SimpSim.routes.append(Route(a, b, job_id, self))
        self.module.new_job(SimpSim.cars, SimpSim.routes)

    def is_finished(self, _id):
        route = list(filter(lambda r: r.id == _id, self.routes))
        assert len(route) == 1, "There should be exactly one route with this id"
        return route[0].is_finished()

    def iterate(self):
        logging.debug("it ...")
        try:
            if SimpSim.running:
                self.work_routes()
                for j in self.routes:
                    if j.is_running():
                        j.new_step(
                            SimpSim.driveSpeed *
                            SimpSim.speedMultiplier *
                            SimpSim.simTime
                        )
                SimpSim.i += 1
        except Exception as _e:
            logging.error("ERROR:" + str(_e))
            raise _e
        logging.debug("... it")

    def work_routes(self):
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            n_queued = 0
            n_to_start = 0
            n_on_route = 0
            n_finished = 0
            for r in self.routes:
                if r.state is RouteState.QUEUED:
                    n_queued += 1
                elif r.state is RouteState.TO_START:
                    n_to_start += 1
                elif r.state is RouteState.ON_ROUTE:
                    n_on_route += 1
                elif r.state is RouteState.FINISHED:
                    n_finished += 1
            assert len(self.routes) == n_queued + n_to_start + n_on_route + n_finished, "Not all routes have s state"
            logging.debug("q:" + str(n_queued) +
                          " | ts:" + str(n_to_start) +
                          " | or:" + str(n_on_route) +
                          " | f:" + str(n_finished))

        for r in self.routes:
            if not (r.is_finished() or r.is_on_route()):  # for all but the finished or on_route ones
                c = self.module.which_car(SimpSim.cars.copy(), r, self.routes)
                if c:
                    r.assign_car(c)

    def check_free(self, car: Car, pose: ndarray):
        cars_to_check = self.cars.copy()
        cars_to_check.remove(car)
        for c in cars_to_check:
            if c.pose[0] == pose[0] and c.pose[1] == pose[1]:
                return False
        return True
