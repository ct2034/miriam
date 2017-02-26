import logging
import time

from PyQt4 import QtCore
from apscheduler.schedulers.background import BackgroundScheduler
from numpy import *

from planner.route import Route, Car, emit_car


class SimpSim(QtCore.QThread):
    """simulation of multiple AGVs"""
    queued_routes = []
    active_routes = []
    finished_routes = []
    old_queue_hash = 0

    cars = []
    driveSpeed = 4
    speedMultiplier = 1
    simTime = .5
    running = False
    scheduler = BackgroundScheduler()
    i = 0
    startTime = time.time()
    replan = True

    def __init__(self, msb_select: bool, mod, parent=None):
        self.module = mod

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
        SimpSim.queued_routes = []
        SimpSim.active_routes = []
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
        SimpSim.queued_routes.append(Route(a, b, None, job_id, self))
        self.module.new_job(SimpSim.cars, SimpSim.queued_routes, SimpSim.active_routes)

    def is_finished(self, id):
        routes = list(filter(lambda r: r.id == id,
                             SimpSim.active_routes +
                             SimpSim.queued_routes +
                             SimpSim.finished_routes))
        assert len(routes) >= 1, "There should be one route with this id"
        return routes[0].finished

    def set_speed_multiplier(self, multiplier):
        SimpSim.speedMultiplier = multiplier

    def iterate(self):
        logging.debug("it ...")
        try:
            if SimpSim.running:
                self.work_queue()
                for j in SimpSim.active_routes:
                    if not j.finished:
                        j.new_step(
                            SimpSim.driveSpeed *
                            SimpSim.speedMultiplier *
                            SimpSim.simTime
                        )
                SimpSim.i += 1
        except Exception as e:
            logging.error("ERROR:" + str(e))
            raise e
        logging.debug("... it")

    def work_queue(self):
        print("q:" + str(len(self.queued_routes)) +
              " | a:" + str(len(self.active_routes)) +
              " | f:" + str(len(self.finished_routes)) +
              " | r:" + str(int(self.replan)))
        for r in SimpSim.queued_routes:
            if self.replan:
                c = self.module.which_car(SimpSim.cars.copy(), r, SimpSim.queued_routes.copy(),
                                          SimpSim.active_routes.copy())
                if c:
                    r.assign_car(c)
                    if r not in SimpSim.active_routes:
                        SimpSim.active_routes.append(r)
                    self.replan = False
                else:  # no car
                    self.replan = True

    def checkfree(self, car: Car, pose: ndarray):
        cars_to_check = self.cars.copy()
        cars_to_check.remove(car)
        for c in cars_to_check:
            if c.pose[0] == pose[0] and c.pose[1] == pose[1]:
                return False
        return True


def get_distance(a, b):
    assert a.size is 2, "A point needs to have two coordinates"
    assert b.size is 2, "B point needs to have two coordinates"
    return linalg.norm(a - b)


def listhash(l):
    return sum(list(map(hash, l)))
