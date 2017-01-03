import logging

logging.basicConfig(level=logging.INFO)

import time

from smartleitstand.route import Route, Car, emit_car
from smartleitstand.which_car import which_car
from smartleitstand import msb
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
        routeTodo.assign_car(which_car(freeCars, routeTodo, []))
        SimpSim.activeRoutes.append(routeTodo)
        # SimpSim.v.update_queue(SimpSim.queue)



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
        SimpSim.queue.append(Route(a, b, None, job_id, self))

    def set_speed_multiplier(self, multiplier):
        SimpSim.speedMultiplier = multiplier


def get_distance(a, b):
    assert a.size is 2, "A point needs to have two coordinates"
    assert b.size is 2, "B point needs to have two coordinates"
    return linalg.norm(a - b)
