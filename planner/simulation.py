import logging
import time

from threading import Lock
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.events import *
from numpy import *

from planner.route import Route, RouteState, Car, emit_car

msb = None


def set_speed_multiplier(multiplier):
    SimpSim.speed_multiplier = multiplier


def get_distance(a, b):
    assert a.size is 2, "A point needs to have two coordinates"
    assert b.size is 2, "B point needs to have two coordinates"
    return linalg.norm(a - b)


def list_hash(l):
    return sum(list(map(hash, l)))


def my_listener(event):
    if event.exception:
        SimpSim.running = False
        SimpSim.scheduler.remove_all_jobs()
        SimpSim.scheduler.shutdown()


class SimpSim():
    """simulation of multiple AGVs"""
    routes = []
    cars = []
    drive_speed = 2.  # m/s
    speed_multiplier = 1
    sim_time = .5  # s
    running = False
    scheduler = BackgroundScheduler()
    i = 0
    start_time = time.time()

    def __init__(self, msb_select: bool, _mod, parent=None):
        # QtCore.QThread.__init__(self, parent)
        logging.info("init Simulation")
        self.lock = Lock()

        self.msb_select = msb_select
        if msb_select:
            global msb
            from planner import msb
            msb.Msb(self)

        self.area = zeros([1])
        self.number_agvs = 1
        self.module = _mod

        SimpSim.scheduler.add_job(
            func=self.iterate,
            trigger='interval',
            id="sim_iterate",
            seconds=SimpSim.sim_time,
            max_instances=1,
            replace_existing=True  # for restarting
        )
        SimpSim.scheduler.add_listener(my_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)

    def start_sim(self, width, height, number_agvs):
        SimpSim.running = True
        self.area = zeros([width, height])
        self.number_agvs = number_agvs
        Car.next_id = 0
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
        # self.emit(QtCore.SIGNAL("open(int, int, PyQt_PyObject)"), width, height, SimpSim.cars)

        SimpSim.i = 0
        self.startTime = time.time()

    def stop_sim(self):
        SimpSim.running = False
        self.area = False
        SimpSim.routes = []
        SimpSim.cars = []
        Car.next_id = 0

        if SimpSim.scheduler.running:
            logging.info("Pause")
            SimpSim.scheduler.pause()

        logging.info('end-start= ' + str(time.time() - self.startTime))
        logging.info('i= ' + str(SimpSim.i))
        logging.info('i*SimTime= ' + str(SimpSim.i * SimpSim.sim_time))
        logging.info('missing: ' + str(time.time() - self.startTime - SimpSim.i * SimpSim.sim_time) + 's')

    def new_job(self, a, b, job_id):
        self.lock.acquire()
        SimpSim.routes.append(Route(a, b, job_id, self))
        self.module.new_job(SimpSim.cars, SimpSim.routes)
        self.lock.release()

    def new_idle_goal(self, goal, stats, id):
        self.lock.acquire()
        SimpSim.routes.append(Route(-1, goal, id, self, stats))
        self.module.new_job(SimpSim.cars, SimpSim.routes)
        self.lock.release()

    def is_finished(self, _id):
        self.lock.acquire()  # TODO: deadlock?
        route = list(filter(lambda r: r.id == _id, self.routes))
        assert len(route) == 1, "There should be exactly one route with this id"
        is_finished = route[0].is_finished()
        self.lock.release()
        return is_finished

    def iterate(self):
        logging.debug("it ...")
        self.lock.acquire()
        try:
            if SimpSim.running:
                self.work_routes()
                for j in self.routes:
                    if j.is_running():
                        j.new_step(
                            SimpSim.drive_speed *
                            SimpSim.speed_multiplier *
                            SimpSim.sim_time
                        )
                if str(self.module.__class__) == "<class 'planner.mod_nearest.Cbsext'>":  # only catching collisions on Cbsext
                    poses = set()
                    for c in self.cars:
                        pose = c.pose
                        assert pose not in poses, "Collision!"
                        poses.add(pose)
                SimpSim.i += 1
        except Exception as _e:
            logging.error("ERROR:" + str(_e))
            raise _e
        self.lock.release()
        logging.debug("... it")

    def work_routes(self):
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            self.print_debug_info()
        for r in self.routes:
            if not (r.is_finished() or r.is_on_route()):  # for all but the finished or on_route ones
                c = self.module.which_car(SimpSim.cars, r, SimpSim.routes)
                if c:
                    r.assign_car(c)

    def print_debug_info(self):
        n_queued = 0
        n_to_start = 0
        n_on_route = 0
        n_finished = 0
        n_ig_queued = 0
        n_ig_running = 0
        for r in self.routes:
            if r.state is RouteState.QUEUED:
                n_queued += 1
            elif r.state is RouteState.TO_START:
                n_to_start += 1
            elif r.state is RouteState.ON_ROUTE:
                n_on_route += 1
            elif r.state is RouteState.FINISHED:
                n_finished += 1
            elif r.state is RouteState.IDLE_GOAL_QUEUED:
                n_ig_queued += 1
            elif r.state is RouteState.IDLE_GOAL_RUNNING:
                n_ig_running += 1
        assert len(self.routes) == n_queued + n_to_start + n_on_route + \
                                   n_finished + n_ig_queued + n_ig_running, "Not all routes have a state"
        logging.debug("q:" + str(n_queued) +
                      " | ts:" + str(n_to_start) +
                      " | or:" + str(n_on_route) +
                      " | f:" + str(n_finished) +
                      " | iq:" + str(n_ig_queued) +
                      " | ir:" + str(n_ig_running))
        assert (n_ig_running + n_on_route + n_to_start) <= len(self.cars), "There can be only so many routes running"
