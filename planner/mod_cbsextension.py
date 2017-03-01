import datetime
import logging
from multiprocessing import Pipe
from multiprocessing import Process
import time
import numpy as np

from planner.cbs_ext.plan import plan, get_paths, comp2condition, comp2state
from planner.mod import Module
from planner.route import Route, Car
from planner.simulation import list_hash

from threading import Lock

FORMAT = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logging.getLogger("apscheduler").setLevel(logging.WARN)


def get_car_i(cars: list, car: Car):
    for i_agent in range(len(cars)):
        if car == cars[i_agent]:
            return i_agent


def plan_process(pipe, agent_pos, jobs, alloc_jobs, idle_goals, grid, fname):
    try:
        (agent_job,
         agent_idle,
         paths) = plan(agent_pos,
                       jobs,
                       alloc_jobs,
                       idle_goals,
                       grid,
                       False,
                       fname)
    except Exception as e:
        # Could not find a solution, returning just anything .. TODO: something better?
        logging.error(str(e))
        agent_job = []
        for a in agent_pos:
            agent_job.append(tuple())
        agent_job[0] = (0,)
        agent_idle = ()
        paths = get_paths(comp2condition(agent_pos, jobs, alloc_jobs, idle_goals, grid),
                          comp2state(tuple(agent_job), agent_idle, ()))

    pipe.send((agent_job,
               agent_idle,
               paths))


class Cbsext(Module):
    def __init__(self, grid):
        # params
        self.agent_job = ()
        self.agent_idle = ()
        self.paths = ()
        self.grid = grid

        # data
        self.fname = "planner/process_test.pkl"
        # if os.path.exists(self.fname):
        #     os.remove(self.fname)
        self.plan_params_hash = False
        self.process = False
        self.lock = Lock()

    def which_car(self, cars: list, route_todo: Route, routes: list) -> Car:
        routes = self.get_routes_to_plan(routes)
        self.update_plan(cars, routes)
        assert len(routes) > 0, "No routes to work with"
        self.lock.acquire()
        i_route = routes.index(route_todo)
        for i_agent in range(len(cars)):
            if len(self.agent_job[i_agent]) > 0:  # has assignment
                if i_route == self.agent_job[i_agent][0]:
                    self.lock.release()
                    return cars[i_agent]
        self.lock.release()
        return False

    def new_job(self, cars, routes):
        self.update_plan(cars, routes)

    def update_plan(self, cars, routes):
        self.lock.acquire()
        routes = self.get_routes_to_plan(routes)
        if list_hash(cars + routes) == self.plan_params_hash:
            self.lock.release()
            return

        if self.process:
            while self.process.is_alive():
                logging.warning("waiting (is already planning)")
                time.sleep(.4)

        agent_pos = []
        for c in cars:
            t = c.toTuple()
            assert not t[0].__class__ is np.ndarray
            assert t[0] == c.pose[0], "Problems with pose"
            agent_pos.append(t)

        jobs = []
        alloc_jobs = []
        for i_route in range(len(routes)):
            r = routes[i_route]
            if not r.is_finished():  # all but the finished ones
                jobs.append(r.to_job_tuple())
                if r.is_on_route():
                    alloc_jobs.append((get_car_i(cars, r.car), i_route))

        idle_goals = [((0, 0), (15, 3)),
                      ((4, 0), (15, 3),),
                      ((9, 0), (15, 3),),
                      ((9, 4), (15, 3),),
                      ((9, 9), (15, 3),),
                      ((4, 9), (15, 3),),
                      ((0, 9), (15, 3),),
                      ((0, 5), (15, 3),)]  # TODO: we have to learn these!

        planning_start = datetime.datetime.now()
        parent_conn, child_conn = Pipe()
        self.process = Process(target=plan_process,
                               args=(child_conn,
                                     agent_pos,
                                     jobs,
                                     alloc_jobs,
                                     idle_goals,
                                     self.grid,
                                     self.fname)
                               )
        self.process.name = "cbs_ext planner"
        self.process.start()
        logging.debug("process started")
        (self.agent_job,
         self.agent_idle,
         self.paths) = parent_conn.recv()
        logging.debug("process received")
        self.process.join(timeout=1)
        logging.debug("process joined")
        self.process.terminate()
        logging.debug("process terminated")

        logging.info("Planning took %.4fs" % (datetime.datetime.now() - planning_start).total_seconds())

        # save the paths in cars
        for i_car in range(len(cars)):
            cars[i_car].setPaths(self.paths[i_car])

        self.plan_params_hash = list_hash(cars + routes)  # how we have planned last time
        self.lock.release()

    def get_routes_to_plan(self, routes):
        return list(filter(lambda r: not r.is_finished(), routes))
