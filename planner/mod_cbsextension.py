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
        logging.warning("Could not find a solution, returning just anything \n", str(e))
        agent_job = []
        for a in agent_pos:
            agent_job.append(tuple(a))
        agent_job[0] = (0,)
        agent_idle = ()
        paths = get_paths(comp2condition(agent_pos, jobs, alloc_jobs, idle_goals, grid),
                          comp2state(tuple(agent_job), agent_idle, ()))

    pipe.send((agent_job,
               agent_idle,
               paths))


def get_routes_to_plan(routes):
    return list(filter(lambda r: not r.is_finished() and not r.is_idle_goal(), routes))


def get_idle_goals_from(routes):
    return list(filter(lambda r: r.is_idle_goal(), routes))


def get_car_from_assignments(assignments, i_to_find, cars):
    for i in range(len(assignments)):
        if len(assignments[i]) and i_to_find == assignments[i][0]:
            return cars[i]
    return False


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
        idle_goals, jobs, routes = self.split_routes(routes)
        self.update_plan(cars, routes)
        assert len(routes) > 0, "No routes to work with"
        self.lock.acquire()
        c = False
        if route_todo in jobs:
            i_j = jobs.index(route_todo)
            c = get_car_from_assignments(self.agent_job, i_j, cars)
            self.lock.release()
        elif route_todo in idle_goals:
            i_ig = idle_goals.index(route_todo)
            c = get_car_from_assignments(self.agent_idle, i_ig, cars)
            self.lock.release()
        else:
            self.lock.release()
        return c

    def split_routes(self, routes):
        jobs = get_routes_to_plan(routes)
        idle_goals = get_idle_goals_from(routes)
        routes = jobs + idle_goals
        return idle_goals, jobs, routes

    def new_job(self, cars, routes):
        self.update_plan(cars, routes)

    def update_plan(self, cars, routes):
        self.lock.acquire()
        idle_goal_routes, jobs, routes = self.split_routes(routes)
        if len(routes) < len(cars):  # to few jobs
            self.lock.release()
            return
        if list_hash(cars + routes) == self.plan_params_hash:  # nothing changed
            self.lock.release()
            return

        if self.process:
            while self.process.is_alive():
                logging.warning("waiting (is already planning)")
                time.sleep(.4)

        agent_pos = []
        for c in cars:
            t = c.to_tuple()
            assert not t[0].__class__ is np.ndarray
            assert t[0] == c.pose[0], "Problems with pose"
            agent_pos.append(t)

        jobs = []
        alloc_jobs = []
        idle_goals = []
        for i_route in range(len(jobs)):
            r = jobs[i_route]
            if not r.is_finished():  # all but the finished ones
                jobs.append(r.to_tuple())
                if r.is_on_route():
                    alloc_jobs.append((get_car_i(cars, r.car), i_route))
        for i_idle_goals in range(len(idle_goal_routes)):
            ig = idle_goal_routes[i_idle_goals]
            idle_goals.append(ig.to_tuple())

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
            cars[i_car].set_paths(self.paths[i_car])

        self.plan_params_hash = list_hash(cars + routes)  # how we have planned last time TODO: idle_goals
        self.lock.release()
