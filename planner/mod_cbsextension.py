import datetime
import logging
from multiprocessing import Pipe
from multiprocessing import Process

import numpy as np

from planner.cbs_ext.plan import plan
from planner.mod import Module
from planner.route import Route, Car
from planner.simulation import list_hash

FORMAT = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logging.getLogger("apscheduler").setLevel(logging.WARN)


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
        self.planning = False
        self.plan_params_hash = False
        self.process = False

    def which_car(self, cars: list, route_todo: Route, routes_queue: list, active_routes) -> Car:
        self.update_plan(cars, routes_queue, active_routes)
        assert len(routes_queue) > 0, "No routes to work with"
        i_route = routes_queue.index(route_todo)
        for i_agent in range(len(cars)):
            if len(self.agent_job[i_agent]) > 0:  # has assignment
                if i_route == self.agent_job[i_agent][0]:
                    return cars[i_agent]
        return False

    def new_job(self, cars, routes_queue, active_routes):
        self.update_plan(cars, routes_queue, active_routes)

    def update_plan(self, cars, routes_queue, active_routes):
        if list_hash(cars + routes_queue + active_routes) == self.plan_params_hash:
            return
        self.cars = cars
        self.queued_routes = routes_queue
        self.active_routes = active_routes

        if self.planning:
            try:
                self.process.terminate()
                logging.warning("terminated (was already planning)")
            except AttributeError as e:
                logging.warning("error terminating: %s" % str(e))
        self.planning = True
        agent_pos = []
        for c in cars:
            t = c.toTuple()
            assert not t[0].__class__ is np.ndarray
            assert t[0] == c.pose[0], "Problems with pose"
            agent_pos.append(t)

        jobs = []
        alloc_jobs = []
        for i_route in range(len(routes_queue)):
            r = routes_queue[i_route]
            jobs.append(r.toJobTuple())
            if r.on_route:
                alloc_jobs.append((self.get_car_i(cars, r.car), i_route))

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
                                     False,
                                     self.fname)
                               )
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

        self.plan_params_hash = list_hash(cars + routes_queue + active_routes)  # how we have planned last time
        self.planning = False

    def get_car_i(self, cars: list, car: Car):
        for i_agent in range(len(cars)):
            if car == cars[i_agent]:
                return i_agent


def plan_process(pipe, agent_pos, jobs, alloc_jobs, idle_goals, grid, plot, fname):
    (agent_job,
     agent_idle,
     paths) = plan(agent_pos,
                   jobs,
                   alloc_jobs,
                   idle_goals,
                   grid,
                   plot,
                   fname)
    pipe.send((agent_job,
               agent_idle,
               paths))
