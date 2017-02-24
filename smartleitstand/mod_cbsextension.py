import datetime
import logging
import os
import time

import numpy as np

from smartleitstand.cbs_ext.plan import plan
from smartleitstand.mod import Module
from smartleitstand.route import Route, Car
from smartleitstand.simulation import listhash


class Cbsext(Module):
    def __init__(self, grid):
        # params
        self.agent_job = ()
        self.agent_idle = ()
        self.paths = ()
        self.grid = grid

        # data
        self.fname = "process_test.pkl"
        if os.path.exists(self.fname):
            os.remove(self.fname)
        self.planning = False
        self.plan_params_hash = False

    def which_car(self, cars: list, route_todo: Route, routes_queue: list, active_routes) -> Car:
        self.update_plan(cars, routes_queue, active_routes)
        assert len(routes_queue) > 0, "No routes to work with"
        for i_route in range(len(routes_queue)):
            if routes_queue[i_route] == route_todo:
                break
        for i_agent in range(len(cars)):
            if len(self.agent_job[i_agent]) > 0:
                if i_route == self.agent_job[i_agent][0]:
                    return cars[i_agent]
        return False

    def new_job(self, cars, routes_queue, active_routes):
        self.update_plan(cars, routes_queue, active_routes)

    def update_plan(self, cars, routes_queue, active_routes):
        if listhash(cars + routes_queue + active_routes) == self.plan_params_hash:
            return
        if self.planning:
            logging.warning("already planning")
            while (self.planning):
                time.sleep(.1)
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

        idle_goals = [((10, 10), (50, 20)), ((10, 11), (50, 20),),
                      ((10, 9), (50, 20),)]  # TODO: we have to learn these!

        planning_start = datetime.datetime.now()
        (self.agent_job,
         self.agent_idle,
         self.paths) = plan(agent_pos,
                            jobs,
                            alloc_jobs,
                            idle_goals,
                            self.grid,
                            plot=False,
                            filename=self.fname)
        logging.info("Planning took %.4fs" % (datetime.datetime.now() - planning_start).total_seconds())

        # allpaths = list(map(lambda x: (x[0], x[1]), sum(sum(self.paths, ()), [])))
        # for j in jobs:
        #     assert j[0] in allpaths, "Start not in paths"
        #     assert j[1] in allpaths, "Goal not in paths"
        # for i_a in range(len(agent_pos)):
        #     assert agent_pos[i_a][0] == cars[i_a].pose[0], "Pose problems"
        #     assert agent_pos[i_a][1] == cars[i_a].pose[1], "Pose problems"

        # save the paths in cars
        for i_car in range(len(cars)):
            cars[i_car].setPaths(self.paths[i_car])

        self.plan_params_hash = listhash(cars + routes_queue + active_routes)  # how we have planned last time
        self.planning = False

    def get_car_i(self, cars: list, car: Car):
        for i_agent in range(len(cars)):
            if car == cars[i_agent]:
                return i_agent
