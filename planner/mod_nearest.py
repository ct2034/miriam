import random

from planner.astar.astar_grid48con import astar_grid4con
from planner.cbs_ext.plan import get_nearest
from planner.mod import Module
from planner.route import Route, Car


class Nearest(Module):
    def __init__(self, grid):
        self.grid = grid

    def which_car(self, cars: list, route_todo: Route, routes: list) -> Car:
        free_cars = []
        free_cars_poses = []
        for c in cars:
            if c.route:
                if not c.route.is_running() and not c.route.is_finished():
                    free_cars.append(c)
                    free_cars_poses.append(c.pose)
            else:  # no route assigned yet
                free_cars.append(c)
                free_cars_poses.append(c.pose)
        if len(free_cars) > 0:
            nearest = get_nearest(free_cars_poses, route_todo.start)
            i_car = free_cars_poses.index(nearest)
            car = free_cars[i_car]
            car.setPaths(self.plan(car, route_todo))
            return car
        else:
            return None  # No free car

    def new_job(self, cars, routes):
        pass

    def plan(self, car, route):
        return (astar_grid4con((car.pose[0], car.pose[1], 0),
                               tuple(route.start), self.grid),
                astar_grid4con((route.start[0], route.start[1], 0),
                               tuple(route.goal), self.grid))
