import random

from planner.astar.astar_grid48con import astar_grid4con
from planner.mod import Module
from planner.route import Route, Car


class Random(Module):
    def __init__(self, grid):
        self.grid = grid

    def which_car(self, cars: list, route_todo: Route, routes_queue: list, active_routes) -> Car:
        rand = random.Random()
        free_cars = []
        for c in cars:
            if not c.route:
                free_cars.append(c)
        if len(free_cars) > 0:
            car = free_cars[rand.randint(0, len(free_cars) - 1)]
            car.setPaths(self.plan(car, route_todo))
            return car
        else:
            return None  # No free car

    def new_job(self, cars, routes_queue, active_routes):
        pass

    def plan(self, car, route):
        return (astar_grid4con((car.pose[0], car.pose[1], 0),
                               tuple(route.start), self.grid),
                astar_grid4con((route.start[0], route.start[1], 0),
                               tuple(route.goal), self.grid))
