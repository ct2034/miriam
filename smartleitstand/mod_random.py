import random

from smartleitstand.mod import Module
from smartleitstand.route import Route, Car


class Random(Module):
    def which_car(self, cars: list, route_todo: Route, routes_queue: list) -> Car:
        rand = random.Random()
        free_cars = []
        for c in cars:
            if not c.route:
                free_cars.append(c)
        if len(free_cars) > 0:
            return free_cars[rand.randint(0, len(free_cars) - 1)]
        else:
            return None  # No free car

    def new_job(self, route_todo: Route):
        pass
