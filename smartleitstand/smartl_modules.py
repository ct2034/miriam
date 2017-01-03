import random

from smartleitstand.route import Route, Car

class Module:
    def which_car(self, cars: list, route_todo: Route, routes_queue: list) -> Car:
        raise NotImplementedError()


class Random(Module):
    def which_car(self, cars: list, route_todo: Route, routes_queue: list) -> Car:
        rand = random.Random()
        return cars[rand.randint(0, len(cars) - 1)]
