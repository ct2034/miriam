from simulation import Route, Car
from smartleitstand.smartl_modules.random import Random


def which_car(cars: list, route_todo: Route, routes_queue: list) -> Car:
    module = Random()
    module.which_car(cars, route_todo, routes_queue)
