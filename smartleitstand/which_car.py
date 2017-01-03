from smartleitstand.smartl_modules import *


def which_car(cars: list, route_todo, routes_queue: list) -> Car:
    module = Random()
    return module.which_car(cars, route_todo, routes_queue)
