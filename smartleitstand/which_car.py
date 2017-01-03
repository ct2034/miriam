from smartleitstand.smartl_modules import *


def which_car(cars: list, route_todo, routes_queue: list) -> Car:
    print("### SMARTLEITSTAND ###")
    print("cars:" + "; ".join(map(str, cars)))
    print("route_todo:" + route_todo.__str__())
    print("routes_queue:" + str(routes_queue))
    print("######################")
    module = Random()
    return module.which_car(cars, route_todo, routes_queue)
