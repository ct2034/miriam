from simulation import Route, Car
from smartleitstand.smartl_modules.module import Module


class Random(Module):
    def which_car(self, cars: list, route_todo: Route, routes_queue: list) -> Car:
        print("### SMARTLEITSTAND ###")
        print("cars:" + str(cars))
        print("route_todo:" + route_todo.__str__())
        print("routes_queue:" + str(routes_queue))
        print("######################")
        return cars[0]
