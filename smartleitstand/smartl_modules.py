from smartleitstand.route import Route, Car

class Module:
    def which_car(self, cars: list, route_todo: Route, routes_queue: list) -> Car:
        raise NotImplementedError()


class Random(Module):
    def which_car(self, cars: list, route_todo: Route, routes_queue: list) -> Car:
        print("### SMARTLEITSTAND ###")
        print("cars:" + "; ".join(map(str, cars)))
        print("route_todo:" + route_todo.__str__())
        print("routes_queue:" + str(routes_queue))
        print("######################")
        return cars[0]
