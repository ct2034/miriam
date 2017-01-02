from simulation import Car, Route


class Module:
    def which_car(self, cars: list, route_todo: Route, routes_queue: list) -> Car:
        raise NotImplementedError()
