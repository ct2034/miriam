from smartleitstand.route import Route, Car


class Module:
    def which_car(self, cars: list, route_todo: Route, routes_queue: list) -> Car:
        raise NotImplementedError()

    def work_queue(self):
        raise NotImplementedError()

    def new_job(self, route_todo: Route):
        raise NotImplementedError()
