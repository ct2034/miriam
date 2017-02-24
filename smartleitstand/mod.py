

class Module:
    def which_car(self, cars, route_todo, routes_queue, active_routes):
        raise NotImplementedError()

    def new_job(self, cars, routes_queue, active_routes):
        raise NotImplementedError()
