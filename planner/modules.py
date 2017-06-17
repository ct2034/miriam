from simple_simulation.route import Car


def __init__(mod):
    global module
    module = mod


def which_car(cars: list, route_todo, routes_queue: list, idle_goals: list) -> Car:
    # print("### SMARTLEITSTAND ###")
    # print("cars:" + "; ".join(map(str, cars)))
    # print("route_todo:" + route_todo.__str__())
    # print("routes_queue:" + str(routes_queue))
    # print("######################")
    return module.which_car(cars, route_todo, routes_queue, idle_goals)


def work_queue():
    module.work_routes()
