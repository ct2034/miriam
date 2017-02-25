from planner.route import Car


def __init__(mod):
    global module
    module = mod

def which_car(cars: list, route_todo, routes_queue: list) -> Car:
    # print("### SMARTLEITSTAND ###")
    # print("cars:" + "; ".join(map(str, cars)))
    # print("route_todo:" + route_todo.__str__())
    # print("routes_queue:" + str(routes_queue))
    # print("######################")
    return module.which_car(cars, route_todo, routes_queue)


def work_queue():
    module.work_queue()
