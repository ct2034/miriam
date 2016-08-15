from numpy import *
import time
import threading
import random


def loop():
    SimpleSimulation.running = True
    while SimpleSimulation.running:
        try:
            print(".")
            for j in SimpleSimulation.routes:
                if not j.finished:
                    j.new_step(SimpleSimulation.driveSpeed)
            time.sleep(SimpleSimulation.simTime)
        except Exception as e:
            print(str(e))


class SimpleSimulation(object):
    """simulation of multiple AGVs"""

    routes = []
    driveSpeed = 5
    simTime = 1
    running = False

    def __init__(self):
        print("init Simulation")

        self.area = zeros([1])
        self.number_agvs = 1
        self.cars = []

    def start(self, width, height, number_agvs):
        self.area = zeros([width, height])
        self.number_agvs = number_agvs
        for i in range(self.number_agvs):
            self.cars.append(Car(self))
        t = threading.Thread(target=loop)
        t.start()

    def stop(self):
        SimpleSimulation.running = False

    def new_job(self, a, b):
        car = False
        for c in self.cars:
            if not c.route:
                car = c
        if not car:
            print("No free Car, waiting")
            time.sleep(SimpleSimulation.simTime)
            self.new_job(a, b)
        else:
            new_job = Route(a, b, car)
            SimpleSimulation.routes.append(new_job)


def get_distance(a, b):
    assert a.size is 2, "A point needs to have two coordinates"
    assert b.size is 2, "B point needs to have two coordinates"
    return linalg.norm(a - b)


class Route(object):
    """a route to be simulated"""

    nextId = 0

    def __init__(self, start, goal, car):
        assert start.__class__ is ndarray, 'Start needs to be a numpy.ndarray'
        self.start = start
        assert goal.__class__ is ndarray, 'Goal needs to be a numpy.ndarray'
        self.goal = goal

        assert car.route == False
        car.route = self
        self.car = car
        self.car.pose = start

        self.distance = get_distance(start, goal)
        self.remaining = self.distance
        self.finished = False

        self.id = Route.nextId
        Route.nextId += 1

        print("Created route with id", str(self.id), "distance:", self.distance)

    def new_step(self, speed):
        self.remaining = self.remaining - speed
        if self.remaining <= 0:
            self.car.pose = self.goal
            self.car.route = False
            self.remaining = 0
            self.finished = True
            print("Route", self.id, "reached Goal", self.goal)


class Car(object):
    """an AGV to be simulated"""

    nextId = 0

    def __init__(self, s):
        assert s.__class__ is SimpleSimulation, "Pass the simulation object to the new car"
        self.pose = random.random() * array(s.area.shape)

        self.route = False

        self.id = Car.nextId
        Car.nextId += 1

        print("New car:", str(self.id), "at", str(self.pose))

if __name__ == '__main__':
    s = SimpleSimulation()

    try:
        mwc = MsbWsClient('ws://atm.virtualfortknox.de/msb', callback)

        f = Function("num",
                     # there can not be capital letters in name
                     DataFormat("start", "Integer").toCol(),
                     "number",
                     "A number")
        s = Application(
            "testclient_uuid2",
            "testclient_name2",
            "testclient_desc2",
            [],
            [f.toOrderedDict()],
            "token2")

        mwc.register(s)

        # wait a bit
        time.sleep(2)

        # mwc.emitEvent(s, e.toOrderedDict(), 3, 2)

        while True:
            time.sleep(2)

    except KeyboardInterrupt:
        print("EXIT ..")
        mwc.disconnect()

    s.start(100, 100, 2)

    print("start sim")
    time.sleep(1)

    s.new_job(array([10, 10]), array([20, 20]))
    s.new_job(array([10, 10]), array([30, 50]))
    s.new_job(array([70, 10]), array([20, 80]))

    time.sleep(15)

    print("sim finished")
    s.stop()