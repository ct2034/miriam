import math
import numpy as np
import pyafai
from pyafai import shapes, objects
from pyflann import FLANN
import random
from sympy.geometry.polygon import Point, Ray, Line, pi


class NnWorld(pyafai.World2D):
    def __init__(self, size):
        super(NnWorld, self).__init__(size, size)
        self.size = size
        self.flann = FLANN()

    def update(self, delta):
        super(NnWorld, self).update(delta)
        poses = []
        for a in self._agents:
            poses.append([a.body.x, a.body.y])
        self.flann.build_index(np.array(poses))

    def get_nearest_agents(self, nn, pos):
        result, dists = self.flann.nn_index(np.array([pos]), nn)
        return [self._agents[i] for i in result[0]], dists[0]


class Robi(pyafai.Agent):
    def __init__(self, prio, size):
        super(Robi, self).__init__()
        random.seed(prio)
        self.prio = prio

        obj = objects.SimplePhysicsObject(
            random.random()*size,
            random.random()*size,
            random.random()*1000 % 360
        )
        shape = shapes.Circle(10)
        obj.add_shape(shape)
        self.would_be_velocity = 20
        obj.velocity = self.would_be_velocity
        self.body = obj
        self.size = size
        self.collisions = 0

    def stop_me(self):
        self.body.velocity = 0
        self.body._shapes[0].color = ('c3B', (255, 0, 0))

    def move_me(self):
        self.body.velocity = self.would_be_velocity
        self.body._shapes[0].color = ('c3B', (255, 255, 255))

    def pos(self):
        return (self.body.x, self.body.y)

    def update(self, delta):
        self.body.x = self.body.x + \
            math.cos(self.body.angle) * self.body.velocity * delta
        self.body.y = self.body.y + \
            math.sin(self.body.angle) * self.body.velocity * delta
        if (self.body.x >= self.world.size or
            self.body.x <= 0 or
            self.body.y >= self.world.size or
                self.body.y <= 0):
            self.collisions += 1
            self.body.angle = random.random()*1000 % 360
        a = self.in_front_of_me()
        if a:
            if a.prio > self.prio:
                self.stop_me()
        else:
            self.move_me()

    def in_front_of_me(self):
        agents, dists = self.world.get_nearest_agents(
            4, [self.body.x, self.body.y])
        for i_a, a in enumerate(agents):
            if a != self and self.would_be_velocity * 5 >= dists[i_a] > 0:
                p_self = Point(self.pos())
                p_other = Point(a.pos())
                angle_other = math.atan2(
                    p_other[1] - p_self[1], p_other[0] - p_self[0])
                if(abs(self.body.angle - angle_other)) < pi / 2:
                    return a
        return False


if __name__ == '__main__':
    size = 1000
    world = NnWorld(size)
    display = pyafai.Display(world)

    for i in range(30):
        robi = Robi(prio=i, size=size)
        world.add_agent(robi)
    # world.update(1)

    pyafai.run()
    world.pause_toggle()
