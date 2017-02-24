import numpy as np
from time import sleep


class Station(object):
    sim_thread = None

    def __init__(self, name="Stat0", pos=[0, 0]):
        self.name = name
        self.pos = pos
        self._is_blocked = False

        print("Station " + name + " wurde erstellt.")

    def process_Product(self, wait_time=1):
        self._is_blocked = True  # blocked till process is done
        assert wait_time < 360, "Wait time is higher than 360 seconds."
        if (wait_time > 0):
            sleep(wait_time)
        self._is_blocked = False  # next Package is allowed

    def is_free(self):
        return not self._is_blocked

    def get_pos(self):
        return self.pos

    def get_name(self):
        return self.name
