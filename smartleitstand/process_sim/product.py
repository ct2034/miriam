import random
from time import clock
import threading

from smartleitstand.process_sim import station
from numpy import linalg as LA
import numpy as np
from time import sleep



class Product(object):
    sim_thread = None
    last_id = 0

    def __init__(self, my_dad, stations=[station], flow=[], n_id=0):
        self.my_dad = my_dad
        self.n_id = n_id
        self.stations = stations
        self.flow_index = 0
        self.flow = flow
        self.__is_finished = False
        self.t_birth = clock()
        print("Product ", n_id, " was created")
        my_dad.register_products(self)
        # self.__start_lifecycle()
        self.thread = threading.Thread(self.__start_lifecycle())
        self.id = 0

    def start_lifecycle(self):
        # runs lifecycle as Thread
        self.thread.start()
        self.thread.join()

    def __start_lifecycle(self):
        # als Eigenen Thread starten
        self.t_life = clock()
        # do flow here
        for i in range(0, self.flow.__len__() - 1):
            if (i >= self.flow.__len__() - 2):
                # Produkt fährt zum Ausgang
                self.transport_to_next_station()
                self.__is_finished = True
                print("Product Nr. ", self.n_id, "is finished")
            else:
                self.transport_to_next_station()
                self.process_Product()

        self.t_end = clock()

    def process_Product(self):
        is_processed = False
        current_flow = self.flow[self.get_flow_index()]
        process_time = current_flow[1]  # Zeit des Prozesschrittes aus Liste
        source_Station = self.stations.__getitem__(self.flow[self.get_flow_index()][0])

        while (is_processed == False):
            if (source_Station.is_free()):
                source_Station.process_Product(process_time)
                is_processed = True
                print("Product ", self.n_id, "on Station", source_Station.get_name(), "was processed in ", process_time,
                      "secs.")
            else:
                # print("Station: ",source_Station.get_name()," is currently blocked.")
                pass
            sleep(1)

    def transport_to_next_station(self):
        self.id = Product.last_id
        Product.last_id += 1
        # Transportiere von Source nach Destination
        # flow = [stat, time]
        # Ermittle Station aus Flow List

        source_Station = self.stations.__getitem__(self.flow[self.get_flow_index()][0])
        dest_Station = self.stations.__getitem__(self.flow[self.get_next_flow_index()][0])
        # Ermittle Position der Station aus Stationsliste
        pos_source = source_Station.get_pos()
        pos_dest = dest_Station.get_pos()

        Product.sim_thread.new_job(
            np.array(pos_source),
            np.array(pos_dest),
            self.id
        )

        print("Product ", self.n_id, " drives from " + source_Station.get_name() + " to " + dest_Station.get_name())

        while not Product.sim_thread.is_finished(self.id):
            sleep(.2)

        # Fzg ist am Ziel, Flusspunkt erreicht
        self.increment_flow_index()

    def is_finished(self):
        return self.__is_finished

    def get_flow_index(self):
        # print("index",self.flow_index)
        return self.flow_index

    def get_next_flow_index(self):
        if (self.flow_index >= self.flow.__len__() - 1):
            return self.flow_index
        else:
            return self.flow_index + 1

    def increment_flow_index(self):
        if (self.flow_index < self.flow.__len__() - 1):
            self.flow_index += 1
