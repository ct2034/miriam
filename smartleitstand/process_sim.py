import logging
import numpy as np
from random import random
import matplotlib.pyplot as plt
import time
from apscheduler.schedulers.background import BackgroundScheduler

from smartleitstand.mod_random import Random

from smartleitstand.simulation import SimpSim

FORMAT = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)
logging.getLogger("apscheduler").setLevel(logging.WARN)


def init(n):
    s = {}
    s['t_d'] = .1
    s['stations_loc'] = [[2, 2], [4, 2], [6, 2], [4, 4]]
    s['order_station'] = [0 for _ in range(n)]
    s['order_state'] = [0 for _ in range(n)]  # 0: transport, 1: assembly
    s['seq_stations'] = [1, 2, 1, 4, 3, 2, 3, 4]
    s['seq_durations'] = [.2, .2, .4, .5, .2, .4, .4, .3]
    return s


def finished(s):
    return all([s['order_station'] == len(s['seq_stations'])])


def iterate(s):
    logging.info('.')


if __name__ == "__main__":
    s = init(5)
    # plt.plot([x[0] for x in s['stations_loc']], [x[1] for x in s['stations_loc']], 'xr')
    # plt.show()

    mod = Random()
    # simThread = SimpSim(msb_select=False, mod=mod)

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        func=iterate,
        trigger='interval',
        args=(s,),
        id="process_sim_iterate",
        seconds=s['t_d'],
        max_instances=1
    )

    scheduler.start()

    while not finished(s):
        time.sleep(1)
