import getpass
import multiprocessing

import psutil


def get_system_parameters():
    user = getpass.getuser()
    print("\n-----\nUser:", user)
    cores = multiprocessing.cpu_count()
    print("CPUs:", cores)
    memory = psutil.virtual_memory().total
    print("Total Memory: %e" % memory)
    print("-----\n")
    return user, cores, memory