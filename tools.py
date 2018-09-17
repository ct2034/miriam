import getpass
import logging
import multiprocessing
import os
import signal
from contextlib import contextmanager
from datetime import datetime
from itertools import product

import numpy as np


class TimeoutException(Exception):
    pass

def get_system_parameters(disp=True):
    import psutil
    user = getpass.getuser()
    cores = multiprocessing.cpu_count()
    memory = psutil.virtual_memory().total
    if disp:
        print("\n-----\nUser:", user)
        print("CPUs:", cores)
        print("Total Memory: %e" % memory)
        print("-----\n")
    return user, cores, memory


def is_travis():
    u, _, _ = get_system_parameters(False)
    print("User: >" + str(u) + "<")
    return u == 'travis'


def is_cch():
    u, _, _ = get_system_parameters(False)
    print("User: >" + str(u) + "<")
    return u == 'cch'


def is_in_docker():
    return 0 == os.system("bash -f /.dockerenv")


def load_map(fname='tcbs/map.png'):
    import png
    r = png.Reader(filename=fname)

    x, y, iter, color = r.read()

    m = np.vstack(map(np.sign, iter))
    m = np.array(m, dtype=np.int8) - 1
    return m


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def benchmark(fun, vals, samples=10, disp=True, timeout=60):
    def benchmark_fun(args):
        res = False
        start = datetime.now()
        try:
            with time_limit(timeout):
                res = fun(*args)
        except TimeoutException:
            print("Timed out!")
        except Exception as e:
            print("#"*10)
            print("Benchmark stopped for EXCEPTION:")
            print(e)
            print("#"*10)
            # raise e
        t = (datetime.now() - start).total_seconds()
        if not res:
            res = None
        return t, res

    assert vals.__class__ == list and vals[0].__class__ == list, "Please provide list of lists per argument"

    lens = list(map(len, vals))
    ts = np.zeros(lens + [samples])
    ress = np.zeros(lens + [samples])

    for i in product(*tuple(map(range, lens))):
        args = tuple()
        ind = tuple()
        for i_v in range(len(i)):
            args += (vals[i_v][i[i_v]],)
            ind += (i[i_v],)
        for i_s in range(samples):
            ts[ind + (i_s,)], ress[ind + (i_s,)] = benchmark_fun(args)

    if disp:
        print("Inputs:")
        print(vals)
        print("Durations: [s]")
        print(ts)
        print("Results")
        print(ress)
    return ts, ress


def get_git():
    from git import Repo
    import os
    # TODO: Set log level for git.cmd to info
    return Repo(os.getcwd(), search_parent_directories=True)


def get_git_sha():
    return get_git().head.commit.hexsha


def get_git_message():
    return get_git().head.commit.message


def mongodb_save(name, data):
    import os
    import pymongo
    import datetime

    if 0 != os.system('ping -c2 -W1 8.8.8.8'):
        logging.warning("No Internet connection -> not saving to mongodb")
        return

    key = get_git_sha()

    client = pymongo.MongoClient(
        "mongodb://testing:6R8IimXpg0TqVDwm" +
        "@ds033607.mlab.com:33607/smartleitstand-results"
    )
    db = client["smartleitstand-results"]
    collection = db.test_collection
    cursor = collection.find({'_id': key})
    print("Saving to MongoDB")
    if cursor.count():  # exists
        print("exists")
        entry = cursor[0]
    else:  # create
        print("creating")
        entry = {'_id': key,
                 'time': datetime.datetime.now(),
                 'git_message': get_git_message()
                 }
        id = collection.insert_one(entry).inserted_id
        assert id == key, "Inserted ID does not match"
    entry.update(
        {name: data}
    )
    collection.find_one_and_replace(
        filter={'_id': key},
        replacement=entry
    )
    print("Saved in mongodb as _id:" + str(key))
    print("name:" + str(name))
    print("data:" + str(data))


BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

# The background is set with 40 plus the number of the color, and the foreground with 30

# These are the sequences need to get colored ouput
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"


def formatter_message(message, use_color=True):
    if use_color:
        message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
    else:
        message = message.replace("$RESET", "").replace("$BOLD", "")
    return message


COLORS = {
    'WARNING': YELLOW,
    'INFO': WHITE,
    'DEBUG': BLUE,
    'CRITICAL': YELLOW,
    'ERROR': RED
}


class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, use_color=True):
        logging.Formatter.__init__(self, msg)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in COLORS:
            levelname_color = COLOR_SEQ % (30 + COLORS[levelname]) + levelname + RESET_SEQ
            record.levelname = levelname_color
        return logging.Formatter.format(self, record)


# Custom logger class with multiple destinations
class ColoredLogger(logging.Logger):
    FORMAT = "[$BOLD%(name)-20s$RESET][%(levelname)-18s]  %(message)s ($BOLD%(filename)s$RESET:%(lineno)d)"
    COLOR_FORMAT = formatter_message(FORMAT, True)

    def __init__(self, name):
        logging.Logger.__init__(self, name, logging.DEBUG)

        color_formatter = ColoredFormatter(self.COLOR_FORMAT)

        console = logging.StreamHandler()
        console.setFormatter(color_formatter)

        self.addHandler(console)
        return


def get_map_str(grid):
    grid = grid[:, :, 0]
    map_str = ""
    for y in range(grid.shape[1]):
        for x in range(grid.shape[0]):
            if grid[x, y] == 0:
                map_str += '.'
            else:
                map_str += '@'
        map_str += '\n'
    return map_str