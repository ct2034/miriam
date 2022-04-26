import getpass
import logging
import multiprocessing
import os
import re
import signal
import subprocess
from contextlib import contextmanager
from datetime import datetime
from hashlib import sha256
from itertools import product
from typing import Callable, Dict, List, Tuple, Union

import networkx as nx
import numpy as np
import yaml

from definitions import POS

STATIC = "static"


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
    return 0 == run_command("bash -f /.dockerenv")


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
        except AssertionError as e:
            print("Benchmark stopped for AssertionError:")
            raise e
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

    assert vals.__class__ == list and vals[0].__class__ == list, \
        "Please provide list of lists per argument"

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

    # TODO: Set log level for git.cmd to info
    return Repo(os.getcwd(), search_parent_directories=True)


def get_git_sha():
    return get_git().head.commit.hexsha


def get_git_message():
    return get_git().head.commit.message


def mongodb_save(name, data):
    import datetime

    import pymongo

    if 0 != run_command('ping -c2 -W1 8.8.8.8'):
        logging.warning("No Internet connection -> not saving to mongodb")
        return

    if 0 != run_command('ping -c2 -W1 ds033607.mlab.com'):
        logging.warning("can not reach mlab -> not saving to mongodb")
        return

    key = get_git_sha()

    client = pymongo.MongoClient(
        "mongodb://testing:6R8IimXpg0TqVDwm" +
        "@ds033607.mlab.com:33607/miriam-results"
    )
    db = client["miriam-results"]
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

# The background is set with 40 plus the number of the color, and the
# foreground with 30

# These are the sequences need to get colored ouput
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"


def formatter_message(message, use_color=True):
    if use_color:
        message = message.replace(
            "$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
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
            levelname_color = COLOR_SEQ % (
                30 + COLORS[levelname]) + levelname + RESET_SEQ
            record.levelname = levelname_color
        return logging.Formatter.format(self, record)


# Custom logger class with multiple destinations
class ColoredLogger(logging.Logger):
    FORMAT = "[$BOLD%(name)-20s$RESET][%(levelname)-18s]  "\
        "%(message)s ($BOLD%(filename)s$RESET:%(lineno)d)"
    COLOR_FORMAT = formatter_message(FORMAT, True)

    def __init__(self, name):
        logging.Logger.__init__(self, name, logging.DEBUG)

        color_formatter = ColoredFormatter(self.COLOR_FORMAT)

        console = logging.StreamHandler()
        console.setFormatter(color_formatter)

        self.addHandler(console)
        return


def get_map_str(grid):
    if len(grid.shape) == 3:
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


def run_command(bashCommand, timeout=None, cwd=None) -> Tuple[str, str, int]:
    """executes given command as subprocess with optional timeout. Returns
    tuple of stdout, stderr and returncode."""
    process = subprocess.Popen(bashCommand.split(" "),
                               stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               cwd=cwd)
    out = process.communicate(timeout=timeout)
    return (out[0].decode(),
            out[1].decode(),
            process.returncode)


def hasher(args, kwargs={}):
    """Hash args that are hashable or np.ndarrays"""
    hashstr = ""
    for i, arg in enumerate(list(args)):
        hashstr += to_string(arg)
        hashstr += str(i)
    for keyw, arg in kwargs.items():
        hashstr += to_string(arg)
        hashstr += str(keyw)
    my_hash = sha256(hashstr.encode('utf-8')).hexdigest()
    return my_hash


def to_string(arg):
    if isinstance(arg, np.ndarray):
        return str(sha256(arg.tobytes()).hexdigest())
    elif isinstance(arg, nx.Graph):
        return str(nx.get_node_attributes(arg, POS))
    else:
        return str(arg)


RED_SEQ = "\033[;31m"
GREEN_SEQ = "\033[;32m"
YELLOW_SEQ = "\033[;33m"
BLUE_SEQ = "\033[;34m"


class ProgressBar(object):
    def __init__(self, name: str, total: int, step_perc: int = 0,
                 print_func: Callable = print):
        """Track progress of something.
        :param name: What is this tracking progress of?
        :param total: How many iterations are there in total?
        :param step_perc: Print info only after a increase by this percent?"""
        self.name = name
        self.total = total
        self.step_perc = step_perc / 100.
        self.last_print = 0.
        self.i = 0
        self.start_time = datetime.now()
        self.t_format = "%H:%M:%S"
        self.print_func = print_func
        self.print_func(BOLD_SEQ + "{} started.".format(
            self.name) + RESET_SEQ)

    def progress(self, i=None):
        """call this after every of `total` iterations. Pass argument
        0 < `i` <= `total` for setting absolute iteration."""
        if i is None:
            self.i += 1
        else:
            self.i = i + 1
        progress = self.i / self.total
        elapsed_time: datetime.timedelta = datetime.now() - self.start_time
        eta_time = (elapsed_time / progress) - elapsed_time
        if progress-self.last_print >= self.step_perc:
            self.last_print += self.step_perc
            self.print_func("{} progress: {}%\n > took: {}, eta: {}".format(
                self.name,
                int(round(progress * 100 - 1E-6)),
                str(elapsed_time),
                str(eta_time)))

    def end(self):
        """call this at the end to get total time."""
        elapsed_time = datetime.now() - self.start_time
        self.print_func(BOLD_SEQ + "{} finished. elapsed time: {}".format(
            self.name,
            str(elapsed_time))+RESET_SEQ)
        return elapsed_time


class StatCollector(object):
    def __init__(self, names: List[str]):
        self.stats: Dict[str,
                         Dict[str,
                              Union[List[float], Union[float, str]]]] = {}
        for name in names:
            self.stats[name] = {
                "t": [],
                "x": []
            }
        self.stats[STATIC] = {}

    def add(self, name: str, t: int, x: float):
        assert name in self.stats.keys()
        assert "t" in self.stats[name].keys()
        assert isinstance(self.stats[name]["t"], list)
        assert "x" in self.stats[name].keys()
        assert isinstance(self.stats[name]["x"], list)
        self.stats[name]["t"].append(t)  # type: ignore
        self.stats[name]["x"].append(x)  # type: ignore

    def add_static(self, name: str, x: Union[float, str]):
        self.stats[STATIC][name] = x

    def add_statics(self, statics: Dict[str, Union[float, str]]):
        self.stats[STATIC].update(statics)

    def get_stats(self, names: Union[List[str], str]):
        if isinstance(names, str):
            names = [names]
        return {name: (
            self.stats[name]["t"],
            self.stats[name]["x"]
        ) for name in names}

    def get_stats_wildcard(self, pattern: str):
        matched_names = []
        regex = re.compile(pattern)
        for n in self.stats.keys():
            if regex.match(n):
                matched_names.append(n)
        if STATIC in matched_names:
            matched_names.remove(STATIC)
        return self.get_stats(matched_names)

    def get_statics(self):
        return self.stats[STATIC]

    def to_yaml(self, filename: str):
        assert filename.endswith(".yaml")
        with open(filename, 'w') as f:
            yaml.dump(self.stats, f)

    def from_yaml(self, filename: str):
        assert filename.endswith(".yaml")
        with open(filename, 'r') as f:
            self.stats = yaml.load(f, Loader=yaml.SafeLoader)
