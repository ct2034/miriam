import os
import random
import re
import subprocess
import sys

# GPU picking
# http://stackoverflow.com/a/41638727/419116
# Nvidia-smi GPU memory parsing.
# Tested on nvidia-smi 370.23

# source: https://github.com/adler-j/adler/blob/master/adler/util/gpu.py


def run_command(cmd):
    """Run command, return output as string."""

    output = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return output.decode("ascii")


def list_available_gpus():
    """Returns list of available GPU ids."""

    output = run_command("nvidia-smi -L")
    # lines of the form GPU 0: TITAN X
    gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
    result = []
    for line in output.strip().split("\n"):
        m = gpu_regex.match(line)
        assert m, "Couldnt parse "+line
        result.append(int(m.group("gpu_id")))
    return result


def gpu_memory_map():
    """Returns map of GPU id to memory allocated on that GPU."""

    output = run_command("nvidia-smi")
    gpu_output = output[output.find("GPU Memory"):]
    # lines of the form
    # |    0      8734    C   python    11705MiB |
    memory_regex = re.compile(
        r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB")
    rows = gpu_output.split("\n")
    result = {gpu_id: 0 for gpu_id in list_available_gpus()}
    for row in gpu_output.split("\n"):
        m = memory_regex.search(row)
        if not m:
            continue
        gpu_id = int(m.group("gpu_id"))
        gpu_memory = int(m.group("gpu_memory"))
        result[gpu_id] += gpu_memory
    return result


def pick_gpu_lowest_memory():
    """Returns GPU with the least allocated memory"""

    memory_gpu_map = [(memory, gpu_id)
                      for (gpu_id, memory) in gpu_memory_map().items()]
    best_memory, best_gpu = sorted(memory_gpu_map)[0]
    best_gpus = []
    for mem, gpu in memory_gpu_map:
        if mem <= best_memory+2048:  # allow 10% difference
            best_gpus.append(gpu)
    return random.choice(best_gpus)


def pick_gpu_low_memory():
    """Returns GPU with little allocated memory"""
    N = 4  # select randomly between N lowest gpus
    memory_gpu_map = [(memory, gpu_id)
                      for (gpu_id, memory) in gpu_memory_map().items()]
    n_best_gpus = sorted(memory_gpu_map)[0:N+1]
    _, good_gpu = random.choice(n_best_gpus)
    return good_gpu


def setup_one_gpu():
    assert 'tensorflow' not in sys.modules, \
        "GPU setup must happen before importing TensorFlow"
    gpu_id = pick_gpu_lowest_memory()
    print("Picking GPU "+str(gpu_id))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def setup_no_gpu():
    if 'tensorflow' in sys.modules:
        print("Warning, GPU setup must happen before importing TensorFlow")
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
