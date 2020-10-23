#!/usr/bin/env python3
import logging
import os
import subprocess
import sys
from typing import List

import docker
from paramiko import SSHClient

logging.basicConfig()
logging.root.setLevel(logging.NOTSET)

logging.getLogger('docker').setLevel(logging.INFO)
logging.getLogger('urllib3').setLevel(logging.INFO)
logging.getLogger('paramiko').setLevel(logging.INFO)


def name_to_hostname(name: str):
    return name + '.local'


def check_for_reachability(remote_pcs: List[str]):
    FNULL = open(os.devnull, 'w')
    for host in remote_pcs:
        command = ['ping', '-c', '1', name_to_hostname(host)]
        if subprocess.call(command, stdout=FNULL, stderr=subprocess.STDOUT) != 0:
            logging.warn("Can not reach {}".format(host))
            return False
    logging.info("All hosts reachable.")
    return True


def make_folders_check_empty(remote_pcs: List[str]):
    all_are_empty = True
    for host in remote_pcs + ["localhost"]:
        path = "./data_" + host
        if not os.path.exists(path):
            os.makedirs(path)
        else:  # exists
            if len(os.listdir(path)):  # not empty
                all_are_empty = False
                logging.error("{} is not empty!".format(path))
    logging.info("All directories exist and are empty.")
    return all_are_empty


def get_local_volume_data(volume_name: str):
    local_path = "./data_localhost/"
    command = [
        "docker",
        "run",
        "-t",  # pseudo-tty to keep container running
        "-d",
        "-v", volume_name + ":/data_from",
        "alpine"
    ]
    out = subprocess.check_output(command)
    container_id = out.decode("utf-8").replace('\n', '')
    logging.debug(container_id)

    command = [
        "docker",
        "cp",
        container_id+":data_from/.",
        local_path
    ]
    out = subprocess.check_output(command)
    logging.debug(out.decode("utf-8"))

    n_received = len(os.listdir(local_path))
    logging.info("Got {} file from local volume".format(n_received))


def get_remote_volumes(remote_pcs: List[str], volume_name: str):
    for host in remote_pcs:
        ssh = SSHClient()
        ssh.load_system_host_keys()
        ssh.connect(name_to_hostname(host), username='ch')
        cmd = 'pwd'
        stdin, stdout, stderr = ssh.exec_command(cmd)
        logging.info("{}\n{}:\n{}".format(host, cmd, stdout.readlines()))
        ssh.close()


if __name__ == "__main__":
    # variables we need
    remote_pcs = [
        "metaverse",
        "marble"
    ]
    volume_name = "policylearn_data_out"

    if not check_for_reachability(remote_pcs):
        sys.exit(1)
    if not make_folders_check_empty(remote_pcs):
        sys.exit(2)
    get_local_volume_data(volume_name)
    # get_remote_volumes(remote_pcs, volume_name)
