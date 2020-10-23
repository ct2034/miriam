#!/usr/bin/env python3
import logging
import os
import subprocess
from typing import List

import docker

logging.basicConfig()
logging.root.setLevel(logging.NOTSET)

logging.getLogger('docker').setLevel(logging.INFO)
logging.getLogger('urllib3').setLevel(logging.INFO)


def name_to_hostname(name: str):
    return name + '.local'


def check_for_reachability(remote_pcs: List[str]):
    FNULL = open(os.devnull, 'w')
    for host in remote_pcs:
        command = ['ping', '-c', '1', name_to_hostname(host)]
        if subprocess.call(command, stdout=FNULL, stderr=subprocess.STDOUT) != 0:
            logging.warn("Can not reach {}".format(host))
            return False
    logging.info("All hosts reachable")
    return True


def get_local_volume(volume_name: str):
    client = docker.from_env()
    volume = client.volumes.get(volume_name)
    path = volume.attrs['Mountpoint']
    logging.info("Local volume path: {}".format(path))


if __name__ == "__main__":
    # variables we need
    remote_pcs = [
        "metaverse",
        "marble"
    ]
    volume_name = "policylearn_data_out"

    if not check_for_reachability(remote_pcs):
        exit
    get_local_volume(volume_name)
