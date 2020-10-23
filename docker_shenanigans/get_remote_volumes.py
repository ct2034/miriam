#!/usr/bin/env python3
import logging
import os
import stat
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

    # getting the files
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

    # cleaning up
    command = [
        "docker",
        "rm",
        "-f",
        container_id
    ]
    out = subprocess.check_output(command)
    logging.debug(out.decode("utf-8"))


def get_remote_volumes(remote_pcs: List[str], volume_name: str):
    for host in remote_pcs:
        their_path = "/tmp/data_" + host + "/"
        local_path = "./data_" + host + "/"
        ssh = SSHClient()
        ssh.load_system_host_keys()
        ssh.connect(name_to_hostname(host), username='ch')
        sftp = ssh.open_sftp()
        try:
            assert len(sftp.listdir(their_path)
                       ) == 0, "remote directory should be empty"
        except IOError:  # if the folder does not even exist
            pass

        command = [
            "docker",
            "run",
            "-t",  # pseudo-tty to keep container running
            "-d",
            "-v", volume_name + ":/data_from",
            "alpine"
        ]
        stdin, stdout, stderr = ssh.exec_command(" ".join(command))
        out = stdout.readlines()
        logging.debug("{}\n{}:\n{}".format(host, command, out))
        container_id = out[0].replace('\n', '')
        logging.debug(container_id)

        # getting the files on their filesystem
        command = [
            "docker",
            "cp",
            container_id+":data_from/.",
            their_path
        ]
        stdin, stdout, stderr = ssh.exec_command(" ".join(command))
        out = stdout.readlines()
        logging.debug("{}\n{}:\n{}".format(host, command, out))

        # copy files to our machine
        for fname in sftp.listdir(their_path):
            if stat.S_ISDIR(sftp.stat(their_path + fname).st_mode):
                logging.error("There is a path in the source folder: {}"
                              .format(their_path + fname))
            else:
                logging.info("Receiving {} from {}".format(fname, host))
                if not os.path.isfile(os.path.join(local_path, fname)):
                    sftp.get(their_path + fname,
                             os.path.join(local_path, fname))
                    sftp.remove(their_path + fname)
                else:
                    logging.info("File exists: {}".format(fname))

        # cleaning up
        command = [
            "docker",
            "rm",
            "-f",
            container_id
        ]
        stdin, stdout, stderr = ssh.exec_command(" ".join(command))
        out = stdout.readlines()
        logging.debug("{}\n{}:\n{}".format(host, command, out))

        # stats
        n_received = len(os.listdir(local_path))
        logging.info("Got {} file from {}".format(n_received, host))

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
    get_remote_volumes(remote_pcs, volume_name)
