import logging

import torch.multiprocessing as tmp

from multi_optim.configs import configs
from multi_optim.multi_optim_run import run_optimization

if __name__ == "__main__":
    # multiprocessing
    # tmp.set_sharing_strategy('file_system')
    tmp.set_start_method('spawn')
    # set_ulimit()  # fix `RuntimeError: received 0 items of ancdata`
    n_processes = min(tmp.cpu_count(), 16)
    pool = tmp.Pool(processes=n_processes)

    for prefix in [
        # "debug",
        "mapf_benchm_random-32-32-10"
    ]:
        if prefix == "debug":
            level = logging.DEBUG
        else:
            level = logging.INFO
        for set_fun in [
            logging.getLogger(__name__).setLevel,
            logging.getLogger(
                "planner.mapf_implementations.plan_cbs_roadmap"
            ).setLevel,
            logging.getLogger(
                "sim.decentralized.policy"
            ).setLevel
        ]:
            set_fun(level)

        # start the actual run
        run_optimization(
            **configs[prefix],
            pool_in=pool)

    pool.close()
    pool.terminate()
