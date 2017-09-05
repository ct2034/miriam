import datetime
import time

from planner.cbs_ext.plan import generate_config, plan, alloc_threads
from planner.cbs_ext_test import get_data_colission
from planner.eval.eval import get_costs
from tools import benchmark


def tcbsnn_for_comparison(config):
    print("Testing with number_nearest=" + str(config['number_nearest']))
    print("Testing with all_collisions=" + str(config['all_collisions']))
    agent_idle, agent_job, agent_pos, grid, idle_goals, jobs = get_data_colission()
    start_time = datetime.datetime.now()
    if 'milp' in config:
        from planner.milp.milp import plan_milp
        res_agent_job, res_agent_idle, res_paths = plan_milp(agent_pos, jobs, grid, config)
    else:
        res_agent_job, res_agent_idle, res_paths = plan(agent_pos, jobs, [], idle_goals, grid, config)
    print("computation time:", (datetime.datetime.now() - start_time).total_seconds(), "s")
    print(res_agent_job)
    get_costs(res_paths, jobs, res_agent_job, True)


def test_tcbsnn_comparison():
    config_opt = generate_config()
    config_opt['filename_pathsave'] = 'colission.pkl'

    config_nn = config_opt.copy()
    config_nn['number_nearest'] = 2

    config_col = config_nn.copy()
    config_col['all_collisions'] = True

    config_milp = config_opt.copy()
    config_milp['milp'] = 1
    benchmark(tcbsnn_for_comparison, [config_col, config_nn, config_opt])


if __name__ == "__main__":
    test_tcbsnn_comparison()
