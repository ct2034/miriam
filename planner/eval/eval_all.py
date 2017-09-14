import os

from planner.cbs_ext.plan import generate_config, plan, alloc_threads, pre_calc_paths
from planner.cbs_ext_test import get_data_colission, get_data_random
from planner.eval.eval import get_costs
from tools import benchmark


def one_planner(config, size):
    print("Testing with number_nearest=" + str(config['number_nearest']))
    print("Testing with all_collisions=" + str(config['all_collisions']))
    agent_pos, grid, idle_goals, jobs = config['params']
    if 'milp' in config:
        print("milp")
        from planner.milp.milp import plan_milp
        res_agent_job, res_paths = plan_milp(agent_pos, jobs, grid, config)
    elif 'greedy' in config:
        print("greedy")
        from planner.greedy.greedy import plan_greedy
        res_agent_job, res_paths = plan_greedy(agent_pos, jobs, grid, config)
    else:
        res_agent_job, res_agent_idle, res_paths = plan(agent_pos, jobs, [], idle_goals, grid, config)
    print(res_agent_job)
    return get_costs(res_paths, jobs, res_agent_job, True)


def print_map(grid):
    grid = grid[:, :, 0]
    map_str = ""
    for y in range(grid.shape[1]):
        for x in range(grid.shape[0]):
            if grid[x, y] == 0:
                map_str += '.'
            else:
                map_str += '@'
        map_str += '\n'
    print(map_str)


def test_planner_comparison():
    fname = '/tmp/random.pkl'
    try:
        os.remove(fname)  # cleaning from maybe last run
    except FileNotFoundError:
        pass

    params = get_data_random(map_res=8,
                             map_fill_perc=20,
                             agent_n=3,
                             job_n=4,
                             idle_goals_n=0)

    agent_pos, grid, idle_goals, jobs = params
    pre_calc_paths(jobs, idle_goals, grid, fname)

    config_opt = generate_config()
    config_opt['params'] = params
    config_opt['filename_pathsave'] = fname

    config_milp = config_opt.copy()
    config_milp['milp'] = 1

    config_greedy = config_opt.copy()
    config_greedy['greedy'] = 1

    config_nn = config_opt.copy()
    config_nn['number_nearest'] = 2

    config_col = config_nn.copy()
    config_col['all_collisions'] = True

    print_map(params[1])

    configs = [config_milp, config_opt]
    sizes = [2, 4]
    ts, ress = benchmark(one_planner, [configs, sizes], samples=2, timeout=600)

    print(ts)
    print(ress)

    os.remove(fname)

if __name__ == "__main__":
    n = 10
    success = n
    for i in range(n):
        print("=" * 60)
        print("%d/%d" % (i, n))
        try:
            test_planner_comparison()
        except AssertionError:
            success -= 1
    print("success: %d\%" % (success / n * 100))
