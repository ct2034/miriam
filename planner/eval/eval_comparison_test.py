import datetime
import hashlib
import json
import math
import random
from threading import Thread, Event

from planner.cbs_ext_test import get_data_random
from planner.eval.eval_scenarios import get_costs
from planner.tcbs.plan import generate_config, plan, pre_calc_paths
from tools import benchmark, mongodb_save, is_cch, get_map_str

class MyThread(Thread):
    def __init__(self, event):
        Thread.__init__(self)
        self.stopped = event

    def run(self):
        while not self.stopped.wait(300):
            print("alive ...")


def one_planner(config, size):
    print("size=" + str(size))
    print("Testing with number_nearest=" + str(config['number_nearest']))
    print("Testing with all_collisions=" + str(config['all_collisions']))
    agent_pos, grid, idle_goals, jobs = config['params']
    agent_pos = agent_pos[0:size]
    jobs = jobs[0:size]
    if 'milp' in config:
        print("milp")
        from planner.milp.milp import plan_milp
        res_agent_job, res_paths = plan_milp(agent_pos, jobs, grid, config)
    elif 'cobra' in config:
        print("cobra")
        from planner.cobra.funwithsnakes import plan_cobra
        res_agent_job, res_paths = plan_cobra(agent_pos, jobs, grid, config)
    elif 'greedy' in config:
        print("greedy")
        from planner.greedy.greedy import plan_greedy
        res_agent_job, res_paths = plan_greedy(agent_pos, jobs, grid, config)
    else:
        res_agent_job, res_agent_idle, res_paths = plan(
            agent_pos, jobs, [], idle_goals, grid, config, plot=False
        )
    print(res_agent_job)
    return get_costs(res_paths, jobs, res_agent_job, True)


def planner_comparison(seed):
    params = get_data_random(seed+1,
                             map_res=8,
                             map_fill_perc=30,
                             agent_n=5,
                             job_n=5,
                             idle_goals_n=0)
    agent_pos, grid, idle_goals, jobs = params
    mapstr= get_map_str(grid)
    print(mapstr)
    maphash = str(hashlib.md5(mapstr.encode('utf-8')).hexdigest())[:8]
    print(maphash)

    fname = "planner/eval/cache/" + str(maphash) + '.pkl'  # unique filename based on map
    pre_calc_paths(jobs, idle_goals, grid, fname)

    config_opt = generate_config()
    config_opt['params'] = params
    config_opt['filename_pathsave'] = fname

    config_milp = config_opt.copy()
    config_milp['milp'] = 1

    config_cobra = config_opt.copy()
    config_cobra['cobra'] = 1

    config_greedy = config_opt.copy()
    config_greedy['greedy'] = 1

    config_nn = config_opt.copy()
    config_nn['number_nearest'] = 2

    config_col = config_nn.copy()
    config_col['all_collisions'] = True

    if is_cch():
        print("Configs: [config_opt, config_nn, config_milp]")
        configs = [config_opt, config_nn, config_milp] #, config_cobra, config_greedy, config_col]
        sizes = [1, 2]
    else:
        print("Configs: [config_opt, config_nn, config_milp, config_cobra, config_greedy]")
        configs = [config_opt, config_nn, config_milp, config_cobra, config_greedy]
        sizes = [2, 3, 4]
    ts, ress = benchmark(one_planner, [configs, sizes], samples=1, timeout=1000)

    return ts, ress


def test_planner_comparison():
    stopFlag = Event()
    thread = MyThread(stopFlag)
    thread.start()

    if is_cch():
        n_samples = 4
    else:
        n_samples = 30

    all_results = []
    all_times = []

    random.seed(datetime.datetime.now())

    for i_s in range(n_samples):
        print("######\nSample Nr: " + str(i_s) + ".\n######")
        if is_cch():
            seed = i_s
        else:
            seed = random.randint(0, 1000)
        ts, ress = planner_comparison(seed)
        if not is_cch():
            mongodb_save(
                'test_planner_comparison_' + str(i_s),
                {
                    'durations': ts.tolist(),
                    'results': ress.tolist()
                }
            )
        for i_size in range(ress.shape[1]):
            if bool(ress[0][i_size][0]) & (not math.isnan(ress[0][i_size][0])):
                for i_comp in range(1, len(ress)):
                    if bool(ress[i_comp][i_size][0]) & (not math.isnan(ress[i_comp][i_size][0])):
                        assert ress[0][i_size][0] <= ress[i_comp][i_size][0], "Optimal Planner not optimal? " + str(i_comp)
                all_results.append(ress.tolist())
                all_times.append(ts.tolist())

    print("all_results:", json.dumps(all_results))
    print("all_times:", json.dumps(all_times))

    # this will stop the timer
    stopFlag.set()


if __name__ == "__main__":
    test_planner_comparison()
