import hashlib

from planner.cbs_ext.plan import plan, generate_config
from planner.cbs_ext_test import get_data_random
from planner.eval.eval_comparison_test import get_map_str
from planner.milp.milp import plan_milp
from tools import load_map
from planner.eval.display import *
from mpl_toolkits.mplot3d import Axes3D

params = get_data_random(20,
                         map_res=8,
                         map_fill_perc=20,
                         agent_n=4,
                         job_n=4,
                         idle_goals_n=0)
agent_pos, grid, idle_goals, jobs = params

mapstr = get_map_str(grid)
maphash = str(hashlib.md5(mapstr.encode('utf-8')).hexdigest())[:8]
fname = "planner/eval/cache/" + str(maphash) + '.pkl'  # unique filename based on map
config = generate_config()
config['filename_pathsave'] = fname

tcbs_agent_job, tcbs_agent_idle, tcbs_paths = plan(
    agent_pos, jobs, [], [], grid, config, plot=False
)

minlp_agent_job, minlp_paths = plan_milp(agent_pos, jobs, grid, config)

f = plt.figure()
f.set_size_inches(8, 4.5)
ani = animate_results(
    f, [], tcbs_paths, tcbs_agent_job, agent_pos, grid, [], jobs, 'TCBS'
)
ani.save("random_tcbs.mp4", writer="ffmpeg", fps=10)

f = plt.figure()
f.set_size_inches(8, 4.5)
ani = animate_results(
    f, [], minlp_paths, minlp_agent_job, agent_pos, grid, [], jobs, 'MINLP'
)
ani.save("random_minlp.mp4", writer="ffmpeg", fps=10)
