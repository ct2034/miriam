from planner.cbs_ext.plan import plan, generate_config
from planner.cbs_ext_test import get_data_random
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

params = {'legend.fontsize': 'small'}
plt.rcParams.update(params)

f = plt.figure()
ax1 = f.add_subplot(111)
f.set_size_inches(4, 4)
plot_inputs(ax1, agent_pos, [], jobs, grid, title='')

f.savefig('random.png')
