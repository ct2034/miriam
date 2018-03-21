import re
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pickle

plt.style.use('bmh')

# client = pymongo.MongoClient(
#     "mongodb://testing:6R8IimXpg0TqVDwm" +
#     "@ds033607.mlab.com:33607/smartleitstand-results"
# )
# db = client["smartleitstand-results"]
# cursor = db.test_collection.find({
#     "_id": "d1614210140a86187ab5438a12ed913942d462e7"
# })
#
# i = 0
# ress = []
#
# ress.append(cursor.next())
# cursor.close()
#
# if len(ress) > 1:
#     for i, r in enumerate(ress):
#         print("%d: %s - %s"%(i, r['time'], r['git_message'].strip()))
#     i_s = input("Which data to load [0] >")
#     if not i_s:
#         i_s = 0
# else:
#     i_s = 0
# data = ress[int(i_s)]
data = False
from planner.eval.comparison_data_raw import all_times, all_results

n_planners = 5
n_sizes = 3

if data:
    print("git message: " + str(data['git_message']).strip())
    print("time: " + str(data['time']))
    keys = data.keys()
    n_results = len(list(filter(lambda s: s.startswith("test_planner_comparison"), keys)))
    durations = np.zeros([n_planners, n_sizes, n_results])
    results = np.zeros([n_planners, n_sizes, n_results])
    p = re.compile('[0-9]{1,2}')
    for k in keys:
        if k.startswith("test_planner_comparison"):
            duration = np.array(data[k]['durations'])
            result = np.array(data[k]['results'])
            res = p.findall(k)
            i_sample = int(res[0])
            durations[:,:,i_sample:i_sample+1] = np.array(duration)
            results[:,:,i_sample:i_sample+1] = np.array(result)
else:
    n_results = len(all_results)
    durations = np.zeros([n_planners, n_sizes, n_results])
    results = np.zeros([n_planners, n_sizes, n_results])
    for i_r in range(n_results):
        duration = np.array(all_times[i_r])
        result = np.array(all_results[i_r])
        durations[:,:,i_r:i_r+1] = np.array(duration)
        results[:,:,i_r:i_r+1] = np.array(result)

try:
    with open("comparison_data.pkl", 'wb') as f:
        pickle.dump([durations, results], f, pickle.HIGHEST_PROTOCOL)
except Exception as e:
    print(e)

for i in product(*tuple(map(range, durations.shape))):
    if durations[i] > 599:
        durations[i] = 0
        results[i] = 0
    if results[i] == np.nan:
        results[i] = 0
    if results[i] == np.inf:
        results[i] = 0

labels = ["TCBS-NN2",   # 1
          "Greedy",     # 4
          "MINLP",      # 2
          "TPTS"]       # 3

solution_quality = np.zeros([4, 3*n_results])
# i_d            0           1          2            3             4
# from source: [config_opt, config_nn, config_milp, config_cobra, config_greedy]
# data -> solution quality
for i_d, i_sq in [(1, 0), (2, 2), (3, 3), (4, 1)]:
    solution_quality[i_sq,0:n_results] = (results[i_d,0,:] - results[0,0,:]) / 2  # mean per tasks
    solution_quality[i_sq,n_results:n_results*2] = (results[i_d,1,:] - results[0,1,:]) / 3
    solution_quality[i_sq,n_results*2:n_results*3] = (results[i_d,2,:] - results[0,2,:]) / 4
solution_quality = np.swapaxes(solution_quality, 0, 1)
solution_quality = np.nan_to_num(solution_quality)
f = plt.figure()
f.set_size_inches(5, 4)
ax = f.add_subplot(111)
violin_parts = ax.violinplot(solution_quality, showmeans=True, showextrema=False)
violin_parts['cmeans'].set_color("C1")

# ax.set_yscale('log')
maxx = int(np.ceil(np.max(solution_quality)))
ax.set_yticks(range(maxx))
ax.set_yticklabels(list(map(str, range(maxx))))
ax.set_xticks([y+1 for y in range(len(labels))])
plt.setp(ax, xticks=[y+1 for y in range(len(labels))],
         xticklabels=labels)
plt.savefig("solutionquality.png")

fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(9, 4))
fig.tight_layout()
fig.subplots_adjust(left=.1, bottom=.15, right=.9, top=.9, wspace=.1, hspace=0)
titles = ["TCBS"] + labels
# data -> time
for i_d, i_t in [(0, 0), (1, 1), (2, 3), (3, 4), (4, 2)]:
    labels = ["2", "3", "4"]
    times = np.zeros([len(labels), n_results])
    times[:,:] = durations[i_d, :, :]
    axes[i_t].set_ylim(np.min(durations), np.max(durations))
    times = np.swapaxes(times, 0, 1)
    violin_parts = axes[i_t].violinplot(times, showmeans=True, showextrema=False)
    violin_parts['cmeans'].set_color("C1")
    axes[i_t].set_title(titles[i_t])
    axes[i_t].set_xticks([y + 1 for y in range(len(labels))])
    axes[i_t].set_yscale('log')
    if i_t > 0:
        axes[i_t].set_yticklabels([])
    if i_t == 2:
        axes[i_t].set_xlabel("Tasks")
    plt.setp(axes[i_t], xticks=[y + 1 for y in range(len(labels))],
             xticklabels=labels)
axes[0].set_ylabel("Duration [s]")

fig.savefig('durations.png')