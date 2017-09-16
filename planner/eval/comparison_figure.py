import re
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pymongo
import pickle

plt.style.use('bmh')

client = pymongo.MongoClient(
    "mongodb://testing:6R8IimXpg0TqVDwm" +
    "@ds033607.mlab.com:33607/smartleitstand-results"
)
db = client["smartleitstand-results"]
cursor = db.test_collection.find({
    "_id": "7d3bca9c03252588e1a6450793f3e0d47bcbae87"
})

n_results = 10
i = 0
ress = []

ress.append(cursor.next())
cursor.close()

if len(ress) > 1:
    for i, r in enumerate(ress):
        print("%d: %s - %s"%(i, r['time'], r['git_message'].strip()))
    i_s = input("Which data to load [0] >")
    if not i_s:
        i_s = 0
else:
    i_s = 0
data = ress[int(i_s)]

print("git message: " + str(data['git_message']).strip())
print("time: " + str(data['time']))
keys = data.keys()

durations = np.zeros([6, 3, 20])
results = np.zeros([6, 3, 20])

p = re.compile('[0-9]{1,2}')

for k in keys:
    if k.startswith("test_planner_comparison"):
        duration = np.array(data[k]['durations'])
        result = np.array(data[k]['results'])
        res = p.findall(k)
        i_sample = int(res[0])
        if "nocobra" in k:  # minlp, greedy, tcbs-nn2, tcbs
            durations[2:6,:,i_sample:i_sample+1] = np.array(duration)
            results[2:6,:,i_sample:i_sample+1] = np.array(result)
        else:               # greedy, cobra
            durations[0:2,:,i_sample:i_sample+1] = np.array(duration)
            results[0:2,:,i_sample:i_sample+1] = np.array(result)

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

labels = ["TCBS-NN2",   # 4
          "MINLP",      # 2
          "TPTS",       # 1
          "Greedy"]     # 0 or 3

solution_quality = np.zeros([4, 60])
# data -> solution quality
for i_d, i_sq in [(4, 0), (2, 1), (1, 2), (0, 3)]:
    solution_quality[i_sq,0:20] = (results[i_d,0,:] - results[5,0,:]) / 2  # mean per tasks
    solution_quality[i_sq,20:40] = (results[i_d,1,:] - results[5,1,:]) / 3
    solution_quality[i_sq,40:60] = (results[i_d,2,:] - results[5,2,:]) / 4
solution_quality = np.swapaxes(solution_quality, 0, 1)
solution_quality = np.nan_to_num(solution_quality)
solution_quality = np.abs(solution_quality)
f = plt.figure()
f.set_size_inches(5, 4)
ax = f.add_subplot(111)
ax.violinplot(solution_quality, showmeans=True)
ax.set_yscale('log')
ax.set_xticks([y+1 for y in range(len(labels))])
plt.setp(ax, xticks=[y+1 for y in range(len(labels))],
         xticklabels=labels)
plt.savefig("solutionquality.png")

fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(9, 4))
fig.tight_layout()
fig.subplots_adjust(left=.1, bottom=.1, right=.9, top=.9, wspace=.3, hspace=0)
titles = ["TCBS"] + labels
# data -> time
for i_d, i_t in [(5, 0), (4, 1), (2, 2), (1, 3), (0, 4)]:
    labels = ["2", "3", "4"]
    times = np.zeros([len(labels), 20])
    times[:,:] = durations[i_d,:,:]
    times = np.swapaxes(times, 0, 1)
    axes[i_t].violinplot(times)
    axes[i_t].set_title(titles[i_t])
    axes[i_t].set_ylim(0,np.max(times))
    axes[i_t].set_xticks([y+1 for y in range(len(labels))])
    axes[i_t].set_xlabel("Tasks")
    plt.setp(axes[i_t], xticks=[y+1 for y in range(len(labels))],
         xticklabels=labels)
axes[0].set_ylabel("Duration [s]")

fig.savefig('durations.png')