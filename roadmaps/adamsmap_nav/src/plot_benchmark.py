#!/usr/bin/env python
import matplotlib.pyplot as plt
import os
import pickle
import sys

ODRM_STR = "ODRM-based"

if __name__ == "__main__":
    plt.style.use('bmh')
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["savefig.dpi"] = 500

    folder = sys.argv[1]  # type: str
    assert folder.endswith("/"), "Please call with folder"
    data = []
    ns_agents = []
    planners = []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".pkl"):
            # exmpl:
            #   adamsmap_global_planner-AdamsmapGlobalPlanner_x_100_4096.pkl_n_agents-8.pkl
            #   navfn-NavfnROS_x_100_4096.pkl_n_agents-8.pkl
            with open(folder + fname, "r") as pkl_file:
                d = pickle.load(pkl_file)
                data.append(d)
                ns_agents.append(fname.split(".")[1].split("-")[-1])
                planner = fname.split("-")[0]
                if planner == "adamsmap_global_planner":
                    planners.append(ODRM_STR)
                else:
                    planners.append("Gridmap-based")

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    violin_parts = ax.violinplot(data, showmeans=True, showextrema=False)
    idx = range(len(data))
    violin_parts['cmeans'].set_color("C1")
    for i in idx:
        if planners[i] == ODRM_STR:
            violin_parts['bodies'][i].set_facecolor("C3")
        else:
            violin_parts['bodies'][i].set_facecolor("C2")
    print(violin_parts['cmeans'])
    ax.set_xticks([x + 1 for x in idx])


    def get_label(x):
        if x == 1 or x == 4:
            return planners[x] + "\n" + ns_agents[x] + " Agents"
        else:
            return "\n" + ns_agents[x] + " Agents"


    ax.set_xticklabels(
        map(get_label, idx)
    )
    ax.set_ylabel("Max Travel Duration [s]")
    # plt.title("Decentralized Planner Travel Duration")
    plt.tight_layout()
    fig.savefig(folder + 'decentralized_travel_times.png')
