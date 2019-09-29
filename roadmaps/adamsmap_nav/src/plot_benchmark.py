#!/usr/bin/env python
import matplotlib.pyplot as plt
import os
import pickle
import sys

ODRM_STR = "ODRM-based"

if __name__ == "__main__":
    plt.style.use('bmh')
    # plt.rcParams["font.family"] = "serif"
    plt.rcParams["savefig.dpi"] = 500

    for rand in [True, False]:
        folder = sys.argv[1]  # type: str
        assert folder.endswith("/"), "Please call with folder"
        data = []
        ns_agents = []
        planners = []
        colors = []
        maps = []
        for fname in sorted(os.listdir(folder)):

            if fname.endswith(".pkl") and (("random" in fname) if rand else (not "random" in fname)):
                # exmpl:
                #   adamsmap_global_planner-AdamsmapGlobalPlanner_x_100_4096.pkl_n_agents-8.pkl
                #   navfn-NavfnROS_x_100_4096.pkl_n_agents-8.pkl
                with open(folder + fname, "r") as pkl_file:
                    d = pickle.load(pkl_file)
                    data.append(d)
                    if rand:
                        ns_agents.append(fname.split(".")[1].split("-")[-1].split("_")[0])
                    else:
                        ns_agents.append(fname.split(".")[1].split("-")[-1])
                    planner = fname.split("-")[0]
                    mapscen = fname.split("-")[1].split("_")[1]
                    maps.append(mapscen)
                    if planner == "adamsmap_global_planner":
                        planners.append(ODRM_STR)
                    else:
                        planners.append(planner)
                    if rand:
                        if mapscen == "x":
                            colors.append("C2")
                        elif mapscen == "o":
                            colors.append("C3")
                        else:
                            colors.append("C4")
                    else:
                        if planner == "adamsmap_global_planner":
                            colors.append("C3")
                        else:
                            colors.append("C2")

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
        violin_parts = ax.violinplot(data, showmeans=True, showextrema=False)
        idx = range(len(data))
        violin_parts['cmeans'].set_color("C1")
        for i in idx:
            violin_parts['bodies'][i].set_facecolor(colors[i])
        print(violin_parts['cmeans'])
        ax.set_xticks([x + 1 for x in idx])


        def get_label(x):
            if rand:
                if x == 1 or x == 4:
                    return planners[x] + "\n" + "Scenario " + str(maps[x]).upper()
                else:
                    return "\n" + "Scenario " + str(maps[x]).upper()
            else:
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
        if rand:
            fig.savefig(folder + 'decentralized_travel_times_random.png')
        else:
            fig.savefig(folder + 'decentralized_travel_times.png')
