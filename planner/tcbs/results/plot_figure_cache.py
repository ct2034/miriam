import pickle

import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.style.use("bmh")
    filename = "figure_cache.pkl"

    try:
        with open(filename, "rb") as f:
            (res, res2) = pickle.load(f)
    except FileNotFoundError:
        print("WARN: File %s does not exist", filename)

    plt.figure(figsize=[5, 3])
    plt.boxplot(res.T)
    plt.xlabel("Agents")
    plt.ylabel("Planning Time Saving [%]")
    ax = plt.gca()
    ax.set_ylim(bottom=18, top=102)

    plt.savefig("figure_cache.png", bbox_inches="tight")

    plt.figure(figsize=[5, 3])
    plt.boxplot(res2.T)
    plt.xlabel("Agents")
    plt.ylabel("Planning Time [s]")
    ax = plt.gca()
    ax.set_yscale("log")

    plt.savefig("figure_planningtime.png", bbox_inches="tight")
