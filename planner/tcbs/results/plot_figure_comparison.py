import matplotlib.pyplot as plt

plt.style.use("bmh")

if __name__ == "__main__":
    # data from process_test.py - test_benchmark() @ tag:iros17_comparison
    dat = [128.016001, 81.024077]
    saved = dat[0] - dat[1]
    print("Saved %1.1f s, %.0f%%" % (saved, 100 * saved / dat[0]))

    ypos = [0, 1]

    plt.figure(figsize=[5, 3])
    plt.barh(ypos, dat, height=0.6)
    plt.xlabel("Production Time [s]")
    # plt.ylabel('Planning Time Saving [%]')
    ax = plt.gca()
    ax.set_yticks(ypos)
    ax.set_yticklabels(["Nearest", "CBSEXT"])
    ax.set_ylim(bottom=-0.6, top=1.6)

    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)

    # plt.show()
    plt.savefig("figure_comparison.png", bbox_inches="tight")
