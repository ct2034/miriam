#!/usr/bin/env python3
import os
import sys

from tools import StatCollector

from multi_optim_run import write_stats_png

if __name__ == "__main__":
    explanation = "Pass yaml path as argument"
    assert len(sys.argv) == 2, explanation
    fpath = sys.argv[1]
    assert fpath.endswith(".yaml"), explanation

    sc = StatCollector.from_yaml(fpath)
    save_folder = os.path.dirname(fpath)
    fname = os.path.basename(fpath)
    prefix = fname.replace("_stats.yaml", "")
    
    print(f"{fname=}")
    print(f"{prefix=}")
    print(f"{save_folder=}")

    write_stats_png(prefix, save_folder, sc)
