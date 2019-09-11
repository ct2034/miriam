#!/usr/bin/env python2
import imageio

from adamsmap_eval.filename_verification import (
    get_graph_csvs,
    is_result_file,
    resolve_mapname,
    get_basename_wo_extension
)
from adamsmap import ()

PATH_ILP = "~/src/optimal-mrppg-journal"   # github.com/ct2034/optimal-mrppg-journal
PATH_ECBS = "~/src/libMultiRobotPlanning"  # github.com/ct2034/libMultiRobotPlanning

def evaluate(fname):
    assert is_result_file(fname), "Please call with result file (*.pkl)"
    fname_graph_adjlist, fname_graph_pos = get_graph_csvs(fname)
    assert os.path.exists(fname_graph_adjlist) and os.path.exists(fname_graph_pos), "Please make csv files first `script/write_graph.py csv res/...pkl`"
    fname_map = resolve_mapname(fname)

    # map
    im = imageio.imread(fname_map)

if __name__ == '__main__':
    fname = sys.argv[1]
    evaluate(fname)
