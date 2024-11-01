#!/usr/bin/env python2
import re

BASE_NAME = ".*\/[a-z]_[0-9]+_[0-9]+\."
RESULT_FILE = re.compile(BASE_NAME + "pkl$")
EVAL_FILE = re.compile(BASE_NAME + "pkl.eval$")
EVAL_CEN_FILE = re.compile(BASE_NAME + "eval_cen.*.pkl$")


def is_result_file(fname):
    return bool(RESULT_FILE.match(fname))


def is_eval_file(fname):
    return bool(EVAL_FILE.match(fname))


def is_eval_cen_file(fname):
    return bool(EVAL_CEN_FILE.match(fname))


def get_basename_wo_extension(fname):
    return fname.split("/")[-1].split(".")[0]


def resolve(fname):
    return get_basename_wo_extension(fname).split("_")


def resolve_mapname(fname):
    return "maps/" + resolve(fname)[0] + ".png"


def resolve_number_of_nodes(fname):
    return int(resolve(fname)[1])


def resolve_number_of_iterations(fname):
    return int(resolve(fname)[2])


def get_graph_csvs(fname):
    return (
        "res/" + get_basename_wo_extension(fname) + ".graph_adjlist.csv",
        "res/" + get_basename_wo_extension(fname) + ".graph_pos.csv",
    )


def get_graph_undir_csv(fname):
    return "res/" + get_basename_wo_extension(fname) + ".graph_undir_adjlist.csv"
