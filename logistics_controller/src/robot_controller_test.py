#!/bin/env python2
import logistics_controller as rc


def conversion_plan_map_test_a():
    assert (-2, -2) == rc.plan_to_map((5, 3))


def conversion_plan_map_test_b():
    assert (8, -8) == rc.plan_to_map((10, 0))


def conversion_map_plan_test_a():
    assert (5, 3) == rc.map_to_plan((-2, -2))


def conversion_map_plan_test_b():
    assert (10, 0) == rc.map_to_plan((8, -8))
