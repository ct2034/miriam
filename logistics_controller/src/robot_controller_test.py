#!/bin/env python2
import robot_controller


def conversion_plan_map_test_a():
    assert (-2, -2) == robot_controller.plan_to_map((5, 3))


def conversion_plan_map_test_b():
    assert (8, -8) == robot_controller.plan_to_map((10, 0))


def conversion_map_plan_test_a():
    assert (5, 3) == robot_controller.map_to_plan((-2, -2))


def conversion_map_plan_test_b():
    assert (10, 0) == robot_controller.map_to_plan((8, -8))
