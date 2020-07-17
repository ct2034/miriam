import unittest
import numpy as np

import planner.policylearn.generate_data as generate_data


class GenerateDataTest(unittest.TestCase):
    def test_generate_random_gridmap(self):
        w = 999
        h = 100
        gridmap_empty = generate_data.generate_random_gridmap(w, h, 0)

        assert gridmap_empty.shape[0] == w
        assert gridmap_empty.shape[1] == h
        assert np.max(gridmap_empty) == 0
        assert np.min(gridmap_empty) == 0

        gridmap_half = generate_data.generate_random_gridmap(w, h, 0.5)

        assert gridmap_half.shape[0] == w
        assert gridmap_half.shape[1] == h
        assert np.max(gridmap_half) == 1
        assert np.min(gridmap_half) == 0

    def test_import_ecbs(self):
        from planner.policylearn.libMultiRobotPlanning.plan_ecbs import (
            plan_in_gridmap, BLOCKS_STR
        )

    def test_add_padding_to_gridmap(self):
        radius = 2  # testing
        my_gridmap = np.zeros([2, 3])
        my_gridmap[1, 2] = 1  # a pixel
        out_gridmap = generate_data.add_padding_to_gridmap(my_gridmap, radius)
        assert out_gridmap[0, 0] == 1  # padding
        assert out_gridmap[1, 1] == 1  # padding
        assert out_gridmap[2, 2] == 0  # free map space
        assert out_gridmap[3, 3] == 0  # free map space
        assert out_gridmap[4, 4] == 1  # padding
        assert out_gridmap[5, 5] == 1  # padding
        assert out_gridmap[5, 6] == 1  # padding
        assert out_gridmap[3, 4] == 1  # a pixel

    def test_make_fovs(self):
        radius = 2  # testing
        my_gridmap = np.zeros([2, 3])
        my_gridmap[1, 2] = 1  # a pixel
        padded_gridmap = generate_data.add_padding_to_gridmap(
            my_gridmap, radius)
        path = np.array(
            [(0, 0, 0), (0, 1, 1), (1, 1, 2), (1, 2, 3), (99, 99, 99)])
        out_fovs = generate_data.make_obstacle_fovs(
            padded_gridmap,
            path,
            3,
            radius
        )
        assert out_fovs.shape[2] == 3 + 1
        # t = 0
        assert out_fovs[0, 0, 0] == 1
        assert out_fovs[1, 1, 0] == 1
        assert out_fovs[2, 2, 0] == 0
        assert out_fovs[3, 3, 0] == 0
        assert out_fovs[4, 4, 0] == 1
        # t = 3
        assert out_fovs[0, 0, 3] == 1
        assert out_fovs[1, 1, 3] == 0
        assert out_fovs[2, 2, 3] == 1  # a pixel
        assert out_fovs[3, 3, 3] == 1
        assert out_fovs[4, 4, 3] == 1

    def test_make_target_deltas(self):
        testing_path = np.array(
            [(0, 0), (0, 1), (1, 1), (1, 2), (99, 99)])
        out_deltas = generate_data.make_target_deltas(testing_path, 3)
        # t = 0
        assert out_deltas[0][0] == 99
        assert out_deltas[0][1] == 99
        # t = 1
        assert out_deltas[1][0] == 99
        assert out_deltas[1][1] == 98
        # t = 3
        assert out_deltas[3][0] == 98
        assert out_deltas[3][1] == 97

    def test_get_path(self):
        testing_path = np.array(
            [(0, 0, 0), (0, 1, 1), (1, 1, 2), (1, 2, 3), (99, 99, 4)])
        # fixed_length_shorter
        fixed_len_path = generate_data.get_path(testing_path, 2)
        assert len(fixed_len_path) == 3
        assert len(fixed_len_path[1]) == 2
        assert len(fixed_len_path[-1]) == 2
        assert fixed_len_path[0, 0] == 0
        assert fixed_len_path[0, 1] == 0
        assert fixed_len_path[1, 0] == 0
        assert fixed_len_path[1, 1] == 1
        assert fixed_len_path[2, 0] == 1
        assert fixed_len_path[2, 1] == 1
        assert fixed_len_path[-1, 0] == 1
        assert fixed_len_path[-1, 1] == 1
        # fixed_length
        full_len_path = generate_data.get_path(testing_path, -1)
        assert len(full_len_path) == len(testing_path)
        assert len(full_len_path[1]) == 2
        assert len(full_len_path[4]) == 2
        assert full_len_path[1, 0] == 0
        assert full_len_path[1, 1] == 1
        assert full_len_path[4, 0] == 99
        assert full_len_path[4, 1] == 99


if __name__ == "__main__":
    unittest.main()
