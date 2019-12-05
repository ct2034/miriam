import numpy as np

import generate_data


def test_make_random_gridmap():
    w = 999
    h = 100
    gridmap_empty = generate_data.make_random_gridmap(w, h, 0)

    assert gridmap_empty.shape[0] == w
    assert gridmap_empty.shape[1] == h
    assert np.max(gridmap_empty) == 0
    assert np.min(gridmap_empty) == 0

    gridmap_half = generate_data.make_random_gridmap(w, h, 0.5)

    assert gridmap_half.shape[0] == w
    assert gridmap_half.shape[1] == h
    assert np.max(gridmap_half) == 1
    assert np.min(gridmap_half) == 0
