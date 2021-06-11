import unittest

import numpy as np
from planner.policylearn.train_model import augment_data


class TrainModelClassTest(unittest.TestCase):
    def test_augment_data(self):
        n = 10
        width = height = 4
        features = 5
        ts = 3

        n_augmentations = 8

        images = np.zeros([n, width, height, features, ts])
        images[0, [
            0, 0, 3, 3
        ], [
            0, 3, 3, 0
        ], 0, 0] = np.arange(4)
        labels = np.arange(n)
        images_out, labels_out = augment_data(images, labels)
        reordering = set([
            (0, 1, 2, 3),
            (3, 0, 1, 2),
            (2, 3, 0, 1),
            (1, 2, 3, 0),
            (3, 2, 1, 0),
            (0, 3, 2, 1),
            (1, 0, 3, 2),
            (2, 1, 0, 3)
        ])

        self.assertEqual(images.shape[0] * n_augmentations, len(images_out))
        for i in range(n_augmentations):
            test_seq = np.array(images_out)[i, [
                0, 0, 3, 3
            ], [
                0, 3, 3, 0
            ], 0, 0]
            self.assertTrue(tuple(test_seq) in reordering)
            reordering.remove(tuple(test_seq))
        self.assertTrue(len(reordering) == 0)


if __name__ == "__main__":
    unittest.main()
