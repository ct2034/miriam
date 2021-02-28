#!/usr/bin/env python3

import argparse

import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.python.ops.gen_math_ops import mod

if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_read_h5', type=argparse.FileType('rb'))
    args = parser.parse_args()

    model: keras.Model = keras.models.load_model(args.model_read_h5.name)
    model.summary()

    empty_sample = np.zeros((1, 7, 7, 3, 5))
    y = model.predict([empty_sample])
    print(y)

    random_sample = np.random.random((100, 7, 7, 3, 5))
    y = model.predict([random_sample])
    plt.hist(y)
    plt.show()
