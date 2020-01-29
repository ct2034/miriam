#!/usr/bin/env python3

import argparse
import pickle
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D, DepthwiseConv2D, Dropout)
from tensorflow.keras.models import Sequential

IMG_SIZE = 11
IMG_DEPTH = 10
TRAININING_PERC = .8


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'fname_read_pkl', type=argparse.FileType('rb'))
    args = parser.parse_args()
    with open(args.fname_read_pkl.name, 'rb') as f:
        d = pickle.load(f)
    print(len(d))
    random.shuffle(d)

    # data
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    for i in range(len(d)):
        if i < TRAININING_PERC * len(d):  # training data
            dat = d[i][0]
            train_images.append(dat)

            # more data
            dat = np.rot90(dat)
            train_images.append(dat)
            dat = np.rot90(dat)
            train_images.append(dat)
            dat = np.rot90(dat)
            train_images.append(dat)
            # <->
            dat = np.flip(dat, 0)
            train_images.append(dat)
            dat = np.rot90(dat)
            train_images.append(dat)
            dat = np.rot90(dat)
            train_images.append(dat)
            dat = np.rot90(dat)
            train_images.append(dat)
            for _ in range(8):
                train_labels.append(d[i][1])
        else:  # test data
            test_images.append(d[i][0])
            test_labels.append(d[i][1])
    # shuffling
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    training_data = np.c_[train_images.reshape(
        len(train_images), -1), train_labels.reshape(len(train_labels), -1)]
    np.random.shuffle(training_data)
    train_images2 = training_data[:, :train_images.size //
                                  len(train_images)].reshape(train_images.shape)
    train_labels2 = training_data[:, train_images.size //
                                  len(train_images):].reshape(train_labels.shape)

    # model
    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, IMG_DEPTH)),
        Flatten(),
	Dense(32, activation='relu'),
	Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # train
    model.fit([train_images2], train_labels2, epochs=3)

    # test
    test_loss, test_acc = model.evaluate([test_images], test_labels)
    print("Loss " + str(test_loss))
    print("Accuracy " + str(test_acc))
