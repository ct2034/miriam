#!/usr/bin/env python3

import argparse
import pickle
import random

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.layers import (Conv2D, Conv3D, Dense, DepthwiseConv2D,
                                     Dropout, Flatten, MaxPooling2D, Reshape)
from tensorflow.keras.models import Sequential

IMG_SIZE = 13
IMG_DEPTH_T = 3
IMG_DEPTH_FRAMES = 4
TRAINING_PERC = 1


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
    train_images_l = []
    train_labels_l = []
    test_images = []
    test_labels = []
    for i in range(len(d)):
        if i < TRAINING_PERC * len(d):  # training data
            dat = d[i][0]
            train_images_l.append(dat)

            # more data
            dat = np.rot90(dat)
            train_images_l.append(dat)
            dat = np.rot90(dat)
            train_images_l.append(dat)
            dat = np.rot90(dat)
            train_images_l.append(dat)
            # <->
            dat = np.flip(dat, 0)
            train_images_l.append(dat)
            dat = np.rot90(dat)
            train_images_l.append(dat)
            dat = np.rot90(dat)
            train_images_l.append(dat)
            dat = np.rot90(dat)
            train_images_l.append(dat)
            for _ in range(8):
                train_labels_l.append(d[i][1])
        else:  # test data
            test_images.append(d[i][0])
            test_labels.append(d[i][1])
    # shuffling
    train_images = np.array(train_images_l)
    train_labels = np.array(train_labels_l)
    training_data = np.c_[train_images.reshape(
        len(train_images), -1), train_labels.reshape(len(train_labels), -1)]
    np.random.shuffle(training_data)
    train_images2 = training_data[:, :train_images.size //
                                  len(train_images)].reshape(
                                      train_images.shape)
    train_labels2 = training_data[:, train_images.size //
                                  len(train_images):].reshape(
                                      train_labels.shape)

    CONV3D_1_LAYERS = 16
    # model
    model = Sequential([
        Conv3D(CONV3D_1_LAYERS, 4, padding='same', activation='relu',
               input_shape=(IMG_SIZE, IMG_SIZE, IMG_DEPTH_T, IMG_DEPTH_FRAMES)),
        Dropout(0.2),
        Reshape((IMG_SIZE, IMG_SIZE, IMG_DEPTH_T * CONV3D_1_LAYERS)),
        Conv2D(32, 4, padding='same', activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # train
    history = model.fit([train_images2], train_labels2,
                        validation_split=0.1, epochs=8, batch_size=1024)

    # test
    # test_loss, test_acc = model.evaluate([test_images], test_labels)
    # print("Loss " + str(test_loss))
    # print("Accuracy " + str(test_acc))

    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
