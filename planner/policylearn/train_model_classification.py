#!/usr/bin/env python3

import argparse
import os
import pickle
import random

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.layers import (Conv2D, Conv3D, Dense, Dropout, Flatten,
                                     LocallyConnected2D, Reshape, 
                                     BatchNormalization, MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow import keras

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

IMG_SIZE = 13
IMG_DEPTH_T = 3
IMG_DEPTH_FRAMES = 4
TRAINING_PERCENTAGE = .9


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'fname_read_pkl', type=argparse.FileType('rb'))
    args = parser.parse_args()
    with open(args.fname_read_pkl.name, 'rb') as f:
        d = pickle.load(f)
    print(len(d))
    random.shuffle(d)

    checkpoint_path = "checkpoint.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

    # data
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    for i in range(len(d)):
        if i < TRAINING_PERCENTAGE * len(d):  # training data
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
                                  len(train_images)].reshape(
                                      train_images.shape)
    train_labels2 = training_data[:, train_images.size //
                                  len(train_images):].reshape(
                                      train_labels.shape)

    CONV3D_1_LAYERS = 32
    # model
    model = Sequential([
        Conv3D(CONV3D_1_LAYERS, 2, padding='same', activation='relu',
               input_shape=(IMG_SIZE, IMG_SIZE,
                            IMG_DEPTH_T, IMG_DEPTH_FRAMES)),
        Conv3D(CONV3D_1_LAYERS, 3, padding='same', activation='relu'),
        Dropout(0.4),
        BatchNormalization(),
        Reshape((IMG_SIZE, IMG_SIZE, IMG_DEPTH_T * CONV3D_1_LAYERS)),
        Conv2D(128, 3, padding='same', activation='relu'),
        Conv2D(128, 4, padding='same', activation='relu'),
        Dropout(0.4),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        Conv2D(64, 4, padding='same', activation='relu'),
        Dropout(0.4),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.4),
        BatchNormalization(),
        Dense(256, activation='relu'),
        Dropout(0.4),
        BatchNormalization(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    adam = keras.optimizers.Adam(
        learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adam,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # train
    history = model.fit([train_images2], train_labels2,
                        validation_split=0.1, epochs=64, batch_size=512,
                        callbacks=[cp_callback])

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
