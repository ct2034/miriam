#!/usr/bin/env python3

import argparse
import pickle
import random

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import (Conv2D, Conv3D, Dense, DepthwiseConv2D,
                                     Dropout, Flatten, MaxPooling2D, Reshape)
from tensorflow.keras.models import Sequential

tf.compat.v1.GPUOptions(allow_growth=True)

IMG_SIZE = 13
IMG_DEPTH_T = 3
IMG_DEPTH_FRAMES = 5


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

    dat_transformers = [
        lambda x: x,
        #lambda x: np.rot90(x, k=1, axes=(0, 1)),
        #lambda x: np.rot90(x, k=2, axes=(0, 1)),
        #lambda x: np.rot90(x, k=3, axes=(0, 1)),
        lambda x: np.flip(x, axis=0),
        #lambda x: np.rot90(np.flip(x, axis=0), k=1, axes=(0, 1)),
        #lambda x: np.rot90(np.flip(x, axis=0), k=2, axes=(0, 1)),
        #lambda x: np.rot90(np.flip(x, axis=0), k=3, axes=(0, 1)),
    ]

    for i in range(int(len(d))):
        dat = d[i][0]
        for transf in dat_transformers:
            train_images_l.append(transf(dat))
            train_labels_l.append(d[i][1])
    train_images = np.array(train_images_l)
    train_labels = np.array(train_labels_l)
    # # shuffling
    # training_data = np.c_[train_images.reshape(
    #     len(train_images), -1), train_labels.reshape(len(train_labels), -1)]
    # np.random.shuffle(training_data)
    # train_images2 = training_data[:, :train_images.size //
    #                               len(train_images)].reshape(
    #                                   train_images.shape)
    # train_labels2 = training_data[:, train_images.size //
    #                               len(train_images):].reshape(
    #                                   train_labels.shape)

    # optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=0.01, epsilon=1)

    # model
    CONV3D_1_FILTERS = 4
    model = Sequential([
        Conv3D(CONV3D_1_FILTERS, 2, padding='same', activation='relu',
               input_shape=(IMG_SIZE, IMG_SIZE, IMG_DEPTH_T, IMG_DEPTH_FRAMES)),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=opt, loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # train
    history = model.fit([train_images], train_labels,
                        validation_split=0.2, epochs=1, batch_size=len(dat_transformers))
    model.save('my_model.h5')

    # test
    # test_loss, test_acc = model.evaluate([test_images], test_labels)
    # print("Loss " + str(test_loss))
    # print("Accuracy " + str(test_acc))

    # Plot training & validation accuracy values
    #plt.plot(history.history['accuracy'])
    #plt.plot(history.history['val_accuracy'])
    #plt.title('Model accuracy')
    #plt.ylabel('Accuracy')
    #plt.xlabel('Epoch')
    #plt.legend(['Train', 'Test'], loc='upper left')
    #plt.savefig('training_accuracy.png')
    #plt.show()

    # Plot training & validation loss values
    #plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    #plt.title('Model loss')
    #plt.ylabel('Loss')
    #plt.xlabel('Epoch')
    #plt.legend(['Train', 'Test'], loc='upper left')
    #plt.savefig('training_loss.png')
    #plt.show()
