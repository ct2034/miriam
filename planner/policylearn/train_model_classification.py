#!/usr/bin/env python3

import argparse
import pickle
from typing import List

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from numpy.core.shape_base import _concatenate_shapes
from tensorflow.keras.layers import (Conv2D, Conv3D, Dense, DepthwiseConv2D,
                                     Flatten, MaxPooling2D, Reshape)
from tensorflow.keras.models import Sequential

tf.compat.v1.GPUOptions(allow_growth=True)


class BatchHistory(tf.keras.callbacks.Callback):
    """Collection of history per batch,
    src: https://stackoverflow.com/a/66401457/1493204"""
    batch_accuracy: List[float] = []  # accuracy at given batch
    batch_loss: List[float] = []  # loss at given batch

    def __init__(self):
        super(BatchHistory, self).__init__()

    def on_train_batch_end(self, batch, logs=None):
        BatchHistory.batch_accuracy.append(logs.get('accuracy'))
        BatchHistory.batch_loss.append(logs.get('loss'))


def construct_model(img_width, img_depth_t, img_depth_frames):
    CONV3D_FILTERS = 8
    CONV2D_FILTERS = 8
    model = Sequential([
        Conv3D(CONV3D_FILTERS, (5, 5, 3), padding='same', activation='relu',
               input_shape=(img_width, img_width, img_depth_t, img_depth_frames)),
        Reshape((img_width, img_width, img_depth_t * CONV3D_FILTERS)),
        Conv2D(CONV2D_FILTERS, 3, padding='same', activation='relu'),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=opt, loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


def augment_data(images_in, labels_in):
    assert len(images_in) == len(images_in)
    n = len(images_in)
    shape = images_in.shape
    augmentations = [
        lambda x: x,
        lambda x: np.rot90(x, k=1),
        lambda x: np.rot90(x, k=2),
        lambda x: np.rot90(x, k=3),
        lambda x: np.flip(x, axis=0),
        lambda x: np.flip(np.rot90(x, k=1), axis=0),
        lambda x: np.flip(np.rot90(x, k=2), axis=0),
        lambda x: np.flip(np.rot90(x, k=3), axis=0)
    ]
    shape_out = list(shape)
    shape_out[0] = shape_out[0] * len(augmentations)
    images_out = np.zeros(shape=shape_out)
    labels_out = np.zeros(shape=len(augmentations)*n)
    for i_d in range(n):
        for i_a, aug in enumerate(augmentations):
            i_out = i_d * len(augmentations) + i_a
            images_out[i_out] = aug(images_in[i_d])
            labels_out[i_out] = labels_in[i_d]
    return images_out, labels_out


if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'fname_read_pkl', type=argparse.FileType('rb'))
    args = parser.parse_args()
    validation_split = .1

    # data
    with open(args.fname_read_pkl.name, 'rb') as f:
        d = pickle.load(f)
    n = len(d)
    print(f'n: {n}')
    n_train = int(n * (1-validation_split))
    print(f'n_train: {n_train}')
    train_images = np.array([d[i][0] for i in range(n_train)])
    train_labels = np.array([d[i][1] for i in range(n_train)])
    assert train_images.shape[0] == n_train, "We must have all data."
    val_images = np.array([d[i][0] for i in range(n_train+1, n)])
    val_labels = np.array([d[i][1] for i in range(n_train+1, n)])
    print(f"train_images.shape: {train_images.shape}")
    (_, img_width, img_height, img_depth_t, img_depth_frames) = train_images.shape
    assert img_width == img_height, "Images must be square."

    # data augmentation
    # train_images_augmented, train_labels_augmented = augment_data(train_images, train_labels,
    #                                                               )
    # print(f"train_images_augmented.shape: {train_images_augmented.shape}")

    # optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=0.05, epsilon=1)

    # model
    model = construct_model(img_width, img_depth_t, img_depth_frames)

    # train
    bcp = BatchHistory()
    history = model.fit([train_images], train_labels,
                        epochs=4, batch_size=4, callbacks=[bcp]
                        )
    model.save('my_model.h5')

    # manual validation
    val_loss, val_acc = model.evaluate([val_images], val_labels)
    print(f"val_loss: {val_loss}")
    print(f"val_acc: {val_acc}")

    # print history
    plt.plot(bcp.batch_accuracy, label="accuracy")
    plt.plot(bcp.batch_loss, label="loss")
    plt.legend(loc='lower left')
    plt.xlabel('Batch')
    plt.savefig("training_history.png")
    plt.show()
