#!/usr/bin/env python3
import argparse
import os
import pickle
from typing import List, Optional

import numpy as np
from importtf import keras, tf
from matplotlib import pyplot as plt
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Conv3D, ConvLSTM2D, Dense,
                                     DepthwiseConv2D, Dropout, Flatten,
                                     MaxPooling2D, MaxPooling3D, Reshape)
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.backend import dropout
from tools import ProgressBar

# workaround, src https://github.com/tensorflow/tensorflow/issues/43174
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# choices of model type
CLASSIFICATION_STR = "classification"
CONVRNN_STR = "convrnn"


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


def construct_model_classification(img_width, img_len_t, img_depth_frames):
    CONV3D_FILTERS = 8
    CONV2D_FILTERS = 8
    model = Sequential([
        Conv3D(CONV3D_FILTERS, (5, 5, 3), padding='same', activation='relu',
               input_shape=(img_width, img_width, img_len_t, img_depth_frames)),
        Reshape((img_width, img_width, img_len_t * CONV3D_FILTERS)),
        Conv2D(CONV2D_FILTERS, 3, padding='same', activation='relu'),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=opt, loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


def construct_model_convrnn(img_width, img_len_t, img_depth_frames):
    dropout = .3
    convlstm2d_filters = 64
    model = Sequential([
        ConvLSTM2D(convlstm2d_filters, kernel_size=(3, 3),
                   padding='valid',
                   return_sequences=False,
                   activation='tanh',
                   #    kernel_regularizer='l2',
                   #    recurrent_regularizer='l2',
                   #    bias_regularizer='l2',
                   input_shape=(
            img_len_t, img_width, img_width, img_depth_frames)
        ),
        BatchNormalization(),
        Dropout(dropout),
        # ConvLSTM2D(convlstm2d_filters, kernel_size=(3, 3),
        #            padding='valid',
        #            return_sequences=True,
        #            activation='tanh'
        #            kernel_regularizer='l2',
        #            recurrent_regularizer='l2',
        #            bias_regularizer='l2',
        #            ),
        # BatchNormalization(),
        # # Dropout(dropout),
        # ConvLSTM2D(convlstm2d_filters, kernel_size=(3, 3),
        #            padding='valid',
        #            return_sequences=False,
        #            activation='tanh',
        #            kernel_regularizer='l2',
        #            #    recurrent_regularizer='l2',
        #            #    bias_regularizer='l2'
        #            ),
        # BatchNormalization(),
        # Dropout(dropout),
        Flatten(),
        Dense(256,
              activation='relu',
              kernel_regularizer='l2',
              bias_regularizer='l2'
              ),
        BatchNormalization(),
        Dropout(dropout),
        Dense(256,
              activation='relu',
              kernel_regularizer='l2',
              bias_regularizer='l2'
              ),
        BatchNormalization(),
        Dropout(dropout),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=opt, loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


def fix_data_convrnn(images):
    return np.moveaxis(images,
                       [1, 2, 3],
                       [-3, -2, -4]
                       )


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
        '-m', '--model_fname', type=str, default="my_model.h5", )
    parser.add_argument(
        '-t', '--model_type', choices=[
            CLASSIFICATION_STR,
            CONVRNN_STR
        ])
    parser.add_argument(
        'fnames_read_pkl', type=str, nargs='+')
    args = parser.parse_args()
    fnames_read_pkl: List[str] = args.fnames_read_pkl
    print(f'fnames_read_pkl: {fnames_read_pkl}')
    model_fname: str = args.model_fname
    print(f'model_fname: {model_fname}')
    model_type: str = args.model_type
    print(f'model_type: {model_type}')
    validation_split: float = .1
    test_split: float = .1

    if tf.test.gpu_device_name():
        print('GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Running on CPU.")

    epochs = 10

    # data
    pb = ProgressBar("epochs * files", len(fnames_read_pkl)*epochs)
    for i_e in range(epochs):
        for fname_read_pkl in fnames_read_pkl:
            print("~"*60)
            print(f"epoch {i_e+1} of {epochs}")
            print(
                f"reading file {fnames_read_pkl.index(fname_read_pkl) + 1} of " +
                f"{len(fnames_read_pkl)} : " +
                f"{fname_read_pkl}")
            with open(fname_read_pkl, 'rb') as f:
                d = pickle.load(f)
            n = len(d)
            n_val = int(n * validation_split)
            print(f'n_val: {n_val}')
            # on first file only
            if fname_read_pkl == fnames_read_pkl[0] and i_e == 0:
                print(f'n: {n}')
                n_test = int(n*test_split)
                print(f'n_test: {n_test}')
                n_train = n - n_val - n_test
                print(f'n_train: {n_train}')
                train_images = np.array([d[i][0] for i in range(n_train)])
                train_labels = np.array([d[i][1] for i in range(n_train)])
                assert train_images.shape[0] == n_train, "We must have all data."
                test_images = np.array(
                    [d[i][0] for i in range(n_train, n_train+n_test)])
                test_labels = np.array(
                    [d[i][1] for i in range(n_train, n_train+n_test)])
                val_images = np.array(
                    [d[i][0] for i in range(n_train+n_test, n)])
                val_labels = np.array(
                    [d[i][1] for i in range(n_train+n_test, n)])
                assert len(d) == len(train_images) + \
                    len(test_images) + len(val_images)
            else:
                n_train = n - n_val
                print(f'n_train: {n_train}')
                train_images = np.array([d[i][0] for i in range(n_train)])
                train_labels = np.array([d[i][1] for i in range(n_train)])
                val_images = np.array([d[i][0] for i in range(n_train, n)])
                val_labels = np.array([d[i][1] for i in range(n_train, n)])
                assert len(d) == len(train_images) + len(val_images)

            if model_type == CLASSIFICATION_STR:
                (n_samples, img_width, img_height, img_len_t,
                 img_channels) = train_images.shape
            elif model_type == CONVRNN_STR:
                print("fixing data for "+CONVRNN_STR)
                train_images = fix_data_convrnn(train_images)
                val_images = fix_data_convrnn(val_images)
                # on first file only
                if fname_read_pkl == fnames_read_pkl[0] and i_e == 0:
                    test_images = fix_data_convrnn(val_images)
                (n_samples, img_len_t, img_width, img_height,
                 img_channels) = train_images.shape

            # on first file only
            if fname_read_pkl == fnames_read_pkl[0] and i_e == 0:
                # info on data shape
                print(f"train_images.shape: {train_images.shape}")
                assert img_width == img_height, "Images must be square."
                print(f"n_samples: {n_samples}")
                print(f"img_width: {img_width}")
                print(f"img_height: {img_height}")
                print(f"img_len_t: {img_len_t}")
                print(f"img_channels: {img_channels}")

                # optimizer
                if model_type == CLASSIFICATION_STR:
                    opt = tf.keras.optimizers.Adam(
                        learning_rate=0.05, epsilon=1)
                elif model_type == CONVRNN_STR:
                    opt = tf.keras.optimizers.Adam(learning_rate=0.005)

                # model
                print(f"model_fname: {model_fname}")
                if os.path.isfile(model_fname):
                    print("model exists. going to load and improve it ...")
                    model: keras.Model = keras.models.load_model(
                        model_fname)
                else:
                    print(
                        "model does not exist. going to make a new one ... of type "+model_type)
                    if model_type == CLASSIFICATION_STR:
                        model = construct_model_classification(
                            img_width, img_len_t, img_channels)
                    elif model_type == CONVRNN_STR:
                        model = construct_model_convrnn(
                            img_width, img_len_t, img_channels)

                # train
                accuracy: List[float] = []
                val_accuracy: Optional[List[float]] = []
                test_accuracy: List[float] = []
                loss: List[float] = []
                val_loss: Optional[List[float]] = []
                test_loss: List[float] = []
                test_x: List[float] = []

                # evaluating untrained model
                pretrain_test_loss, pretrain_test_accuracy = model.evaluate(
                    [test_images], test_labels)
                print(f"pretrain_test_loss: {pretrain_test_loss}")
                print(f"pretrain_test_accuracy: {pretrain_test_accuracy}")
                test_x.append(0)
                test_accuracy.append(pretrain_test_accuracy)
                test_loss.append(pretrain_test_loss)
            # (if) on first file only

            if model_type == CLASSIFICATION_STR:
                bcp = BatchHistory()
                history = model.fit([train_images], train_labels,
                                    epochs=1, batch_size=4, callbacks=[bcp]
                                    )
                # stats from bcp
                accuracy.extend(bcp.batch_accuracy)
                val_accuracy = None
                loss.extend(bcp.batch_loss)
                val_loss = None
            elif model_type == CONVRNN_STR:
                history = model.fit([train_images], train_labels,
                                    epochs=1, batch_size=512,
                                    validation_data=([val_images], val_labels)
                                    )
                # normal stats
                accuracy.extend(history.history['accuracy'])
                assert val_accuracy is not None
                val_accuracy.extend(history.history['val_accuracy'])
                loss.extend(history.history['loss'])
                assert val_loss is not None
                val_loss.extend(history.history['val_loss'])

            del d
            del train_images
            del train_labels
            del val_images
            del val_labels
            pb.progress()

        # manual validation (testing) after each epoch
        one_test_loss, one_test_accuracy = model.evaluate(
            [test_images], test_labels)
        print(f"one_test_loss: {one_test_loss}")
        print(f"one_test_accuracy: {one_test_accuracy}")
        test_x.append(len(accuracy)-1)
        test_accuracy.append(one_test_accuracy)
        test_loss.append(one_test_loss)

    model.save(model_fname)
    pb.end()

    # print history
    fig, axs = plt.subplots(2)
    axs[0].plot(accuracy, label="accuracy")
    axs[1].plot(loss, label="loss")
    if val_accuracy is not None:
        axs[0].plot(val_accuracy, label="val_accuracy")
    if val_loss is not None:
        axs[1].plot(val_loss, label="val_loss")
    axs[0].plot(test_x, test_accuracy, label="test_accuracy")
    axs[1].plot(test_x, test_loss, label="test_loss")
    axs[0].legend(loc='lower left')
    axs[0].set_xlabel('Batch')
    axs[1].legend(loc='lower left')
    axs[1].set_xlabel('Batch')
    fig.savefig("training_history.png")
    fig.show()
