#!/usr/bin/env python3

import random
from typing import *

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.layers import (Dense, DepthwiseConv2D, Dropout, Flatten,
                                     Input, MaxPooling2D, Reshape)
from tensorflow.keras.models import Model

size_polynome = 4  # how many parameters has the polynome
learn_res = 50     # with of input samples (x) == neurons of input and output
n_encoding = 8     # neurons in encoding layer
epochs = 16
batch_size = 256


def make_random_poly():
    return np.polynomial.Polynomial(np.random.random(size_polynome)-.5)


def get_sample(poly, t):
    return poly(t), poly.coef


def get_poly_with_res(res, model):
    while True:
        poly = make_random_poly()
        X, Y = get_sample(poly)
        pred = model.predict_classes(np.array([X]))[0][0]
        if (pred == Y) == res:
            return poly


def make_data(n):
    sample_end = 1  # where does X data stop

    t = np.linspace(-1, sample_end, learn_res)
    x = []
    y = []
    for _ in range(n):
        poly = make_random_poly()
        X, Y = get_sample(poly, t)
        x.append(X)
        y.append(Y)
    return x, y


def train_pred(x, y):
    """train model that predicts polynome coefficients"""
    # helpers
    init = initializers.RandomNormal(mean=0.0,
                                     stddev=0.1, seed=None)
    reg_sparse = regularizers.l2(.01)

    # layers
    input_data = Input(shape=x[0].shape)
    e1 = Dense(learn_res, kernel_initializer=init,
               activation='relu')(input_data)
    encoded_input = Dense(n_encoding, kernel_initializer=init,
                          kernel_regularizer=reg_sparse, activation='relu')(e1)
    predicted_coeff = Dense(size_polynome, kernel_initializer=init,
                            activation='linear')(encoded_input)

    pred_model = Model(input_data, predicted_coeff, name='pred')
    pred_model.compile(optimizer='adam',
                       loss='mean_squared_error',
                       metrics=['accuracy'])
    pred_model.summary()

    # train
    history = pred_model.fit([x], [y],
                             validation_split=0.3, epochs=epochs,
                             batch_size=batch_size)
    return pred_model, history


def train_autoenc(x) -> (Model, Any):
    """train model that autoencodes polynoms"""
    # helpers
    init = initializers.RandomNormal(mean=0.0,
                                     stddev=0.1, seed=None)
    reg_sparse = regularizers.l2(.01)

    # layers
    input_data = Input(shape=x[0].shape)
    e1 = Dense(learn_res, kernel_initializer=init,
               activation='relu')(input_data)
    encoded_input = Dense(n_encoding, kernel_initializer=init,
                          kernel_regularizer=reg_sparse, activation='relu')(e1)
    decoded = Dense(learn_res, kernel_initializer=init,
                    activation='linear')(encoded_input)

    autoenc_model = Model(input_data, decoded, name='autoenc')
    autoenc_model.compile(optimizer='adam',
                          loss='mean_squared_error',
                          metrics=['accuracy'])
    autoenc_model.summary()

    # train
    history = autoenc_model.fit([x], [x],
                                validation_split=0.3, epochs=epochs,
                                batch_size=batch_size)
    return autoenc_model, history


def train_transfer(x, y, encoder_layer):
    # helpers
    init = initializers.RandomNormal(mean=0.0,
                                     stddev=0.1, seed=None)

    # layers
    input_data = Input(shape=x[0].shape)
    encoded_input = encoder_layer(input_data)
    predicted_coeff = Dense(size_polynome, kernel_initializer=init,
                            activation='linear')(encoded_input)

    transfer_model = Model(input_data, predicted_coeff)
    transfer_model.compile(optimizer='adam',
                           loss='mean_squared_error',
                           metrics=['accuracy'])
    transfer_model.summary()

    # train
    history = transfer_model.fit([x], [y],
                                 validation_split=0.3, epochs=epochs,
                                 batch_size=batch_size)
    return transfer_model, history


def run_an_example_and_plot_info():
    # samples per model
    n_pred = 2 ** 16
    n_autoenc = 2 ** 19
    n_transfer = 2 ** 14
    x_autoenc, _ = make_data(n_autoenc)
    x_pred, y_pred = make_data(n_pred)
    x_transfer, y_transfer = make_data(n_transfer)

    # normal prediction model
    pred_model, pred_history = train_pred(x_pred, y_pred)

    # autoencoder model
    autoenc_model, autoenc_history = train_autoenc(x_autoenc)

    # the encoder for the transfer learning
    encoder_layer = autoenc_model.layers[0]
    encoder_layer.trainable = False

    # transfer learning using autoencode info
    transfer_model, transfer_history = train_transfer(
        x_transfer, y_transfer, encoder_layer)

    # the decoder for fun
    encoded_input = Input(shape=(n_encoding,))
    decoder_layer = autoenc_model.layers[-1]
    decoder_model = Model(encoded_input, decoder_layer(encoded_input))

    plt.subplot(321)
    # Plot training & validation accuracy values
    plt.plot(pred_history.history['acc'])
    plt.plot(pred_history.history['val_acc'])
    plt.ylim(0, 1.1)
    plt.title('Prediction Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.subplot(322)
    # Plot training & validation loss values
    plt.plot(pred_history.history['loss'])
    plt.plot(pred_history.history['val_loss'])
    plt.title('Prediction Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.subplot(323)
    # Plot training & validation accuracy values
    plt.plot(autoenc_history.history['acc'])
    plt.plot(autoenc_history.history['val_acc'])
    plt.ylim(0, 1.1)
    plt.title('Autoenc Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.subplot(324)
    # Plot training & validation loss values
    plt.plot(autoenc_history.history['loss'])
    plt.plot(autoenc_history.history['val_loss'])
    plt.title('Autoenc Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.subplot(325)
    # Plot training & validation accuracy values
    plt.plot(transfer_history.history['acc'])
    plt.plot(transfer_history.history['val_acc'])
    plt.ylim(0, 1.1)
    plt.title('Transfer Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.subplot(326)
    # Plot training & validation loss values
    plt.plot(transfer_history.history['loss'])
    plt.plot(transfer_history.history['val_loss'])
    plt.title('Transfer Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # The End
    plt.show()


if __name__ == "__main__":
    run_an_example_and_plot_info()
