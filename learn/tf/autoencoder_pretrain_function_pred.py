#!/usr/bin/env python3

import random

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
epochs = 8
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


def train_autoenc(x):
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


def run_an_example_and_plot_info():
    n = 2 ** 17  # samples
    x, y = make_data(n)

    pred_model, pred_history = train_pred(x, y)
    autoenc_model, autoenc_history = train_autoenc(x)

    # stuff we need out of the autoenc model
    encoded_input = Input(shape=(n_encoding,))
    decoder_layer = autoenc_model.layers[-1]
    decoder_layer = autoenc_model.layers[-1]
    decoder_model = Model(encoded_input, decoder_layer(encoded_input))

    plt.subplot(221)
    # Plot training & validation accuracy values
    plt.plot(pred_history.history['acc'])
    plt.plot(pred_history.history['val_acc'])
    plt.ylim(0, 1)
    plt.title('Prediction Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.subplot(222)
    # Plot training & validation loss values
    plt.plot(pred_history.history['loss'])
    plt.plot(pred_history.history['val_loss'])
    plt.title('Prediction Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.subplot(223)
    # Plot training & validation accuracy values
    plt.plot(autoenc_history.history['acc'])
    plt.plot(autoenc_history.history['val_acc'])
    plt.ylim(0, 1)
    plt.title('Autoenc Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.subplot(224)
    # Plot training & validation loss values
    plt.plot(autoenc_history.history['loss'])
    plt.plot(autoenc_history.history['val_loss'])
    plt.title('Autoenc Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # The End
    plt.show()


if __name__ == "__main__":
    run_an_example_and_plot_info()
