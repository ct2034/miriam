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
n_encoding = 5     # neurons in encoding layer


def make_random_poly():
    return np.polynomial.Polynomial(np.random.random(size_polynome)-.5)


def get_sample(poly, t, t_pred):
    return poly(t), int(poly(t_pred) > 0)


def get_poly_with_res(res, model, t, t_pred):
    while True:
        poly = make_random_poly()
        X, Y = get_sample(poly, t, t_pred)
        pred = model.predict_classes(np.array([X]))[0][0]
        if (pred == Y) == res:
            return poly


def train_autoenc(n, learn_res, sample_end, t_pred):
    t = np.linspace(-1, sample_end, learn_res)
    x = []
    y = []
    for _ in range(n):
        poly = make_random_poly()
        X, Y = get_sample(poly, t, t_pred)
        x.append(X)
        y.append(Y)

    # helpers
    init = initializers.RandomNormal(mean=0.0,
                                     stddev=0.05, seed=None)
    reg_sparse = regularizers.l2(.01)

    # layers
    input_data = Input(shape=x[0].shape)
    e1 = Dense(learn_res, kernel_initializer=init,
               activation='relu')(input_data)
    encoded_input = Dense(n_encoding, kernel_initializer=init,
                          kernel_regularizer=reg_sparse, activation='relu')(e1)
    decoded = Dense(learn_res, kernel_initializer=init,
                    activation='linear')(encoded_input)

    autoencoder_model = Model(input_data, decoded)
    autoencoder_model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    autoencoder_model.summary()

    # train
    history = autoencoder_model.fit([x], [x],
                        validation_split=0.3, epochs=32, batch_size=256)
    return autoencoder_model, history


def run_an_example_and_plot_info():
    n = 2 ** 17  # samples
    sample_end = 1  # where does X data stop
    t_pred = 1  # where to measure y

    autoencoder_model, history = train_autoenc(n, learn_res, sample_end, t_pred)

    encoded_input = Input(shape=(n_encoding,))
    decoder_layer = autoencoder_model.layers[-1]
    decoder_model = Model(encoded_input, decoder_layer(encoded_input))

    plt.subplot(211)
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.subplot(212)
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # original data and its reconstruction
    plt.figure()
    n = 8
    plot_res = 1000
    t_plot = np.linspace(-1.2, 1.2, plot_res)
    t_sample = np.linspace(-1, sample_end, learn_res)
    for i in range(n):
        poly = make_random_poly()
        X, Y = get_sample(poly, t_sample, t_pred)
        pred = autoencoder_model.predict(np.array([X]))[0]
        plt.subplot(100 * n / 2 + 20 + i)
        plt.plot(t_plot, poly(t_plot), 'k--')
        plt.plot(t_sample, pred, 'r')
        plt.title(str(poly.coef))
        plt.grid(True)

    plt.figure()
    i = 1
    for pm in [-1, 1]:
        for i_e in range(n_encoding):
            plt.subplot(2, n_encoding, i)
            i += 1
            code = np.zeros((n_encoding,))
            code[i_e] += pm
            decoded = decoder_model.predict(np.array([code]))[0]
            plt.title(str(code))
            plt.plot(t_sample, decoded, 'r')

    # The End
    plt.show()


if __name__ == "__main__":
    run_an_example_and_plot_info()
