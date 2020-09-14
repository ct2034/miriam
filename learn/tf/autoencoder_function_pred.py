#!/usr/bin/env python3

import argparse
import pickle
import random
from itertools import product

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import initializers
from tensorflow.keras.layers import (Dense, DepthwiseConv2D,
                                     Dropout, Flatten, MaxPooling2D, Reshape)
from tensorflow.keras.models import Sequential

size_polynome = 4  # how many parameters has the polynome
learn_res = 50    # with of input samples (x) == neurons of input and output
n_encoding = 10    # neurons in encoding layer


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


def train(n, learn_res, sample_end, t_pred):
    t = np.linspace(-1, sample_end, learn_res)
    x = []
    y = []
    for _ in range(n):
        poly = make_random_poly()
        X, Y = get_sample(poly, t, t_pred)
        x.append(X)
        y.append(Y)

    # model
    my_init = initializers.RandomNormal(mean=0.0,
                                        stddev=0.05, seed=None)
    layers = [
        Dense(learn_res, kernel_initializer=my_init,
              activation='relu', input_shape=x[0].shape),
        # Dense(learn_res / 2, kernel_initializer=my_init, activation='relu'),
        Dense(n_encoding, kernel_initializer=my_init, activation='relu'),
        # Dense(learn_res / 2, kernel_initializer=my_init, activation='relu'),
        Dense(learn_res, kernel_initializer=my_init, activation='linear'),
    ]
    model = Sequential(layers)
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    model.summary()

    # train
    history = model.fit([x], [x],
                        validation_split=0.3, epochs=128, batch_size=256)
    return model, history


def run_an_example_and_plot_info():
    n = 2 ** 15  # samples
    sample_end = 0  # where does X data stop
    t_pred = 1  # where to measure y

    model, history = train(n, learn_res, sample_end, t_pred)

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

    # having fun with our model
    plt.figure()
    n = 8
    plot_res = 1000
    t_plot = np.linspace(-1, 1, plot_res)
    t_sample = np.linspace(-1, sample_end, learn_res)
    for i in range(n):
        poly = make_random_poly()
        X, Y = get_sample(poly, t_sample, t_pred)
        pred = model.predict(np.array([X]))[0]
        plt.subplot(100 * n / 2 + 20 + i)
        plt.plot(t_plot, poly(t_plot), 'k--')
        plt.plot(t_sample, pred, 'r')
        # if pred == Y:
        #     col = 'g'
        # else:
        #     col = 'r'
        # plt.plot(t_pred, poly(t_pred), col+'X')
        # plt.plot(t_sample, X, 'b')
        # plt.title('True label: {} | Predicted: {}'.format(Y, pred))
        plt.grid(True)
        # plt.axhline(y=0, color='k')

    # The End
    plt.show()


def see_accuracy_per_learn_res():
    n = 2 ** 10  # samples
    sample_end = .5  # where does X data stop
    t_pred = 1  # where to measure y
    # -----------
    n_res = 10  # how often to run with each res
    learn_resolutions = range(1, 10)
    res = [list()] * n_res
    for i_n, learn_res in product(range(n_res), learn_resolutions):
        model, history = train(n, learn_res, sample_end, t_pred)
        acc = history.history['val_acc'][-1]
        res[i_n].append(acc)
    plt.violinplot(res)
    plt.show()


if __name__ == "__main__":
    run_an_example_and_plot_info()
    # see_accuracy_per_learn_res()
