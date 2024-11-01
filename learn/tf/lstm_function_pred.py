#!/usr/bin/env python3

import argparse
import pickle
import random
from itertools import product

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

# data
#       .--.
#      '    '
#     /      '   o
#  \ /        '''
#   -
#  -------------------
#  |+++x+++|     y
#                > 0 : 1
#                <=0 : 0


def make_random_poly():
    return np.polynomial.Polynomial(np.random.random(5) - 0.5)


def get_sample(poly, t, t_pred):
    return poly(t), int(poly(t_pred) > 0)


def get_poly_with_res(res, model, t, t_pred):
    while True:
        poly = make_random_poly()
        X, Y = get_sample(poly, t, t_pred)
        X = np.reshape(X, (X.shape[0], 1))
        pred = int(model.predict(np.array([X]))[0][0] > 0.5)
        if (pred == Y) == res:
            return poly


def train(n, learn_res, sample_end, t_pred):
    t = np.linspace(-1, sample_end, learn_res)
    x = []
    y = []
    for _ in range(n):
        poly = make_random_poly()
        X, Y = get_sample(poly, t, t_pred)
        X = np.reshape(X, (learn_res, 1))
        x.append(X)
        y.append(Y)
    x = np.array(x)
    y = np.array(y)
    print(f"x.shape {x.shape}")
    print(f"y.shape {y.shape}")

    # model
    units = 64
    n_layers = 4
    layers = (
        [
            LSTM(
                units,
                activation="relu",
                return_sequences=True,
                input_shape=(learn_res, 1),
            )
        ]  # (timesteps, features)
        + [
            LSTM(units, activation="relu", return_sequences=True)
            for _ in range(n_layers - 2)
        ]
        + [
            LSTM(units, activation="relu", return_sequences=False),
            Dense(1, activation="sigmoid"),
        ]
    )
    model = Sequential(layers)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    # train
    history = model.fit(
        x, y, validation_split=0.2, epochs=16, batch_size=256, verbose=1
    )
    return model, history


def run_an_example_and_plot_info():
    n = 2**15  # samples
    learn_res = 8  # with of input samples (x)
    sample_end = 0  # where does X data stop
    t_pred = 1  # where to measure y

    model, history = train(n, learn_res, sample_end, t_pred)

    plt.subplot(2, 1, 1)
    # Plot training & validation accuracy values
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")

    plt.subplot(2, 1, 2)
    # Plot training & validation loss values
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")

    # having fun with our model
    plt.figure()
    n = 8
    plot_res = 1000
    t_plot = np.linspace(-1, 1, plot_res)
    t_sample = np.linspace(-1, sample_end, learn_res)
    for i in range(n):
        poly = get_poly_with_res(i % 2, model, t_sample, t_pred)
        X, Y = get_sample(poly, t_sample, t_pred)
        X = np.reshape(X, (learn_res, 1))
        pred = int(model.predict(np.array([X]))[0][0] > 0.5)
        plt.subplot(int(n / 2), 2, i + 1)
        plt.plot(t_plot, poly(t_plot), "k--")
        if pred == Y:
            col = "g"
        else:
            col = "r"
        plt.plot(t_pred, poly(t_pred), col + "X")
        plt.plot(t_sample, X, "b")
        plt.title("True label: {} | Predicted: {}".format(Y, pred))
        plt.grid(True)
        plt.axhline(y=0, color="k")

    # The End
    plt.show()


def see_accuracy_per_learn_res():
    n = 2**10  # samples
    sample_end = 0.5  # where does X data stop
    t_pred = 1  # where to measure y
    # -----------
    n_res = 10  # how often to run with each res
    learn_resolutions = range(1, 10)
    res = [list()] * n_res
    for i_n, learn_res in product(range(n_res), learn_resolutions):
        model, history = train(n, learn_res, sample_end, t_pred)
        acc = history.history["val_accuracy"][-1]
        res[i_n].append(acc)
    plt.violinplot(res)
    plt.show()


if __name__ == "__main__":
    run_an_example_and_plot_info()
    # see_accuracy_per_learn_res()
