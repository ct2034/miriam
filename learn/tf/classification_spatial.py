#!/usr/bin/env python3

import argparse
import pickle
import random
from itertools import product

import cachier
import networkx as nx
import numpy as np
import sim
import tensorflow as tf
from matplotlib import pyplot as plt
from scenarios.evaluators import is_well_formed
from scenarios.generators import FREE, random_fill
from sim.decentralized.agent import gridmap_to_nx
from tensorflow.keras import initializers, layers, regularizers
from tensorflow.keras.models import Sequential
from tools import hasher


@cachier.cachier(hash_params=hasher)
def make_sample(size, fill_scale, seed):
    random.seed(seed)
    np.random.seed(int(seed * 1000))
    fill = random.random() * fill_scale
    env = sim.decentralized.runner.initialize_environment(size, fill)
    free = np.array(np.where(env == FREE))
    env_nx = gridmap_to_nx(env)
    le = len(free[0])
    smpls = np.random.randint(0, le, (2))
    start = np.transpose(free[:, smpls[0]])
    goal = np.transpose(free[:, smpls[1]])
    Y = int(nx.has_path(env_nx, tuple(start), tuple(goal)))
    X = np.zeros(env.shape + (2,), dtype=np.int8)
    X[:, :, 0] = env
    X[start[0], start[1], 1] = 1
    X[goal[0], goal[1], 1] = 1
    return X, Y


@cachier.cachier()
def make_data(n):
    x = []
    y = []
    prev_perc = 0
    for i in range(n):
        perc = int(i / n * 100.0)
        if perc > prev_perc:
            print(str(perc) + "%")
        prev_perc = perc
        X, Y = make_sample(8, 0.6, i)
        x.append(X)
        y.append(Y)

    print("Y fill: {}%".format(100 * sum(y) / n))
    return x, y


def train(x, y):
    # model
    layer_width = 64
    init = initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
    reg = regularizers.l2(0.001)
    ls = [
        layers.Conv2D(
            8,
            5,
            input_shape=x[0].shape,
            kernel_initializer=init,
            activity_regularizer=reg,
        ),
        layers.Dropout(0.3),
        layers.Conv2D(8, 3, kernel_initializer=init, activity_regularizer=reg),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(
            layer_width,
            kernel_initializer=init,
            activity_regularizer=reg,
            activation="relu",
        ),
        layers.Dense(
            1, kernel_initializer=init, activity_regularizer=reg, activation="sigmoid"
        ),
    ]
    model = Sequential(ls)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    # train
    history = model.fit([x], y, validation_split=0.3, epochs=512, batch_size=128)
    return model, history


def run_an_example_and_plot_info():
    n = 10000  # samples
    x, y = make_data(n)

    model, history = train(x, y)

    plt.subplot(211)
    # Plot training & validation accuracy values
    plt.plot(history.history["acc"])
    plt.plot(history.history["val_acc"])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")

    plt.subplot(212)
    # Plot training & validation loss values
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")

    # having fun with our model
    plt.figure()
    n_p = 8
    for i in range(n_p):
        success_desired = i < n_p / 2
        success = not success_desired
        Y_desired = np.floor(i / 4) % 2 == 0
        Y = not Y_desired
        while success != success_desired or Y != Y_desired:
            X, Y = make_sample(8, 0.6, random.randint(1, 1000))
            pred = model.predict_classes(np.array([X]))[0][0]
            success = Y == pred
        plt.subplot(2, n_p / 2, i + 1)
        # map
        swapped_map = np.swapaxes(X[:, :, 0], 0, 1)
        plt.imshow(swapped_map, cmap="Greys", origin="lower")
        # path / points
        start_goal = np.array(np.where(X[:, :, 1] == 1))
        if pred == Y:
            col = "g"
        else:
            col = "r"
        if Y == 1 and start_goal.shape[1] == 2:  # has a path
            env_nx = gridmap_to_nx(X[:, :, 0])
            p = nx.shortest_path(
                env_nx, tuple(start_goal[:, 0]), tuple(start_goal[:, 1])
            )
            path = np.array(p)
            plt.plot(path[:, 0], path[:, 1], col)
        elif Y == 1 and start_goal.shape[1] == 1:
            plt.plot(start_goal[0], start_goal[1], col + "x")
        else:
            plt.plot(start_goal[0], start_goal[1], col + "x")
        plt.title("True label: {} | Predicted: {}".format(Y, pred))

    # # The End
    plt.show()


if __name__ == "__main__":
    run_an_example_and_plot_info()
