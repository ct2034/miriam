#!/usr/bin/env python3

import logging
import pickle
import sys
from datetime import datetime

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

from generate_data import OTHERS_STR, OWN_STR


def get_training_step(training_data, step):
    X, y = training_data[step]
    x = []
    for i_t in range(len(X[OWN_STR])):
        xt = X[OWN_STR][i_t]
        for i_a in range(len(X[OTHERS_STR])):
            xt = np.append(xt, X[OTHERS_STR][i_a][i_t])
        x.append(xt)
    return x, [y]  # TODO: bigger batches


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # tf Logging
    logdir = datetime.now().strftime("%Y%m%d-%H%M%S")

    fname_pkl = sys.argv[-1]
    assert fname_pkl.endswith(".pkl"), "to read the training data,\
        give filname of pickle file as argument"

    training_data = []
    with open(fname_pkl, 'rb') as f:
        training_data = pickle.load(f)

    assert len(training_data), "No training data loaded"
    logger.info("Training Data len: {}".format(len(training_data)))

    # --------------------------------------------------------------------------------
    # model
    learning_rate = 0.01
    training_steps = int(len(training_data) * .8)
    validation_steps = len(training_data) - training_steps
    # batch_size = 10
    display_step = 100

    num_inputs_other = training_data[0][0][OTHERS_STR][0][0].shape[0]
    num_others = len(training_data[0][0][OTHERS_STR])
    num_inputs_self = training_data[0][0][OWN_STR][0].shape[0]
    num_input = num_inputs_self + num_others * num_inputs_other  # TODO: use blurmap
    num_com_channels = 3  # how many colors has the image  TODO: use blurmap
    num_hidden = 64  # hidden layers other and self
    num_classes = 1  # one class for 0. .. 1.

    timesteps = None  # to be defined at runtime ...

    X = tf.placeholder("float", [timesteps, num_input], name="X")
    Y = tf.placeholder("float", [num_classes], name="Y")

    weights = {
        'out': tf.Variable(tf.random_normal([num_hidden, num_classes]), name="weights")
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_classes]), name="biases")
    }

    def LSTM(x, weights, biases):    
        # Define a lstm cell with tensorflow
        lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, [x], dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']

    pred_cont = LSTM(X, weights, biases)
    prediction = tf.math.round(pred_cont)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.square(Y - pred_cont))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(prediction, Y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)

        for step in range(training_steps):
            batch_x, batch_y = get_training_step(training_data, step)
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " +
                      "{:.4f}".format(loss) + ", Training Accuracy= " +
                      "{:.3f}".format(acc))

        print("Optimization Finished!")
        file_writer = tf.compat.v1.summary.FileWriter(logdir, sess.graph)

        # Calculate accuracy for 128 mnist test images
        # test_len = 128
        # test_data = mnist.test.images[:test_len].reshape(
        #     (-1, timesteps, num_input))
        # test_label = mnist.test.labels[:test_len]
        # test_label = argmax_label(test_label)
        # print("Testing Accuracy:",
        #       sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))

