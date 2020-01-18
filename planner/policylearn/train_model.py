#!/usr/bin/env python3

import logging
import pickle
import sys
from datetime import datetime

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

from generate_data import OTHERS_STR, OWN_STR
from gaussian_map_layer import GaussianMapLayer


def get_training_steps(training_data, start_step, batch_size, num_input, num_timesteps):
    xy_batch = training_data[start_step:(start_step+batch_size)]
    X_out = []
    Y_out = []
    timesteps = []
    for i_b in range(batch_size):
        timesteps.append(len(xy_batch[i_b][0][OWN_STR]))
    assert num_timesteps > max(timesteps),  "must have at least the timesteps"
    for i_b in range(batch_size):
        x = []
        this_timesteps = len(xy_batch[i_b][0][OWN_STR])
        # padding, tipp by https://github.com/keras-team/keras/issues/85#issuecomment-96425996
        pad = num_timesteps - this_timesteps
        for i_t in range(pad):
            x.append(np.zeros(num_input))
        for i_t in range(this_timesteps):
            xt = xy_batch[i_b][0][OWN_STR][i_t]
            for i_a in range(len(xy_batch[i_b][0][OTHERS_STR])):
                xt = np.append(xt, xy_batch[i_b][0][OTHERS_STR][i_a][i_t])
            x.append(xt)
        X_out.append(x)
        Y_out.append([xy_batch[i_b][1], ])
    return X_out, Y_out


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # tf Logging
    logdir = datetime.now().strftime("logs/%Y%m%d-%H%M%S")

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
    training_precentage = .9
    training_steps = int(len(training_data) * training_precentage)
    test_steps = len(training_data) - training_steps
    batch_size = 100
    display_step = 1

    num_inputs_other = training_data[0][0][OTHERS_STR][0][0].shape[0]
    num_others = len(training_data[0][0][OTHERS_STR])
    num_inputs_self = training_data[0][0][OWN_STR][0].shape[0]
    num_com_channels = 3  # how many colors has the image  TODO: use blurmap
    num_hidden = num_inputs_self * 2  # hidden layers other and self
    num_classes = 1  # one class for 0. .. 1.
    num_timesteps = 20  # TODO: get from generated data
    num_input = num_inputs_self + num_others * num_inputs_other

    # blurmap size
    map_width = 10
    map_height = 10  # TODO: read from somewhere

    X = tf.compat.v1.placeholder(
        tf.float32, [batch_size, num_timesteps, num_input], name="X")
    Y = tf.compat.v1.placeholder(
        tf.float32, [batch_size, num_classes], name="Y")

    weights = {
        'out': tf.Variable(tf.random.normal([num_hidden, num_classes]), name="weights")
    }
    biases = {
        'out': tf.Variable(tf.random.normal([num_classes]), name="biases")
    }

    def LSTM(x, weights, biases):
        x = tf.unstack(x, None, 1)

        # Define a lstm cell with tensorflow
        map_layer = GaussianMapLayer(
            num_inputs_other,
            num_inputs_self,
            num_others,
            num_com_channels,
            num_hidden,
            map_width,
            map_height)

        # Get lstm cell output
        # outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
        layer = tf.keras.layers.RNN(map_layer)

        # maybe: https://www.tensorflow.org/guide/keras/rnn#define_a_custom_cell_that_support_nested_inputoutput

        for i_t in range(len(x)):
            inputs = x[i_t]
            if i_t == 0:
                state = map_layer.get_initial_state(inputs, batch_size)
            outputs, state = layer.call(inputs, state)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']

    pred_cont = LSTM(X, weights, biases)  # continous variable predicted
    pred_disc = tf.math.round(pred_cont)  # discrete prediction

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.square(Y - pred_cont))
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(pred_disc, Y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.compat.v1.global_variables_initializer()

    # Start training
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)

        assert 0 == training_steps % batch_size, "training steps must be divisible by batch size"
        for batch_step in range(int(training_steps / batch_size)):
            start_step = batch_step * batch_size
            batch_x, batch_y = get_training_steps(
                training_data, start_step, batch_size,
                num_input, num_timesteps)
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x,
                                          Y: batch_y})
            if batch_step % display_step == 0 or batch_step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy],
                                     feed_dict={X: batch_x,
                                                Y: batch_y})
                print("Step " + str(batch_step) + ", Minibatch Loss= " +
                      "{:.4f}".format(loss) + ", Training Accuracy= " +
                      "{:.3f}".format(acc))

        print("Optimization Finished!")
        file_writer = tf.compat.v1.summary.FileWriter(logdir, sess.graph)

        # Calculate accuracy
        test_x, test_y = get_training_steps(
            training_data, training_steps, batch_size,
            num_input, num_timesteps)
        print("Testing Accuracy:",
              sess.run(accuracy, feed_dict={X: test_x, Y: test_y}))
