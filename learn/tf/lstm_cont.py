""" Recurrent Neural Network.
A Recurrent Neural Network (LSTM) implementation example using TensorFlow
library. This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Links:
    [Long Short Term Memory](
        http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Logging
logdir = "learn/tf/logs/"+datetime.now().strftime("%Y%m%d-%H%M%S")

# Import MNIST data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Training Parameters
learning_rate = 0.01
training_steps = 1000
batch_size = 128
display_step = 100

# Network Parameters
num_input = 28  # MNIST data input (img shape: 28*28)
# timesteps = 28 # timesteps
num_hidden = 64  # hidden layer num of features
num_classes = 1  # one class for 0 .. 9

# tf Graph input
X = tf.placeholder("float", [None, None, num_input], name="X")
Y = tf.placeholder("float", [None, num_classes], name="Y")

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]),
                       name="weights")
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]), name="biases")
}


def LSTM(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape
    # (batch_size, n_input)
    timesteps = 28
    x = tf.unstack(x, timesteps, 1)
    x = x[:-2]  # training with one row less

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

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


def argmax_label(y):
    return [[np.argmax(line)] for line in y]


# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    # defining "at runtime"
    timesteps = 28

    for step in range(1, training_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_y = argmax_label(batch_y)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
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
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape(
        (-1, timesteps, num_input))
    test_label = mnist.test.labels[:test_len]
    test_label = argmax_label(test_label)
    print("Testing Accuracy:",
          sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))

    # trying
    for dat in mnist.test.images[:2]:
        pred = sess.run(prediction, feed_dict={
                        X: dat.reshape((-1, timesteps, num_input))})[0][0]
        plt.imshow(dat.reshape([28, 28]))
        plt.title("pred: " + str(pred))
        plt.show()
