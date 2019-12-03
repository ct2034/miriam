""" Recurrent Neural Network.
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import random

# Training Parameters
learning_rate = 0.001
training_steps = 1000
batch_size = 128
display_step = 10

# Network Parameters
num_inputs = 1 # one y value per timestep
timesteps = 1000 # timesteps
num_hidden = 128 # hidden layer num of features
num_outputs = 1 # function value

t = np.linspace(0., 2*np.pi, timesteps-1)[:-1]
s = np.sin(t) + np.sin(2*t - 3/2) + np.sin(3*t - 4)

def get_sample():
    d = random.randint(0, timesteps-1)
    x = s[d:] + s[:d]
    y = s[d]
    return tf.constant(x), tf.constant(y)

def get_batch(n):
    xs = []
    ys = []
    for _ in range(n):
        x, y = get_sample()
        xs.append(x)
        ys.append(y)
    return tf.constant(xs), tf.constant(ys) 

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_inputs])
Y = tf.placeholder("float", [None, num_outputs])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_outputs]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_outputs]))
}


def RNN(x, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

prediction = RNN(X, weights, biases)

# Define loss and optimizer
loss_op = tf.pow(tf.subtract(Y, prediction), 2)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):
        batch_x, batch_y = get_batch(batch_size)

        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_inputs))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))