from collections import namedtuple

import tensorflow as tf


class GaussianMapLayer(tf.keras.layers.Layer):
    """
    (from tf.keras.layers.Layer)
    * `__init__()`: Save configuration in member variables
    * `build()`: Called once from `__call__`, when we know the shapes of inputs
    and `dtype`. Should have the calls to `add_weight()`, and then
    call the super's `build()` (which sets `self.built = True`, which is
    nice in case the user wants to call `build()` manually before the
    first `__call__`).
    * `call()`: Called in `__call__` after making sure `build()` has been called
    once. Should actually perform the logic of applying the layer to the
    input tensors (which should be passed in as the first argument)."""

    def __init__(self,
                 num_inputs_other: int,
                 num_inputs_self: int,
                 num_others: int,
                 num_com_channels: int,
                 num_hidden: int,
                 map_width: int,
                 map_height: int,
                 **kwargs):
        self.num_inputs_other = num_inputs_other
        self.num_inputs_self = num_inputs_self
        self.num_others = num_others
        self.num_com_channels = num_com_channels
        self.num_hidden = num_hidden
        num_classes = 1  # we have one output

        self.NestedInput = namedtuple('NestedInput', [
                                      'feature_self', ] + ['feature_other' +
                                                           str(i) for i in
                                                           range(num_others)])
        self.NestedState = namedtuple('NestedState', [
                                      'state_self', ] + ['state_other' +
                                                         str(i) for i in
                                                         range(num_others)])

        self.blurmap = tf.zeros([map_width, map_height, num_com_channels])

        self.weights_other_to_map = tf.Variable(
            tf.random_normal([num_hidden, num_com_channels]))
        self.biases_other_to_map = tf.Variable(
            tf.random_normal([num_com_channels]))

        self.weights_self_out = tf.Variable(
            tf.random_normal([num_hidden, num_classes]))
        self.biases_self_out = tf.Variable(tf.random_normal([num_classes]))

        state_sizes = (num_hidden,) * (num_others + 1)
        self.state_size = self.NestedState(*state_sizes)
        self.output_size = 1
        super(GaussianMapLayer, self).__init__(**kwargs)

    def build(self, input_shapes):
        input_self = input_shapes[0]
        input_others = input_shapes[1:]

        self.cell_self = tf.keras.layers.LSTMCell(self.num_hidden)
        self.cells_others = [tf.keras.layers.LSTMCell(
            self.num_hidden)] * self.num_others

        self.cell_self.build(input_self)
        for i_a in range(self.num_others):
            self.cells_others[i_a].build(input_others[i_a])

    def call(self, inputs, states):
        inputs_self = inputs[0]
        inputs_others = inputs[1:]
        states_self = states[0]
        states_others = states[1:]

        output_self, new_state_self = self.cell_self.call(
            inputs_self, states_self)

        pos_self = inputs_self[2:]  # last two data fields have pose
        self.blurmap[pos_self] = output_self

        outputs_and_new_states_others = [self.cells_others[i_a].call(
            inputs_others[i_a], states_others[i_a])
            for i_a in range(self.num_others)]

        return output, new_states

    def _getGaussValue(self, kerStd, posX, posY):
        return (1./(2. *
                    math.pi*(np.power(kerStd, 2))) *
                math.exp(-(np.power(posX, 2)+np.power(posY, 2)) /
                         (2.*(np.power(kerStd, 2)))))

    def _getGaussKernel(self, kerStd, datSize):
        d = int(6*kerStd)
        d_idxs = range(int(-d), int(d+1), 1)
        kerSize = 2 * d + 1
        kernel = np.zeros([kerSize, kerSize, datSize, datSize])

        for ix, iy in product(range(kerSize), repeat=2):
            dx = d_idxs[ix]
            dy = list(reversed(d_idxs))[iy]
            kernel[ix, iy] = (np.eye(datSize) *
                self._getGaussValue(kerStd, dx, dy))

        return tf.constant(kernel, dtype=tf.float32)

    def _blur(self, g, imageData, kernel):
        if imageData.dtype is not tf.float32:
            imageData = tf.cast(imageData, dtype=tf.float32)
        y = tf.cast(tf.nn.conv2d(imageData, kernel, strides=[
            1, 1, 1, 1], padding="SAME"), dtype=tf.int32)
        init_op = tf.global_variables_initializer()
        with tf.Session(graph=g) as sess:
            return sess.run(y)
