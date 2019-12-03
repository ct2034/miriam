import tensorflow as tf
from PIL import Image
import numpy as np
import math
from itertools import product

# src: https://dmyhaspl.github.io/2018/10/20/Gaussian-blur-2018/


def getGaussValue(kerStd, posX, posY):
    return (1./(2. *
                math.pi*(np.power(kerStd, 2))) *
            math.exp(-(np.power(posX, 2)+np.power(posY, 2)) /
                     (2.*(np.power(kerStd, 2)))))


def getGaussKernel(kerStd, kerSize, datSize):
    assert kerSize % 2 == 1, "use with odd kernel size"
    d = (kerSize-1)/2
    d_idxs = range(int(-d), int(d+1), 1)
    kernel = np.zeros([kerSize, kerSize, datSize, datSize])

    for ix, iy in product(range(kerSize), repeat=2):
        dx = d_idxs[ix]
        dy = list(reversed(d_idxs))[iy]
        kernel[ix, iy] = np.eye(datSize) * getGaussValue(kerStd, dx, dy)

    return tf.constant(kernel, dtype=tf.float32)


def getImageData(fileNameList):
    imageData = []
    for fn in fileNameList:
        testImage = Image.open(fn)
        imageData.append(np.array(testImage))
    return np.array(imageData, dtype=np.float32)


def blur(g, imageData, kernel):
    if imageData.dtype is not tf.float32:
        imageData = tf.cast(imageData, dtype=tf.float32)
    y = tf.cast(tf.nn.conv2d(imageData, kernel, strides=[
                1, 1, 1, 1], padding="SAME"), dtype=tf.int32)
    init_op = tf.global_variables_initializer()
    with tf.Session(graph=g) as sess:
        return sess.run(y)


def blur_repeatedly(n, g, imageData, kernel):
    for _ in range(n):
        imageData = blur(g, imageData, kernel)
    return imageData


def show(dat):
    img = Image.fromarray(np.uint8(dat[0]))
    img.show()


g = tf.Graph()
with g.as_default():
    imageData = getImageData(("tf/map.png",))
    show(imageData)
    imageData = tf.constant(imageData)
    kernel = getGaussKernel(1.2, 9, 3)

    # first 10 run
    resultData10 = blur_repeatedly(10, g, imageData, kernel)
    show(resultData10)

    # second 10 (20)
    resultData20 = blur_repeatedly(10, g, resultData10, kernel)
    show(resultData20)

    # third 10 (30)
    resultData30 = blur_repeatedly(10, g, resultData20, kernel)
    show(resultData30)

    # fourth 10 (40)
    resultData40 = blur_repeatedly(10, g, resultData30, kernel)
    show(resultData40)

    # fifth 10 (50)
    resultData50 = blur_repeatedly(10, g, resultData40, kernel)
    show(resultData50)