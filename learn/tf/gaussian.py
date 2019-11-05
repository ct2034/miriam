import tensorflow as tf
from PIL import Image
import numpy as np
import math
from itertools import product


def getGaussValue(kerStd, posX, posY):
    return 1./(2.*math.pi*(np.power(kerStd, 2)))*math.exp(-(np.power(posX, 2)+np.power(posY, 2))/(2.*(np.power(kerStd, 2))))


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
        testImage.show()
        imageData.append(np.array(testImage))
    return np.array(imageData, dtype=np.float32)


def blur(g, y):
    with tf.Session(graph=g) as sess:
        return sess.run(y)[0]


g = tf.Graph()
with g.as_default():
    imageData = getImageData(("tf/dog.png",))
    testData = tf.constant(imageData)
    kernel = getGaussKernel(2.0, 11, 3)
    y = tf.cast(tf.nn.conv2d(testData, kernel, strides=[
                1, 1, 1, 1], padding="VALID"), dtype=tf.int32)
    init_op = tf.global_variables_initializer()

print(testData.get_shape())
resultData = blur(g, y)
resulImage = Image.fromarray(np.uint8(resultData))
resulImage.show()