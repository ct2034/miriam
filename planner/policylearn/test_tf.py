import unittest
import tensorflow as tf
from tensorflow.python.platform import build_info as tf_build_info

print(tf_build_info.cuda_version_number)
print(tf_build_info.cudnn_version_number)

@unittest.skip("currently not working on tf nor gps")
def test_gpu():
    assert tf.test.is_gpu_available()