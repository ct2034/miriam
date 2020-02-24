import tensorflow as tf
from tensorflow.python.platform import build_info as tf_build_info

print(tf_build_info.cuda_version_number)
print(tf_build_info.cudnn_version_number)

assert tf.test.is_gpu_available()