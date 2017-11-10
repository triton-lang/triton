import time
import numpy as np
import tensorflow as tf
import isaac as sc
import timeit
from tensorflow.python.client import timeline

isaac = tf.load_op_library(sc.tensorflow)


# Session
sess = tf.Session()
options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

# Shapes to benchmark
shapes = [(7, 7, 512, 8, 512, 3, 3)]
print('Shapes\t\t\t\tISAAC\tcuDNN')
for shape in shapes:
    (W, H, C, N, K, R, S) = shape
    # Graph
    A = tf.Variable(tf.random_uniform(shape=[N, C, H, W], seed=1), dtype=tf.float32)
    filters = tf.Variable(tf.random_uniform(shape=[R, S, C, K], seed=1), dtype=tf.float32)
    trans_filters = tf.Variable(tf.transpose(filters, [2, 0, 1, 3]))
    y_tf = tf.nn.conv2d(input=A, filter=filters, strides=[1, 1, 1, 1], padding="SAME", data_format='NCHW')
    y_sc = isaac.conv(input=A, filter=trans_filters, strides=[1, 1, 1, 1], padding="SAME", data_format='NCHW')
    # Initialize
    sess.run(tf.global_variables_initializer())
    # Compute
    z_sc = sess.run(y_sc)
    z_tf = sess.run(y_tf)
    error = np.linalg.norm(z_tf - z_sc) /  np.linalg.norm(z_tf)
    t_sc = timeit.repeat(lambda: sess.run(tf.group(y_sc)), repeat=10, number=1)
    t_tf = timeit.repeat(lambda: sess.run(tf.group(y_tf)), repeat=10, number=1)
    # Log
    num_ops = 2*N*C*H*W*R*S*K*1e-12
    print('{}\t{:.5f}\t{:.5f}'.format(shape, min(t_sc), min(t_tf)))

