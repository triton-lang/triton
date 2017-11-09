import tensorflow as tf
import isaac as sc
import numpy as np
import itertools

isaac = tf.load_op_library(sc.tensorflow)

N, C, H, W =  1, 512, 64, 64
R, S, K = 3, 3, 512

A = tf.placeholder(tf.float32, [N, C, H, W])
filters = tf.placeholder(tf.float32, [R, S, K, C])
filters_isaac = tf.placeholder(tf.float32, [C, R, S, K])
y_tf = tf.nn.conv2d(input=A, filter=filters, strides=[1, 1, 1, 1], padding="SAME", data_format="NCHW")
y_sc = isaac.conv(input=A, filter=filters_isaac, strides=[1, 1, 1, 1], padding="SAME", data_format="NCHW")

# Session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
# Random tensors
A_random = tf.random_normal(shape=[N, C, H, W], seed=1)
filters_random = tf.random_normal(shape=[R, S, K, C], seed=1)
# Compute
A_in = sess.run(A_random)
filters_in = sess.run(filters_random)
z_tf = sess.run(y_tf, feed_dict={A: A_in, filters: filters_in})
z_sc = sess.run(y_sc, feed_dict={A: A_in, filters_isaac: np.transpose(filters_in, [2,0,1,3])})
error = np.linalg.norm(z_tf - z_sc)/np.max(z_tf)
print('Relative error:', error)

