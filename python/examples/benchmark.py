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
shapes = [([7, 7], 512, 8, 512, [3, 3])]
print('Shapes\t\t\t\tISAAC\tcuDN\tError')
for shape in shapes:
    # Extract shapes
    Ishape, C, N, K, Fshape = shape
    dim = len(Ishape)
    fmt = {2: 'NCHW', 3: 'NCDHW'}[dim]
    tf_op = {2: tf.nn.conv2d, 3: tf.nn.conv3d}[dim]
    sc_op = {2: isaac.conv2d, 3: isaac.conv3d}[dim]
    # Graph
    A = tf.Variable(tf.random_uniform(shape=[N, C] + Ishape, seed=1), dtype=tf.float32)
    filters = tf.Variable(tf.random_uniform(shape= Fshape + [C, K], seed=1), dtype=tf.float32)
    trans_filters = tf.Variable(tf.transpose(filters, [dim] + list(range(dim)) + [dim + 1]))
    y_tf = tf_op(input=A, filter=filters, strides=[1]*(2 + dim), padding="SAME", data_format=fmt)
    y_sc = sc_op(input=A, filter=trans_filters, strides=[1]*(2 + dim), padding="SAME", data_format=fmt)
    # Initialize
    sess.run(tf.global_variables_initializer())
    # Compute
    z_sc = sess.run(y_sc)
    z_tf = sess.run(y_tf)
    error = np.linalg.norm(z_tf - z_sc) /  np.linalg.norm(z_tf)
    t_sc = timeit.repeat(lambda: sess.run(tf.group(y_sc)), repeat=10, number=1)
    t_tf = timeit.repeat(lambda: sess.run(tf.group(y_tf)), repeat=10, number=1)
    # Log
    num_ops = 2*C*N*K*np.prod(Ishape)*np.prod(Fshape)*1e-12
    print('{}\t{:.2f}\t{:.2f}\t{:.3f}'.format(shape, min(t_sc)*1e3, min(t_tf)*1e3, error))

