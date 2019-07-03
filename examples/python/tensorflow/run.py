import os
import tensorflow as tf
import numpy as np
from time import time
data_files_path = tf.resource_loader.get_data_files_path()
library_dir = os.path.dirname(os.path.realpath(__file__))
module = tf.load_op_library(os.path.join(library_dir, 'libtf_blocksparse.so'))

def run_dot():
    M, N, K = 128,128,128
    a = tf.placeholder(tf.float16, shape=[M, K])
    b = tf.placeholder(tf.float16, shape=[N, K])
    locks = tf.placeholder(tf.int32, shape=[4096])
    # c = tf.matmul(a, b, transpose_a=True)
    c = module.dot(a, b, locks)
    # Reference
    ha = np.random.rand(M, K).astype(np.float16)
    hb = np.random.rand(N, K).astype(np.float16)
    # Run
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    result = sess.run([c], feed_dict = {locks: np.zeros(4096),
                                        a: ha,
                                        b: hb})[0]
    # Test
    hresult = np.dot(ha.T, hb).T
    dif = np.abs(result - hresult)
    print("dif: %f" % np.max(dif))

def run_conv():
    B, C, H, W = 16, 32, 32, 32
    R, S, NF = 3, 3, 32
    a = tf.placeholder(tf.float32, shape=[B, C, H, W])
    b = tf.placeholder(tf.float32, shape=[C, R, S, NF])
    c = module.conv2d(a, b)
    # Reference
    ha = np.random.rand(B, C, H, W)
    hb = np.random.rand(C, R, S, NF)
    # Run
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    result = sess.run([c], feed_dict = {a: ha,
                                        b: hb})[0]

def run_shift():
    B, C, H, W = 16, 32, 32, 32
    R, S, F = 3, 3, 32
    a = tf.placeholder(tf.float32, shape=[C, H, W, B])
    b = tf.placeholder(tf.float32, shape=[C, F])
    shift_h = tf.zeros(C, tf.int32)
    shift_w = tf.zeros(C, tf.int32)
    hshift_h = np.zeros(C, np.int32)
    hshift_w = np.zeros(C, np.int32)
    c = module.shift_conv(a, b, shift_h=tf.make_tensor_proto(hshift_h), shift_w=tf.make_tensor_proto(hshift_w))
    # Reference
    ha = np.random.rand(C, H, W, B)
    hb = np.random.rand(C, F)
    # Run
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    result = sess.run([c], feed_dict = {a: ha,
                                        b: hb})[0]

run_shift()
