import os
import tensorflow as tf
from tensorflow.python.framework import ops
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


@ops.RegisterGradient('ShiftConv')
def blocksparse_matmul_grad(op, dy):
    shift_h = op.get_attr('shift_h')
    shift_w = op.get_attr('shift_w')
    x = op.inputs[0]
    w = op.inputs[1]
    dx = module.shift_conv_dx(dy, w, shift_h=shift_h, shift_w=shift_w)
    dw = module.shift_conv_dw(dy, x, shift_h=shift_h, shift_w=shift_w)
    return (dx, dw)

def run_shift():
    B, C, H, W = 1, 16, 4, 4
    R, S, F = 3, 3, 16
    np.random.seed(2)
    a = tf.placeholder(tf.float32, shape=[C, H, W, B])
    b = tf.placeholder(tf.float32, shape=[C, F])
    #hshift_h = np.random.randint(- (R//2), R//2 + 1, size=C, dtype=np.int32)
    #hshift_w = np.random.randint(- (S//2), R//2 + 1, size=C, dtype=np.int32)
    hshift_h = -1*np.ones(C, dtype=np.int32)
    hshift_w = -1*np.ones(C, dtype=np.int32)
    print(hshift_h)
    print(hshift_w)
    c = module.shift_conv(a, b, shift_h=tf.make_tensor_proto(hshift_h), shift_w=tf.make_tensor_proto(hshift_w))
    c = tf.math.reduce_sum(c)
    # Reference
    ha = np.ones((C, H, W, B), dtype=np.float32)
    hb = np.ones((C, F), dtype=np.float32)
    #ha = np.ones((C, H, W, B), dtype=np.int32)
    #hb = np.ones((C, F), dtype=np.int32)
    sess = tf.InteractiveSession()
    grads = tf.test.compute_gradient([a, b], [(C, H, W, B), (C, F)], c, (1,),
                                    extra_feed_dict={a: ha, b: hb})
    dx_t, dx_n = grads[0]
    dw_t, dw_n = grads[1]
    #print(dw_t - dw_n)
    #np.savetxt('diff.dat', dw_t - dw_n, fmt='%2.4f')
    #np.savetxt('theoretical.dat', dw_t, fmt='%2.4f')
    #np.savetxt('numerical.dat', dw_n, fmt='%2.4f')
    print(np.max(np.abs(dw_t - dw_n)))
    print(np.max(np.abs(dx_t - dx_n)))
    np.savetxt('diff.dat', dx_t - dx_n, fmt='%2.4f')
    np.savetxt('theoretical.dat', dx_t, fmt='%2.4f')
    np.savetxt('numerical.dat', dx_n, fmt='%2.4f')
    # Run
    sess.run(tf.global_variables_initializer())
    result = sess.run([c], feed_dict = {a: ha,
                                        b: hb})[0]
    #print(result)

run_shift()
