import os
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
from time import time

data_files_path = tf.resource_loader.get_data_files_path()
library_dir = os.path.dirname(os.path.realpath(__file__))
module = tf.load_op_library(os.path.join(library_dir, 'libtf_blocksparse.so'))

def run_dot():
    M, N, K = 128, 128, 128
    a = tf.placeholder(tf.float16, shape=[M, K])
    b = tf.placeholder(tf.float16, shape=[N, K])
    # c = tf.matmul(a, b, transpose_a=True)
    c = module.dot(a, b)
    # Reference
    ha = np.random.rand(M, K).astype(np.float16)
    hb = np.random.rand(N, K).astype(np.float16)
    # Run
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    result = sess.run([c], feed_dict = {a: ha,
                                        b: hb})[0]
    # Test
    hresult = np.dot(ha.T, hb).T
    dif = np.abs(result - hresult)
    np.savetxt('dif.dat', dif, '%2.4f')
    print(hresult)
    print(result)
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
    stride_h = op.get_attr('stride_h')
    stride_w = op.get_attr('stride_w')
    x = op.inputs[0]
    w = op.inputs[1]
    dx = module.shift_conv_dx(dy, w, stride_h=stride_h, stride_w=stride_w, shift_h=shift_h, shift_w=shift_w)
    dw = module.shift_conv_dw(dy, x, stride_h=stride_h, stride_w=stride_w, shift_h=shift_h, shift_w=shift_w)
    return (dx, dw)

def run_shift():
    B, C, H, W = 2, 16, 4, 4
    R, S, F = 3, 3, 16
    stride_h, stride_w = 1, 1
    np.random.seed(2)
    a = tf.placeholder(tf.float16, shape=[B, C, H, W])
    b = tf.placeholder(tf.float16, shape=[C, F])
    hshift_h = np.random.randint(- (R//2), R//2 + 1, size=C, dtype=np.int32)
    hshift_w = np.random.randint(- (S//2), R//2 + 1, size=C, dtype=np.int32)
    c = module.shift_conv(a, b, stride_h=stride_h, stride_w=stride_w, shift_h=tf.make_tensor_proto(hshift_h), shift_w=tf.make_tensor_proto(hshift_w))
    # feed values
    ha = np.random.rand(B, C, H, W)*0.1
    hb = np.random.rand(C, F)*0.1
    sess = tf.InteractiveSession()
    # check gradients
    grads = tf.test.compute_gradient([a, b], [(B, C, H, W), (C, F)], c, (B, F, H//stride_h, W//stride_w),
                                     extra_feed_dict = {a: ha, b: hb}, delta=1e-2)
    dw_t, dw_n = grads[1]
    dx_t, dx_n = grads[0]
    #import sys
    #np.set_printoptions(threshold=sys.maxsize)
    print(dw_t)
    print(dw_n)
    print(np.max(np.abs(dw_t - dw_n)))
    print(np.max(np.abs(dx_t - dx_n)))
    # Run
    sess.run(tf.global_variables_initializer())
    result = sess.run([c], feed_dict = {a: ha,
                                        b: hb})[0]
    #print(result)


def batch_norm(x, g, b, epsilon=1e-6):
    shape = x.shape
    C     = int(shape[1])
    assert g.get_shape().num_elements() == C
    assert b.get_shape().num_elements() == C
    return module.batchnorm_forward(x, g, b, eps=epsilon)

@ops.RegisterGradient("BatchnormForward")
def batch_norm_grad(op, dy, mean, var):
    eps = op.get_attr("eps")
    return module.batchnorm_backward(dy, op.inputs[0], op.inputs[1],
                                     op.outputs[1], op.outputs[2], eps=eps)


def run_batchnorm():
    C, H, W, B = 8, 4, 4, 32
    np.random.seed(0)
    # Placeholders
    x = tf.placeholder(tf.float32, shape=[C, H, W, B])
    g = tf.placeholder(tf.float32, shape=[C])
    b = tf.placeholder(tf.float32, shape=[C])
    # Feed values
    hx = np.random.rand(C, H, W, B)
    hg = np.random.rand(C)
    hb = np.random.rand(C)
    # batchnorm
    y, m, v = module.batchnorm_forward(x, g, b, eps=1e-5)
    loss = np.sum(y)
    # Run
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    result = sess.run([y, m, v], feed_dict = {x: hx, g: hg, b: hb})
    grads = tf.test.compute_gradient([x, g, b], [(C, H, W, B), (C, ), (C, )], y, (C, H, W, B),
                                     extra_feed_dict = {x: hx, g: hg, b: hb})
    dx_t, dx_n = grads[0]
    dg_t, dg_n = grads[1]
    db_t, db_n = grads[2]
    print(np.max(np.abs(dx_t - dx_n)))
    print(np.max(np.abs(dg_t - dg_n)))
    print(np.max(np.abs(db_t - db_n)))

run_dot()
#run_shift()
#run_batchnorm()
