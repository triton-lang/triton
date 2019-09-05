import numpy as np
import triton

def run_tf():
  import tensorflow as tf
  M, N, K = 128, 128, 128
  a = tf.placeholder(tf.float32, shape=[M, K])
  b = tf.placeholder(tf.float32, shape=[N, K])
  tr_c = triton.ops.dot(a, b, transpose_a = False, transpose_b = True)
  tr_d = triton.ops.dot(tr_c, b, transpose_a = True, transpose_b = False)
  tf_c = tf.matmul(a, b, transpose_a = False, transpose_b = True)
  tf_d = tf.matmul(tf_c, b, transpose_a = True, transpose_b = False)
  # Gradient
  tr_da = tf.gradients(tr_d, [a])
  tf_da = tf.gradients(tf_d, [a])
  # Reference
  ha = np.random.rand(M, K).astype(np.float32)
  hb = np.random.rand(K, N).astype(np.float32)
  # Run
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())
  result = sess.run([tr_da, tf_da], feed_dict = {a: ha,
                                      b: hb})
  # Test
  print(result[0][0])
  print(result[1][0])
  dif = np.abs(result[0][0] - result[1][0])
  print("dif: %f" % np.max(dif))

def run_torch():
  import torch as th
  th.manual_seed(0)
  M, N, K = 128, 128, 128
  a = th.randn(M, K).cuda()
  b = th.randn(K, N).cuda()
  b.requires_grad_(True)
  #th_c = th.matmul(a, th.t(b))
  #th_d = th.matmul(th.t(th_c), b)
  tr_c = triton.ops.dot(a, b, False, True)
  #tr_d = triton.ops.dot(tr_c, b, True, False)
  y = th.sum(tr_c)
  #print('backprop', y)
  y.backward()
  #print('backward done')
  print(b.grad)
  #th_d.backward()
  #print(a.grad)


run_torch()