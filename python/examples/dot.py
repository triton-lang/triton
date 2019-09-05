import numpy as np
import tensorflow as tf
import triton

def run_dot():
  M, N, K = 128, 128, 128
  a = tf.placeholder(tf.float32, shape=[M, K])
  b = tf.placeholder(tf.float32, shape=[N, K])
  _dot = triton.ops.dot.apply
  tr_c = _dot(a, b, transpose_a = False, transpose_b = True)
  tr_d = _dot(tr_c, b, transpose_a = True, transpose_b = False)
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

run_dot()