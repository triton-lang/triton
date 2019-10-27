import numpy as np
import triton

def run_tf():
  M, N, K = 2048, 2048, 2048
  a = tf.placeholder(tf.float32, shape=[M, K])
  b = tf.placeholder(tf.float32, shape=[N, K])
  tr_c = triton.ops.dot(a, b, transpose_a = False, transpose_b = True, bench=10)
  tr_d = triton.ops.dot(tr_c, b, transpose_a = True, transpose_b = False, bench=10)
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
  # Benchmark
  nanosec = triton.bench_registry[tr_d]
  print('NANOSEC: ', nanosec)
  print('TFLOPS:', 2. * M * N * K / nanosec * 1e-3)
  # Test
  print(result[0][0])
  print(result[1][0])
  dif = np.abs(result[0][0] - result[1][0])
  print("dif: %f" % np.max(dif))


def run_torch():
  torch.manual_seed(0)
  M, N, K = 2048, 2048, 2048
  a = torch.randn(M, K).cuda()
  b = torch.randn(K, N).cuda()
  a.requires_grad_(True)
  b.requires_grad_(True)
  torch_c = torch.matmul(a, torch.t(b))
  torch_d = torch.matmul(torch.t(torch_c), b)
  torch_y = torch.mean(torch_d)
  triton_c = triton.ops.dot(a, b, False, True)
  triton_d = triton.ops.dot(triton_c, b, True, False, 1)
  triton_y = torch.mean(triton_d)
  # torch gradient
  torch_y.backward()
  torch_da = a.grad.clone()
  torch_db = b.grad.clone()
  # triton gradient
  a.grad.zero_()
  b.grad.zero_()
  triton_y.backward()
  triton_da = a.grad.clone()
  triton_db = b.grad.clone()

  nanosec = triton.bench_registry[triton_d]
  print(nanosec)
  print('TFLOPS:', 2. * M * N * K / nanosec * 1e-3)
  print('Diff DA:', (torch_da - triton_da).max())
  print('Diff DB:', (torch_db - triton_db).max())

try:
  import tensorflow as tf
  run_tf()
except ModuleNotFoundError:
  pass

try:
  import torch
  run_torch()
except ModuleNotFoundError:
  pass
