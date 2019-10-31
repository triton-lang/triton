import numpy as np
import triton

def run_tf():
  M, N, K = 2048, 2048, 2048
  a = tf.placeholder(tf.float32, shape=[M, K])
  b = tf.placeholder(tf.float32, shape=[N, K])
  triton_c = triton.ops.dot(a, b, False, True, 1)
  triton_d = triton.ops.dot(triton_c, b, True, False, 1)
  triton_y = tf.math.reduce_mean(triton_d)
  fw_c = tf.matmul(a, b, False, True)
  fw_d = tf.matmul(fw_c, b, True, False)
  fw_y = tf.math.reduce_mean(fw_d)
  # Gradient
  triton_da, triton_db = tf.gradients(triton_y, [a, b])
  fw_da, fw_db = tf.gradients(fw_y, [a, b])
  # Reference
  feed_dict = {a: np.random.rand(M, K).astype(np.float32),
               b: np.random.rand(K, N).astype(np.float32)}
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())
  result = sess.run([triton_da, fw_da, triton_db, fw_db, fw_y, triton_y], feed_dict = feed_dict)
  triton_da, fw_da = result[0][0], result[1][0]
  triton_db, fw_db = result[2][0], result[3][0]
  # Benchmark
  nanosec = triton.bench_registry[triton_d]
  print('TFLOPS:', 2. * M * N * K / nanosec * 1e-3)
  print('Diff DA:', (triton_da - fw_da).max())
  print('Diff DB:', (triton_db - fw_db).max())


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
  triton_c = triton.ops.dot(a, b, False, True, 1)
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

  #nanosec = triton.bench_registry[triton_d]
  #print('TFLOPS:', 2. * M * N * K / nanosec * 1e-3)
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
