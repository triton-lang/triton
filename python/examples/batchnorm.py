import triton
import numpy as np
from enum import Enum

class MODE(Enum):
  TF = 1
  TORCH = 2

try:
  import tensorflow as tf
  mode = MODE.TF
except ModuleNotFoundError:
  pass

try:
  import torch
  mode = MODE.TORCH
except ModuleNotFoundError:
  pass


C, H, W, B = 32, 1, 1, 128

x = np.random.uniform(-1, 1, (C, H, W, B)).astype(np.float32)
gamma = np.random.uniform(-1, 1, C).astype(np.float32)
beta = np.random.uniform(-1, 1, C).astype(np.float32)
dy = np.random.uniform(-1, 1, (C, H, W, B)).astype(np.float32)

if mode == MODE.TORCH:
    fw_x = torch.from_numpy(x).cuda()
    fw_gamma = torch.from_numpy(gamma).cuda()
    fw_beta = torch.from_numpy(beta).cuda()
    fw_dy = torch.from_numpy(dy).cuda()
    # register gradients
    fw_x.requires_grad_(True)
    fw_gamma.requires_grad_(True)
    fw_beta.requires_grad_(True)
    # execute
    fw_y = triton.ops.batchnorm(fw_x, fw_gamma, fw_beta, 1e-4)
    fw_y.backward(fw_dy)
 
if mode == MODE.TF:
   fw_x = tf.placeholder(shape=x.shape, dtype=x.dtype)
   fw_gamma = tf.placeholder(shape=gamma.shape, dtype=gamma.dtype)
   fw_beta = tf.placeholder(shape=beta.shape, dtype=beta.dtype)
   fw_dy = tf.placeholder(shape=dy.shape, dtype=dy.dtype)
   # execute
   fw_y = triton.ops.batchnorm(fw_x, fw_gamma, fw_beta, 1e-4)
   fw_mean, fw_var = tf.nn.moments(fw_x, [1, 2, 3])
   fw_dx, fw_dgamma, fw_dbeta = tf.gradients(fw_y, [fw_x, fw_gamma, fw_beta], fw_dy)
   # run
   sess = tf.InteractiveSession()
   feed_dict = {fw_x: x, fw_gamma: gamma, fw_beta: beta, fw_dy: dy}
   sess.run(tf.global_variables_initializer())
   result = sess.run([fw_dx, fw_dgamma, fw_dbeta], feed_dict=feed_dict)
   print(result)