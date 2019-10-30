#!/usr/bin/env python

import numpy       as np
from enum import Enum
import triton

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

cases = []
# Matmul
cases += [[[4, 1024, 1024], [1024, 1024], [4, 1024, 1024], "btc,ck->btk"]]
# Attention
# cases += [[[4, 256, 8, 2, 64], [8, 2, 512, 64], [4, 256, 8, 2, 512], "bchak,hank->bchan"]]

if mode == MODE.TF:
    sess = tf.InteractiveSession()

for a_shape, b_shape, c_shape, einsum in cases:

    A = np.random.uniform(-1.0, 1.0, a_shape).astype(np.float16).astype(np.float32)
    B = np.random.uniform(-1.0, 1.0, b_shape).astype(np.float16).astype(np.float32)
    E = np.random.uniform(-1.0, 1.0, c_shape).astype(np.float16).astype(np.float32)

    # Execute (tensorflow)
    if mode == MODE.TF:
        a = tf.placeholder(tf.float32, a_shape, name="a")
        b = tf.placeholder(tf.float32, b_shape, name="b")
        e = tf.placeholder(tf.float32, c_shape, name="e")
        c = triton.ops.einsum(einsum, a, b, 1)
        da, db = tf.gradients(c, [a, b], e)
        feed_dict = { a: A.astype(np.float32), 
                    b: B.astype(np.float32), 
                    e: E }
        sess.run(tf.global_variables_initializer())
        result = sess.run([c, da, db], feed_dict = feed_dict)
    # Execute (torch)
    if mode == MODE.TORCH:
        a = torch.from_numpy(A).cuda()
        b = torch.from_numpy(B).cuda()
        e = torch.from_numpy(E).cuda()
        a.requires_grad_(True)
        b.requires_grad_(True)
        c = triton.ops.einsum(einsum, a, b, 1)
        torch.autograd.backward(c, e)
        da = a.grad
        db = b.grad
        result = [c.cpu().detach().numpy(), da.cpu().detach().numpy(), db.cpu().detach().numpy()]
        
    # benchmark 
    nanosec = triton.bench_registry[c]
    ctx = triton.ctx_registry[c]
    b, m, n, k = tuple((ctx.bmnk[i] for i in range(0, 4)))
    ops = 2.*b*m*n*k
    print('C TFLOPS:', ops / triton.bench_registry[c] * 1e-3)
    #print('DA TFLOPS:', ops / triton.bench_registry[da] * 1e-3)
    #print('DB TFLOPS:', ops / triton.bench_registry[db] * 1e-3)

    # test
    ctx = triton.ctx_registry[c]
    t_a = ctx.trans_a
    t_b = ctx.trans_b
    e_a = ctx.einsum_a
    e_b = ctx.einsum_b
    e_c = ctx.einsum_c
    C = np.einsum(einsum, A, B)
    if not t_a and not t_b: # NN
        DA = np.einsum(f"{e_c},{e_b}->{e_a}", E, B)
        DB = np.einsum(f"{e_a},{e_c}->{e_b}", A, E)
    elif not t_a and t_b:   # NT
        DA = np.einsum(f"{e_c},{e_b}->{e_a}", E, B)
        DB = np.einsum(f"{e_c},{e_a}->{e_b}", E, A)
    elif t_a and not t_b:   # TN
        DA = np.einsum(f"{e_b},{e_c}->{e_a}", B, E)
        DB = np.einsum(f"{e_a},{e_c}->{e_b}", A, E)
    c, da, db = result[0], result[1], result[2]
    print('C diff:',  np.abs((C - c)).max())
    print('DA diff:', np.abs((DA - da)).max())
    print('DB diff:', np.abs((DB - db)).max())