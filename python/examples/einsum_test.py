#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy       as np
import tensorflow  as tf
import triton
import blocksparse as bs
from tensorflow.python.ops import gradient_checker

one = 0
out = 0
bench = 0

class ProdKeyTest(tf.test.TestCase):

    def testEinsum(self):
        # multi-threading screws up benchmarking
        conf = tf.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1)

        with self.test_session(config=conf) as sess, tf.device("/gpu:0"):

            batch_dim = 4
            ctx_dim   = 256
            head_dim  = 8
            n_keys    = 512
            key_dim   = 128

            # batch_dim = 2
            # ctx_dim   = 8
            # head_dim  = 2
            # n_keys    = 16
            # key_dim   = 16

            for a_shape, b_shape, c_shape, einsum in [
                [ [ 4, 8, 8 ], [ 8, 8 ], [ 4, 8, 8 ], "btc,ck->btk" ],
                [ [4, 1024, 1024], [ 1024, 1024 ], [4, 1024, 1024 ], "btc,ck->btk" ],
                [ (batch_dim, ctx_dim, head_dim, 2, key_dim//2),(head_dim, 2, n_keys,  key_dim//2), (batch_dim, ctx_dim, head_dim, 2, n_keys), "bchak,hank->bchan" ],
            ]:

                if one:
                    A = np.ones(a_shape, dtype=np.float16).astype(np.float32)
                    B = np.ones(b_shape, dtype=np.float16).astype(np.float32)
                    E = np.ones(c_shape, dtype=np.float32)
                else:
                    # QK = np.random.normal(loc=0.0, scale=1.0, size=qk_shape).astype(np.float16).astype(np.float32)
                    # V  = np.random.normal(loc=0.0, scale=1.0, size=vw_shape).astype(np.float16).astype(np.float32)
                    A = np.random.uniform(-1.0, 1.0, a_shape).astype(np.float16).astype(np.float32)
                    B = np.random.uniform(-1.0, 1.0, b_shape).astype(np.float16).astype(np.float32)
                    E = np.random.uniform(-1.0, 1.0, c_shape).astype(np.float16).astype(np.float32)

                a = tf.placeholder(tf.float32, a_shape, name="a")
                b = tf.placeholder(tf.float32, b_shape, name="b")
                e = tf.placeholder(tf.float32, c_shape, name="e")
                feed_dict = { a: A.astype(np.float32), 
                              b: B.astype(np.float32), 
                              e: E }

                c = triton.ops.einsum(einsum, a, b, bench=bench)

                # error = gradient_checker.compute_gradient_error(a, a_shape, c, c_shape, delta=1e-1, extra_feed_dict={ b:B }) #
                # print(error)
                # error = gradient_checker.compute_gradient_error(b, b_shape, c, c_shape, delta=1e-1, extra_feed_dict={ a:A }) #
                # print(error)
                # return

                with tf.control_dependencies([c.op]):
                    da, db = tf.gradients(c, [a, b], e)

                # c, = sess.run( [ c, ], feed_dict )
                rc, rda, rdb = sess.run( [ c, da, db ], feed_dict )
                
                if bench > 0:
                    nanosec = triton.bench_registry[c]
                    ctx = triton.ctx_registry[c]
                    b, m, n, k = tuple((ctx.bmnk[i] for i in range(0, 4)))
                    ops = 2. * b * m * n * k
                    print('C TFLOPS:', ops / triton.bench_registry[c] * 1e-3)
                    print('DA TFLOPS:', ops / triton.bench_registry[da] * 1e-3)
                    print('DB TFLOPS:', ops / triton.bench_registry[db] * 1e-3)

                else:
                    C = np.einsum(einsum, A, B)
                    ctx = triton.ctx_registry[c]
                    t_a = ctx.trans_a
                    t_b = ctx.trans_b
                    e_a = ctx.einsum_a
                    e_b = ctx.einsum_b
                    e_c = ctx.einsum_c

                    if not t_a and not t_b: # NN
                        DA = np.einsum(f"{e_c},{e_b}->{e_a}", E, B)
                        DB = np.einsum(f"{e_a},{e_c}->{e_b}", A, E)
                    elif not t_a and t_b:   # NT
                        DA = np.einsum(f"{e_c},{e_b}->{e_a}", E, B)
                        DB = np.einsum(f"{e_c},{e_a}->{e_b}", E, A)
                    elif t_a and not t_b:   # TN
                        DA = np.einsum(f"{e_b},{e_c}->{e_a}", B, E)
                        DB = np.einsum(f"{e_a},{e_c}->{e_b}", A, E)

                    print("testProdKey", einsum)
                    if not bench:
                        for op, dev, cpu in [
                            [   "C",   rc,   C ],
                            [  "DA",  rda,  DA ],
                            [  "DB",  rdb,  DB ],
                        ]:
                            self.compare_results(op, dev, cpu)

    def compare_results(self, op, dev, cpu):
        dev = dev.astype(np.float64)
        cpu = cpu.astype(np.float64)

        # print(dev.reshape(-1)[0:4])
        # print(cpu.reshape(-1)[0:4])

        dif     = np.abs(cpu - dev)
        maxval  = np.max(abs(cpu))
        avgval  = np.average(abs(cpu))
        maxdif  = dif.max()
        max_err = maxdif if avgval == 0 else maxdif / avgval
        l2_err  = 0.0    if avgval == 0 else np.sqrt(np.square(dif).sum()) / np.sqrt(np.square(cpu).sum())

        print("op:%3s, max:%18.12f, avg:%18.12f, dif:%18.12f, err:%18.12f, l2_err:%18.12f shape:%15s" % (op, maxval, avgval, maxdif, max_err, l2_err, str(cpu.shape)))

        if out:
            dim = cpu.shape[-1]
            np.savetxt("%s_dif.txt" % op, dif.reshape((-1,dim)), fmt='%4.1f') #7.5 5.3
            np.savetxt("%s_cpu.txt" % op, cpu.reshape((-1,dim)), fmt='%4.1f') #7.5 5.3
            np.savetxt("%s_dev.txt" % op, dev.reshape((-1,dim)), fmt='%4.1f') #7.5 5.3
            exit()

if __name__ == "__main__":
  tf.test.main()

