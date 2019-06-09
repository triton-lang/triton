import os
import tensorflow as tf
import numpy as np
from time import time
data_files_path = tf.resource_loader.get_data_files_path()
library_dir = '/home/philippe/development/triton/build/examples/python/tensorflow'
module = tf.load_op_library(os.path.join(library_dir, 'libtf_blocksparse.so'))

M, N, K = 256,256,256
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

#bench = tf.test.Benchmark().run_op_benchmark(sess=sess, 
#                                             op_or_tensor=c, 
#                                             feed_dict={a: ha, b: hb},
#                                             min_iters=100)
#print(end - start)
#print(2*M*N*K / (end - start) * 1e-12)
hresult = np.dot(ha.T, hb).T
dif = np.abs(result - hresult)
print("dif: %f" % np.max(dif))

#np.savetxt("dif.txt", dif, fmt="%5.2f")
#np.savetxt("gpu.txt", result, fmt="%5.2f")
#np.savetxt("cpu.txt", hresult, fmt="%5.2f")
