import os
import tensorflow as tf
import numpy as np

data_files_path = tf.resource_loader.get_data_files_path()
library_dir = '/home/philippe/development/triton/build/examples/python/tensorflow'
module = tf.load_op_library(os.path.join(library_dir, 'libtf_blocksparse.so'))

M, N, K = 512, 512, 512
a = tf.placeholder(tf.float32, shape=[M, K])
b = tf.placeholder(tf.float32, shape=[N, K])
locks = tf.placeholder(tf.int32, shape=[4096])
c = module.block_sparse_mat_mul(a, b, locks)
# Run
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
result = sess.run([c], feed_dict = {locks: np.zeros(4096),
									a: np.random.rand(M, K),
									b: np.random.rand(N, K)})
print(result)
