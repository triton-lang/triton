import os
import tensorflow as tf
import numpy as np

data_files_path = tf.resource_loader.get_data_files_path()
library_dir = '/home/philippe/development/triton/build/examples/python/tensorflow'
module = tf.load_op_library(os.path.join(library_dir, 'libtf_blocksparse.so'))

M, N, K = 16, 16, 16
a = tf.placeholder(tf.float16, shape=[M, K])
b = tf.placeholder(tf.float16, shape=[N, K])
locks = tf.placeholder(tf.int32, shape=[4096])
c = module.dot(a, b, locks)
# Reference
ha = np.ones((M, K)).astype(np.float16)
hb = np.ones((N, K)).astype(np.float16)
hresult = np.dot(hb.T, ha)

# Run
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
result = sess.run([c], feed_dict = {locks: np.zeros(4096),
                                    a: ha,
                                    b: hb})
print(result - hresult)
