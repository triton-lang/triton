import libtriton
import tensorflow as tf
import distutils
import distutils.log
import setuptools.command.build_ext
import setuptools
import numpy as np
import os
import tempfile
import shutil
import hashlib

src = """
const tunable int TM = {128};
const tunable int TN = {128};
const tunable int TK = {32};

void matmul(restrict read_only align(16) half *A,
            restrict read_only align(16) half *B,
            restrict read_only align(16) half *C,
            int M, int N, int K,
            multiple_of(8) int lda, multiple_of(8) int ldb, int ldc) {
  int ridx = get_range_id(0);
  int ridy = get_range_id(1);
  int rxa[TM] = ridx * TM + (0 ... TM);
  int ryb[TN] = ridy * TN + (0 ... TN);
  int rka[TK] = 0 ... TK;
  int rkb[TK] = 0 ... TK;
  float xc[TM, TN] = 0;
  half* pa[TM, TK] = A + rka[newaxis, :]*lda + rxa[:, newaxis];
  half* pb[TN, TK] = B + rkb[newaxis, :]*ldb + ryb[:, newaxis];
  half a[TM, TK] = *pa;
  half b[TN, TK] = *pb;
  for(int k = K; k > 0; k = k - TK){
    xc = dot(a, trans(b), xc);
    pa = pa + TK*lda;
    pb = pb + TK*ldb;
    a = *pa;
    b = *pb;
  }
  int rxc[TM] =  ridx * TM + (0 ... TM);
  int ryc[TN] =  ridy * TN + (0 ... TN);
  half* pc[TM, TN] = C + ryc[newaxis, :]*ldc + rxc[:, newaxis];
  half c[TM, TN] = xc;
  bool checkc0[TM] = rxc < M;
  bool checkc1[TN] = ryc < N;
  bool checkc[TM, TN] = checkc0[:, newaxis] && checkc1[newaxis, :];
  @checkc *pc = c;
}
"""


extra_ops = tf.load_op_library('/home/philippe/development/triton/python/build/lib.linux-x86_64-3.6/libextra_tf_ops.so')


def make_bindings(src, outputs, grids):
  return libtriton.make_tensorflow_src(src, outputs, grids)

def make_cache_path(src):
  md5 = hashlib.sha1(src.encode())
  hexhash = md5.hexdigest()
  home = os.path.expanduser('~')
  cacheroot = os.path.join(home, '.triton', 'cache')
  cachepath = os.path.join(cacheroot, str(hexhash))
  if not os.path.exists(cachepath):
    os.makedirs(cachepath)
  print(cachepath)
  return cachepath

def write_bindings(src, root):
  cpp = os.path.join(root, 'tensorflow.cpp')
  so = os.path.join(root, 'tensorflow.so')
  recompile = False
  # recompile if .so does not exist
  if not os.path.exists(cpp) or not os.path.exists(so):
    recompile = True
  # recompile if cpp was modified after .so
  elif max(cpp, so, key=os.path.getctime) == cpp:
    recompile = True
  # write cpp file
  if recompile:
    with open(cpp, 'w+') as handle:
      handle.writelines(src)
  # return path of cpp file
  return cpp
  
def build(src, path):
  # include directories
  triton_include_dirs = ['/home/philippe/development/triton/include']
  tensorflow_include_dirs = [tf.sysconfig.get_include()]
  cuda_include_dirs = ['/usr/local/cuda-10.1/targets/x86_64-linux/include/']
  include_dirs = triton_include_dirs + tensorflow_include_dirs + cuda_include_dirs
  # library directories
  triton_library_dirs = [os.path.realpath(os.path.join(libtriton.__file__, os.path.pardir))]
  tensorflow_library_dirs = [tf.sysconfig.get_lib()]
  library_dirs = triton_library_dirs + tensorflow_library_dirs
  # libraries
  libraries = ['tensorflow_framework', 'triton']
  # extra arguments
  extra_compile_args = []
  extra_link_args = []
  # create extension module
  ext = setuptools.Extension(
      name = 'test',
      language = 'c++',
      sources = [src],
      include_dirs = include_dirs,
      extra_compile_args = extra_compile_args,
      extra_link_args = extra_link_args,
      library_dirs = library_dirs,
      libraries = libraries
  )
  # build extension module
  args = ['build_ext']
  tmp = tempfile.mkdtemp()
  args.append('--build-temp=' + tmp)
  args.append('--build-lib=' + path)
  args.append('-q')
  args = dict(
      name = 'test',
      ext_modules = [ext],
      script_args = args,
  ) 
  setuptools.setup(**args)
  shutil.rmtree(tmp)

def make_tensorflow_op(src, outputs, grids):
  bindings = make_bindings(src, outputs, grids)
  cache_path = make_cache_path(bindings)
  cpp = write_bindings(bindings, cache_path)
  build(cpp, cache_path)
  result = tf.load_op_library(os.path.join(cache_path, 'test.cpython-36m-x86_64-linux-gnu.so'))
  return result


library_dir = os.path.dirname(os.path.realpath(__file__))
module = make_tensorflow_op(src, ['C'], ['(M + #TM - 1)/#TM', '(N + #TN - 1)/#TN'])
print(module.matmul)


class dot:

  def __init__(self):
    trans_a = True
    trans_b = False
  
  def __call__(self, a, b):
    shape_a = tf.shape(a)
    shape_b = tf.shape(b)
    M = shape_a[0]
    K = shape_a[1]
    N = shape_b[0]
    lda = M
    ldb = K
    ldc = M
    c = extra_ops.alloc_empty(tf.stack([M, N]))
    return module.matmul(a, b, c, M, N, K, lda, ldb, ldc)

dot_nt = dot()
def run_dot():
  M, N, K = 128, 128, 128
  a = tf.placeholder(tf.float16, shape=[M, K])
  b = tf.placeholder(tf.float16, shape=[N, K])
  # c = tf.matmul(a, b, transpose_a=True)
  c = dot_nt(a, b)
  # Reference
  ha = np.random.rand(M, K).astype(np.float16)
  hb = np.random.rand(N, K).astype(np.float16)
  # Run
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())
  result = sess.run([c], feed_dict = {a: ha,
                                      b: hb})[0]
  # Test
  hresult = np.dot(ha.T, hb).T
  dif = np.abs(result - hresult)
  np.savetxt('dif.dat', dif, '%2.4f')
  print(hresult)
  print(result)
  print("dif: %f" % np.max(dif))

run_dot()