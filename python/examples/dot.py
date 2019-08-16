import libtriton
import tensorflow as tf
import distutils
import distutils.log
import setuptools.command.build_ext
import setuptools
import os

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

with open('test.cpp', 'w+') as test:
  src = libtriton.make_tensorflow_src(src, [2], '(M + #TM - 1)/#TM, (N + #TN - 1)/#TN, 1')
  test.writelines(src)

triton_include_dirs = ['/home/philippe/development/triton/include']
tensorflow_include_dirs = [tf.sysconfig.get_include()]
llvm_include_dirs = ['/usr/include/llvm-8/', '/usr/include/llvm-c-8/']
cuda_include_dirs = ['/usr/local/cuda-10.1/targets/x86_64-linux/include/']

triton_library_dirs = [os.path.realpath(libtriton.__file__)]
tensorflow_library_dirs = [tf.sysconfig.get_lib()]

include_dirs = triton_include_dirs + tensorflow_include_dirs + cuda_include_dirs
extra_compile_args = []
extra_link_args = []
library_dirs = tensorflow_library_dirs
libraries = ['tensorflow_framework']

ext = setuptools.Extension(
    name = 'test',
    language = 'c++',
    sources = ['/home/philippe/development/triton/python/examples/test.cpp'],
    include_dirs = include_dirs,
    extra_compile_args = extra_compile_args,
    extra_link_args = extra_link_args,
    library_dirs = library_dirs,
    libraries = libraries
)

build_path = '.'
args = ['build_ext']
#args.append('--build-temp=' + build_path)
#args.append('--build-lib=' + build_path)
args.append('-q')
args = dict(
    name = 'test',
    ext_modules = [ext],
    script_args = args,
    cmdclass = {
      'build_ext': setuptools.command.build_ext.build_ext
  }

) 

setuptools.setup(**args)
library_dir = os.path.dirname(os.path.realpath(__file__))
module = tf.load_op_library(os.path.join(library_dir, 'build/lib.linux-x86_64-3.6/test.cpython-36m-x86_64-linux-gnu.so'))
print(module.matmul)