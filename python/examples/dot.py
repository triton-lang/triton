import triton
import tensorflow as tf
import numpy as np

src = """
const tunable int TM = {128};
const tunable int TN = {128};
const tunable int TK = {32};

void matmul(restrict read_only align(16) half *A,
            restrict read_only align(16) half *B,
            restrict read_only align(16) half *C,
            int M, int N, int K,
            multiple_of(8) int lda, multiple_of(8) int ldb, int ldc) 
{
  int ridx = get_program_id(0);
  int ridy = get_program_id(1);
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
  half* pc[TM, TN] = C + ryc[newaxis, :] + rxc[:, newaxis]*ldc;
  half c[TM, TN] = xc;
  bool checkc0[TM] = rxc < M;
  bool checkc1[TN] = ryc < N;
  bool checkc[TM, TN] = checkc0[:, newaxis] && checkc1[newaxis, :];
  @checkc *pc = c;
}
"""

class dot:

  def __init__(self):
    self.matmul = triton.make_tensorflow_op(src, ['C'], ['(M + #TM - 1)/#TM', '(N + #TN - 1)/#TN'])
  
  def __call__(self, a, b):
    shape_a = tf.shape(a)
    shape_b = tf.shape(b)
    M = shape_a[0]
    K = shape_a[1]
    N = shape_b[0]
    lda = M
    ldb = K
    ldc = N
    c = triton.empty([M, N])
    return self.matmul.matmul(a, b, c, M, N, K, lda, ldb, ldc)

dot_tn = dot()
def run_dot():
  M, N, K = 128, 128, 128
  a = tf.placeholder(tf.float16, shape=[M, K])
  b = tf.placeholder(tf.float16, shape=[N, K])
  # c = tf.matmul(a, b, transpose_a=True)
  c = dot_tn(a, b)
  # Reference
  ha = np.random.rand(M, K).astype(np.float16)
  hb = np.random.rand(N, K).astype(np.float16)
  # Run
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())
  result = sess.run([c], feed_dict = {a: ha,
                                      b: hb})[0]
  # Test
  hresult = np.dot(ha.T, hb)
  dif = np.abs(result - hresult)
  np.savetxt('dif.dat', dif, '%2.4f')
  print(hresult)
  print(result)
  print("dif: %f" % np.max(dif))

run_dot()