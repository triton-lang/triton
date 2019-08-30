import triton
import tensorflow as tf
import numpy as np

src = """
#if AT == 1
#define USEA ^a
#else
#define USEA a
#endif

#if BT == 1
#define USEB ^b
#else
#define USEB b
#endif

void dot(TYPE * A __noalias __readonly __aligned(16),
         TYPE * B __noalias __readonly __aligned(16),
         TYPE * C __noalias __readonly __aligned(16),
         int M, int N, int K,
         int lda __multipleof(8),
         int ldb __multipleof(8),
         int ldc) {

  /* prologue */
  int ridx = get_program_id(0);
  int ridy = get_program_id(1);
  int rxa[TM] = ridx * TM + 0 ... TM;
  int ryb[TN] = ridy * TN + 0 ... TN;
  int rka[TK] = 0 ... TK;
  int rkb[TK] = 0 ... TK;
  float xc[TM, TN] = 0;

  /* pointers for A */
#if AT == 1
  TYPE* pa[TK, TM] = A + rka[:, newaxis]*lda + rxa[newaxis, :];
  TYPE a[TK, TM] = *pa;
#else
  TYPE* pa[TM, TK] = A + rka[newaxis, :] + rxa[:, newaxis]*lda;
  TYPE a[TM, TK] = *pa;
#endif

  /* pointers for B */
#if BT == 1
  TYPE* pb[TN, TK] = B + rkb[newaxis, :] + ryb[:, newaxis]*ldb;
  TYPE b[TN, TK] = *pb;
#else
  TYPE* pb[TK, TN] = B + rkb[:, newaxis]*ldb + ryb[newaxis, :];
  TYPE b[TK, TN] = *pb;
#endif

  /* reduction loop */
  for(int k = K; k > 0; k = k - TK){
    xc = USEA @ USEB + xc;
#if AT == 1
    pa = pa + TK*lda;
#else
    pa = pa + TK;
#endif
#if BT == 1
    pb = pb + TK;
#else
    pb = pb + TK*ldb;
#endif
    a = *pa;
    b = *pb;
  }

  /* epilogue */
  int rxc[TM] =  ridx * TM + (0 ... TM);
  int ryc[TN] =  ridy * TN + (0 ... TN);
  TYPE* pc[TM, TN] = C + ryc[newaxis, :] + rxc[:, newaxis] * ldc;
  TYPE c[TM, TN] = xc;
  bool checkc0[TM] = rxc < M;
  bool checkc1[TN] = ryc < N;
  bool checkc[TM, TN] = checkc0[:, newaxis] && checkc1[newaxis, :];
  *?(checkc) pc = c;
}
"""

def cdiv(a, b):
    return -(-a // b)

class dot_op:

  def __init__(self, trans_a = False, trans_b = False):
    self.dot = triton.op(src, ['C'])
    self.trans_a = trans_a
    self.trans_b = trans_b
  
  def __call__(self, a, b):
    shape_a = triton.shape(a)
    shape_b = triton.shape(b)
    M = shape_a[0]
    Ka = shape_a[1]
    Kb = shape_b[0]
    N = shape_b[1]
    # transpose shapes
    if self.trans_a:
      M, Ka = Ka, M
    if self.trans_b:
      Kb, N = N, Kb
    K = Ka
    # contiguous dimensions
    lda = Ka
    ldb = N
    ldc = N
    c = triton.empty([M, N])
    return self.dot(a, b, c, M, N, K, lda, ldb, ldc, 
                    lambda opt: [cdiv(M, opt.d('TM')), cdiv(N, opt.d('TN'))],             
                    AT = self.trans_a, BT = self.trans_b, TYPE = tf.float16, 
                    TM = [128], TN = [ 128], TK = [32])

dot_nt = dot_op(False, True)
dot_nn = dot_op(False, False)
dot_tn = dot_op(True, False)
dot_tt = dot_op(True, True)

# @triton.register_gradient(dot_op)
# def _dot_grad(op, dy):
#   a = op.inputs[0]
#   b = op.inputs[1]
#   return [dot_tn(dy, b), dot_nt(a, dy), None, None, None, None, None, None, None]

def run_dot():
  M, N, K = 128, 128, 128
  a = tf.placeholder(tf.float16, shape=[M, K])
  b = tf.placeholder(tf.float16, shape=[N, K])
  # c = tf.matmul(a, b, transpose_a=True)
  c = dot_nt(a, b)
  # grads = tf.gradients(c, [a])
  # Reference
  ha = np.random.rand(M, K).astype(np.float16)
  hb = np.random.rand(K, N).astype(np.float16)
  # Run
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())
  result = sess.run([c], feed_dict = {a: ha,
                                      b: hb})[0]
  # Test
  hresult = np.dot(ha, hb.T)
  dif = np.abs(result - hresult)
  np.savetxt('dif.dat', dif, '%2.4f')
  print(hresult)
  print(result)
  print("dif: %f" % np.max(dif))

run_dot()