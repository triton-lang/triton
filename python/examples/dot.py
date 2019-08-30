import tensorflow as tf
import triton
import numpy as np

src = """
#if AT == 1
#define USEA ^a
#define STRIDE_AK lda
#define STRIDE_AM 1
#define BROADCAST_AK :, newaxis
#define BROADCAST_AM newaxis, :
#define SHAPE_A TK, TM
#else
#define USEA a
#define STRIDE_AK 1
#define STRIDE_AM lda
#define BROADCAST_AK newaxis, :
#define BROADCAST_AM :, newaxis
#define SHAPE_A TM, TK
#endif

#if BT == 1
#define USEB ^b
#define STRIDE_BK 1
#define STRIDE_BN ldb
#define BROADCAST_BK newaxis, :
#define BROADCAST_BN :, newaxis
#define SHAPE_B TN, TK
#else
#define USEB b
#define STRIDE_BK ldb
#define STRIDE_BN 1
#define BROADCAST_BK :, newaxis
#define BROADCAST_BN newaxis, :
#define SHAPE_B TK, TN
#endif

void dot(TYPE * A,
         TYPE * B,
         TYPE * C,
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
  /* pointers for operands */
  TYPE* pa[SHAPE_A] = A + rka[BROADCAST_AK] * STRIDE_AK + rxa[BROADCAST_AM] * STRIDE_AM;
  TYPE* pb[SHAPE_B] = B + rkb[BROADCAST_BK] * STRIDE_BK + ryb[BROADCAST_BN] * STRIDE_BN;
  /* prefetches operands */
  TYPE a[SHAPE_A] = *pa;
  TYPE b[SHAPE_B] = *pb;
  /* reduction loop */
  for(int k = K; k > 0; k = k - TK){
    xc = USEA @ USEB + xc;
    pa = pa + TK * STRIDE_AK;
    pb = pb + TK * STRIDE_BK;
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
  *pc = c;
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


def dot(a, b, trans_a = False, trans_b = False):
  if (trans_a, trans_b) not in dot.ops:
    dot.ops[trans_a, trans_b] = dot_op(trans_a, trans_b)
  return dot.ops[trans_a, trans_b](a, b)
dot.ops = dict()

# @triton.register_gradient(dot_op)
# def _dot_grad(op, dy):
#   a = op.inputs[0]
#   b = op.inputs[1]
#   return [dot_tn(dy, b), dot_nt(a, dy), None, None, None, None, None, None, None]

def run_dot():
  M, N, K = 128, 128, 128
  a = tf.placeholder(tf.float16, shape=[M, K])
  b = tf.placeholder(tf.float16, shape=[N, K])
  c = dot(a, b, trans_a = False, trans_b = True)
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
  print("dif: %f" % np.max(dif))

run_dot()