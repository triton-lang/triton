import tensorflow as tf
import triton
import numpy as np

src = """
// Templates for accessing A
#if AT == 1
#define USE_A ^a
#define STRIDE_AK lda
#define STRIDE_AM 1
#define BROADCAST_AK :, newaxis
#define BROADCAST_AM newaxis, :
#define SHAPE_A TK, TM
#else
#define USE_A a
#define STRIDE_AK 1
#define STRIDE_AM lda
#define BROADCAST_AK newaxis, :
#define BROADCAST_AM :, newaxis
#define SHAPE_A TM, TK
#endif

// Templates for accessing B
#if BT == 1
#define USE_B ^b
#define STRIDE_BK 1
#define STRIDE_BN ldb
#define BROADCAST_BK newaxis, :
#define BROADCAST_BN :, newaxis
#define SHAPE_B TN, TK
#else
#define USE_B b
#define STRIDE_BK ldb
#define STRIDE_BN 1
#define BROADCAST_BK :, newaxis
#define BROADCAST_BN newaxis, :
#define SHAPE_B TK, TN
#endif

void dot(TYPE * A, TYPE * B, TYPE * C,
         int M, int N, int K,
         int lda __multipleof(8),
         int ldb __multipleof(8),
         int ldc) {
  // prologue
  int ridx = get_program_id(0);
  int ridy = get_program_id(1);
  int rxa[TM] = ridx * TM + 0 ... TM;
  int ryb[TN] = ridy * TN + 0 ... TN;
  int rka[TK] = 0 ... TK;
  int rkb[TK] = 0 ... TK;
  float c[TM, TN] = 0;
  // pointers to operands
  TYPE* pa[SHAPE_A] = A + rka[BROADCAST_AK] * STRIDE_AK + rxa[BROADCAST_AM] * STRIDE_AM;
  TYPE* pb[SHAPE_B] = B + rkb[BROADCAST_BK] * STRIDE_BK + ryb[BROADCAST_BN] * STRIDE_BN;
  // prefetches operands
  TYPE a[SHAPE_A] = *pa;
  TYPE b[SHAPE_B] = *pb;
  // reduction loop
  for(int k = K; k > 0; k-= TK){
    c += USE_A @ USE_B;
    pa = pa + TK * STRIDE_AK;
    pb = pb + TK * STRIDE_BK;
    a = *pa;
    b = *pb;
  }
  // epilogue
  int rxc[TM] =  ridx * TM + 0 ... TM;
  int ryc[TN] =  ridy * TN + 0 ... TN;
  TYPE* pc[TM, TN] = C + ryc[newaxis, :] + rxc[:, newaxis] * ldc;
  bool checkc[TM, TN] = (rxc < M)[:, newaxis] && (ryc < N)[newaxis, :];
  *?(checkc) pc = c;
}
"""

class dot_op(triton.op2):

  def __init__(self, transpose_a = False, transpose_b = False):
    self.dot = triton.op(src, ['C'])
    self.transpose_a = transpose_a
    self.transpose_b = transpose_b
  
  def forward(self, a, b):
    dtype = a.dtype
    # extract shapes
    shape_a = triton.shape(a)
    shape_b = triton.shape(b)
    M, Ka = shape_a[0], shape_a[1]
    Kb, N = shape_b[0], shape_b[1]
    # transpose shapes
    if self.transpose_a:
      M, Ka = Ka, M
    if self.transpose_b:
      Kb, N = N, Kb
    # contiguous dimensions
    lda = M if self.transpose_a else Ka
    ldb = Kb if self.transpose_b else N
    ldc = N
    # allocate output
    c = triton.empty([M, N], dtype = dtype)
    # compute
    return self.dot(a, b, c, M, N, Ka, lda, ldb, ldc, 
                    lambda opt: [triton.cdiv(M, opt.d('TM')), triton.cdiv(N, opt.d('TN'))],             
                    AT = self.transpose_a, BT = self.transpose_b, TYPE = dtype, 
                    TM = [128], TN = [128], TK = [8])

  def backward(self, op, dy):
    a = op.inputs[0]
    b = op.inputs[1]
    da = dot_op(self.transpose_a, self.transpose_b).forward(dy, b)
    db = dot_op(self.transpose_a, self.transpose_b).forward(a, dy)
    return [da, db, None, None, None, None, None, None, None]


def dot(a, b, transpose_a = False, transpose_b = False):
  if (transpose_a, transpose_b) not in dot.ops:
    dot.ops[transpose_a, transpose_b] = dot_op(transpose_a, transpose_b)
  return dot.ops[transpose_a, transpose_b](a, b)
dot.ops = dict()


def run_dot():
  M, N, K = 128, 128, 128
  a = tf.placeholder(tf.float32, shape=[M, K])
  b = tf.placeholder(tf.float32, shape=[N, K])
  c = dot(a, b, transpose_a = False, transpose_b = False)
  da = tf.gradients(c, [a])
  # Reference
  ha = np.random.rand(M, K).astype(np.float32)
  hb = np.random.rand(K, N).astype(np.float32)
  # Run
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())
  result = sess.run([da], feed_dict = {a: ha,
                                      b: hb})[0]
  # Test
  print(result)
  hresult = np.dot(ha, hb)
  dif = np.abs(result - hresult)
  np.savetxt('dif.dat', dif, '%2.4f')
  print("dif: %f" % np.max(dif))

run_dot()