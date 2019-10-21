import numpy as np
import torch
import triton

class _dot(triton.function):

  src = """
__global__ void dot(TYPE * A, TYPE * B, TYPE * C,
                   int sb, int sh, int sa, int sk, int sn) {
  // program id
  int pidx = get_program_id(0);
  int pidy = get_program_id(1);
  int pidz = get_program_id(2);
  // ranges
  int rxa[TM] = pidx * TM + 0 ... TM;
  int ryb[TN] = pidy * TN + 0 ... TN;
  int rza[TZ] = pidz * TZ + 0 ... TZ;
  int rzb[TZ] = pidz * TZ + 0 ... TZ;
  int rka[TK] = 0 ... TK;
  int rkb[TK] = 0 ... TK;
  // accumulator
  float c[TM, TN, TZ] = 0;
  // pointers to A
  TYPE* pa[TM, TK, TZ] = A + rka[newaxis, :, newaxis] * 1             // reduction
                           + rxa[:, newaxis, newaxis] * sk * sa * sh  // outer
                           + rza[newaxis, newaxis, :] * sk;           // batch
  // pointers to B
  TYPE* pb[TK, TN, TZ] = B + rkb[:, newaxis, newaxis] * 1             // reduction
                           + ryb[newaxis, :, newaxis] * sk            // outer
                           + rzb[newaxis, newaxis, :] * sk  * sn;     // batch      
  // reduction loop
  for(int k = sk; k > 0; k -= TK){
    TYPE a[TM, TK, TZ] = *pa;
    TYPE b[TK, TN, TZ] = *pb;
    c += a @ b;
    pa += TK;
    pb += TK;
  }     
  // epilogue
  int rxc[TM] = pidx * TM + 0 ... TM;
  int ryc[TN] = pidy * TN + 0 ... TN;
  int rzc[TZ] = pidz * TZ + 0 ... TZ;
  TYPE* pc[TM, TN, TZ] = C + rxc[:, newaxis, newaxis] * sn * sa * sh   // outer[0]
                           + ryc[newaxis, :, newaxis] * 1              // outer[1]
                           + rzc[newaxis, newaxis, :] * sn;
  *pc = c;
}
"""

  kernel = triton.kernel(src, ['C'])

  @staticmethod
  def _call(a, b, transpose_a, transpose_b):
    # extract shapes
    shape_a = triton.shape(a)
    shape_b = triton.shape(b)
    B, H, A, K = shape_a[0], shape_a[1], shape_a[2], shape_a[3]
    H, A, N, K = shape_b[0], shape_b[1], shape_b[2], shape_b[3]
    # allocate output
    dtype = a.dtype
    c = triton.empty([B, H, A, N], dtype = dtype)
    # SPMD grid
    grid = lambda opt: [triton.cdiv(B, opt.d('TM')), 
                        triton.cdiv(N, opt.d('TN')), 
                        triton.cdiv(H*A, opt.d('TZ'))]
    # launch kernel
    return _dot.kernel(a, b, c, B, H, A, K, N, grid,           
                  AT = transpose_a, BT = transpose_b, TYPE = dtype, 
                  TM = [32], TN = [32], TK = [8], TZ = [8])

  @staticmethod
  def forward(ctx, a, b, transpose_a = False, transpose_b = False):
    ctx.save_for_backward(a, b)
    ctx.t_a = transpose_a
    ctx.t_b = transpose_b
    return _dot._call(a, b, transpose_a, transpose_b)


dot = _dot.apply


batch_dim  = 16
ctx_dim    = 32
head_dim   = 8
state_dim  = 32
key_dim    = 32
n_keys     = 32
bs         = batch_dim * ctx_dim

# shapes
x_shape  = (bs, state_dim)
qw_shape = (state_dim, head_dim * key_dim)
kw_shape = (head_dim, 2, n_keys,  key_dim // 2)

x  = np.random.uniform(-1.0, 1.0,  x_shape).astype(np.float32) # layer input
qw = np.random.uniform(-1.0, 1.0, qw_shape).astype(np.float32) # query weights
kw = np.random.uniform(-1.0, 1.0, kw_shape).astype(np.float32) # key   weights
# (bs, head_dim * key_dim) = (bs, state_dim) * (state_dim, head_dim * key_dim)
# (bs, head_dim, 2, key_dim//2) <==  (bs, head_dim * key_dim)
q = np.dot(x, qw).reshape(bs, head_dim, 2, key_dim//2) # normal matmul

# (bs, head_dim, 2, n_keys) = (bs, head_dim, 2, key_dim//2) * (head_dim, 2, n_keys,  key_dim//2)
# outer: bs, n_keys
# inner: key_dim//2
# batch: head_dim, 2 (key_axis)
qk = np.einsum("bhak,hank->bhan", q, kw)

tq = torch.from_numpy(q).contiguous().cuda()
tkw = torch.from_numpy(kw).contiguous().cuda()
tqk = triton.ops.einsum("bhak,hank->bhan", tq, tkw)
diff = qk - tqk.cpu().numpy()
print(np.max(diff))

