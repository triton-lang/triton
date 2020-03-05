import triton
import torch

class _linear(torch.autograd.Function):

  src = '''
  __global__ void main (TYPE* A __readonly  __noalias __aligned(16),
                       TYPE* B __readonly  __noalias __aligned(16),
                       TYPE* C __writeonly __noalias __aligned(16),
                       int lda, int ldb, int ldc,
                       int M, int Kmax,
                       int* lut,
                       int* locks, int nlocks) {
    /* ---------------- */
    /*    Prologue      */
    /* ---------------- */
    // program ids
    int pid0 = get_program_id(0);
    int pid1 = get_program_id(1);
#ifdef DW
    // load LUT header
    int *header = lut + pid0 * 2;
    int i = *(header + 0);
    int j = *(header + 1);
    int K = Kmax / TZ;
    int lockid = select(TZ > 1, 1, 0);
    int offk = pid1 * K;
    int offm = i * TM;
    int offn = j * TN;
    int maxid = get_num_programs(1);
#else
    // load LUT header
    int *header = lut + pid1 * 5;
    int offset = *(header + 0);
    int K      = *(header + 1);
    int column = *(header + 2);
    int lockid = *(header + 3);
    int maxid = *(header + 4);
    int *pinc   = lut + offset;
    int offk = (*pinc) * TK;
    int offm = pid0 * TM;
    int offn = column * TN;
#endif
    // initialize a, b pointers
    int rka[TK] = offk + 0 ... TK;
    int rkb[TK] = offk + 0 ... TK;
    int ram[TM] = offm + (0 ... TM);
    int rbn[TN] = offn + (0 ... TN);
    TYPE* pa[TM, TK] = A + ram[:, newaxis] * STRIDE_AM + rka[newaxis, :] * STRIDE_AK;
    TYPE* pb[TK, TN] = B + rbn[newaxis, :] * STRIDE_BN + rkb[:, newaxis] * STRIDE_BK;
    // pre-fetch
    bool checka[TM, TK] = ram[:, newaxis] < M;
    bool checkb[TK, TN] = 1;
    TYPE a[TM, TK] = checka ? *pa : 0;
    TYPE b[TK, TN] = checkb ? *pb : 0;

    /* ---------------- */
    /*    Inner Loop    */
    /* ---------------- */
    // create result tile
    float acc[TM, TN] = 0;
#ifdef DW
    int step = TK;
#else
    int step = 1;
#endif
    for(int k = K; k > 0; k -= step) {
      acc += a @ b;
      // update pointers
#ifdef DW
      int inc_a = TK * STRIDE_AK;
      int inc_b = TK * STRIDE_BK;
#else
      pinc += 1;
      int inc_a = (*pinc) * TK * STRIDE_AK;
      int inc_b = (*pinc) * TK * STRIDE_BK;
#endif
      pa += inc_a;
      pb += inc_b;
      // pre-fetch
      bool checka[TM, TK] = k > 1;
      bool checkb[TK, TN] = k > 1;
      a = *?(checka)pa;
      b = *?(checkb)pb;
    }
    TYPE c[TM, TN] = acc;

    /* ---------------- */
    /*    Epilogue      */
    /* ---------------- */
    // initialize c pointers
    int   rcm[TM]    = offm + (0 ... TM);
    int   rcn[TN]    = offn + (0 ... TN);
    TYPE* pc[TM, TN] = C + rcm[:, newaxis]*ldc + rcn[newaxis, :];
    bool  checkc[TM, TN] = rcm[:, newaxis] < M;
    // write-back directly
    if(lockid == 0) {
      *?(checkc) pc = c;
    }
    // accumulate partial result using spin-locks
    else {
      int *plock = locks + get_program_id(0)*nlocks + lockid - 1;
      int *pcount = plock + get_num_programs(0)*nlocks;
      for(int repeat = 1; repeat == 1; repeat = atomic_cas(plock, 0, 1));
      int count = *pcount;
      if(count == 0)
        *?(checkc) pc = c;
      else
        *?(checkc) pc = c + *?(checkc)pc;
      atomic_xchg(pcount, (count + 1) % maxid);
      atomic_xchg(plock, 0);
    }
  }
'''

  # dictionaries for cached triton kernels
  y_kernel = dict()
  dx_kernel = dict()
  dw_kernel = dict()

  # Given an array sizes representing reduction size for each
  # column of a block-sparse matrix multiplication,
  # performs load-balancing to achieve more smaller reductions
  # of size seg_size
  @staticmethod
  def load_balance(sizes, seg_size=8):
    div = sizes // seg_size
    rem = sizes % seg_size
    packs = div + (rem != 0).long()
    width = packs.sum()
    # split reduction into segments
    segments = torch.empty(width, dtype=sizes.dtype)
    column = torch.empty_like(segments)
    lockid = torch.zeros_like(segments)
    maxid = torch.zeros_like(segments)
    nlocks = 0
    current = 0
    col_idx = 0
    for i in range(len(sizes)):
      d, r = div[i], rem[i]
      last = current + d + (r > 0)
      # column id
      column[current:last] = col_idx
      # lock id
      if d > 1 or (d == 1 and r > 0):
        nlocks += 1
        lockid[current:last] = nlocks
        maxid[current:last] = last - current
      # segment size
      segments[current:current+d] = seg_size
      if r > 0:
        segments[current+d] = r
      current = last
      col_idx += 1
    offsets = torch.zeros_like(segments)
    offsets[1:] = torch.cumsum(segments[:-1], dim=0)
    return segments, column, lockid, maxid, offsets

  # Given a binary mask of 0s and 1s,
  # Construct look-up table for efficient execution on GPUs
  @staticmethod
  def make_ydx_lut(mask, block_size):
    # offsets in lookup table
    sizes = torch.sum(mask, 0)
    offsets = torch.zeros_like(sizes)
    offsets[1:] = torch.cumsum(sizes[:-1], dim=0)
    # load-balancing
    segments, column, lockid, maxid, offsets = dot.load_balance(sizes)
    # pointer increments
    nnz = torch.nonzero(mask.T)
    idx = nnz[:, 1]
    incs = idx.clone() 
    incs[1:] -= idx[:-1]
    incs[offsets] = idx[offsets]
    # create header
    width = column.size(0)
    offsets += 5*width
    header = torch.stack((offsets, segments, column, lockid, maxid), dim=1).view(-1).contiguous()
    # create lut
    lut = torch.cat((header, incs)).type(torch.int32).cuda()
    # create locks
    num_locks = max(1, lockid.max())
    locks = torch.zeros((2*mask.size(0), num_locks), dtype=torch.int32).cuda()
    return lut, locks, width
  
  @staticmethod
  def make_dw_lut(mask, depth, block_size):
    nnz = torch.nonzero(mask)
    # create lut
    width = nnz.size(0)
    i = nnz[:, 0]
    j = nnz[:, 1]
    lut = torch.stack((i, j), dim=1).view(-1).contiguous()
    lut = lut.type(torch.int32).cuda()
    # create locks
    num_locks = 1
    locks = torch.zeros((2*width, num_locks), dtype=torch.int32).cuda()
    return lut, locks, width

  @staticmethod
  def forward(ctx, x, w, block_size, 
              y_lut, y_locks, y_width,
              dx_lut, dx_locks, dx_width,
              dw_lut, dw_locks, dw_width):
    M, Kx = x.size()
    Kw, N = w.size()
    dtype = x.dtype
    # memory strides
    lda = Kx
    ldb = N
    ldc = N
    # create kernel
    key = (dtype, block_size)
    if key not in dot.y_kernel:
      defines = {'TM': 64, 'TN': block_size, 'TK': block_size, 'TYPE': dtype,
                'STRIDE_AM': 'lda', 'STRIDE_AK': '1',
                'STRIDE_BN': '1', 'STRIDE_BK': 'ldb'}
      dot.y_kernel[key] = triton.kernel(dot.src, defines=defines)
    kernel = dot.y_kernel[key]
    # allocate output
    y = torch.empty((M, N), dtype=dtype, device=x.device)
    # launch kernel
    grid = lambda opt: [triton.cdiv(M, opt.d('TM')), y_width]
    kernel(x, w, y, lda, ldb, ldc, M, K, y_lut, y_locks, y_locks.size(1), grid=grid)
    # save information in context
    ctx.dx_width = dx_width
    ctx.dw_width = dw_width
    ctx.kernel = kernel
    ctx.block_size = block_size
    ctx.save_for_backward(x, w, dx_lut, dx_locks, dw_lut, dw_locks)
    return y
  
  @staticmethod
  def backward(ctx, dy):
    # retrieve information in context
    x, w, dx_lut, dx_locks, dw_lut, dw_locks = ctx.saved_tensors
    dx_width = ctx.dx_width
    dw_width = ctx.dw_width
    block_size = ctx.block_size
    kernel = ctx.kernel
    # shapes
    M, N = dy.size()
    _, K = x.size()
    dtype = x.dtype
    ################
    # input gradient
    ################
    dx = None
    if ctx.needs_input_grad[0]:
      # create kernel
      key = (dtype, block_size)
      if key not in dot.dx_kernel:
        defines =  {'TM': 64, 'TN': block_size, 'TK': block_size, 'TYPE': dtype,
                    'STRIDE_AM': 'lda', 'STRIDE_AK': '1',
                    'STRIDE_BN': 'ldb', 'STRIDE_BK': '1'}
        dot.dx_kernel[key] = triton.kernel(dot.src, defines=defines)
      kernel = dot.dx_kernel[key]
      # allocate output
      dx = torch.empty_like(x)
      # launch kernel
      grid = lambda opt: [triton.cdiv(M, opt.d('TM')), dx_width]
      kernel(dy, w, dx, N, N, K, M, N, dx_lut, dx_locks, dx_locks.size(1), grid=grid)
    #################
    # weight gradient
    #################
    dw = None
    if ctx.needs_input_grad[1]:
      # create kernel
      key = (dtype, block_size)
      if key not in dot.dw_kernel:
        defines =  {'TM': block_size, 'TN': block_size, 'TK': 8, 'TYPE': dtype,
                    'STRIDE_AM': '1', 'STRIDE_AK': 'lda',
                    'STRIDE_BN': '1', 'STRIDE_BK': 'ldb',
                    'DW': True, 'TZ': 2}
        dot.dw_kernel[key] = triton.kernel(dot.src, defines=defines)
      kernel = dot.dw_kernel[key]
      # allocate output
      dw = torch.zeros_like(w)
      # launch kernel
      grid = lambda opt: [dw_width, opt.d('TZ')]
      kernel(x, dy, dw, K, N, N, K, M, dw_lut, dw_locks, dw_locks.size(1), grid=grid)
    # done
    return dx, dw, None,\
          None, None, None,\
          None, None, None,\
          None, None, None
linear = _linear.apply

class Linear(torch.nn.Module):

  def __init__(self, in_features, out_features, block_size, mask):
    super(Linear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
    self.reset_parameter()
    # create look-up tables
    self.y_lut, self.y_locks, self.y_width = _linear.make_ydx_lut(mask, block_size)
    self.dx_lut, self.dx_locks, self.dx_width = _linear.make_ydx_lut(mask.T, block_size)
    self.dw_lut, self.dw_locks, self.dw_width = _linear.make_dw_lut(mask, M, block_size)
  
   def reset_parameters(self):
      init.kaiming_uniform_(self.weight, a=math.sqrt(5))
      if self.bias is not None:
          fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
          bound = 1 / math.sqrt(fan_in)
          init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input):
      return linear(input, self.weight, self.block_size,
                    self.y_lut, self.y_locks, self.y_width,
                    self.dx_lut, self.dx_locks, self.dx_width,
                    self.dw_lut, self.dw_locks, self.dw_width)


def reference_dot(x, w, mask):
  WS0, WS1 = w.size()
  MS0, MS1 = mask.size()
  assert WS0 % MS0 == 0
  assert WS1 % MS1 == 0
  block_size_0 = WS0 // MS0
  block_size_1 = WS1 // MS1
  assert block_size_0 == block_size_1
  maskedw = w.clone()
  for bi, wi in enumerate(range(0, WS0, block_size_0)):
    for bj, wj in enumerate(range(0, WS1, block_size_1)):
      maskedw[wi : wi+block_size_0,
              wj : wj+block_size_1] *= mask[bi, bj]
  return torch.matmul(x, maskedw)

torch.manual_seed(0)
# parameters
M, N, K = 256, 256, 256
BS = 16
# initialize inputs
mask = torch.randint(0, 2, (K//BS, N//BS))
x = torch.rand((M, K), dtype=torch.float32, requires_grad=True).cuda()
w = torch.rand((K, N), dtype=torch.float32, requires_grad=True).cuda()
x.retain_grad()
w.retain_grad()
# reference result
ry = reference_dot(x, w, mask)
dy = torch.rand_like(ry)
ry.backward(dy)
rdx = x.grad.clone()
rdw = w.grad.clone()
# reset gradients
x.grad.zero_()
w.grad.zero_()
# triton result
y_lut, y_locks, y_width = _linear.make_ydx_lut(mask, BS)
dx_lut, dx_locks, dx_width = _linear.make_ydx_lut(mask.T, BS)
dw_lut, dw_locks, dw_width = _linear.make_dw_lut(mask, M, BS)
ty = _linear.apply(x, w, BS, 
               y_lut, y_locks, y_width,
               dx_lut, dx_locks, dx_width,
               dw_lut, dw_locks, dw_width)
ty.backward(dy)
tdx = x.grad.clone()
tdw = w.grad.clone()
# test
print((ty - ry).abs().max())
print((tdx - rdx).abs().max())
print((tdw - rdw).abs().max())