import torch
import triton

class _dot(torch.autograd.Function):
    src = """
    __global__ void dot(TYPE *A __noalias __readonly __aligned(16), 
                        TYPE *B __noalias __readonly __aligned(16), 
                        TYPE *C __noalias __aligned(16), 
                        float alpha,
                        int M __retune,
                        int N __retune,
                        int K __retune,
                        int lda __multipleof(8), 
                        int ldb __multipleof(8), 
                        int ldc __multipleof(8)) {
      // prologue
      int ridx = get_program_id(0);
      int ridy = get_program_id(1);
      int ridz = get_program_id(2);
      int gridx = M / TM;
      int gridy = N / TN;
      int rid = ridx + ridy * gridx;
      ridx = rid / gridy;
      ridy = rid % gridy;
      int rm[TM] = ridx * TM + 0 ... TM;
      int rn[TN] = ridy * TN + 0 ... TN;

      // reduction splitting
      K           = K / TZ;
      int rk[TK]  = ridz * K + 0 ... TK;

      // pointers to operands
      int offa[TM, TK] = rk[newaxis, :] * STRIDE_AK + rm[:, newaxis] * STRIDE_AM;
      int offb[TK, TN] = rk[:, newaxis] * STRIDE_BK + rn[newaxis, :] * STRIDE_BN;
      TYPE* pa[TM, TK] = A + offa;
      TYPE* pb[TK, TN] = B + offb;

      // prefetches operands
      bool checka[TM, TK] = rk[newaxis, :] < K;
      bool checkb[TK, TN] = rk[:, newaxis] < K;
      TYPE a[TM, TK] = checka ? *pa : 0;
      TYPE b[TK, TN] = checkb ? *pb : 0;

      // reduction loop
      float acc[TM, TN] = 0;
      for(int k = K; k > 0; k -= TK){
        acc += a @ b;
        bool checka[TM, TK] = k > TK;
        bool checkb[TK, TN] = k > TK;
        pa += TK * STRIDE_AK;
        pb += TK * STRIDE_BK;
        a = *?(checka)pa;
        b = *?(checkb)pb;
      }
      acc = acc * alpha;
      TYPE c[TM, TN] = acc;

      // epilogue
      int rxm[TM] = ridx * TM + 0 ... TM;
      int rxn[TN] = ridy * TN + 0 ... TN;
      int offc[TM, TN] = rxm[:, newaxis] * ldc + rxn[newaxis, :];
      TYPE* pc[TM, TN] = C + offc;
      bool checkc[TM, TN] = (rxm[:, newaxis] < M) && (rxn[newaxis, :] < N);

#if (TZ==1)
      *?(checkc) pc = c;
#else
      // accumulate partial result using spin-locks
      int *plock  = locks + rid;
      int *pcount = plock + get_num_programs(0) * get_num_programs(1);
      for(int repeat = 1; repeat == 1; repeat = atomic_cas(plock, 0, 1));
      int count = *pcount;
      if(count == 0)
        *?(checkc) pc = c;
      else
        *?(checkc) pc = c + *?(checkc)pc;
      atomic_xchg(pcount, (count + 1) % TZ);
      atomic_xchg(plock, 0);
#endif
}
    """

    @staticmethod
    def forward(ctx, a, b):
        c = _dot._call(a,b)
        return c

    
    kernel = dict()

    @staticmethod
    def _call(a, b):
        # create kernel if necessary
        dtype = a.dtype
        if dtype not in _dot.kernel:
            defines = {
                'TYPE' : dtype,
                'STRIDE_AM': 'lda', 'STRIDE_AK': '1',
                'STRIDE_BN': '1', 'STRIDE_BK': 'ldb',
                'TM'   : [64, 128],
                'TN'   : [64, 128],
                'TK'   : [8, 16],
                'TZ'   : [1]
            }
            _dot.kernel[dtype] = triton.kernel(_dot.src, num_warps=[4], defines=defines)
        kernel = _dot.kernel[dtype]
        # allocate output
        M, K = a.shape
        K, N = b.shape
        c = torch.empty([M,N], dtype=dtype, device=a.device)
        # enqueue
        grid = lambda opt: [triton.cdiv(M, opt.d('TM')), 
                            triton.cdiv(N, opt.d('TN'))]
        time = kernel(a, b, c, 1., M, N, K, 
                      a.stride(0), b.stride(0), c.stride(0), grid=grid)
        return c


dot = _dot.apply

torch.manual_seed(0)

M, N, K = 2048, 2048, 2048
a = torch.rand((M, K)).cuda()
b = torch.rand((K, N)).cuda()

#a[:] = 1
#b[:] = 1

zc  = torch.matmul(a,b)
zc_ = dot(a,b)

print(torch.allclose(zc, zc_))
