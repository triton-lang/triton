import torch
import triton

class _conv(torch.autograd.Function):
    src = """
    __global__ void conv(TYPE *A __noalias __readonly __aligned(16), 
                         TYPE *B __noalias __readonly __aligned(16), 
                         TYPE *C __noalias __aligned(16), 
                         float alpha,
                         // equivalent matmul
                         int M __retune,
                         int N __retune,
                         int K __retune,
                         // convolution properties
                         int pad_h, int pad_w, int stride_h, int stride_w,
                         // pointer increment
                         int *ADELTA,
                         // memory strides
                         int lda_z  __multipleof(8), int lda_ci __multipleof(8), int lda_h __multipleof(8), int lda_w __multipleof(8),
                         int ldb_ci __multipleof(8), int ldb_r  __multipleof(8), int ldb_s __multipleof(8), int ldb_co __multipleof(8),
                         int ldc_z  __multipleof(8), int ldc_co __multipleof(8), int ldc_p __multipleof(8), int ldc_q __multipleof(8)) {
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

      // unpack aggregate rows
      // m = (z, p, q)
      int rq[TM]   = rm  % QQ;
      int rzp[TM]  = rm  / QQ;
      int rp[TM]   = rzp % PP;
      int rz[TM]   = rzp / PP;
      // unpack aggregate reduction
      // k = (ci, r, s)
      int rs  [TK] = rk % SS;
      int rcir[TK] = rk / SS;
      int rr  [TK] = rcir % RR;
      int rci [TK] = rcir / RR;

      // padding / striding
      int rh_0[TM] = rp * stride_h - pad_h;
      int rw_0[TM] = rq * stride_w - pad_w;
      int rh[TM, TK] = rh_0[:, newaxis] + rr[newaxis, :];
      int rw[TM, TK] = rw_0[:, newaxis] + rs[newaxis, :];

      // pointers to lhs
      int offa[TM, TK] = rz [:, newaxis]  * lda_z  +
                         rci[newaxis, :]  * lda_ci +
                         rh               * lda_h  +
                         rw               * 1;
      TYPE* pa[TM, TK] = A + offa;
      int* padelta[TK] = ADELTA + rk;
      // pointers to rhs
      int offb[TK, TN] = rci[:, newaxis] * ldb_ci +
                         rr [:, newaxis] * ldb_r  +
                         rs [:, newaxis] * ldb_s  +
                         rn [newaxis, :] * 1;
      TYPE* pb[TK, TN] = B + offb;

      // prefetches operands
      bool checkam[TM, TK] = rm[:, newaxis] < M;
      bool checka[TM, TK] = checkam && rh >= 0 && rh < HH && rw >= 0 && rw < WW;
      bool checkb[TK, TN] = rk[:, newaxis] < K;
      TYPE a[TM, TK] = checka ? *pa : 0;
      TYPE b[TK, TN] = checkb ? *pb : 0;
      int total = 0;

      // reduction loop
      float acc[TM, TN] = 0;
      for(int k = K; k > 0; k -= TK){
        acc += a @ b;
        // increment A
        int adelta[TK] = *padelta;
        padelta += TK;
        pa += adelta[newaxis, :];
        // bounds-checking A
        rk += TK;
        rs = rk % SS;
        rcir = rk / SS;
        rr = rcir % RR;
        rh = rh_0[:, newaxis] + rr[newaxis, :];
        rw = rw_0[:, newaxis] + rs[newaxis, :];
        bool checka[TM, TK] = checkam && rh >= 0 && rh < HH && rw >= 0 && rw < WW;
        // increment B
        pb += TK * ldb_s;
        // bounds-checking B
        bool checkb[TK, TN] = k > TK;
        a = checka ? *pa : 0;
        b = *?(checkb)pb;
      }
      acc = acc * alpha;
      TYPE c[TM, TN] = acc;

      // epilogue
      rm  = ridx * TM + 0 ... TM;
      rn  = ridy * TN + 0 ... TN;
      rq  = rm  % QQ;
      rzp = rm  / QQ;
      rp  = rzp % PP;
      rz  = rzp / PP;
      int offc[TM, TN] = rz [:, newaxis] * ldc_z +
                         rn [newaxis, :] * ldc_co+
                         rp [:, newaxis] * ldc_p +
                         rq [:, newaxis] * 1;
      TYPE* pc[TM, TN] = C + offc;
      bool checkc[TM, TN] = rm[:, newaxis] < M && rn[newaxis, :] < N;

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

    kernel = dict()

    @staticmethod
    def unpack(IDX, CI, R, S):
      s  = IDX %  S
      cr = IDX // S
      r  = cr  %  R
      ci = cr  // R
      return ci, r, s

    @staticmethod
    def forward(ctx, a, b, pad, stride, time):
      # create kernel if necessary
      dtype = a.dtype
      # shapes
      Z, CI, H, W = a.shape
      _, R, S, CO = b.shape
      P = (H + 2*pad[0] - R)//stride[0] + 1
      Q = (W + 2*pad[1] - S)//stride[1] + 1
      # compile kernel
      if dtype not in _conv.kernel:
          TK = 8
          defines = {
              'TYPE' : dtype,
              'TM'   : [16, 32, 64, 128],
              'TN'   : [16, 32, 64, 128],
              'TK'   : [TK],
              'TZ'   : [1],
              'HH': H, 'WW': W, 'PP': P, 'QQ': Q, 'SS': S, 'RR': R,
          }
          idx = torch.arange(CI*R*S)
          ci,   r,  s = _conv.unpack(idx, CI, R, S)
          nci, nr, ns = _conv.unpack(idx + TK, CI, R, S)
          delta = (nci - ci)*a.stride(1) + (nr - r)*a.stride(2) + (ns - s)*a.stride(3)
          delta = delta.type(torch.int32).cuda()
          _conv.kernel[dtype] = (delta, triton.kernel(_conv.src, num_warps=[2, 4], defines=defines))
      delta, kernel = _conv.kernel[dtype]
      # allocate output
      c = triton.empty([Z, CO, P, Q], dtype=dtype)
      # enqueue
      grid = lambda opt: [triton.cdiv(Z*P*Q, opt.d('TM')), 
                          triton.cdiv(CO, opt.d('TN'))]
      time[0] = kernel(a, b, c, 1., Z*P*Q, CO, CI*R*S, 
                    pad[0], pad[1], stride[0], stride[1],
                    delta,
                    a.stride(0), a.stride(1), a.stride(2), a.stride(3),
                    b.stride(0), b.stride(1), b.stride(2), b.stride(3),
                    c.stride(0), c.stride(1), c.stride(2), c.stride(3),
                    grid=grid, bench=100)
      return c



conv = _conv.apply
torch.manual_seed(0)
Z, H, W, CI, CO, R, S = 1, 56, 56, 1024, 1024, 3, 3
pad = (1, 1)
stride = (1, 1)
a = torch.rand((Z, CI, H, W)).cuda()
b = torch.rand((CI, R, S, CO)).cuda()
time = [None]
cc = torch.nn.functional.conv2d(a, b.permute(3,0,1,2), None, stride, pad, [1, 1])
c = conv(a, b, pad, stride, time)
print((cc - c).abs().max() / max(cc.max(), c.max()))
print(time[0], 2*Z*H*W*CI*CO*R*S/(time[0]*1e-9)*1e-12)
#zc  = torch.matmul(a,b)
#zc_ = dot(a,b)
