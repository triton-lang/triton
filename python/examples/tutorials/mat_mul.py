import torch
import triton

class _dot(torch.autograd.Function):
    src = """
    __global__ void dot(TYPE *A, TYPE *B, TYPE *C, int M, int N, int K,
                        int lda __multipleof(8), int ldb __multipleof(8), int ldc __multipleof(8)) {
        int pm = get_program_id(0);
        int pn = get_program_id(1);

        // ranges
        int rm[TM] = pm * TM + 0 ... TM;
        int rn[TN] = pn * TN + 0 ... TN;
        int rk[TK] = 0 ... TK;

        // accumulator
        float c[TM, TN] = 0;

        // pointers
        TYPE* pa[TM, TK] = A + rk[newaxis, :] * 1 + rm[:, newaxis] * lda;
        TYPE* pb[TK, TN] = B + rk[:, newaxis] * ldb + rn[newaxis, :] * 1;

        for(int k=K; k>0; k-=TK) {
            TYPE a[TM, TK] = *pa;
            TYPE b[TK, TN] = *pb;
            c += a @ b;
            pa = pa + TK * 1;
            pb = pb + TK * ldb;
        }

        TYPE* pc[TM,TN] = C + rn[newaxis, :] + rm[:,newaxis] * ldc;
        *pc = c;
    }
    """

    @staticmethod
    def forward(ctx, a, b):
        c = _dot._call(a,b)
        return c

    
    kernel = dict()

    @staticmethod
    def _call(a, b):
        # shapes
        M, K = a.shape
        K, N = b.shape
        # leading dimension
        lda = K
        ldb = N
        ldc = N
        dtype = a.dtype
        # create kernel if necessary
        if dtype not in _dot.kernel:
            defines = {
                'TYPE' : dtype,
                'TM'   : [64, 128],
                'TN'   : [64, 128],
                'TK'   : [8, 16],
            }
            _dot.kernel[dtype] = triton.kernel(_dot.src, num_warps=[2, 4], defines=defines)
        kernel = _dot.kernel[dtype]
        # allocate output
        c = triton.empty([M,N], dtype=dtype)
        # enqueue
        grid = lambda opt: [triton.cdiv(M, opt.d('TM')), 
                            triton.cdiv(N, opt.d('TN'))]
        kernel(a, b, c, M, N, K, lda, ldb, ldc, grid=grid)
        return c


dot = _dot.apply

torch.manual_seed(0)

M, N, K = 128, 512, 256
a = torch.rand((M, K)).cuda()
b = torch.rand((K, N)).cuda()


zc  = torch.matmul(a,b)
zc_ = dot(a,b)

print(torch.allclose(zc, zc_))