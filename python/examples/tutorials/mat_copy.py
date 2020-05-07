import torch
import triton

class _copy(torch.autograd.Function):
    src = """
    __global__ void copy(TYPE * X, TYPE * Y,
                                  int M, int N, int ldx __multipleof(8)) {
          // extract program ID
          int pidm = get_program_id(0); //(1)
          int pidn = get_program_id(1); //(2)

          // create 1D range along the two matrix's axes
          int rm[TM] = pidm * TM + 0 ... TM; //(3)
          int rn[TN] = pidn * TN + 0 ... TN; //(4)

          // create 2D array of pointers
          TYPE* px[TM, TN] = X + rm[:, newaxis] + rn[newaxis, :] * ldx; //(5)
          TYPE* py[TM, TN] = Y + rm[:, newaxis] + rn[newaxis, :] * ldx; //(6)

          *py = *px;
        }
    """

    kernel = None ### initialize later when we know the sizes

    @staticmethod
    def forward(ctx, x):

       M, N = x.shape

       ldx = N;

       dtype = x.dtype

       y = torch.empty((M,N)).cuda()

       defines= {
           'TYPE' : dtype,
           'TM'   : [32,64,128],
           'TN'   : [32,64,128],
       }
       grid = lambda opt: [triton.cdiv(M, opt.d('TM')), triton.cdiv(N, opt.d('TN'))]

       if _copy.kernel is None:
           _copy.kernel = triton.kernel(_copy.src, defines=defines, num_warps=[4])

       _copy.kernel(x, y, M, N, ldx, grid=grid)

       return y

copy = _copy.apply

# test
torch.manual_seed(0)
x = torch.randn(8,4).cuda()

print(x)

ya = x
yb = copy(x)

print()
print(ya)
print()
print(yb)
print(torch.allclose(ya, yb))

print(ya == yb)
