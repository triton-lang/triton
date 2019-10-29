import triton
import math

class _batchnorm(triton.function):

  fwd_src = """
void batchnormForward(float *Y, float *M, float *V,
                  float *X, float *G, float *B,
                  int N, float rcpN, float eps) {
  int rx[TM] = 0 ... TM;
  float *px[TM];
  float x[TM] = 0;
  int c = get_program_id(1);
  float g = *(G + c);
  float b = *(B + c);

  float mean[TM] = 0;
  px = X + rx + c*N;
  for(int i = 0; i < N; i = i + TM){
    x = *px;
    mean = mean + x;
    px = px + TM;
  }
  float *pm = M + c;
  float m = mean[+] * rcpN;
  *pm = m;

  float var[TM] = 0;
  px = X + rx + c*N;
  for(int i = 0; i < N; i = i + TM){
    x = *px;
    x = x - m;
    var = var + x*x;
    px = px + TM;
  }
  float v = var[+] * rcpN;
  float *pv = V + c;
  *pv = v;
  float rstdg = 1 / sqrtf(v + eps) * g;

  px = X + rx + c*N;
  float* py[TM] = Y + rx + c*N;
  for(int i = 0; i < N; i = i + TM){
    x = *px;
    float y[TM] = (x - m)*rstdg + b;
    *py = y;
    px = px + TM;
    py = py + TM;
  }
}
"""
  
  fwd_kernel = triton.kernel(fwd_src, ['Y', 'M', 'V'])

  @staticmethod
  def forward(ctx, x, gamma, beta, eps):
    shape = triton.shape(x)
    dtype = x.dtype
    # allocate outputs
    C, H, W, B = shape[0], shape[1], shape[2], shape[3]
    y = triton.empty(shape, dtype=dtype)
    mean = triton.empty([C], dtype=dtype)
    var = triton.empty([C], dtype=dtype)
    # execute kernels
    N = H*W*B
    _batchnorm.fwd_kernel(y, mean, var, x, gamma, beta, N, 1./N, eps,
                        lambda opt: [1, C],
                        TM = 128)
    # save
    ctx.eps = eps
    ctx.save_for_backward(x, gamma, beta, mean, var)
    return y, mean, var


batchnorm = _batchnorm.apply