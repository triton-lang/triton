import triton
import math

class _batchnorm(triton.function):

  fwd_src = """
void fwdbatchnorm(float *Y, float *M, float *V,
                  float *X, float *G, float *B,
                  int N, float eps) {
  // pointers
  int c = get_program_id(1);
  int rm[TM] = 0 ... TM;
  float *px[TM] = X + rm + c*N;
  float* py[TM] = Y + rm + c*N;

  // compute mean
  float accm[TM] = 0;
  for(int i = 0; i < N; i = i + TM)
    accm = accm + *(px + i);
  float mean = (float)accm[+] / N;
  *(M + c) = mean;

  // compute variance
  float accv[TM] = 0;
  for(int i = 0; i < N; i = i + TM){
    float x[TM] = *(px + i);
    x = x - mean;
    accv = accv + x*x;
  }
  float var = (float)accv[+] / N;
  *(V + c) = var;

  // Normalize batch
  float gamma = *(G + c);
  float beta = *(B + c);
  float rstdg = 1 / sqrtf(var + eps) * gamma;
  for(int i = 0; i < N; i = i + TM){
    float x[TM] = *(px + i);
    float y[TM] = (x - mean)*rstdg + beta;
    *(py + i) = y;
  }
}
"""
  fwd_kernel = triton.kernel(fwd_src)

  bwd_src = """
void bwdbatchnorm(float *DX, float *DG, float *DB,
                  float *DY, float *X, float *G,
                  float *M, float *V,
                  int N, float epsilon) {

   // pointers
  int c = get_program_id(1);
  int rx[TM] = 0 ... TM;
  int offset = c*N;
  float* px[TM]  =  X + rx + offset;
  float* pdy[TM] = DY + rx + offset;
  float* pdx[TM] = DX + rx + offset;

  // fetch statistics
  float gamma = *(G + c);
  float mean = *(M + c);
  float var = *(V + c);
  float rstd = 1 / sqrtf(var + epsilon);

  // compute dgamma and dbeta
  float  acc_dg[TM] = 0;
  float  acc_db[TM] = 0;
  for(int i = 0; i < N; i = i + TM){
    float x[TM] = *(px + i);
    float dy[TM] = *(pdy + i);
    acc_dg += dy*(x - mean)*rstd;
    acc_db += dy;
  }
  float dg = acc_dg[+];
  float db = acc_db[+];
  *(DG + c) = dg;
  *(DB + c) = db;

  // compute dx
  for(int i = 0; i < N; i = i + TM){
    float x[TM] = *(px + i);
    float dy[TM] = *(pdy + i);
    float xhat[TM] = (x - mean) * rstd;
    float xtmp[TM] = (xhat * dg + db) / N;
    float dx[TM] = (dy - xtmp) * rstd * gamma;
    *(pdx + i) = dx;
  }
}
"""
  bwd_kernel = triton.kernel(bwd_src)

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
    _batchnorm.fwd_kernel(y, mean, var, x, gamma, beta, H*W*B, eps,
                          grid = lambda opt: [1, C],
                          defines = {'TM': 128})
    # save
    ctx.save_for_backward(x, gamma, beta, mean, var)
    ctx.eps = eps
    return y

  @staticmethod
  def backward(ctx, dy):
    # retrieve info
    x, gamma, beta, mean, var = ctx.saved_tensors
    eps = ctx.eps
    # allocate result
    dx = triton.empty(triton.shape(x), dtype=x.dtype)
    dgamma = triton.empty(triton.shape(gamma), dtype=gamma.dtype)
    dbeta = triton.empty(triton.shape(beta), dtype=beta.dtype)
    # execute
    C, H, W, B = triton.shape(x)
    _batchnorm.bwd_kernel(dx, dgamma, dbeta, dy, 
                          x, gamma, mean, var, 
                          H*W*B, eps,
                          grid = lambda opt: [1, C],
                          defines = {'TM': 128})
    return dx, dgamma, dbeta, None

batchnorm = _batchnorm.apply