import triton
import math

class _batchnorm(triton.function):

  fwd_src = """
void fwdbatchnorm(float *Y, float *M, float *V,
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

  bwd_src = """
void batchnormBackward(float *DX, float *DG, float *DB,
                       float *DY, float *X, float *G,
                       float *M, float *V,
                       int DHWN, float rcpDHWN, float epsilon) {
  int rx[TM] = 0 ... TM;
  int c = get_program_id(1);
  int offset = c*DHWN;
  float g = *(G + c);
  float mean = *(M + c);
  float var = *(V + c);
  float rstd = 1 / sqrtf(var + epsilon);
  float* px[TM];
  float* pdx[TM];
  float* pdy[TM];
  px  =  X + rx + offset;
  pdy = DY + rx + offset;
  float  dg[TM] = 0;
  float  db[TM] = 0;
  for(int i = 0; i < DHWN; i = i + TM){
    float x[TM] = *px;
    float dy[TM] = *pdy;
    dg = dg + dy*(x - mean)*rstd;
    db = db + dy;
    px = px + TM;
    pdy = pdy + TM;
  }
  float sdg = dg[+];
  float sdb = db[+];
  float *pdg = DG + c;
  float *pdb = DB + c;
  *pdg = sdg;
  *pdb = sdb;
  px  =  X + rx + offset;
  pdy = DY + rx + offset;
  pdx = DX + rx + offset;
  for(int i = 0; i < DHWN; i = i + TM){
    float x[TM] = *px;
    float dy[TM] = *pdy;
    float xhat[TM] = (x - mean) * rstd;
    float xtmp[TM] = (xhat * dg + db) * rcpDHWN;
    float dx[TM] = (dy - xtmp) * rstd * g;
    *pdx = dx;
    px = px + TM;
    pdy = pdy + TM;
    pdx = pdx + TM;
  }
}
"""

  bwd_kernel = triton.kernel(bwd_src, ['DX', 'DG', 'DB'])

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
    y, mean, var = _batchnorm.fwd_kernel(y, mean, var, x, gamma, beta, N, 1./N, eps,
                        lambda opt: [1, C],
                        TM = 128)
    # save
    ctx.eps = eps
    ctx.save_for_backward(x, gamma, beta, mean, var)
    return y

  @staticmethod
  def backward(ctx, dy):
    eps = ctx.eps
    x, gamma, beta, mean, var = ctx.saved_tensors
    dx = triton.empty(x.shape, dtype=x.dtype)
    dgamma = triton.empty(gamma.shape, dtype=gamma.dtype)
    dbeta = triton.empty(beta.shape, dtype=beta.dtype)
    # launch
    C, H, W, B = x.shape
    N = H*W*B
    _batchnorm.bwd_kernel(dx, dgamma, dbeta, dy, 
                          x, gamma, mean, var, 
                          N, 1./N, eps,
                          lambda opt: [1, C],
                          TM = 128)
    return dx, dgamma, dbeta, None

batchnorm = _batchnorm.apply