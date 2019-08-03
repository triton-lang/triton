/* Copyright 2015-2019 Philippe Tillet
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files
* (the "Software"), to deal in the Software without restriction,
* including without limitation the rights to use, copy, modify, merge,
* publish, distribute, sublicense, and/or sell copies of the Software,
* and to permit persons to whom the Software is furnished to do so,
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "triton/dnn/batchnorm.h"

namespace triton{
namespace dnn{

/* ---------------
 *    Forward
 * --------------- */

batchnorm_forward::batchnorm_forward(int C, int D, int H, int W, int B, std::string ty, float eps)
  : base("batchnorm_forward"),
    C_(C), D_(D), H_(H), W_(W), B_(B), ty_(ty), eps_(eps) {
  DHWB_ = D_*H_*W_*B_;
  rcpDHWB_ = (float)1 / DHWB_;
}

size_t batchnorm_forward::num_flops() const {
  return C_*DHWB_;
}


std::vector<int64_t> batchnorm_forward::retune_params() const {
  return {C_, D_, H_, W_, B_};
}

base* batchnorm_forward::clone() const {
  return new batchnorm_forward(*this);
}

void batchnorm_forward::enqueue_impl(driver::stream *stream, driver::kernel *kernel,
                                     std::vector<driver::buffer*> args,
                                     runtime::launch_information info)
{
  driver::buffer *y = args[0], *m = args[1], *v = args[2];
  driver::buffer *x = args[3], *g = args[4], *b = args[5];
  std::array<size_t, 3> grid = {1, (size_t)C_, 1};
  kernel->setArg(0, y);
  kernel->setArg(1, m);
  kernel->setArg(2, v);
  kernel->setArg(3, x);
  kernel->setArg(4, g);
  kernel->setArg(5, b);
  kernel->setArg(6, DHWB_);
  kernel->setArg(7, rcpDHWB_);
  kernel->setArg(8, eps_);
  stream->enqueue(kernel, grid, {info.num_threads, 1, 1});
}

void batchnorm_forward::triton_c_src(std::ostream &os) const {
  os <<
R"(
const tunable int TM = {32, 64, 128};

void batchnorm_forward(float *Y, float *M, float *V,
               restrict read_only float *X,
               restrict read_only float *G,
               restrict read_only float *B,
               int DHWN,
               float rcpDHWN, float eps) {
  int rx[TM] = 0 ... TM;
  float *px[TM];
  float x[TM] = 0;
  int c = get_range_id(1);
  float g = *(G + c);
  float b = *(B + c);

  float mean[TM] = 0;
  px = X + rx + c*DHWN;
  for(int i = 0; i < DHWN; i = i + TM){
    x = *px;
    mean = mean + x;
    px = px + TM;
  }
  float *pm = M + c;
  float m = __sum(mean) * rcpDHWN;
  *pm = m;

  float var[TM] = 0;
  px = X + rx + c*DHWN;
  for(int i = 0; i < DHWN; i = i + TM){
    x = *px;
    x = x - m;
    var = var + x*x;
    px = px + TM;
  }
  float v = __sum(var) * rcpDHWN;
  float *pv = V + c;
  *pv = v;
  float rstdg = 1 / sqrt(v + eps) * g;

  px = X + rx + c*DHWN;
  float* py[TM] = Y + rx + c*DHWN;
  for(int i = 0; i < DHWN; i = i + TM){
    x = *px;
    float y[TM] = (x - m)*rstdg + b;
    *py = y;
    px = px + TM;
    py = py + TM;
  }
})";
}

/* ---------------
 *    Backward
 * --------------- */

batchnorm_backward::batchnorm_backward(int C, int D, int H, int W, int B, std::string ty, float eps)
  : base("batchnorm_backward"),
    C_(C), D_(D), H_(H), W_(W), B_(B),
    ty_(ty), eps_(eps)
{ }

size_t batchnorm_backward::num_flops() const {
  return C_*D_*H_*W_*B_;
}

std::vector<int64_t> batchnorm_backward::retune_params() const {
  return {C_, D_, H_, W_, B_};
}

base* batchnorm_backward::clone() const {
  return new batchnorm_backward(*this);
}

void batchnorm_backward::enqueue_impl(driver::stream *stream, driver::kernel *kernel,
                                      std::vector<driver::buffer *> args,
                                      runtime::launch_information info) {
  driver::buffer *dx = args[0], *dg = args[1], *db = args[2], *dy = args[3];
  driver::buffer *x = args[4], *g = args[5], *m = args[6], *v = args[7];
  std::array<size_t, 3> grid = {1, (size_t)C_, 1};
  kernel->setArg(0, dx);
  kernel->setArg(1, dg);
  kernel->setArg(2, db);
  kernel->setArg(3, dy);
  kernel->setArg(4, x);
  kernel->setArg(5, g);
  kernel->setArg(6, m);
  kernel->setArg(7, v);
  kernel->setArg(8, (int32_t)(D_*H_*W_*B_));
  kernel->setArg(9, (float)1/(D_*H_*W_*B_));
  kernel->setArg(10, eps_);
  stream->enqueue(kernel, grid, {info.num_threads, 1, 1});
}

void batchnorm_backward::triton_c_src(std::ostream &os) const {
  os <<
R"(
const tunable int TM = {32, 64, 128};

void batchnorm_backward(float *DX, float *DG, float *DB,
               restrict read_only float *DY,
               restrict read_only float *X,
               restrict read_only float *G,
               restrict read_only float *M,
               restrict read_only float *V,
               int DHWN, float rcpDHWN, float epsilon) {
  int rx[TM] = 0 ... TM;
  int c = get_range_id(1);
  int offset = c*DHWN;
  float g = *(G + c);
  float mean = *(M + c);
  float var = *(V + c);
  float rstd = 1 / sqrt(var + epsilon);
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
  float sdg = __sum(dg);
  float sdb = __sum(db);
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
})";
}

}
}
