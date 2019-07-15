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
  : base("batchnorm"),
    C_(C), D_(D), H_(H), W_(W), B_(B), ty_(ty), eps_(eps) {
  DHWB_ = D_*H_*W_*B_;
  rcpDHWB_ = (float)1 / DHWB_;
}

size_t batchnorm_forward::num_flops() const {
  return C_*DHWB_;
}

bool batchnorm_forward::operator <(const base& other) const {
  auto *y = dynamic_cast<const batchnorm_forward*>(&other);
  if(!y)
    return true;
  return std::tie(C_, D_, H_, W_, B_, ty_)
       < std::tie(y->C_, y->D_, y->H_, y->W_, y->B_, y->ty_);
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
const tunable int32 TM = {32, 64, 128};

void batchnorm(fp32 *Y, fp32 *M, fp32 *V,
               restrict read_only fp32 *X,
               restrict read_only fp32 *G,
               restrict read_only fp32 *B,
               int32 DHWN,
               fp32 rcpDHWN, fp32 eps) {
  int32 rx[TM] = 0 ... TM;
  fp32 *px[TM];
  fp32 x[TM];
  int32 c = get_range_id(1);
  fp32 g = *(G + c);
  fp32 b = *(B + c);

  fp32 mean[TM] = 0;
  px = X + rx + c*DHWN;
  for(int32 i = 0; i < DHWN; i = i + TM){
    x = *px;
    mean = mean + x;
    px = px + TM;
  }
  fp32 *pm = M + c;
  fp32 m = __sum(mean) * rcpDHWN;
  *pm = m;

  fp32 var[TM] = 0;
  px = X + rx + c*DHWN;
  for(int32 i = 0; i < DHWN; i = i + TM){
    x = *px;
    x = x - m;
    var = var + x*x;
    px = px + TM;
  }
  fp32 v = __sum(var) * rcpDHWN;
  fp32 *pv = V + c;
  *pv = v;
  fp32 rstdg = 1 / sqrt(v + eps) * g;

  px = X + rx + c*DHWN;
  fp32* py[TM] = Y + rx + c*DHWN;
  for(int32 i = 0; i < DHWN; i = i + TM){
    x = *px;
    fp32 y[TM] = (x - m)*rstdg + b;
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
  : base("batchnorm"),
    C_(C), D_(D), H_(H), W_(W), B_(B),
    ty_(ty), eps_(eps)
{ }

size_t batchnorm_backward::num_flops() const {
  return C_*D_*H_*W_*B_;
}

bool batchnorm_backward::operator <(const base& other) const {
  auto *y = dynamic_cast<const batchnorm_backward*>(&other);
  if(!y)
    return true;
  return std::tie(C_, D_, H_, W_, B_, ty_)
       < std::tie(y->C_, y->D_, y->H_, y->W_, y->B_, y->ty_);
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
const tunable int32 TM = {32, 64, 128};

void batchnorm(fp32 *DX, fp32 *DG, fp32 *DB,
               restrict read_only fp32 *DY,
               restrict read_only fp32 *X,
               restrict read_only fp32 *G,
               restrict read_only fp32 *M,
               restrict read_only fp32 *V,
               int32 DHWN, fp32 rcpDHWN, fp32 epsilon) {
  int32 rx[TM] = 0 ... TM;
  int32 c = get_range_id(1);
  int32 offset = c*DHWN;
  fp32 g = *(G + c);
  fp32 mean = *(M + c);
  fp32 var = *(V + c);
  fp32 rstd = 1 / sqrt(var + epsilon);
  fp32* px[TM];
  fp32* pdx[TM];
  fp32* pdy[TM];

  px  =  X + rx + offset;
  pdy = DY + rx + offset;
  fp32  dg[TM] = 0;
  fp32  db[TM] = 0;
  for(int32 i = 0; i < DHWN; i = i + TM){
    fp32 x[TM] = *px;
    fp32 dy[TM] = *pdy;
    dg = dg + dy*(x - mean)*rstd;
    db = db + dy;
    px = px + TM;
    pdy = pdy + TM;
  }
  fp32 sdg = __sum(dg);
  fp32 sdb = __sum(db);
  fp32 *pdg = DG + c;
  fp32 *pdb = DB + c;
  *pdg = sdg;
  *pdb = sdb;

  px  =  X + rx + offset;
  pdy = DY + rx + offset;
  pdx = DX + rx + offset;
  for(int32 i = 0; i < DHWN; i = i + TM){
    fp32 x[TM] = *px;
    fp32 dy[TM] = *pdy;
    fp32 xhat[TM] = (x - mean) * rstd;
    fp32 xtmp[TM] = (xhat * dg + db) * rcpDHWN;
    fp32 dx[TM] = (dy - xtmp) * rstd * g;
    *pdx = dx;
    px = px + TM;
    pdy = pdy + TM;
    pdx = pdx + TM;
  }
})";
}

}
}
