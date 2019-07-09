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

batchnorm_forward::batchnorm_forward(int C, int D, int H, int W, int B, std::string ty)
  : C_(C), D_(D), H_(H), W_(W), B_(B), ty_(ty) { }

void batchnorm_forward::enqueue(driver::stream *stream, driver::kernel *kernel,
                        driver::buffer *y, driver::buffer *m, driver::buffer *v,
                        driver::buffer *x, driver::buffer *g, driver::buffer *b,
                        size_t, size_t nthreads) {

  std::array<size_t, 3> grid = {(size_t)C_, 1, 1};
  kernel->setArg(0, y);
  kernel->setArg(1, m);
  kernel->setArg(2, v);
  kernel->setArg(3, x);
  kernel->setArg(4, g);
  kernel->setArg(5, b);
  kernel->setArg(6, (int32_t)(D_*H_*W_*B_));
  stream->enqueue(kernel, grid, {nthreads, 1, 1});
}

void batchnorm_forward::src(std::ostream &os) {
  os <<
R"(
const tunable int32 TM = {32, 64, 128};

void batchnorm(fp32 *Y, fp32 *M, fp32 *V,
               restrict read_only fp32 *X,
               restrict read_only fp32 *G,
               restrict read_only fp32 *B,
               int32 DHWN) {
  int32 rx[TM] = 0 ... TM;
  fp32 *px[TM];
  fp32 x[TM];
  int32 c = get_range_id(0);
  fp32 g = *(G + c);
  fp32 b = *(B + c);

  fp32 mean[TM] = 0;
  px = X + rx + c*DHWN;
  for(int32 i = 0; i < DHWN; i = i + TM){
    x = *px;
    mean = mean + x;
    px = px + TM;
  }
  fp32 m = __sum(mean);
  fp32 *pm = M + c;
  *pm = m;

  fp32 var[TM] = 0;
  px = X + rx + c*DHWN;
  for(int32 i = 0; i < DHWN; i = i + TM){
    x = *px;
    x = x - mean;
    var = var + x*x;
    px = px + TM;
  }
  fp32 v = __sum(var);
  fp32 *pv = V + c;
  *pv = v;
})";
}

/* ---------------
 *    Backward
 * --------------- */

batchnorm_backward::batchnorm_backward(int C, int D, int H, int W, int B, std::string ty)
  : C_(C), D_(D), H_(H), W_(W), B_(B), ty_(ty)
{ }

void batchnorm_backward::enqueue(driver::stream *stream, driver::kernel *kernel,
                        driver::buffer *dx, driver::buffer *dg, driver::buffer *db, driver::buffer *dy,
                        driver::buffer *x, driver::buffer *g, driver::buffer *m, driver::buffer *v,
                        size_t, size_t nthreads) {

  std::array<size_t, 3> grid = {(size_t)C_, 1, 1};
  kernel->setArg(0, dx);
  kernel->setArg(1, dg);
  kernel->setArg(2, db);
  kernel->setArg(3, dy);
  kernel->setArg(4, x);
  kernel->setArg(5, g);
  kernel->setArg(6, m);
  kernel->setArg(7, v);
  kernel->setArg(8, (int32_t)(D_*H_*W_*B_));
  stream->enqueue(kernel, grid, {nthreads, 1, 1});
}

void batchnorm_backward::src(std::ostream &os) {
  os <<
R"(
const tunable int32 TM = {32, 64, 128};

void batchnorm(fp32 *DX, fp32 *DG, fp32 *DB,
               restrict read_only fp32 *DY,
               restrict read_only fp32 *X,
               restrict read_only fp32 *G,
               restrict read_only fp32 *M,
               restrict read_only fp32 *V,
               int32 DHWN) {
  int32 rx[TM] = get_global_range[TM](0);
  int32 c = get_range_id(0);
  int32 offset = c*DHWN;
  fp32 g = *(G + c);
  fp32 mean = *(M + c);
  fp32 var = *(V + c);
  fp32 rstd = var;
  fp32* px[TM];
  fp32* pdx[TM];
  fp32* pdy[TM];

  px  =  X + rx + offset;
  pdy = DY + rx + offset;
  fp32  dg[TM] = 0;
  fp32  db[TM] = 0;
  for(int32 i = 0; i < DHWN; i += TM){
    fp32 x[TM] = *px;
    fp32 dy[TM] = *pdy;
    dg = dg + dy*(x - mean)*rstd;
    db = db + dy;
  }

  px  =  X + rx + offset;
  pdy = DY + rx + offset;
  pdx = DX + rx + offset;
  for(int32 i = 0; i < DHWN; i += TM){
    fp32 xhat[TM] = (x - mean) * rstd;
    fp32 xtmp[TM] = (xhat * dg + db) * NDHW;
    fp32 dx[TM] = (dy - xtmp) * rstd * g;
    *pdx = dx;
  }
})";
}

}
}
