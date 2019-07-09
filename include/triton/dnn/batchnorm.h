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

#ifndef TDL_INCLUDE_DNN_BATCHNORM_H
#define TDL_INCLUDE_DNN_BATCHNORM_H

#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include "triton/driver/stream.h"
#include "triton/driver/kernel.h"

namespace triton{
namespace dnn{

class batchnorm_forward {
public:
  // constructor
  batchnorm_forward(int C, int D, int H, int W, int B, std::string ty = "fp32");
  // enqueue
  void enqueue(driver::stream *stream, driver::kernel *kernel,
               driver::buffer *y, driver::buffer *m, driver::buffer *v,
               driver::buffer *x, driver::buffer *g, driver::buffer *b,
               size_t TM, size_t nthreads);
  // triton-c source code
  void src(std::ostream &os);

private:
  int32_t C_;
  int32_t D_;
  int32_t H_;
  int32_t W_;
  int32_t B_;
  std::string ty_;
  float eps_;
  int32_t DHWB_;
  float rcpDHWB_;
};

class batchnorm_backward {
public:
  // constructor
  batchnorm_backward(int C, int D, int H, int W, int B, std::string ty = "fp32");
  // enqueue
  void enqueue(driver::stream *stream, driver::kernel *kernel,
               driver::buffer *dx, driver::buffer *dg, driver::buffer *db, driver::buffer *dy,
               driver::buffer *x, driver::buffer *g, driver::buffer *m, driver::buffer *v,
               size_t TM, size_t nthreads);
  // triton-c source code
  void src(std::ostream &os);


private:
  int32_t C_;
  int32_t D_;
  int32_t H_;
  int32_t W_;
  int32_t B_;
  std::string ty_;
};

}
}

#endif
