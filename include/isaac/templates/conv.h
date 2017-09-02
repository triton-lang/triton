/* Copyright 2015-2017 Philippe Tillet
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

#ifndef ISAAC_TEMPLATES_CONV_H_
#define ISAAC_TEMPLATES_CONV_H_

#include <cstddef>
#include <string>
#include "isaac/templates/common.hpp"

namespace isaac{

namespace templates{

class Conv: public Generator{
public:
  Conv(DType dtype, param_t C, param_t H, param_t W, param_t N, param_t K, param_t P, param_t Q, param_t R, param_t S, param_t pad_h, param_t pad_w, param_t stride_h, param_t stride_w,
       param_t vec, param_t bp, param_t bq, param_t bn, param_t bk,  param_t bf_n, param_t ps, param_t qs, param_t ns, param_t ks, param_t crs_l, param_t crs_s, param_t cs, param_t bc, param_t gridc);
  // Execution
  std::string id() const;
  std::string dump(driver::Device const & device, std::string const & name);
  std::vector<param_t> tuning_params() const;
  void enqueue(driver::Kernel& kernel, driver::Stream& queue, scalar const & alpha, driver::Buffer const & I, driver::Buffer const & F, scalar const & beta, driver::Buffer& O);
  // Validity
  static void check_valid(driver::Device const & device, size_t M, param_t* params, uint8_t* valid);
  // Benchmark
  static double tflops(param_t P, param_t Q, param_t K, param_t N, param_t C, param_t R, param_t S, double time);

private:
  DType dtype_;
  //input shapes
  param_t C_;
  param_t H_;
  param_t W_;
  param_t N_;
  param_t K_;
  param_t P_;
  param_t Q_;
  param_t R_;
  param_t S_;
  param_t pad_h_;
  param_t pad_w_;
  param_t stride_h_;
  param_t stride_w_;
  //parameters
  param_t vec_;
  param_t bp_;
  param_t bq_;
  param_t bn_;
  param_t bk_;
  param_t bf_n_;
  param_t ps_;
  param_t qs_;
  param_t ns_;
  param_t ks_;
  param_t crs_l_;
  param_t crs_s_;
  param_t cs_;
  param_t bc_;
  param_t gridc_;
};

}
}

#endif
