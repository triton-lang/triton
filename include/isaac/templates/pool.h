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

#ifndef ISAAC_TEMPLATES_POOL_H_
#define ISAAC_TEMPLATES_POOL_H_

#include <cstddef>
#include <string>
#include "isaac/templates/common.hpp"

namespace isaac{

enum PoolType{
  MaxPool,
  AvgPool
};


namespace templates{

class Pool: public Generator{
private:
  void init_constant_memory(std::vector<int32_t>& delta, std::vector<uint32_t> &masks, size_t nlut, int32_t strideIc, int32_t strideIw, int32_t strideIh, int32_t strideId);

public:
  static const std::string id;
  static const size_t Nshapes;
  static const size_t Ntune;
  static const size_t Nparams;

public:
  Pool(DType in_dtype, DType out_dtype, PoolType pool_type,
       param_t C, param_t D, param_t H, param_t W, param_t N, param_t M, param_t P, param_t Q, param_t T, param_t R, param_t S,
       param_t pad_d, param_t pad_h, param_t pad_w,
       param_t stride_d, param_t stride_h, param_t stride_w,
       param_t vec = 1, param_t bc0 = 32, param_t cs0 = 4, param_t u = 1);
  // Execution
  std::string dump(driver::Device const & device, std::string const & name);
  static void check_valid(driver::Device const & device, size_t M, param_t* params, uint8_t* valid);
  void enqueue(driver::Kernel& kernel, driver::Stream& queue, driver::Buffer const & I, driver::Buffer &O, float i_scale = 1, float o_scale = 1);
  std::vector<unsigned int> tuning_params() const;
  static double tflops(param_t P, param_t Q, param_t M, param_t K, param_t N, param_t T, param_t R, param_t S, double time);

private:
  DType in_dtype_;
  DType out_dtype_;
  PoolType pool_type_;
  // Shapes
  param_t Cin_;
  param_t Cout_;
  param_t D_;
  param_t H_;
  param_t W_;
  param_t N_;
  param_t M_;
  param_t P_;
  param_t Q_;
  param_t T_;
  param_t R_;
  param_t S_;
  param_t pad_d_;
  param_t pad_h_;
  param_t pad_w_;
  param_t stride_d_;
  param_t stride_h_;
  param_t stride_w_;
  // Tuning params
  param_t vec_;
  param_t bc0_;
  param_t cs0_;
  param_t u_;
  // Constant buffer
  std::vector<int32_t> cLUT;
  std::vector<uint32_t> masks_;
};

}

}

#endif
