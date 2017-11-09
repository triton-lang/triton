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

enum ActivationType{
  ReLU,
  LeakyReLU
};

class Conv: public Generator{
public:
  static const std::string id;
  static const size_t Nshapes;
  static const size_t Ntune;
  static const size_t Nparams;

private:
  void init_constant_memory(std::vector<int32_t>& delta, std::vector<uint32_t> &masks, int32_t strideIc, int32_t strideIw, int32_t strideIh, int32_t strideId);

public:
  Conv(DType dtype, param_t C, param_t D, param_t H, param_t W, param_t N, param_t K, param_t M, param_t P, param_t Q, param_t T, param_t R, param_t S, param_t pad_h, param_t pad_w, param_t pad_d, param_t stride_h, param_t stride_w, param_t stride_d,
       param_t vec, param_t bpqn, param_t bk, param_t pqns, param_t ks, param_t crs_l, param_t cs, param_t bc, param_t gridc);
  // Execution
  std::string dump(driver::Device const & device, std::string const & name);
  std::vector<param_t> tuning_params() const;
  void enqueue(driver::Kernel& kernel, driver::Stream& queue, scalar const & alpha, driver::Buffer const & I, driver::Buffer const & F, scalar const & beta, driver::Buffer& O);
  // Validity
  static void output_shapes(param_t D, param_t H, param_t W, param_t T, param_t R, param_t S, param_t pad_d,
                            param_t pad_h, param_t pad_w, param_t stride_d, param_t stride_h, param_t stride_w, param_t& M,
                            param_t& P, param_t& Q);
  static void check_valid(driver::Device const & device, size_t M, param_t* params, uint8_t* valid);
  // Benchmark
  static double tflops(param_t P, param_t Q, param_t M, param_t K, param_t N, param_t C, param_t R, param_t S, param_t T, double time);

private:
  DType dtype_;
  //input shapes
  param_t C_;
  param_t N_;
  param_t K_;
  // Input dimensions
  param_t D_;
  param_t H_;
  param_t W_;
  // Output Dimensions
  param_t M_;
  param_t P_;
  param_t Q_;
  // Filter Dimensions
  param_t T_;
  param_t R_;
  param_t S_;
  // Pad
  param_t pad_d_;
  param_t pad_h_;
  param_t pad_w_;
  // stride
  param_t stride_d_;
  param_t stride_h_;
  param_t stride_w_;
  //parameters
  param_t vec_;
  param_t bc0_;
  param_t bc1_;
  param_t cs0_;
  param_t cs1_;
  param_t bf_n_;
  param_t u_;
  param_t us_;
  param_t zs_;
  param_t bz_;
  param_t gridz_;
  // constant memory
  std::vector<int32_t> cLUT;
  std::vector<uint32_t> masks_;
};

}
}

#endif
