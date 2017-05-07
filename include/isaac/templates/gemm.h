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

#ifndef ISAAC_TEMPLATES_GEMM_H_
#define ISAAC_TEMPLATES_GEMM_H_

#include <cstddef>
#include <string>
#include "isaac/templates/common.hpp"
#include "isaac/scalar.h"

namespace isaac{

namespace driver{
  class Device;
  class Stream;
  class Kernel;
  class Buffer;
}

enum IsaacOperation_t{
  ISAAC_OP_N = 1,
  ISAAC_OP_T = 2
};


namespace templates{

class GEMM: public Generator{
public:
  GEMM(DType dtype, IsaacOperation_t AT, IsaacOperation_t BT, param_t M, param_t N, param_t K, param_t offa, param_t lda, param_t offb, param_t ldb, param_t offc, param_t ldc,
       param_t vec, param_t bm, param_t kl, param_t bn, param_t ms, param_t ks, param_t ns, param_t a_bf0, param_t a_bf1, param_t b_bf0, param_t b_bf1,
       param_t rs, param_t br, param_t gridr);
  std::string dump(driver::Device const & device, std::string const & name);
  std::string id() const;
  void enqueue(driver::Kernel& kernel, driver::Stream& queue, scalar const & alpha, driver::Buffer const & A, driver::Buffer const & B, scalar const & beta, driver::Buffer& C);
  static void check_valid(driver::Device const & device, size_t M, param_t* params, uint8_t* valid);
  static double tflops(param_t M, param_t N, param_t K, double time);

private:
  DType dtype_;
  //transposition
  IsaacOperation_t AT_;
  IsaacOperation_t BT_;
  //input shapes
  param_t M_;
  param_t N_;
  param_t K_;
  param_t offa_;
  param_t lda_;
  param_t offb_;
  param_t ldb_;
  param_t offc_;
  param_t ldc_;
  //parameters
  param_t vec_;
  param_t bm_;
  param_t kl_;
  param_t bn_;
  param_t ms_;
  param_t ks_;
  param_t ns_;
  param_t a_bf0_;
  param_t a_bf1_;
  param_t b_bf0_;
  param_t b_bf1_;
  param_t rs_;
  param_t br_;
  param_t gridr_;
};

}
}

#endif
