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

#include <tuple>

#include "isaac/runtime/predict.h"
#include "isaac/driver/backend.h"
#include "isaac/driver/cublas.h"
#include "isaac/driver/context.h"
#include "isaac/driver/kernel.h"
#include "isaac/driver/buffer.h"
#include "isaac/driver/stream.h"
#include "isaac/tools/bench.hpp"
#include "isaac/tools/collections.hpp"
#include "isaac/templates/conv.h"
#include "isaac/templates/gemm.h"
#include "isaac/templates/pool.h"

namespace isaac{

void GEMM(driver::Device const & device, driver::Stream & stream,
          DType dtype, IsaacOperation_t AT, IsaacOperation_t BT, param_t M, param_t N, param_t K,
          size_t offa, size_t lda, size_t offb, size_t ldb, size_t offc, size_t ldc,
          scalar const & alpha, driver::Buffer const & A, driver::Buffer const & B, scalar const & beta, driver::Buffer& C,
          templates::GEMM* generator = NULL);

void CONV(driver::Device const & device, driver::Stream & stream,
          DType dtype, param_t N, param_t K, param_t M, param_t P, param_t Q, param_t C, param_t T, param_t R, param_t S,
          param_t D, param_t H, param_t W, param_t pad_d, param_t pad_h, param_t pad_w, param_t stride_d, param_t stride_h, param_t stride_w,
          scalar const & alpha, driver::Buffer const & I, driver::Buffer const & F, scalar const & beta, driver::Buffer& O,
          templates::Conv* generator = NULL);


void POOL(driver::Device const & device, driver::Stream & stream,
          DType dtype, param_t C, param_t M, param_t P, param_t Q, param_t N, param_t T, param_t R, param_t S,
          param_t D, param_t H, param_t W, param_t pad_d, param_t pad_h, param_t pad_w, param_t stride_d, param_t stride_h, param_t stride_w,
          driver::Buffer const & I, driver::Buffer& O,
          templates::Pool* generator = NULL);

}
