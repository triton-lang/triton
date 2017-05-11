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

#include "isaac/runtime/predict.h"
#include "isaac/driver/backend.h"
#include "isaac/driver/cublas.h"
#include "isaac/driver/context.h"
#include "isaac/driver/kernel.h"
#include "isaac/driver/buffer.h"
#include "isaac/driver/stream.h"
#include "isaac/tools/collections.hpp"
#include "isaac/templates/conv.h"
#include "isaac/templates/gemm.h"

namespace isaac{

void GEMM(driver::Device const & device, driver::Stream & stream,
          DType dtype, IsaacOperation_t AT, IsaacOperation_t BT, param_t M, param_t N, param_t K,
          size_t offa, size_t lda, size_t offb, size_t ldb, size_t offc, size_t ldc,
          scalar const & alpha, driver::Buffer const & A, driver::Buffer const & B, scalar const & beta, driver::Buffer& C)
{
  typedef std::tuple<driver::Stream, DType, IsaacOperation_t, IsaacOperation_t,
                    param_t, param_t, param_t, size_t, size_t, size_t, size_t, size_t, size_t> key_type;
  typedef std::pair<std::shared_ptr<templates::GEMM>, std::shared_ptr<driver::Kernel>> value_type;

  static std::function<value_type()> compile = [&](){
    //Fetch profile
    runtime::GEMMProfile* profile = (runtime::GEMMProfile*)runtime::database.at({device.architecture(), runtime::GEMM}).get();
    templates::GEMM generator = profile->predict(stream, device, dtype, AT, BT, M, N, K, offa, lda, offb, ldb, offc, ldc);
    //Execute
    std::string src = generator.dump(device, "gemm");
    driver::Module module(stream.context(), src);
    return value_type(std::make_shared<templates::GEMM>(generator), std::make_shared<driver::Kernel>(module, "gemm"));
  };
  static cpp::CachedMap<key_type, value_type> cache(compile);

  //Retrieve profile/kernel and execute
  value_type const & value = cache.get(key_type(stream, dtype, AT, BT, M, N, K, offa, lda, offb, ldb, offc, ldc));
  value.first->enqueue(*value.second, stream, alpha, A, B, beta, C);
}

void CONV(driver::Device const & device, driver::Stream & stream,
          DType dtype, param_t N, param_t K, param_t P, param_t Q, param_t C, param_t R, param_t S,
          param_t H, param_t W, param_t pad_h, param_t pad_w, param_t stride_h, param_t stride_w,
          scalar const & alpha, driver::Buffer const & I, driver::Buffer const & F, scalar const & beta, driver::Buffer& O)
{
  typedef std::tuple<driver::Stream, DType, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t> key_type;
  typedef std::pair<std::shared_ptr<templates::Conv>, std::shared_ptr<driver::Kernel>> value_type;

  static std::function<value_type()> compile = [&](){
    //Fetch profile
    runtime::ConvProfile* profile = (runtime::ConvProfile*)runtime::database.at({device.architecture(), runtime::CONV}).get();
    templates::Conv generator = profile->predict(stream, device, dtype, C, H, W, N, K, P, Q, R, S, pad_h, pad_w, stride_h, stride_w);
    //Execute
    std::string src = generator.dump(device, "conv");
    driver::Module module(stream.context(), src);
    return value_type(std::make_shared<templates::Conv>(generator), std::make_shared<driver::Kernel>(module, "conv"));
  };
  static cpp::CachedMap<key_type, value_type> cache(compile);
  //Retrieve profile/kernel and execute
  value_type const & value = cache.get(key_type(stream, dtype, N, K, P, Q, C, R, S, pad_h, pad_w, stride_h, stride_w));
  value.first->enqueue(*value.second, stream, alpha, I, F, beta, O);
}



}
