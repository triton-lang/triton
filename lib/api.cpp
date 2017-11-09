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

#include "isaac/api.h"

namespace isaac{

void GEMM(driver::Device const & device, driver::Stream & stream,
          DType dtype, IsaacOperation_t AT, IsaacOperation_t BT, param_t M, param_t N, param_t K,
          size_t offa, size_t lda, size_t offb, size_t ldb, size_t offc, size_t ldc,
          scalar const & alpha, driver::Buffer const & A, driver::Buffer const & B, scalar const & beta, driver::Buffer& C,
          templates::GEMM* generator)
{
  typedef std::tuple<driver::Stream&, DType, IsaacOperation_t, IsaacOperation_t,
                    param_t, param_t, param_t, size_t, size_t, size_t, size_t, size_t, size_t> key_type;
  // Build the generator if necessary
  static cpp::CachedMap<key_type, std::shared_ptr<templates::GEMM>> inference([&](key_type const & x){
    runtime::GEMMProfile* profile = (runtime::GEMMProfile*)runtime::database.at({device.architecture(), runtime::GEMM}).get();
    templates::GEMM result = profile->predict(std::get<0>(x), std::get<1>(x), std::get<2>(x), std::get<3>(x), std::get<4>(x), std::get<5>(x), std::get<6>(x), std::get<7>(x), std::get<8>(x), std::get<9>(x), std::get<10>(x), std::get<11>(x), std::get<12>(x));
    return std::make_shared<templates::GEMM>(result);
  });

  // Build the kernel
  static cpp::CachedMap<templates::GEMM*, std::shared_ptr<driver::Kernel>> kernels([&](templates::GEMM* key){
    driver::Module module(stream.context(), key->dump(device, "gemm"));
    return std::make_shared<driver::Kernel>(module, "gemm");
  });

  //Retrieve profile/kernel and execute
  if(generator == NULL)
    generator = inference.get(key_type(stream, dtype, AT, BT, M, N, K, offa, lda, offb, ldb, offc, ldc)).get();
  generator->enqueue(*kernels.get(generator), stream, alpha, A, B, beta, C);

}

void CONV(driver::Device const & device, driver::Stream & stream,
          DType dtype, param_t N, param_t K, param_t M, param_t P, param_t Q, param_t C, param_t T, param_t R, param_t S,
          param_t D, param_t H, param_t W, param_t pad_d, param_t pad_h, param_t pad_w, param_t stride_d, param_t stride_h, param_t stride_w,
          scalar const & alpha, driver::Buffer const & I, driver::Buffer const & F, scalar const & beta, driver::Buffer& O,
          templates::Conv* generator)
{
  typedef std::tuple<driver::Stream*, DType, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t,
                     param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t> key_type;
  // Build the generator if necessary
  static cpp::CachedMap<key_type, std::shared_ptr<templates::Conv>> inference([&](key_type const & key){
    runtime::ConvProfile* profile = (runtime::ConvProfile*)runtime::database.at({device.architecture(), runtime::CONV}).get();
    driver::Stream* stream;
    DType dtype;
    param_t C, D, H, W, N, K, M, P, Q, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w;
    std::tie(stream, dtype, C, D, H, W, N, K, M, P, Q, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w) = key;
    templates::Conv result = profile->predict(*stream, dtype, C, D, H, W, N, K, M, P, Q, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w);
    return std::make_shared<templates::Conv>(result);
  });

  // Build the kernel
  static cpp::CachedMap<templates::Conv*, std::shared_ptr<driver::Kernel>> kernels([&](templates::Conv* key){
    driver::Module module(stream.context(), key->dump(device, "Conv"));
    return std::make_shared<driver::Kernel>(module, "Conv");
  });


  //Retrieve profile/kernel and execute
  if(generator == NULL)
    generator = inference.get(key_type(&stream, dtype, C, D, H, W, N, K, M, P, Q, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w)).get();

  generator->enqueue(*kernels.get(generator), stream,  alpha, I, F, beta, O);
}


void POOL(driver::Device const & device, driver::Stream & stream,
          DType dtype, param_t C, param_t M, param_t P, param_t Q, param_t N, param_t T, param_t R, param_t S,
          param_t D, param_t H, param_t W, param_t pad_d, param_t pad_h, param_t pad_w, param_t stride_d, param_t stride_h, param_t stride_w,
          driver::Buffer const & I, driver::Buffer& O,
          templates::Pool* generator)
{
  typedef std::tuple<driver::Stream*, DType, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t> key_type;
  // Build the generator if necessary
  static cpp::CachedMap<key_type, std::shared_ptr<templates::Pool>> inference([&](key_type const & key){
    driver::Stream* stream;
    DType dtype;
    param_t C, D, H, W, N, M, P, Q, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w;
    std::tie(stream, dtype, C, D, H, W, N, M, P, Q, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w) = key;
    runtime::PoolProfile* profile = (runtime::PoolProfile*)runtime::database.at({device.architecture(), runtime::POOL}).get();
    templates::Pool result = profile->predict(*stream, dtype, C, D, H, W, N, M, P, Q, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w);
    return std::make_shared<templates::Pool>(result);
  });
  // Build the kernel
  static cpp::CachedMap<templates::Pool*, std::shared_ptr<driver::Kernel>> kernels([&](templates::Pool* key){
    driver::Module module(stream.context(), key->dump(device, "Pool"));
    return std::make_shared<driver::Kernel>(module, "Pool");
  });
  //Retrieve profile/kernel and execute
  if(generator == NULL)
    generator = inference.get(key_type(&stream, dtype, C, D, H, W, N, M, P, Q, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w)).get();
  generator->enqueue(*kernels.get(generator), stream, I, O);
}

}
