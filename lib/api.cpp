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

inline size_t num_re_evaluate(size_t optimization_level){
    if(optimization_level <= 1)
        return 1;
    return 5*optimization_level;
}

void GEMM(driver::Device const &, driver::Stream & stream,
          DType in_dtype, DType out_dtype, IsaacOperation_t AT, IsaacOperation_t BT, param_t M, param_t N, param_t K,
          param_t offa, param_t lda, param_t offb, param_t ldb, param_t offc, param_t ldc,
          scalar const & alpha, driver::Buffer const & A, driver::Buffer const & B, scalar const & beta, driver::Buffer& C,
          float a_scale, float b_scale, float c_scale,
          const driver::Buffer *bias,
          templates::GEMM* generator, size_t optimization_level)
{
  typedef std::tuple<driver::Stream, DType, DType, IsaacOperation_t, IsaacOperation_t, std::vector<param_t>> key_type;
  // Build the generator if necessary
  static cpp::CachedMap<key_type, std::shared_ptr<templates::GEMM>> inference([&](key_type const & key){
    driver::Stream & stream = (driver::Stream&)std::get<0>(key);
    DType in_dtype = std::get<1>(key);
    DType out_dtype = std::get<2>(key);
    IsaacOperation_t AT = std::get<3>(key), BT = std::get<4>(key);
    runtime::GEMMProfile* profile = (runtime::GEMMProfile*)runtime::database.at({stream.context().device().architecture(), runtime::GEMM}).get();
    std::vector<param_t> const & x = std::get<5>(key);
    templates::GEMM result = profile->predict(stream, in_dtype, out_dtype, AT, BT, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], num_re_evaluate(optimization_level));
    return std::make_shared<templates::GEMM>(result);
  });

  // Build the kernel
  static cpp::CachedMap<std::pair<driver::Stream, templates::GEMM*>, std::shared_ptr<driver::Kernel>> kernels([&](std::pair<driver::Stream, templates::GEMM*> key){
    driver::Context const & context = key.first.context();
    driver::Module module(context, key.second->dump(context.device(), "gemm"));
    return std::make_shared<driver::Kernel>(module, "gemm");
  });

  //Retrieve profile/kernel and execute
  if(generator == NULL)
    generator = inference.get(key_type(stream, in_dtype, out_dtype, AT, BT, {M, N, K, offa, lda, offb, ldb, offc, ldc})).get();
  generator->enqueue(*kernels.get(std::make_pair(stream, generator)), stream, alpha, A, B, beta, C, a_scale, b_scale, c_scale, bias);

}

void CONV(driver::Device const &, driver::Stream & stream,
          DType in_dtype, DType out_dtype, param_t N, param_t K, param_t M, param_t P, param_t Q, param_t C, param_t T, param_t R, param_t S,
          param_t D, param_t H, param_t W,
          param_t pad_d, param_t pad_h, param_t pad_w,
          param_t stride_d, param_t stride_h, param_t stride_w,
          param_t upsample_d, param_t upsample_h, param_t upsample_w,
          driver::Buffer const & I, driver::Buffer const & F, driver::Buffer* O, param_t num_outputs,
          driver::Buffer const * bias,
          ActivationType activation, float alpha,
          float iscale, float fscale, std::vector<float> const & oscale, float z_scale,
          ResidualType residual, param_t Zk, param_t crop_z_m0, param_t crop_z_m1, param_t crop_z_p0, param_t crop_z_p1, param_t crop_z_q0, param_t crop_z_q1, driver::Buffer const *Z,
          templates::Conv* generator, size_t optimization_level)
{
  typedef std::tuple<driver::Stream, DType, DType, std::vector<param_t>> key_type;
  // Build the generator if necessary
  static cpp::CachedMap<key_type, std::shared_ptr<templates::Conv>> inference([&](key_type const & key){
    driver::Stream & stream = (driver::Stream&)std::get<0>(key);
    DType in_dtype = std::get<1>(key);
    DType out_dtype = std::get<2>(key);
    std::vector<param_t> const & x = std::get<3>(key);
    runtime::ConvProfile* profile = (runtime::ConvProfile*)runtime::database.at({stream.context().device().architecture(), runtime::CONV}).get();
    templates::Conv result = profile->predict(stream, in_dtype, out_dtype, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15], x[16], x[17], x[18], x[19], x[20], (ActivationType)x[21], x[22], (ResidualType)x[23], x[24], x[25], x[26], x[27], x[28], x[29], x[30], num_re_evaluate(optimization_level));
    return std::make_shared<templates::Conv>(result);
  });

  // Build the kernel
  static cpp::CachedMap<std::pair<driver::Stream, templates::Conv*>, std::shared_ptr<driver::Kernel>> kernels([&](std::pair<driver::Stream, templates::Conv*> const & key){
    driver::Context const & context = key.first.context();
    driver::Module module(context, key.second->dump(context.device(), "conv"));
    return std::make_shared<driver::Kernel>(module, "conv");
  });


  //Retrieve profile/kernel and execute
  if(generator == NULL)
    generator = inference.get(key_type(stream, in_dtype, out_dtype, {C, D, H, W, N, K, M, P, Q, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, upsample_d, upsample_h, upsample_w, activation, num_outputs, residual, Zk, crop_z_m0, crop_z_m1, crop_z_p0, crop_z_p1, crop_z_q0, crop_z_q1})).get();
  generator->enqueue(*kernels.get(std::make_pair(stream, generator)), stream,  I, F, O, bias, alpha, iscale, fscale, oscale, z_scale, Z);
}


void POOL(driver::Device const &, driver::Stream & stream,
          DType in_dtype, DType out_dtype, PoolType pool_type, param_t C, param_t M, param_t P, param_t Q, param_t N, param_t T, param_t R, param_t S,
          param_t D, param_t H, param_t W, param_t pad_d, param_t pad_h, param_t pad_w, param_t stride_d, param_t stride_h, param_t stride_w,
          driver::Buffer const & I, driver::Buffer& O,
          float iscale, float oscale,
          templates::Pool* generator, size_t optimization_level)
{
  typedef std::tuple<driver::Stream, DType, DType, std::vector<param_t>> key_type;
  // Build the generator if necessary
  static cpp::CachedMap<key_type, std::shared_ptr<templates::Pool>> inference([&](key_type const & key){
    driver::Stream & stream = (driver::Stream&)std::get<0>(key);
    runtime::PoolProfile* profile = (runtime::PoolProfile*)runtime::database.at({stream.context().device().architecture(), runtime::POOL}).get();
    DType in_dtype = std::get<1>(key);
    DType out_dtype = std::get<2>(key);
    std::vector<param_t> const & x = std::get<3>(key);
    templates::Pool result = profile->predict(stream, in_dtype, out_dtype, (PoolType)x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15], x[16], x[17], num_re_evaluate(optimization_level));
    return std::make_shared<templates::Pool>(result);
  });
  // Build the kernel
  static cpp::CachedMap<std::pair<driver::Stream, templates::Pool*>, std::shared_ptr<driver::Kernel>> kernels([&](std::pair<driver::Stream, templates::Pool*> const & key){
    driver::Context const & context = key.first.context();
    driver::Module module(context, key.second->dump(context.device(), "pool"));
    return std::make_shared<driver::Kernel>(module, "pool");
  });

  //Retrieve profile/kernel and execute
  if(generator == NULL)
    generator = inference.get(key_type(stream, in_dtype, out_dtype, {pool_type, C, D, H, W, N, M, P, Q, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w})).get();
  generator->enqueue(*kernels.get(std::make_pair(stream, generator)), stream, I, O, iscale, oscale);
}



}
