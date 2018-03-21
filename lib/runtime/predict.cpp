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

#include <unistd.h>
#include <fstream>
#include <vector>
#include <memory>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <sstream>
#include "isaac/runtime/predict.h"
#include "isaac/templates/conv.h"
#include "isaac/tools/bench.hpp"
#include "isaac/driver/module.h"
#include "isaac/driver/kernel.h"
#include "isaac/driver/stream.h"
#include "isaac/driver/buffer.h"

namespace isaac{
namespace runtime{

// Layers
Layer* Layer::read(u_char*& current){
  uint32_t type;
  read_inc((void*)&type, current, 4);
  if(type==Activation::BINARY_CODE){
    read_inc((void*)&type, current, 4);
    if(type==ReLU::BINARY_CODE) return new ReLU();
    if(type==Linear::BINARY_CODE) return new Linear();
    throw;
  }
  else if(type==Dense::BINARY_CODE)
    return new Dense(current);
  throw;
}

// Activation
size_t Activation::n_outs(size_t n_outs_prev)
{ return n_outs_prev; }

// Relu
void ReLU::forward(matrix<float> const & X, matrix<float> & Y){
  for(size_t i = 0; i < X.shapes()[0]; ++i)
    for(size_t j = 0; j < X.shapes()[1]; ++j)
      Y(i, j) = std::max<float>(X(i,j), 0);
}

// Linear
void Linear::forward(matrix<float> const & X, matrix<float> & Y){
  for(size_t i = 0; i < X.shapes()[0]; ++i)
    for(size_t j = 0; j < X.shapes()[1]; ++j)
      Y(i, j) = X(i,j);
}

// Dense
Dense::Dense(u_char*& data) : W_(data){
  b_.resize(W_.shapes()[1]);
  read_inc((void*)b_.data(), data, b_.size()*4);
}

size_t Dense::n_outs(size_t)
{ return W_.shapes()[1]; }

void Dense::forward(matrix<float> const & X, matrix<float> & Y){
  gemm<float>(Y.shapes()[0], Y.shapes()[1], X.shapes()[1], 1, X.data(), X.ld(), W_.data(), W_.ld(), 1, Y.data(), Y.ld(), b_.data());
}

// Network
Network::Network(u_char* data){
  uint32_t nlayers;
  read_inc((void*)&nlayers, data, 4);
  for(size_t i = 0; i < nlayers; ++i)
    layers_.push_back(std::shared_ptr<Layer>(Layer::read(data)));
}

void Network::predict(matrix<float> const & X, matrix<float> & Y){
  uint32_t N = X.shapes()[0], M = X.shapes()[1];
  size_t nlayers = layers_.size();
  std::vector<uint32_t> n_outs(nlayers+1, M);
  for(size_t i = 0; i < nlayers; ++i)
    n_outs[i+1] = layers_[i]->n_outs(n_outs[i]);
  //Pre-allocate a big buffer to stay in cache memory
  size_t nhid_max = *std::max_element(n_outs.begin(), n_outs.end());
  std::vector<float> scratch(2*N*nhid_max);
  std::vector<size_t> off(nlayers+1, 0);
  //Predict
  for(size_t i = 0; i < nlayers; ++i){
    bool is_dense = dynamic_cast<Dense*>(layers_[i].get());
    off[i+1] = off[i];
    if(is_dense)
      off[i+1] = (off[i] + scratch.size()/2) % scratch.size();
    matrix<float> I({N, n_outs[i]}, (i==0)?n_outs[i]:nhid_max, (i==0)?X.data():(scratch.data() + off[i]));
    matrix<float> O({N, n_outs[i+1]}, (i==nlayers-1)?n_outs[i+1]:nhid_max, (i==nlayers-1)?Y.data():(scratch.data() + off[i+1]));
    layers_[i]->forward(I, O);
  }
}

// Profile
Profile::Profile(u_char* data, size_t nshapes): kernels_(pad_left(matrix<param_t>(data), nshapes)), predictor_(data)
{}

matrix<param_t> const & Profile::kernels() const
{ return kernels_; }

std::vector<param_t> Profile::predict(driver::Device const & device, std::vector<param_t> const & shapes, validator_t const & validator, benchmark_t const & benchmark, size_t num_re_evaluate)
{
  // Get valid profiles
  uint32_t nkernels = kernels_.shapes()[0];
  uint32_t nparams = kernels_.shapes()[1];
  for(size_t i = 0; i < nkernels; ++i)
    for(size_t j = 0; j < shapes.size(); ++j)
      kernels_(i, j) = shapes[j];
  std::vector<uint8_t> valid(nkernels);
  validator(device, nkernels, kernels_.data(), valid.data());
  uint32_t num_valid = std::accumulate(valid.begin(), valid.end(), 0);

  if(num_valid == 0)
    throw std::runtime_error("Something went wrong. No valid profiles found.");

  // Get valid indices
  std::vector<size_t> map;
  map.reserve(num_valid);
  for(size_t i = 0; i < nkernels; ++i)
    if(valid[i]) map.push_back(i);

  // Predictor input
  matrix<float> X({num_valid, nparams});
  for(size_t i = 0; i < num_valid; ++i)
    for(size_t j = 0; j < nparams; ++j)
        X(i, j) = std::log2(kernels_(map[i], j));

  // Do prediction
  matrix<float> Y({num_valid, 1});
  predictor_.predict(X, Y);

  // Sort prediction
  std::vector<size_t> idx(num_valid);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&Y](size_t i1, size_t i2) {return Y(i1,0) > Y(i2, 0);});
  std::vector<param_t> x(nparams);

  //Re-evaluate
  size_t best = 0;
  if(benchmark && num_re_evaluate > 1)
  {
    std::vector<double> time;
    for(size_t i = 0; i < std::min<size_t>(num_valid, num_re_evaluate); ++i){
      for(size_t j = 0; j < nparams; ++j)
        x[j] = kernels_(map[idx[i]], shapes.size() + j);
      time.push_back(benchmark(x));
    }
    best = std::min_element(time.begin(), time.end()) - time.begin();
  }

  // Output
  for(size_t j = 0; j < nparams; ++j)
    x[j] = kernels_(map[idx[best]], shapes.size() + j);
  return x;
}


// Convolution
ConvProfile::ConvProfile(u_char* data): Profile(data, templates::Conv::Nshapes){}

templates::Conv ConvProfile::predict(driver::Stream& stream, DType in_dtype, DType out_dtype, param_t C, param_t D, param_t H, param_t W, param_t N, param_t K, param_t M, param_t P, param_t Q, param_t T, param_t R, param_t S,
                                    param_t pad_d, param_t pad_h, param_t pad_w,
                                    param_t stride_d, param_t stride_h, param_t stride_w,
                                    param_t upsample_d, param_t upsample_h, param_t upsample_w,
                                    ActivationType activation, size_t num_outputs,
                                    ResidualType residual, param_t Zk, param_t crop_z_m0, param_t crop_z_m1, param_t crop_z_p0, param_t crop_z_p1, param_t crop_z_q0, param_t crop_z_q1,
                                    size_t num_re_evaluate)
{
  param_t PACK_IN = (in_dtype==INT8X4_TYPE)?4:1;
  param_t PACK_OUT = (out_dtype==INT8X4_TYPE)?4:1;

  driver::Device const & device = stream.context().device();
  benchmark_t benchmark;
  std::unique_ptr<driver::Buffer> O, I, F;
  if(num_re_evaluate > 1)
  {
    O.reset(new driver::Buffer(stream.context(), K*M*P*Q*N*size_of(out_dtype)/PACK_OUT));
    I.reset(new driver::Buffer(stream.context(), C*D*H*W*N*size_of(in_dtype)/PACK_IN));
    F.reset(new driver::Buffer(stream.context(), C*K*T*R*S*size_of(in_dtype)/PACK_IN));
    benchmark = [&](std::vector<param_t> const& x){
      templates::Conv generator(in_dtype, out_dtype, C, D, H, W, N, K, M, P, Q, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, upsample_d, upsample_h, upsample_w, activation, 1,
                                NoResidual, 0, 0, 0, 0, 0, 0, 0,
                                x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]);
      std::string src = generator.dump(device, "kernel");
      driver::Module module(stream.context(), src);
      driver::Kernel kernel(module, "kernel");
      return bench([&](){ generator.enqueue(kernel, stream, *I, *F, O.get()); }, [&](){ stream.synchronize(); }, device);
    };
  }
  std::vector<param_t> shapes{out_dtype, N*M*P*Q, K, C/PACK_IN, T*R*S};
  std::vector<param_t> x = Profile::predict(device, shapes, templates::Conv::check_valid, benchmark, num_re_evaluate);
  return templates::Conv(in_dtype, out_dtype, C, D, H, W, N, K, M, P, Q, T, R, S,
                         pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, upsample_d, upsample_h, upsample_w, activation, num_outputs,
                         residual, Zk, crop_z_m0, crop_z_m1, crop_z_p0, crop_z_p1, crop_z_q0, crop_z_q1,
                         x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]);
}

// Pooling
PoolProfile::PoolProfile(u_char* data): Profile(data, templates::Pool::Nshapes){}

templates::Pool PoolProfile::predict(driver::Stream& stream, DType in_dtype, DType out_dtype, PoolType pool_type, param_t C, param_t D, param_t H, param_t W, param_t N, param_t M, param_t P, param_t Q, param_t T, param_t R, param_t S,
                      param_t pad_d, param_t pad_h, param_t pad_w, param_t stride_d, param_t stride_h, param_t stride_w, size_t num_re_evaluate)
{
    param_t PACK = (in_dtype==INT8X4_TYPE)?4:1;

    driver::Device const & device = stream.context().device();
    benchmark_t benchmark;
    std::vector<param_t> shapes{in_dtype, N*M*P*Q*C/PACK, T*R*S};
    std::vector<param_t> x = Profile::predict(device, shapes, templates::Pool::check_valid, benchmark, num_re_evaluate);
    return templates::Pool(in_dtype, out_dtype, pool_type, C, D, H, W, N, M, P, Q, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w,
                           x[0], x[1], x[2], x[3]);
}

// GEMM
GEMMProfile::GEMMProfile(u_char* data): Profile(data, templates::GEMM::Nshapes){}

templates::GEMM GEMMProfile::predict(driver::Stream& stream, DType in_dtype, DType out_dtype, IsaacOperation_t AT, IsaacOperation_t BT, param_t M, param_t N, param_t K,
                                     param_t offa, param_t lda, param_t offb, param_t ldb, param_t offc, param_t ldc, size_t num_re_evaluate)
{
  driver::Device const & device = stream.context().device();
  benchmark_t benchmark;
  std::unique_ptr<driver::Buffer> C, A, B;
  std::unique_ptr<scalar> alpha, beta;
  if(num_re_evaluate > 1)
  {
    C.reset(new driver::Buffer(stream.context(), M*N*size_of(out_dtype)));
    A.reset(new driver::Buffer(stream.context(), M*K*size_of(in_dtype)));
    B.reset(new driver::Buffer(stream.context(), K*N*size_of(in_dtype)));
    DType ab_dtype = (out_dtype == INT8X4_TYPE)?FLOAT_TYPE:out_dtype;
    alpha.reset(new scalar(1., ab_dtype));
    beta.reset(new scalar(0., ab_dtype));
    benchmark = [&](std::vector<param_t> const& x){
      templates::GEMM generator(in_dtype, out_dtype, AT, BT, M, N, K, offa, lda, offb, ldb, offc, ldc,
                                x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13]);
      std::string src = generator.dump(device, "kernel");
      driver::Module module(stream.context(), src);
      driver::Kernel kernel(module, "kernel");
      return bench([&](){ generator.enqueue(kernel, stream, *alpha, *A, *B, *beta, *C); }, [&](){ stream.synchronize(); }, device);
    };
  }

  std::vector<param_t> shapes{out_dtype, AT, BT, M, N, K};
  std::vector<param_t> x = Profile::predict(device, shapes, templates::GEMM::check_valid, benchmark, num_re_evaluate);
  return templates::GEMM(in_dtype, out_dtype, AT, BT, M, N, K, offa, lda, offb, ldb, offc, ldc,
                         x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13]);
}

}
}
