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

#include <fstream>
#include <vector>
#include <memory>
#include <iostream>
#include <cstring>
#include <algorithm>
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
  for(size_t i = 0; i < nlayers; ++i){
    n_outs[i+1] = layers_[i]->n_outs(n_outs[i]);
  }
  //Pre-allocate a big buffer to stay in cache memory
  size_t nhid_max = *std::max_element(n_outs.begin(), n_outs.end());
  std::vector<float> scratch(2*N*nhid_max);
  std::vector<size_t> off(nlayers+1, 0);
  //Predict
  for(size_t i = 0; i < nlayers; ++i){
    bool is_dense = dynamic_cast<Dense*>(layers_[i].get());
    off[i+1] = off[i];
    if(is_dense) off[i+1] = (off[i] + scratch.size()/2) % scratch.size();
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

std::vector<param_t> Profile::predict(driver::Device const & device, std::vector<param_t> const & shapes, validator_t const & validator, benchmark_t const & benchmark, uint32_t nkeep)
{
  // Get valid profiles
  uint32_t nkernels = kernels_.shapes()[0];
  uint32_t nparams = kernels_.shapes()[1];
  for(size_t i = 0; i < nkernels; ++i)
    for(size_t j = 0; j < shapes.size(); ++j)
      kernels_(i, j) = shapes[j];
  std::vector<uint8_t> valid(nkernels);
  validator(device, nkernels, kernels_.data(), valid.data());
  uint32_t nvalid = std::accumulate(valid.begin(), valid.end(), 0);

  // Get valid indices
  std::vector<size_t> map;
  map.reserve(nvalid);
  for(size_t i = 0; i < nkernels; ++i)
    if(valid[i]) map.push_back(i);

  // Predictor input
  matrix<float> X({nvalid, nparams});
  for(size_t i = 0; i < nvalid; ++i)
    for(size_t j = 0; j < nparams; ++j)
        X(i, j) = std::log2(kernels_(map[i], j));

  // Do prediction
  matrix<float> Y({nvalid, 1});
  predictor_.predict(X, Y);

  // Sort prediction
  std::vector<size_t> idx(nvalid);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&Y](size_t i1, size_t i2) {return Y(i1,0) > Y(i2, 0);});

  // Return best
  matrix<param_t> best({nkeep, nparams});
  for(size_t i = 0; i < std::min(nvalid, nkeep); ++i)
    for(size_t j = 0; j < nparams; ++j)
      best(i, j) = kernels_(map[idx[i]], shapes.size() + j);

  //Re-Benchmark
  size_t argmin = 0;
  std::vector<param_t> x(nparams);
  if(nkeep > 1){
    std::vector<double> time;
    for(size_t i = 0; i < std::min(nvalid, nkeep); ++i){
      for(size_t j = 0; j < nparams; ++j)
        x[j] = best(i,j);
      time.push_back(benchmark(x));
    }
    argmin = std::min_element(time.begin(), time.end()) - time.begin();
  }
  for(size_t j = 0; j < nparams; ++j)
    x[j] = best(argmin,j);

  return x;
}


ConvProfile::ConvProfile(u_char* data): Profile(data, 8){}

templates::Conv ConvProfile::predict(driver::Stream& stream, driver::Device const & device, DType dtype, param_t C, param_t H, param_t W, param_t N, param_t K, param_t P, param_t Q, param_t R, param_t S,
                      param_t pad_h, param_t pad_w, param_t stride_h, param_t stride_w)
{
  std::vector<param_t> shapes{dtype, N, K, P, Q, C, R, S};

  driver::Buffer O(stream.context(), N*K*P*Q*size_of(dtype));
  driver::Buffer I(stream.context(), C*H*W*N*size_of(dtype));
  driver::Buffer F(stream.context(), C*K*R*S*size_of(dtype));
  scalar alpha(1., dtype);
  scalar beta(0., dtype);
  std::function<double(std::vector<param_t> const&)> benchmark = [&](std::vector<param_t> const& x){
    templates::Conv generator(dtype, C, H, W, N, K, P, Q, R, S, pad_h, pad_w, stride_h, stride_w,
                              x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14]);
    std::string src = generator.dump(device, "conv");
    driver::Module module(stream.context(), src);
    driver::Kernel kernel(module, "conv");
    return bench([&](){ generator.enqueue(kernel, stream, alpha, I, F, beta, O); }, [&](){ stream.synchronize(); }, device);
  };

  std::vector<param_t> x = Profile::predict(device, shapes, templates::Conv::check_valid, benchmark);
  return templates::Conv(dtype, C, H, W, N, K, P, Q, R, S, pad_h, pad_w, stride_h, stride_w,
                         x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14]);
}


GEMMProfile::GEMMProfile(u_char* data): Profile(data, 6){}

templates::GEMM GEMMProfile::predict(driver::Stream& stream, driver::Device const & device, DType dtype, IsaacOperation_t AT, IsaacOperation_t BT, param_t M, param_t N, param_t K,
                                     param_t offa, param_t lda, param_t offb, param_t ldb, param_t offc, param_t ldc)
{
  std::vector<param_t> shapes{dtype, AT, BT, M, N, K};

  driver::Buffer C(stream.context(), M*N*size_of(dtype));
  driver::Buffer A(stream.context(), M*K*size_of(dtype));
  driver::Buffer B(stream.context(), K*N*size_of(dtype));
  scalar alpha(1., dtype);
  scalar beta(0., dtype);
  std::function<double(std::vector<param_t> const&)> benchmark = [&](std::vector<param_t> const& x)
  {
    templates::GEMM generator(dtype, AT, BT, M, N, K, offa, lda, offb, ldb, offc, ldc,
                             x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13]);
    std::string src = generator.dump(device, "gemm");
    driver::Module module(stream.context(), src);
    driver::Kernel kernel(module, "gemm");
    return bench([&](){ generator.enqueue(kernel, stream, alpha, A, B, beta, C); }, [&](){ stream.synchronize(); }, device);
  };

  std::vector<param_t> x = Profile::predict(device, shapes, templates::GEMM::check_valid, benchmark);
  return templates::GEMM(dtype, AT, BT, M, N, K, offa, lda, offb, ldb, offc, ldc,
                         x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13]);
}

}
}
