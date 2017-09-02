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

#ifndef ISAAC_RUNTIME_PREDICT_H_
#define ISAAC_RUNTIME_PREDICT_H_

#include <fstream>
#include <vector>
#include <memory>
#include <iostream>
#include <cstring>
#include <algorithm>
#include "isaac/tools/matrix.hpp"
#include "isaac/driver/device.h"
#include "isaac/templates/common.hpp"
#include "isaac/templates/conv.h"
#include "isaac/templates/gemm.h"
#include <map>

namespace isaac{
namespace runtime{

// Layers
class Layer{
public:
  static Layer* read(u_char*& current);
  virtual void forward(matrix<float> const & X, matrix<float> & Y) = 0;
  virtual size_t n_outs(size_t n_outs_prev) = 0;
};

class Activation: public Layer{
public:
  static const int BINARY_CODE = 0;
  size_t n_outs(size_t n_outs_prev);

private:
};

class ReLU: public Activation{
public:
  static const int BINARY_CODE = 0;
  void forward(matrix<float> const & X, matrix<float> & Y);
};

// Dense
class Dense: public Layer{
public:
  static const int BINARY_CODE = 1;
  Dense(u_char*& data);
  size_t n_outs(size_t);
  void forward(matrix<float> const & X, matrix<float> & Y);

private:
  matrix<float> W_;
  std::vector<float> b_;
};

// Network
class Network{
public:
  Network(u_char* data);
  void predict(const matrix<float>& X, matrix<float>& Y);

private:
  std::vector<std::shared_ptr<Layer>> layers_;
};

enum OperationType{
  GEMM,
  CONV
};

//Profile
class Profile{
protected:
  typedef void (&validator_t)(driver::Device const &, size_t, param_t*, uint8_t*);
  typedef std::function<double(std::vector<param_t> const&)> benchmark_t;

public:
  Profile(u_char* data, size_t nshapes);
  std::vector<param_t> predict(driver::Device const & device, std::vector<param_t> const & shapes, validator_t const & validator);
  matrix<param_t> const & kernels() const;

private:
  matrix<param_t> kernels_;
  driver::Device device_;
  Network predictor_;
};

class ConvProfile: public Profile{
public:
  ConvProfile(u_char* data);
  templates::Conv predict(driver::Stream& stream, DType dtype, param_t C, param_t H, param_t W, param_t N, param_t K, param_t P, param_t Q, param_t R, param_t S,
                        param_t pad_h, param_t pad_w, param_t stride_h, param_t stride_w);
};

class GEMMProfile: public Profile{
public:
  GEMMProfile(u_char* data);
  templates::GEMM predict(driver::Stream& stream, DType dtype, IsaacOperation_t AT, IsaacOperation_t BT, param_t M, param_t N, param_t K,
                          param_t offa, param_t lda, param_t offb, param_t ldb, param_t offc, param_t ldc);
};

//Database
extern const std::map<std::pair<driver::Device::Architecture, OperationType>, std::shared_ptr<Profile> > database;

}
}

#endif
