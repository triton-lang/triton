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

#ifndef ISAAC_TEMPLATES_COMMON_HPP_
#define ISAAC_TEMPLATES_COMMON_HPP_

#include <cstddef>
#include <cstdint>
#include "isaac/scalar.h"

namespace isaac{

inline int32_t ceil(int32_t num, int32_t div){
  return (num + div - 1)/div;
}

inline size_t log2(size_t x){
  size_t res = 0;
  while((x>>=1)>0) res++;
  return res;
}

inline size_t next_pow2(size_t N){
  size_t res = 1;
  while(res < N)
    res*=2;
  return res;
}

inline std::string arith_str(DType dtype){
  switch (dtype) {
  case FLOAT_TYPE: return "f32";
  case DOUBLE_TYPE: return "f64";
  default: throw;
  }
}

inline std::string io_str(DType dtype){
  switch (dtype) {
  case FLOAT_TYPE: return "b32";
  case DOUBLE_TYPE: return "b64";
  default: throw;
  }
}

typedef uint32_t param_t;

namespace driver{
  class Device;
  class Stream;
  class Kernel;
  class Buffer;
}

namespace templates{

class Generator{
public:
  Generator(){}
  virtual std::string dump(driver::Device const & device, std::string const & name) = 0;
  virtual std::vector<param_t> tuning_params() const = 0;
};

}
}

#endif
