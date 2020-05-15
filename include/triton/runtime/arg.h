#pragma once

#ifndef _TRITON_RUNTIME_ARG_H_
#define _TRITON_RUNTIME_ARG_H_

#include <string>
#include <stdexcept>

namespace triton{
namespace ir{
  class type;
}

namespace driver{
  class buffer;
}

namespace runtime {

enum arg_type {
  INT1_T,
  INT8_T,
  INT16_T,
  INT32_T,
  INT64_T,
  HALF_T,
  FLOAT_T,
  DOUBLE_T,
  BUFFER_T
};

arg_type convert(ir::type *ty);


inline size_t size_of(arg_type ty){
  switch(ty){
    case INT1_T: return 1;
    case INT8_T: return 1;
    case INT16_T: return 2;
    case INT32_T: return 4;
    case INT64_T: return 8;
    case HALF_T: return 2;
    case FLOAT_T: return 4;
    case DOUBLE_T: return 8;
    case BUFFER_T: return 8;
    default: throw std::runtime_error("unknown type");
  }
}

inline bool is_int_type(arg_type ty){
  return ty == INT1_T || ty == INT8_T || ty == INT16_T ||
         ty == INT32_T || ty == INT64_T;
}

class arg {
public:
  union value_t {
    bool    int1;
    int8_t  int8;
    int16_t int16;
    int32_t int32;
    int64_t int64;
    uint16_t fp16;
    float   fp32;
    double  fp64;
    driver::buffer* buf;
  };

public:
  // construct from primitive types
  arg(arg_type ty, value_t val): ty_(ty) { val_ = val; }
  arg(int32_t x): ty_(INT32_T) { val_.int32 = x; }
  arg(int64_t x): ty_(INT64_T) { val_.int64 = x; }
  arg(float x): ty_(FLOAT_T)   { val_.fp32 = x; }
  arg(double x): ty_(DOUBLE_T) { val_.fp64 = x; }
  arg(driver::buffer* x): ty_(BUFFER_T) { val_.buf = x; }
  // accessors
  arg_type type() const { return ty_; }
  void* data() const { return (void*)&val_; }
  driver::buffer* buffer() const { return val_.buf; }


private:
  arg_type ty_;
  value_t val_;
};

}
}

#endif
