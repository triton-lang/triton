#pragma once

#ifndef _TRITON_IR_CONTEXT_IMPL_H_
#define _TRITON_IR_CONTEXT_IMPL_H_

#include <map>
#include "triton/ir/type.h"

namespace triton{
namespace ir{

class context;
class constant;
class constant_int;
class constant_fp;
class undef_value;

/* Context impl */
class context_impl {
public:
  // constructors
  context_impl(context &ctx);

public:
  // non-numeric types
  type void_ty, label_ty;
  // floating point types
  type fp8_ty, fp16_ty, bf16_ty, fp32_ty, fp64_ty;
  // integer types
  integer_type int1_ty, int8_ty, int16_ty, int32_ty, int64_ty, int128_ty;
  integer_type uint8_ty, uint16_ty, uint32_ty, uint64_ty;
  // Pointer types
  std::map<std::pair<type*, unsigned>, pointer_type*> ptr_tys;
  // Block types
  std::map<std::pair<type*, type::block_shapes_t>, block_type*> block_tys;

  // Int constants
  std::map<std::pair<type*, uint64_t>, constant_int*> int_constants_;
  // Float constants
  std::map<std::pair<type*, double>, constant_fp*> fp_constants_;
  // undef values
  std::map<type*, undef_value*> uv_constants_;

};

}
}

#endif
