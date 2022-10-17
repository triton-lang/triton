#pragma once

#ifndef _TRITON_IR_CONTEXT_IMPL_H_
#define _TRITON_IR_CONTEXT_IMPL_H_

#include "triton/ir/type.h"
#include "triton/ir/constant.h"
#include <map>
#include <memory>

namespace triton{
namespace ir{

class context;

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
  // Pointer types
  std::map<std::pair<type*, unsigned>, std::unique_ptr<pointer_type>> ptr_tys;
  // Block types
  std::map<std::pair<type*, type::block_shapes_t>, std::unique_ptr<block_type>> block_tys;
  // Struct types
  std::map<type::contained_tys_vec_t, struct_type*> struct_tys;
  // Int constants
  std::map<std::pair<type*, uint64_t>, std::unique_ptr<constant_int>> int_constants_;
  // Float constants
  std::map<std::pair<type*, double>, std::unique_ptr<constant_fp>> fp_constants_;
  // undef values
  std::map<type*, std::unique_ptr<undef_value>> uv_constants_;

};

}
}

#endif
