#ifndef TDL_INCLUDE_IR_CONTEXT_IMPL_H
#define TDL_INCLUDE_IR_CONTEXT_IMPL_H

#include <map>
#include "triton/ir/type.h"

namespace triton{
namespace ir{

class context;
class constant;
class constant_expression;
class constant_int;
class constant_fp;
class undef_value;
class metaparameter;

/* Context impl */
class context_impl {
public:
  // constructors
  context_impl(context &ctx);

public:
  // primitive types
  type void_ty, label_ty, half_ty, float_ty, double_ty;
  // derived types
  integer_type int1_ty, int8_ty, int16_ty, int32_ty, int64_ty, int128_ty;
  // Pointer types
  std::map<std::pair<type*, unsigned>, pointer_type*> ptr_tys;
  std::map<std::pair<type*, type::tile_shapes_t>, tile_type*> tile_tys;
  // Int constants
  std::map<std::pair<type*, uint64_t>, constant_int*> int_constants_;
  // Float constants
  std::map<std::pair<type*, double>, constant_fp*> fp_constants_;
  // undef values
  std::map<type*, undef_value*> uv_constants_;
  // Metaparameters
  std::vector<metaparameter*> mp_constants_;
  // Expr constants
  std::map<std::tuple<int, constant*, constant*>, constant_expression*> expr_constants_;
};

}
}

#endif
