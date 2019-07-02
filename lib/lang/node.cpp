#include "triton/lang/node.h"
#include "triton/ir/builder.h"
#include "triton/ir/module.h"
#include "triton/ir/constant.h"

namespace triton{

namespace lang{

/* node */
ir::value *node::explicit_cast(ir::builder &builder, ir::value *src, ir::type *dst_ty){
  ir::type *src_scalar_ty = src->get_type()->get_scalar_ty();
  ir::type *dst_scalar_ty = dst_ty->get_scalar_ty();
  if(src->get_type()->is_tile_ty())
    dst_ty = ir::tile_type::get_same_shapes(dst_scalar_ty, src->get_type());
  bool src_signed = false;
  bool dst_signed = false;
  if(src_scalar_ty == dst_scalar_ty)
    return src;
  else if(src_scalar_ty->is_integer_ty() && src_signed && dst_scalar_ty->is_floating_point_ty())
    return builder.create_si_to_fp(src, dst_ty);

  else if(src_scalar_ty->is_integer_ty() && !src_signed && dst_scalar_ty->is_floating_point_ty())
    return builder.create_ui_to_fp(src, dst_ty);

  else if(src_scalar_ty->is_floating_point_ty() && dst_scalar_ty->is_integer_ty() && dst_signed)
    return builder.create_fp_to_si(src, dst_ty);

  else if(src_scalar_ty->is_floating_point_ty() && dst_scalar_ty->is_integer_ty() && !dst_signed)
    return builder.create_fp_to_ui(src, dst_ty);

  else if(src_scalar_ty->is_floating_point_ty() && dst_scalar_ty->is_floating_point_ty() &&
          src_scalar_ty->get_fp_mantissa_width() < dst_scalar_ty->get_fp_mantissa_width())
    return builder.create_fp_ext(src, dst_ty);

  else if(src_scalar_ty->is_floating_point_ty() && dst_scalar_ty->is_floating_point_ty() &&
          src_scalar_ty->get_fp_mantissa_width() > dst_scalar_ty->get_fp_mantissa_width())
    return builder.create_fp_trunc(src, dst_ty);

  else if(src_scalar_ty->is_integer_ty() && dst_scalar_ty->is_integer_ty() &&
          src_scalar_ty->get_integer_bitwidth())
    return builder.create_int_cast(src, dst_ty, dst_signed);

  else
    throw std::runtime_error("unreachable");
}


void node::implicit_cast(ir::builder &builder, ir::value *&lhs, ir::value *&rhs,
                          bool &is_float, bool &is_ptr, bool &is_int, bool &is_signed){
  // Input types
  ir::type *left_ty = lhs->get_type()->get_scalar_ty();
  ir::type *right_ty = rhs->get_type()->get_scalar_ty();
  // One operand is pointer
  if(left_ty->is_pointer_ty() || right_ty->is_pointer_ty()){
    if(right_ty->is_pointer_ty())
      std::swap(lhs, rhs);
    is_ptr = true;
  }
  // One operand is double
  else if(left_ty->is_double_ty() || right_ty->is_double_ty()){
    ir::value *&to_convert = left_ty->is_double_ty()?rhs:lhs;
    to_convert = explicit_cast(builder, to_convert, builder.get_double_ty());
    is_float = true;
  }
  // One operand is float
  else if(left_ty->is_float_ty() || right_ty->is_float_ty()){
    ir::value *&to_convert = left_ty->is_float_ty()?rhs:lhs;
    to_convert = explicit_cast(builder, to_convert, builder.get_float_ty());
    is_float = true;
  }
  // One operand is half
  else if(left_ty->is_half_ty() || right_ty->is_half_ty()){
    ir::value *&to_convert = left_ty->is_half_ty()?rhs:lhs;
    to_convert = explicit_cast(builder, to_convert, builder.get_half_ty());
    is_float = true;
  }
  // Both operands are integers
  else if(left_ty->is_integer_ty() && right_ty->is_integer_ty()){
    is_int = true;
    is_signed = true; // always signed for now
    if(left_ty->get_integer_bitwidth() != right_ty->get_integer_bitwidth()){
      ir::value *&to_convert = (left_ty->get_integer_bitwidth() > right_ty->get_integer_bitwidth())?rhs:lhs;
      ir::type *dst_ty = (to_convert==lhs)?right_ty:left_ty;
      to_convert = explicit_cast(builder, to_convert, dst_ty);
    }
  }
  // Not reachable
  else
    throw std::runtime_error("unreachable");
}

void node::implicit_broadcast(ir::module *mod, ir::value *&lhs, ir::value *&rhs) {
  ir::type *lhs_ty = lhs->get_type();
  ir::type *rhs_ty = rhs->get_type();
  ir::type *res_ty = nullptr;
  if(!lhs_ty->is_tile_ty() && !rhs_ty->is_tile_ty())
    return;
  else if(lhs_ty->is_tile_ty() && !rhs_ty->is_tile_ty())
    res_ty = lhs_ty;
  else if(!lhs_ty->is_tile_ty() && rhs_ty->is_tile_ty())
    res_ty = rhs_ty;
  else{
    auto lhs_shapes = lhs_ty->get_tile_shapes();
    auto rhs_shapes = rhs_ty->get_tile_shapes();
    size_t lhs_size = lhs_shapes.size();
    size_t rhs_size = rhs_shapes.size();
    size_t res_size = std::max(lhs_size, rhs_size);
    ir::type::tile_shapes_t res_shapes(res_size);
    ir::type::tile_shapes_t::value_type one = ir::tile_type::make_one(mod->get_context());
    for(int i = 0; i < res_size; i++){
      if(i >= res_size - lhs_size && i >= res_size - rhs_size)
        res_shapes[i] = lhs_shapes[i]==one?rhs_shapes[i]:lhs_shapes[i];
      else if(i >= res_size - lhs_size)
        res_shapes[i] = lhs_shapes[i];
      else if(i >= res_size - rhs_size)
        res_shapes[i] = rhs_shapes[i];
    }
    res_ty = ir::tile_type::get(lhs_ty->get_scalar_ty(), res_shapes);
  }
  implicit_broadcast(mod, res_ty, rhs);
  implicit_broadcast(mod, res_ty, lhs);
}

void node::implicit_broadcast(ir::module *mod, ir::type *ty, ir::value *&src){
  ir::builder &builder = mod->get_builder();
  ir::type *src_ty = src->get_type();
  ir::type::tile_shapes_t::value_type one = ir::tile_type::make_one(mod->get_context());
  // Both are scalar
  if(!ty->is_tile_ty() && !src_ty->is_tile_ty())
    return;
  // Broadcast scalar
  if(ty->is_tile_ty() && !src_ty->is_tile_ty()){
    src = builder.create_splat(src, ty->get_tile_shapes());
    return;
  }
  // Downcast tile
  if(!ty->is_tile_ty() && src_ty->is_tile_ty()){
    for(ir::constant *shape: src_ty->get_tile_shapes())
      if(shape != one)
        throw std::runtime_error("cannot downcast");
    src = builder.create_downcast(src);
    return;
  }
  // Both are arrays
  auto dst_shapes = ty->get_tile_shapes();
  auto src_shapes = src_ty->get_tile_shapes();
  int dst_dim = dst_shapes.size();
  int src_dim = src_shapes.size();
  // Pad
  int off = dst_dim - src_dim;
  for(size_t i = 0; i < off; i++)
    src_shapes.insert(src_shapes.begin(), one);
  if(off > 0)
    src = builder.create_reshape(src, src_shapes);
  // Broadcast
  for(int i = dst_dim - 1; i>= 0; i--)
    if(dst_shapes[i] != src_shapes[i] && dst_shapes[i] != one && src_shapes[i] != one)
      throw std::runtime_error("cannot broadcast");
  if(dst_shapes != src_shapes)
    src = builder.create_broadcast(src, dst_shapes);
}

}

}
