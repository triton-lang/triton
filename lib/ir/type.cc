#include <cassert>
#include <stdexcept>
#include "triton/ir/type.h"
#include "triton/ir/context.h"
#include "triton/ir/context_impl.h"
#include "triton/ir/value.h"
#include "triton/ir/constant.h"

namespace triton{
namespace ir{

//===----------------------------------------------------------------------===//
//                              type class
//===----------------------------------------------------------------------===//

// attributes
type *type::get_scalar_ty() const {
  if(is_block_ty())
    return get_tile_element_ty();
  return const_cast<type*>(this);
}

unsigned type::get_primitive_size_in_bits() const {
  switch (id_) {
    case FP8TyID: return 8;
    case FP16TyID: return 16;
    case BF16TyID: return 16;
    case FP32TyID: return 32;
    case FP64TyID: return 64;
    case IntegerTyID: return ((integer_type*)(this))->get_bitwidth();
    case BlockTyID:  return ((block_type*)(this))->get_bitwidth();
    default: return 0;
  }
}

unsigned type::get_integer_bitwidth() const
{ assert(id_ == IntegerTyID); return ((integer_type*)(this))->get_bitwidth(); }

unsigned type::get_tile_bitwidth() const
{ return ((block_type*)(this))->get_bitwidth(); }

unsigned type::get_fp_mantissa_width() const {
  id_t id = get_scalar_ty()->id_;
  assert(is_floating_point_ty() && "Not a floating point type!");
  if (id == FP8TyID) return 3;
  if (id == FP16TyID) return 10;
  if (id == BF16TyID) return 7;
  if (id == FP32TyID) return 23;
  if (id == FP64TyID) return 53;
  throw std::runtime_error("unreachable");
}

type* type::get_tile_element_ty() const {
  assert(is_block_ty());
  return contained_tys_[0];
}

unsigned type::get_pointer_address_space() const {
  assert(is_pointer_ty());
  return ((pointer_type*)this)->get_address_space();
}

type * type::get_pointer_element_ty() const {
  type *ptr_ty = get_scalar_ty();
  assert(ptr_ty->is_pointer_ty());
  type *scalar_ty = ((pointer_type*)ptr_ty)->get_element_ty();
  if(is_block_ty())
    return block_type::get_same_shapes(scalar_ty, (type*)this);
  return scalar_ty;
}


type::block_shapes_t type::get_block_shapes() const {
  assert(is_block_ty());
  return ((block_type*)this)->get_shapes();
}

const size_t type::get_tile_rank() const {
  return get_block_shapes().size();
}

const size_t type::get_tile_ranks1() const {
  int ret = 0;
  for(int s: get_block_shapes())
    ret += s > 1;
  return ret;
}


unsigned type::get_tile_num_elements() const {
  const block_shapes_t& shapes = get_block_shapes();
  unsigned result = 1;
  for(auto shape: shapes)
    result *= shape;
  return result;
}


// composite predicates
bool type::is_int_or_tileint_ty()
{ return get_scalar_ty()->is_integer_ty(); }

bool type::is_integer_ty(unsigned width) const
{ return is_integer_ty() && get_integer_bitwidth()== width; }


bool type::is_floating_point_ty() const
{ return is_fp8_ty() || is_fp16_ty() || is_bf16_ty() || is_fp32_ty() || is_fp64_ty(); }

bool type::is_sized() const {
  // primitive types are sized
  if(is_integer_ty() || is_floating_point_ty() ||
     is_pointer_ty()){
    return true;
  }
  // tile types are sizes
  if(is_block_ty())
    return get_scalar_ty()->is_sized();
  return false;
}

// primitive types
type *type::get_void_ty(context &ctx) { return &ctx.p_impl->void_ty; }
type *type::get_label_ty(context &ctx) { return &ctx.p_impl->label_ty; }
// floating point
type *type::get_fp8_ty(context &ctx) { return &ctx.p_impl->fp8_ty; }
type *type::get_fp16_ty(context &ctx) { return &ctx.p_impl->fp16_ty; }
type *type::get_bf16_ty(context &ctx) { return &ctx.p_impl->bf16_ty; }
type *type::get_fp32_ty(context &ctx) { return &ctx.p_impl->fp32_ty; }
type *type::get_fp64_ty(context &ctx) { return &ctx.p_impl->fp64_ty; }
// integer types
integer_type *type::get_int1_ty(context &ctx) { return &ctx.p_impl->int1_ty; }
integer_type *type::get_int8_ty(context &ctx) { return &ctx.p_impl->int8_ty; }
integer_type *type::get_int16_ty(context &ctx) { return &ctx.p_impl->int16_ty; }
integer_type *type::get_int32_ty(context &ctx) { return &ctx.p_impl->int32_ty; }
integer_type *type::get_int64_ty(context &ctx) { return &ctx.p_impl->int64_ty; }
integer_type *type::get_int128_ty(context &ctx) { return &ctx.p_impl->int128_ty; }



pointer_type::pointer_type(type *ty, unsigned address_space)
    : type(ty->get_context(), PointerTyID), address_space_(address_space){
  contained_tys_.push_back(ty);
}

bool pointer_type::is_valid_elt_ty(type *ty){
  return !ty->is_void_ty() && !ty->is_label_ty() &&
         !ty->is_metadata_ty() && !ty->is_token_ty();
}

pointer_type* pointer_type::get(type *elt_ty, unsigned address_space){
  assert(elt_ty && "Can't get a pointer to <null> type!");
  assert(is_valid_elt_ty(elt_ty) && "Invalid type for pointer element!");
  // look-up
  context_impl *impl = elt_ty->get_context().p_impl.get();
  std::unique_ptr<pointer_type> &entry = impl->ptr_tys[std::make_pair(elt_ty, address_space)];
  if(!entry)
    entry.reset(new pointer_type(elt_ty, address_space));
  return entry.get();
}

//===----------------------------------------------------------------------===//
//                               composite_type class
//===----------------------------------------------------------------------===//

type* composite_type::get_type_at_index(value *) const{
  assert(is_block_ty());
  return get_scalar_ty();
}

bool composite_type::index_valid(value *idx) const{
  assert(is_block_ty());
  return idx->get_type()->is_int_or_tileint_ty();
}

//===----------------------------------------------------------------------===//
//                               struct_type class
//===----------------------------------------------------------------------===//

struct_type::struct_type(const contained_tys_vec_t& tys, bool is_packed)
  : composite_type(tys[0]->get_context(), StructTyID), is_packed_(is_packed) {
 contained_tys_ = tys;
}

struct_type* struct_type::get(const contained_tys_vec_t& tys, bool is_packed) {
  assert(tys.size());
  context_impl* impl = tys[0]->get_context().p_impl.get();
  struct_type *& entry = impl->struct_tys[tys];
  if(!entry)
    entry = new struct_type(tys, is_packed);
  return  entry;
}


//===----------------------------------------------------------------------===//
//                               block_type class
//===----------------------------------------------------------------------===//

block_type::block_type(type *ty, const block_shapes_t &shapes)
    : composite_type(ty->get_context(), BlockTyID), shapes_(shapes) {
  contained_tys_.push_back(ty);
}

bool block_type::is_valid_elt_ty(type *ty) {
  return ty->is_pointer_ty() || ty->is_floating_point_ty() || ty->is_integer_ty();
}

unsigned block_type::get_num_elements() const {
  unsigned res = 1;
  for(auto shape: shapes_)
    res *= shape;
  return res;
}

unsigned block_type::get_bitwidth() const {
  return get_num_elements() * get_tile_element_ty()->get_primitive_size_in_bits();
}

block_type* block_type::get(type *elt_ty, const block_shapes_t &shapes) {
  assert(elt_ty && "Can't get a tile of <null> type!");
  assert(shapes.size() && "Can't create a tile with empty shapes!");
  assert(is_valid_elt_ty(elt_ty) && "Invalid type for tile element!");
  // look-up
  context_impl *impl = elt_ty->get_context().p_impl.get();
  std::unique_ptr<block_type> &entry = impl->block_tys[std::make_pair(elt_ty, shapes)];
  if(!entry)
    entry.reset(new block_type(elt_ty, shapes));
  return entry.get();
}

block_type* block_type::get_same_shapes(type *ty, type *ref){
  assert(ref->is_block_ty());
  return get(ty, ref->get_block_shapes());
}

//===----------------------------------------------------------------------===//
//                               function_type class
//===----------------------------------------------------------------------===//

function_type::function_type(type *ret_ty, const std::vector<type*> &param_tys):
   type(ret_ty->get_context(), FunctionTyID) {
  contained_tys_.push_back(ret_ty);
  for(type *ty: param_tys)
    contained_tys_.push_back(ty);
}

function_type* function_type::get(type *ret_ty, const std::vector<type *> &param_tys) {
  return new function_type(ret_ty, param_tys);
}

}
}
