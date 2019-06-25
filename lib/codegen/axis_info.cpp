#include "triton/codegen/axis_info.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/ir/basic_block.h"
#include "triton/ir/instructions.h"
#include "triton/ir/type.h"

namespace triton {
namespace codegen{


template<class T>
inline T add_to_cache(ir::value *i, T value, std::map<ir::value*, T> &map) {
  return map.insert(std::make_pair(i, value)).first->second;
}


bool axis_info::is_first_axis_unit(ir::value *x){
  if(x->get_type()->is_tile_ty())
    return x->get_type()->get_tile_shapes()[0]->get_value() == 1;
  else
    return true;
}

bool axis_info::populate_is_constant(ir::value *v) {
  // helper for the cache
  auto cache = [this,v](bool value){ return add_to_cache(v, value, is_constant_); };
  // populate
  if(v->get_type()->is_tile_ty()){
    if(auto *x = dynamic_cast<ir::retile_inst*>(v)){
      bool value = populate_is_constant(x->get_operand(0));
      // check if broadcast (i.e., constant) along contiguous dimension
      if(is_first_axis_unit(x->get_operand(0))
         && !is_first_axis_unit(x))
        return cache(value);
    }
    // otherwise the tile is not constant in the contiguous dimension
    return cache(false);
  }
  // scalars are always constant in the contiguous dimension
  return cache(true);
}

unsigned axis_info::populate_max_contiguous(ir::value *v){
  // helper for the cache
  auto cache = [this,v](unsigned value){ return add_to_cache(v, value, max_contiguous_); };
  // populate
  if(v->get_type()->is_tile_ty()){
    auto shapes = v->get_type()->get_tile_shapes();
    if(dynamic_cast<ir::get_global_range_inst*>(v))
      return cache(shapes[0]->get_value());
    if(auto *x = dynamic_cast<ir::binary_operator*>(v)){
      ir::value* lhs = x->get_operand(0);
      ir::value* rhs = x->get_operand(1);
      unsigned lhs_max_contiguous = populate_max_contiguous(lhs);
      bool lhs_has_cst = populate_is_constant(lhs);
      unsigned rhs_max_contiguous = populate_max_contiguous(rhs);
      bool rhs_has_cst = populate_is_constant(rhs);
      if(x->is_int_add_sub()){
        if(lhs_has_cst)
          return cache(rhs_max_contiguous);
        if(rhs_has_cst)
          return cache(lhs_max_contiguous);
      }
    }
  }
  return cache(1);
}

unsigned axis_info::populate_multiple_of(ir::value *v){
  auto cache = [this,v](unsigned value){ return add_to_cache(v, value, max_contiguous_); };

  if(auto *x = dynamic_cast<ir::argument*>(v)){
    std::set<ir::attribute> attributes = x->get_parent()->get_attributes(x);
    for(auto attr: attributes){
      if(attr.get_kind() == ir::multiple_of)
        return cache(attr.get_value());
    }
  }
  if(auto *x = dynamic_cast<ir::binary_operator*>(v)){
    int lhs = populate_multiple_of(x->get_operand(0));
    int rhs = populate_multiple_of(x->get_operand(1));
    if(x->is_int_mult())
      return cache(lhs * rhs);
    if(x->is_int_add_sub())
      return cache(std::min(lhs, rhs));
    if(x->is_int_div())
      return cache(std::max(lhs / rhs, 1));
    if(x->is_int_rem())
      return cache(std::max(lhs % rhs, 1));
    if(x->is_shl())
      return cache(lhs << rhs);
    if(x->is_shr())
      return cache(std::max(lhs >> rhs, 1));
  }
  if(auto *x = dynamic_cast<ir::retile_inst*>(v)){
    return cache(populate_multiple_of(x->get_operand(0)));
  }
  return cache(1);
}



void axis_info::run(ir::module &mod) {
  // populate constant
  for(ir::function *fn: mod.get_function_list())
  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction *i: block->get_inst_list()){
    populate_is_constant(i);
  }

  // populate multiple_of
  for(ir::function *fn: mod.get_function_list())
  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction *i: block->get_inst_list()){
    populate_multiple_of(i);
  }

  // populate maximum contiguous
  for(ir::function *fn: mod.get_function_list())
  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction *i: block->get_inst_list()){
    populate_max_contiguous(i);
  }
}


}
}
