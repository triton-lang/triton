#ifndef TDL_INCLUDE_IR_BUILDER_H
#define TDL_INCLUDE_IR_BUILDER_H

#include <iostream>
#include <vector>
#include <string>
#include "instructions.h"
#include "basic_block.h"
#include "type.h"

namespace triton{
namespace ir{

class basic_block;
class value;
class type;
class constant_int;
class instruction;
class context;
class phi_node;

/* Builder */
class builder{
  typedef basic_block::iterator iterator;

public:
  // Constructor
  builder(context &ctx);
  // Setters
  void set_insert_point(iterator instr);
  void set_insert_point(instruction* i);
  void set_insert_point(basic_block* block);
  basic_block* get_insert_block() { return block_; }
  iterator get_insert_point() { return insert_point_;}
  // Constants
  value *get_int32(unsigned val);
  // Types
  type *get_int1_ty();
  type *get_int8_ty();
  type *get_int16_ty();
  type *get_int32_ty();
  type *get_int64_ty();
  type *get_half_ty();
  type *get_float_ty();
  type *get_double_ty();
  // Insert
  template<typename InstTy>
  InstTy* insert(InstTy *inst, const std::string &name = ""){
    assert(block_);
    block_->get_inst_list().insert(insert_point_, inst);
    inst->set_parent(block_);
    inst->set_name(name);
    return inst;
  }
  // terminator instructions
  value* create_br(basic_block *dest);
  value* create_cond_br(value *cond, basic_block* if_dest, basic_block* else_dest);
  value* create_ret_void();
  // Tile-level control flow
//  value *create_mask(value *pred, const std::string &name = "");
//  value *create_merge(value *mask_true, value *value_true, value *mask_false, value *value_false, const std::string &name = "");
  // Cast instructions
  value *create_cast(cast_inst::op_t op, value *v, type *dst_ty, const std::string &name = "");
  value* create_si_to_fp(value *src, type *dst_ty, const std::string &name = "");
  value* create_ui_to_fp(value *src, type *dst_ty, const std::string &name = "");
  value* create_fp_to_si(value *src, type *dst_ty, const std::string &name = "");
  value* create_fp_to_ui(value *src, type *dst_ty, const std::string &name = "");
  value* create_fp_ext(value *src, type *dst_ty, const std::string &name = "");
  value* create_fp_trunc(value *src, type *dst_ty, const std::string &name = "");
  value* create_int_cast(value *src, type *dst_ty, bool is_signed, const std::string &name = "");
  value *create_downcast(value *arg, const std::string &name = "");
  // Phi instruction
  phi_node* create_phi(type *ty, unsigned num_reserved, const std::string &name = "");
  // Binary instructions
  value *create_insert_nuwnswb_binop(binary_operator::op_t op, value *lhs, value *rhs, const std::string &name, bool has_nuw, bool has_nsw);
  value *create_fmul(value *lhs, value *rhs, const std::string &name = "");
  value *create_fdiv(value *lhs, value *rhs, const std::string &name = "");
  value *create_frem(value *lhs, value *rhs, const std::string &name = "");
  value *create_fadd(value *lhs, value *rhs, const std::string &name = "");
  value *create_fsub(value *lhs, value *rhs, const std::string &name = "");
  value *create_mul(value *lhs, value *rhs, const std::string &name = "", bool has_nuw = false, bool has_nsw = false);
  value *create_sdiv(value *lhs, value *rhs, const std::string &name = "");
  value *create_udiv(value *lhs, value *rhs, const std::string &name = "");
  value *create_srem(value *lhs, value *rhs, const std::string &name = "");
  value *create_urem(value *lhs, value *rhs, const std::string &name = "");
  value *create_add(value *lhs, value *rhs, const std::string &name = "", bool has_nuw = false, bool has_nsw = false);
  value *create_sub(value *lhs, value *rhs, const std::string &name = "", bool has_nuw = false, bool has_nsw = false);
  value *create_shl(value *lhs, value *rhs, const std::string &name = "", bool has_nuw = false, bool has_nsw = false);
  value *create_ashr(value *lhs, value *rhs, const std::string &name = "", bool has_nuw = false, bool has_nsw = false);
  // GEP
  value *create_gep(value *ptr, const std::vector<value*>& idx_list, const std::string &name = "");
  // Comparison (int)
  value *create_icmp(cmp_inst::pred_t pred, value *lhs, value *rhs, const std::string &name = "");
  value *create_icmpSLE(value *lhs, value *rhs, const std::string &name = "");
  value *create_icmpSLT(value *lhs, value *rhs, const std::string &name = "");
  value *create_icmpSGE(value *lhs, value *rhs, const std::string &name = "");
  value *create_icmpSGT(value *lhs, value *rhs, const std::string &name = "");
  value *create_icmpULE(value *lhs, value *rhs, const std::string &name = "");
  value *create_icmpULT(value *lhs, value *rhs, const std::string &name = "");
  value *create_icmpUGE(value *lhs, value *rhs, const std::string &name = "");
  value *create_icmpUGT(value *lhs, value *rhs, const std::string &name = "");
  value *create_icmpEQ(value *lhs, value *rhs, const std::string &name = "");
  value *create_icmpNE(value *lhs, value *rhs, const std::string &name = "");
  // Comparison (float)
  value *create_fcmp(cmp_inst::pred_t pred, value *lhs, value *rhs, const std::string &name = "");
  value *create_fcmpOLT(value *lhs, value *rhs, const std::string &name = "");
  value *create_fcmpOGT(value *lhs, value *rhs, const std::string &name = "");
  value *create_fcmpOLE(value *lhs, value *rhs, const std::string &name = "");
  value *create_fcmpOGE(value *lhs, value *rhs, const std::string &name = "");
  value *create_fcmpOEQ(value *lhs, value *rhs, const std::string &name = "");
  value *create_fcmpONE(value *lhs, value *rhs, const std::string &name = "");
  // Logical
  value *create_and(value *lhs, value *rhs, const std::string &name = "");
  value *create_xor(value *lhs, value *rhs, const std::string &name = "");
  value *create_or(value *lhs, value *rhs, const std::string &name = "");
  // Side effects
  value *create_fneg(value *arg, const std::string &name = "");
  value *create_neg(value *arg, const std::string &name = "");
  value *create_not(value *arg, const std::string &name = "");
  // Input/Output
  value *create_load(value *arg, const std::string &name = "");
  value *create_store(value *ptr, value *val, const std::string &name = "");
  // Tile instruction
  value *create_splat(value *arg, const type::tile_shapes_t &shapes, const std::string &name = "");
  value *create_reshape(value *arg, const type::tile_shapes_t &shapes, const std::string &name = "");
  value *create_broadcast(value *arg, const type::tile_shapes_t &shapes, const std::string &name = "");
  // Built-in instruction
  value *create_get_global_range(unsigned axis, type::tile_shapes_t::value_type size, const std::string &name = "");
  value *create_get_range_id(unsigned axis, const std::string &name = "");
  value *create_atomic_cas(value *ptr, value *cmp, value *val, const std::string &name = "");
  value *create_atomic_add(value *ptr, value *val, const std::string &name = "");
  value *create_dot(value *A, value *B, value *C, const std::string &name = "");
  value *create_trans(value *A, const std::string &name = "");
  value *create_sqrt(value *A, const std::string &name = "");
  value *create_reduce(value *A, const std::string &name = "");
  value *create_select(value *pred, value *if_value, value *else_value, const std::string &name = "");
  // Intrinsics
  value *create_copy_to_shared(value *arg, const std::string &name = "");
  value *create_vectorize(value *arg, const std::string &name = "");
  value *create_barrier(const std::string &name = "");

private:
  context &ctx_;
  basic_block *block_;
  iterator insert_point_;
};

}
}

#endif
