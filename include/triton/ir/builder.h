#pragma once

#ifndef _TRITON_IR_BUILDER_H_
#define _TRITON_IR_BUILDER_H_

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
public:
  typedef basic_block::iterator iterator;

public:
  // Constructor
  builder(context &ctx);
  // Getters
  // const context& get_context() const { return ctx_; }
  context& get_context() { return ctx_; }

  // Setters
  void set_insert_point(iterator instr);
  void set_insert_point(instruction* i);
  void set_insert_point_after(instruction* i);
  void set_insert_point(basic_block* block);
  basic_block* get_insert_block() { return block_; }
  iterator get_insert_point() { return insert_point_;}
  // Constants
  value *get_int1(bool val);
  value *get_int32(uint32_t val);
  value *get_int64(uint64_t val);
  value *get_float16(float val);
  value *get_float32(float val);
  value *get_range(int32_t lo, int32_t hi);
  // Types
  type *get_void_ty();
  type *get_int1_ty();
  type *get_int8_ty();
  type *get_int16_ty();
  type *get_int32_ty();
  type *get_int64_ty();
  type *get_fp8_ty();
  type *get_half_ty();
  type *get_bf16_ty();
  type *get_float_ty();
  type *get_double_ty();
  // Insert
  template<typename InstTy>
  InstTy* insert(InstTy *inst){
    assert(block_);
    block_->get_inst_list().insert(insert_point_, inst);
    inst->set_parent(block_);
//    for(ir::value* op: inst->ops())
//      op->add_use(inst);
    return inst;
  }
  // terminator instructions
  value* create_br(basic_block *dest);
  value* create_cond_br(value *cond, basic_block* if_dest, basic_block* else_dest);
  value* create_ret_void();
  value* create_ret(value *ret);
  // Dequantize instructions
  value* create_dequantize(value *src, value *scale, value *shift, type *dest_ty);
  // Cast instructions
  value* create_bitcast(value *src, type *dest_ty);
  value *create_cast(cast_op_t op, value *v, type *dst_ty);
  value* create_int_to_ptr(value *src, type *dst_ty);
  value* create_ptr_to_int(value *src, type *dst_ty);
  value* create_si_to_fp(value *src, type *dst_ty);
  value* create_ui_to_fp(value *src, type *dst_ty);
  value* create_fp_to_si(value *src, type *dst_ty);
  value* create_fp_to_ui(value *src, type *dst_ty);
  value* create_fp_ext(value *src, type *dst_ty);
  value* create_fp_trunc(value *src, type *dst_ty);
  value* create_int_cast(value *src, type *dst_ty, bool is_signed);
  value *create_downcast(value *arg);
  // Call instruction
  value* create_call(function* fn, const std::vector<value*>& args);
  value* create_launch(function* fn, const std::vector<value*>& args, const std::vector<value*>& grid, value* num_warps);
  // Phi instruction
  phi_node* create_phi(type *ty, unsigned num_reserved);
  // Binary instructions
  value *create_insert_nuwnswb_binop(binary_op_t op, value *lhs, value *rhs, bool has_nuw, bool has_nsw);
  value *create_fmul(value *lhs, value *rhs);
  value *create_fdiv(value *lhs, value *rhs);
  value *create_frem(value *lhs, value *rhs);
  value *create_fadd(value *lhs, value *rhs);
  value *create_fsub(value *lhs, value *rhs);
  value *create_sdiv(value *lhs, value *rhs);
  value *create_udiv(value *lhs, value *rhs);
  value *create_srem(value *lhs, value *rhs);
  value *create_urem(value *lhs, value *rhs);
  value *create_mul(value *lhs, value *rhs, bool has_nuw = false, bool has_nsw = false);
  value *create_add(value *lhs, value *rhs, bool has_nuw = false, bool has_nsw = false);
  value *create_sub(value *lhs, value *rhs, bool has_nuw = false, bool has_nsw = false);
  value *create_shl(value *lhs, value *rhs, bool has_nuw = false, bool has_nsw = false);
  value *create_lshr(value *lhs, value *rhs, bool has_nuw = false, bool has_nsw = false);
  value *create_ashr(value *lhs, value *rhs, bool has_nuw = false, bool has_nsw = false);
  // GEP
  value *create_gep(value *ptr, const std::vector<value*>& idx_list);
  // Comparison (int)
  value *create_icmp(cmp_pred_t pred, value *lhs, value *rhs);
  value *create_icmpSLE(value *lhs, value *rhs);
  value *create_icmpSLT(value *lhs, value *rhs);
  value *create_icmpSGE(value *lhs, value *rhs);
  value *create_icmpSGT(value *lhs, value *rhs);
  value *create_icmpULE(value *lhs, value *rhs);
  value *create_icmpULT(value *lhs, value *rhs);
  value *create_icmpUGE(value *lhs, value *rhs);
  value *create_icmpUGT(value *lhs, value *rhs);
  value *create_icmpEQ(value *lhs, value *rhs);
  value *create_icmpNE(value *lhs, value *rhs);
  // Comparison (float)
  value *create_fcmp(cmp_pred_t pred, value *lhs, value *rhs);
  value *create_fcmpOLT(value *lhs, value *rhs);
  value *create_fcmpOGT(value *lhs, value *rhs);
  value *create_fcmpOLE(value *lhs, value *rhs);
  value *create_fcmpOGE(value *lhs, value *rhs);
  value *create_fcmpOEQ(value *lhs, value *rhs);
  value *create_fcmpONE(value *lhs, value *rhs);
  value *create_fcmpULT(value *lhs, value *rhs);
  value *create_fcmpUGT(value *lhs, value *rhs);
  value *create_fcmpULE(value *lhs, value *rhs);
  value *create_fcmpUGE(value *lhs, value *rhs);
  value *create_fcmpUEQ(value *lhs, value *rhs);
  value *create_fcmpUNE(value *lhs, value *rhs);
  // Logical
  value *create_and(value *lhs, value *rhs);
  value *create_xor(value *lhs, value *rhs);
  value *create_or(value *lhs, value *rhs);
  // Input/Output
  value *create_load(value *arg, load_inst::CACHE_MODIFIER cache, load_inst::EVICTION_POLICY eviction, bool is_volatile);
  value *create_store(value *ptr, value *val, store_inst::EVICTION_POLICY eviction);
  value *create_masked_load(value *arg, value *mask, value *false_value, load_inst::CACHE_MODIFIER cache, load_inst::EVICTION_POLICY eviction, bool is_volatile);
  value *create_masked_store(value *ptr, value *val, value *mask, store_inst::EVICTION_POLICY eviction);
  // Struct instructions
  value *create_insert_value(value* val, value *elt, size_t idx);
  value *create_extract_value(value* val, size_t idx);
  // Block instruction
  value *create_splat(value *arg, const type::block_shapes_t &shapes);
  value *create_reshape(value *arg, const type::block_shapes_t &shapes);
  value *create_cat(value *lhs, value *rhs);
  value *create_broadcast(value *arg, const type::block_shapes_t &shapes);
  // Atomic instruction
  value *create_atomic_cas(value *ptr, value *cmp, value *val);
  value *create_atomic_rmw(atomic_rmw_op_t op, value *ptr, value *val, value *msk);
  value *create_atomic_max(value *ptr, value *val, value *msk);
  value *create_atomic_umax(value *ptr, value *val, value *msk);
  value *create_atomic_min(value *ptr, value *val, value *msk);
  value *create_atomic_umin(value *ptr, value *val, value *msk);
  value *create_atomic_fadd(value *ptr, value *val, value *msk);
  value *create_atomic_add(value *ptr, value *val, value *msk);
  value *create_atomic_and(value *ptr, value *val, value *msk);
  value *create_atomic_or(value *ptr, value *val, value *msk);
  value *create_atomic_xor(value *ptr, value *val, value *msk);
  value *create_atomic_xchg(value *ptr, value *val, value *msk);
  // Utilities
  value *create_clock();
  value *create_globaltimer();
  // Extern instruction
  value *create_extern_elementwise(const std::string &lib_name,
                                   const std::string &lib_path,
                                   const std::string &symbol_name,
                                   const std::vector<value *> &args,
                                   type *ret_ty);
  // Built-in instruction
  value *create_get_program_id(unsigned axis);
  value *create_get_num_programs(unsigned axis);
  value *create_exp(value* arg);
  value *create_cos(value* arg);
  value *create_sin(value* arg);
  value *create_log(value* arg);
  value *create_dot(value *A, value *B, value *C, bool trans_a, bool trans_b, bool allow_tf32);
  value *create_trans(value *A, const std::vector<int> &perm = {});
  value *create_sqrt(value *A);
  value *create_reduce(value *A, reduce_inst::op_t op, unsigned axis);
  value *create_select(value *pred, value *if_value, value *else_value);
  // Intrinsics
  // These have no place in the IR, and hopefully they can be removed at some point
  value *create_umulhi(value* lhs, value* rhs);
  value *create_copy_to_shared(value *arg);
  value *create_masked_load_async(value *arg, value *mask, value *false_value, load_inst::CACHE_MODIFIER cache, load_inst::EVICTION_POLICY);
  value *create_copy_from_shared(value *arg);
  value *create_barrier(const std::string &name = "");
  value *create_async_wait(int N);
  value *create_prefetch_s(value *arg, int inc);

private:
  context &ctx_;
  basic_block *block_;
  iterator insert_point_;
};


}
}

#endif
