#pragma once

#ifndef _TRITON_IR_INSTRUCTIONS_H_
#define _TRITON_IR_INSTRUCTIONS_H_

#include <vector>
#include <map>
#include "triton/ir/enums.h"
#include "triton/ir/constant.h"
#include "triton/ir/value.h"
#include "triton/ir/type.h"
#include "triton/ir/metadata.h"

namespace triton{
namespace ir{

class constant_int;
class constant;
class constant_range;
class basic_block;
class context;

//===----------------------------------------------------------------------===//
//                               instruction classes
//===----------------------------------------------------------------------===//

class result_reference;
class instruction: public user{
public:
  virtual std::string repr_impl() const = 0;

protected:
  // constructors
  instruction(type *ty, unsigned num_ops, unsigned num_results = 1, const std::string &name = "", instruction *next = nullptr);

public:
  // parent
  void set_parent(basic_block *block)                         { parent_ = block; }
  const basic_block *get_parent() const                       { return parent_;  }
  basic_block *get_parent()                                   { return parent_;  }
  void erase_from_parent();
  // helpers
  bool has_tile_result_or_op();
  // repr
  std::string repr() const                                    { return repr_impl(); }
  // results
  unsigned get_num_results() const                            { return results_.size(); }
  value* get_result(unsigned i)                               { return results_.at(i); }
  // metadata
  void set_metadata(ir::metadata::kind_t kind,
                    unsigned value)                           { metadatas_[kind] = value;}
  unsigned get_metadata(ir::metadata::kind_t kind)            { return metadatas_[kind];}
private:
  basic_block *parent_;
  std::vector<value*> results_;
  std::map<ir::metadata::kind_t, unsigned> metadatas_;
};

// result reference
class result_reference: public value {
public:
  result_reference(instruction *ref, unsigned arg_id, const std::string &name = "");
  instruction *get_ref();
  unsigned     get_arg_id();

private:
  instruction *ref_;
  unsigned arg_id_;
};

//===----------------------------------------------------------------------===//
//                               phi_node classes
//===----------------------------------------------------------------------===//

class phi_node: public instruction{
private:
  phi_node(type *ty, unsigned num_reserved, const std::string &name, instruction *next);
  std::string repr_impl() const { return "phi"; }

public:
  void set_incoming_value(unsigned i, value *v);
  void set_incoming_block(unsigned i, basic_block *block);
  value *get_incoming_value(unsigned i) { return get_operand(i); }
  basic_block *get_incoming_block(unsigned i) { return blocks_[i]; }
  unsigned get_num_incoming() { return get_num_operands(); }
  void add_incoming(value *v, basic_block *block);

  // Type
  void set_type(type *ty) { ty_ = ty; }

  // Factory methods
  static phi_node* create(type *ty, unsigned num_reserved, const std::string &name = "", instruction *next = nullptr);

private:
  unsigned num_reserved_;
  std::vector<basic_block*> blocks_;
};

//===----------------------------------------------------------------------===//
//                               binary_operator classes
//===----------------------------------------------------------------------===//
class binary_operator: public instruction{
public:
  typedef binary_op_t op_t;

private:
  std::string repr_impl() const;

protected:
  // Constructors
  binary_operator(binary_op_t op, value *lhs, value *rhs, type *ty, const std::string &name, instruction *next);

public:
  // Get operand
  binary_op_t get_op() const { return op_; }

  // Bool
  bool is_terminator()  const;
  bool is_binary_op()   const;
  bool is_int_div_rem() const;
  bool is_shift()       const;
  bool is_cast()        const;
  bool is_int_mult()    const;
  bool is_int_add_sub() const;
  bool is_int_div()     const;
  bool is_int_rem()     const;
  bool is_shl()         const;
  bool is_shr()         const;

  // Wraps
  void set_has_no_unsigned_wrap(bool b = true) { has_no_unsigned_wrap_ = b; }
  void set_has_no_signed_wrap(bool b = true)   { has_no_signed_wrap_ = b; }

  // Factory methods
  static binary_operator *create(binary_op_t op, value *lhs, value *rhs,
                                 const std::string &name = "", instruction *next = nullptr);
  static binary_operator *create_fneg(value *arg, const std::string &name = "", instruction *next = nullptr);
  static binary_operator *create_neg(value *arg, const std::string &name = "", instruction *next = nullptr);
  static binary_operator *create_not(value *arg, const std::string &name = "", instruction *next = nullptr);

public:
  binary_op_t op_;
  bool has_no_unsigned_wrap_;
  bool has_no_signed_wrap_;
};


//===----------------------------------------------------------------------===//
//                               cmp_inst classes
//===----------------------------------------------------------------------===//

class cmp_inst: public instruction{
public:
  typedef cmp_pred_t pred_t;
private:
  std::string repr_impl() const;

protected:
  cmp_inst(type *ty, cmp_pred_t pred, value *lhs, value *rhs, const std::string &name, instruction *next);
  static bool is_fp_predicate(cmp_pred_t pred);
  static bool is_int_predicate(cmp_pred_t pred);
  static type* make_cmp_result_type(type *ty);

public:
  cmp_pred_t get_pred() const { return pred_; }

private:
  cmp_pred_t pred_;
};

class icmp_inst: public cmp_inst{
  using cmp_inst::cmp_inst;

public:
  static icmp_inst* create(cmp_pred_t pred, value *lhs, value *rhs,
                    const std::string &name = "", instruction *next = nullptr);
};

class fcmp_inst: public cmp_inst{
  using cmp_inst::cmp_inst;

public:
  static fcmp_inst* create(cmp_pred_t pred, value *lhs, value *rhs,
                    const std::string &name = "", instruction *next = nullptr);
};

//===----------------------------------------------------------------------===//
//                               unary_inst classes
//===----------------------------------------------------------------------===//

class unary_inst: public instruction {
protected:
  unary_inst(type *Ty, value *v, const std::string &name, instruction *next);
};


//===----------------------------------------------------------------------===//
//                               cast_inst classes
//===----------------------------------------------------------------------===//

class cast_inst: public unary_inst{
private:
  std::string repr_impl() const;

protected:
  cast_inst(type *ty, value *v, const std::string &name, instruction *next, cast_op_t op)
    : unary_inst(ty, v, name, next), op_(op) { }

private:
  static bool is_valid(cast_op_t op, value *arg, type *ty);

public:
  // accessors
  cast_op_t get_op() const { return op_; }

  // factory methods
  static cast_inst *create(cast_op_t op, value *arg, type *ty,
                           const std::string &name = "", instruction *next = nullptr);
  static cast_inst *create_integer_cast(value *arg, type *ty, bool is_signed,
                           const std::string &name = "", instruction *next = nullptr);

private:
  cast_op_t op_;
};

#define TRITON_IR_DECLARE_CAST_INST_SIMPL(name, op) \
class name : public cast_inst{ \
  friend class cast_inst; \
  name(type *ty, value *v, const std::string &name, instruction *next) \
    : cast_inst(ty, v, name, next, op){ } \
};

TRITON_IR_DECLARE_CAST_INST_SIMPL(trunc_inst, cast_op_t::Trunc)
TRITON_IR_DECLARE_CAST_INST_SIMPL(z_ext_inst, cast_op_t::ZExt)
TRITON_IR_DECLARE_CAST_INST_SIMPL(s_ext_inst, cast_op_t::SExt)
TRITON_IR_DECLARE_CAST_INST_SIMPL(fp_trunc_inst, cast_op_t::FPTrunc)
TRITON_IR_DECLARE_CAST_INST_SIMPL(fp_ext_inst, cast_op_t::FPExt)
TRITON_IR_DECLARE_CAST_INST_SIMPL(ui_to_fp_inst, cast_op_t::UIToFP)
TRITON_IR_DECLARE_CAST_INST_SIMPL(si_to_fp_inst, cast_op_t::SIToFP)
TRITON_IR_DECLARE_CAST_INST_SIMPL(fp_to_ui_inst, cast_op_t::FPToUI)
TRITON_IR_DECLARE_CAST_INST_SIMPL(fp_to_si_inst, cast_op_t::FPToSI)
TRITON_IR_DECLARE_CAST_INST_SIMPL(ptr_to_int_inst, cast_op_t::PtrToInt)
TRITON_IR_DECLARE_CAST_INST_SIMPL(int_to_ptr_inst, cast_op_t::IntToPtr)
TRITON_IR_DECLARE_CAST_INST_SIMPL(bit_cast_inst, cast_op_t::BitCast)
TRITON_IR_DECLARE_CAST_INST_SIMPL(addr_space_cast_inst, cast_op_t::AddrSpaceCast)

//===----------------------------------------------------------------------===//
//                               terminator_inst classes
//===----------------------------------------------------------------------===//

class terminator_inst: public instruction{
  using instruction::instruction;
};

// return instruction
class return_inst: public terminator_inst{
private:
  std::string repr_impl() const { return "ret"; }
  return_inst(context &ctx, value *ret_val, instruction *next);

public:
  // accessors
  value *get_return_value()
  { return get_num_operands() ? get_operand(0) : nullptr; }

  unsigned get_num_successors() const { return 0; }

  // factory methods
  static return_inst* create(context &ctx, value *ret_val = nullptr, instruction *next = nullptr);
};

// base branch instruction
class branch_inst: public terminator_inst{
private:
  std::string repr_impl() const { return "br"; }

protected:
  using terminator_inst::terminator_inst;

public:
  static branch_inst* create(basic_block *dest,
                             instruction *next = nullptr);
  static branch_inst* create(value *cond, basic_block *if_dest, basic_block *else_dest,
                             instruction *next = nullptr);
};

// conditional branch
class cond_branch_inst: public branch_inst {
private:
  friend class branch_inst;
  cond_branch_inst(basic_block *if_dst, basic_block *else_dst, value *cond, instruction *next);

public:
  basic_block *get_true_dest()  { return (basic_block*)get_operand(0); }
  basic_block *get_false_dest() { return (basic_block*)get_operand(1); }
  value *get_cond()             { return get_operand(2); }
};

// unconditional branch
class uncond_branch_inst: public branch_inst {
private:
  friend class branch_inst;
  uncond_branch_inst(basic_block *dst, instruction *next);

public:
  basic_block *get_dest()  { return (basic_block*)get_operand(0); }
};

// ternary
class ternary_inst: public instruction {
private:
  std::string repr_impl() const { return "ternary"; }
  ternary_inst(value *cond, value *true_value, value *false_value,
               const std::string &name, instruction *next);

public:
  value *get_cond() { return get_operand(0); }
  value *get_true_value() { return get_operand(1); }
  value *get_false_value() { return get_operand(2); }
  static ternary_inst* create(value *cond, value *true_value, value *false_value,
                              const std::string &name = "", instruction *next = nullptr);
};

//===----------------------------------------------------------------------===//
//                               getelementptr_inst classes
//===----------------------------------------------------------------------===//

class getelementptr_inst: public instruction{
private:
  std::string repr_impl() const { return "getelementptr"; }
  getelementptr_inst(type *pointee_ty, value *ptr, const std::vector<value*> &idx, const std::string &name, instruction *next);

private:
  static type *get_return_type(type *ty, value *ptr, const std::vector<value*> &idx);
  static type *get_indexed_type_impl(type *ty, const std::vector<value *> &idx);
  static type *get_indexed_type(type *ty, const std::vector<value*> &idx);

public:
  // accessors
  type *get_source_elt_ty() { return source_elt_ty; }
  op_iterator idx_begin()       { return op_begin() + 1; }
  op_iterator idx_end()         { return op_end(); }
  value *get_pointer_operand()  { return *op_begin(); }

  // factory methods
  static getelementptr_inst* create(value *ptr, const std::vector<value*> &idx,
                                    const std::string &name = "", instruction *next = nullptr);

private:
  type *source_elt_ty;
  type *res_elt_ty;
};

//===----------------------------------------------------------------------===//
//                          load_inst/store_inst classes
//===----------------------------------------------------------------------===//

class io_inst: public instruction {
protected:
  io_inst(type *ty, unsigned num_ops, unsigned num_results = 1, const std::string &name = "", instruction *next = nullptr);
public:
//  value *get_mask() const;
//  value *get_false_value() const;
};

class load_inst: public io_inst{
protected:
  load_inst(value *ptr, unsigned num_extra_ops, const std::string &name, instruction *next);

private:
  std::string repr_impl() const { return "load"; }
  static type *get_pointee_type(type *ty);

public:
  // accessors
  value *get_pointer_operand() { return get_operand(0); }
  // factory method
  static load_inst* create(value *ptr,
                           const std::string &name = "",
                           instruction *next = nullptr);
};

class masked_load_inst: public load_inst{
private:
  std::string repr_impl() const { return "masked_load"; }
  masked_load_inst(value *ptr, value *mask, value *false_value,
                   const std::string &name, instruction *next);

public:
  // accessors
  value *get_mask_operand() { return get_operand(1); }
  value *get_false_value_operand() { return get_operand(2); }
  // factory method
  static masked_load_inst* create(value *ptr, value *mask, value *false_value,
                                  const std::string &name = "",
                                  instruction *next = nullptr);
};

class store_inst: public io_inst{
protected:
  store_inst(value *ptr, value *v, unsigned num_extra_ops,
             const std::string &name, instruction *next);

private:
  std::string repr_impl() const { return "store"; }

public:
  // accessors
  value *get_pointer_operand() { return get_operand(0); }
  value *get_value_operand() { return get_operand(1); }
  // factory method
  static store_inst* create(value* ptr, value *v,
                            const std::string &name = "",
                            instruction *next = nullptr);
};

class masked_store_inst: public store_inst{
private:
  std::string repr_impl() const { return "masked_store"; }
  masked_store_inst(value *ptr, value *v, value *mask,
                    const std::string &name, instruction *next);

public:
  // accessors
  value *get_mask_operand() { return get_operand(2); }
  // factory method
  static masked_store_inst* create(value *ptr, value *v, value *mask,
                                   const std::string &name = "",
                                   instruction *next = nullptr);
};

//===----------------------------------------------------------------------===//
//                               retile_inst classes
//===----------------------------------------------------------------------===//

// retile

class retile_inst: public unary_inst {
protected:
  retile_inst(value *arg, const type::tile_shapes_t &shapes, const std::string &name, instruction *next);
  static std::string shape_suffix(ir::type* ty);
};

// reshape

class reshape_inst: public retile_inst {
private:
  using retile_inst::retile_inst;
  std::string repr_impl() const { return "reshape" + shape_suffix(get_type()); }

public:
  static instruction* create(value *arg, const type::tile_shapes_t &shape_suffix,
                      const std::string &name = "", instruction *next = nullptr);
};

// splat

class splat_inst: public retile_inst {
private:
  using retile_inst::retile_inst;
  std::string repr_impl() const { return "splat" + shape_suffix(get_type()); }

public:
  static instruction* create(value *arg, const type::tile_shapes_t &shape_suffix,
                      const std::string &name = "", instruction *next = nullptr);
};

// broadcast

class broadcast_inst: public retile_inst {
private:
  using retile_inst::retile_inst;
  std::string repr_impl() const { return "broadcast" + shape_suffix(get_type()); }

public:
  static instruction* create(value *arg, const type::tile_shapes_t &shape_suffix,
                      const std::string &name = "", instruction *next = nullptr);
};


// downcast

class downcast_inst: public unary_inst {
private:
  using unary_inst::unary_inst;
  std::string repr_impl() const { return "downcast"; }

public:
  static instruction* create(value *arg, const std::string &name = "", instruction *next = nullptr);
};

//===----------------------------------------------------------------------===//
//                               builtin_inst classes
//===----------------------------------------------------------------------===//

class builtin_inst: public instruction{
protected:
  using instruction::instruction;
};

class get_program_id_inst: public builtin_inst {
private:
  get_program_id_inst(type *ty, unsigned axis, const std::string &name, instruction *next);
  std::string repr_impl() const { return "get_program_id(" + std::to_string(axis_) + ")"; }

public:
  static instruction* create(context &ctx, unsigned axis, const std::string &name = "", instruction *next = nullptr);
  unsigned get_axis() const { return axis_; }

private:
  unsigned axis_;
};

class get_num_program_inst: public builtin_inst {
private:
  get_num_program_inst(type *ty, unsigned axis, const std::string &name, instruction *next);
  std::string repr_impl() const { return "get_num_program(" + std::to_string(axis_) + ")"; }

public:
  static instruction* create(context &ctx, unsigned axis, const std::string &name = "", instruction *next = nullptr);
  unsigned get_axis() const { return axis_; }

private:
  unsigned axis_;
};

class atomic_cas_inst: public builtin_inst {
private:
  atomic_cas_inst(value *ptr, value *cmp, value *val, const std::string &name, instruction *next);
  std::string repr_impl() const { return "atomic_cas"; }

public:
  static instruction* create(value *ptr, value *cmp, value *val, const std::string &name = "", instruction *next = nullptr);
};

class atomic_exch_inst: public builtin_inst {
private:
  atomic_exch_inst(value *ptr, value *val, const std::string &name = "", instruction *next = nullptr);
  std::string repr_impl() const { return "atomic_exch"; }

public:
  static instruction* create(value *ptr, value *val, const std::string &name = "", instruction *next = nullptr);
};

class atomic_add_inst: public builtin_inst {
private:
  atomic_add_inst(value *ptr, value *val, const std::string &name = "", instruction *next = nullptr);
  std::string repr_impl() const { return "atomic_add"; }

public:
  static instruction* create(value *ptr, value *val, const std::string &name = "", instruction *next = nullptr);
};

class dot_inst: public builtin_inst {
public:
  enum TransT { NoTrans, Trans };

private:
  dot_inst(value *A, value *B, value *C, TransT AT, TransT BT, const std::string &name, instruction *next);
  std::string repr_impl() const { return std::string("dot.") + ((AT_==NoTrans)?"n":"t") + ((BT_==NoTrans)?"n":"t"); }

public:
  static instruction *create(value *A, value *B, value *C, bool AT, bool BT, const std::string &name = "", instruction *next = nullptr);
  static instruction* create_nn(value *A, value *B, value *C, const std::string &name = "", instruction *next = nullptr);
  static instruction* create_nt(value *A, value *B, value *C, const std::string &name = "", instruction *next = nullptr);
  static instruction* create_tn(value *A, value *B, value *C, const std::string &name = "", instruction *next = nullptr);
  static instruction* create_tt(value *A, value *B, value *C, const std::string &name = "", instruction *next = nullptr);
  bool is_a_trans() { return AT_ == Trans; }
  bool is_b_trans() { return BT_ == Trans; }

private:
  TransT AT_;
  TransT BT_;
};

//class outer_inst: public builtin_inst {
//private:
//  outer_inst(value *A, value *B, value *C, const std::string &name, instruction *next);
//public:
//  static instruction* create(value *A, value *B, value *C, const std::string &name = "", instruction *next = nullptr);
//};

class trans_inst: public builtin_inst {
public:
  ir::type* get_res_ty(ir::type* in, std::vector<constant_int *> perm);
  std::vector<constant_int*> init_perm(ir::type* ty, const std::vector<constant_int*>& perm);

private:
  trans_inst(value *arg, const std::vector<constant_int*>& perm, const std::string& name, instruction* next);
  std::string repr_impl() const {
    std::string res = "trans<";
    //for(ir::constant_int *x: perm_)
    //  res += x->repr() + ",";
    res[res.size()-1] = '>';
    return res;
  }

public:
  static instruction* create(value *arg, const std::vector<constant_int*>& perm = {}, const std::string &name = "", instruction *next = nullptr);
  const std::vector<constant_int*> get_perm() const;

private:
  std::vector<constant_int*> perm_;
};

class sqrt_inst: public builtin_inst {
private:
  sqrt_inst(value *arg, const std::string& name, instruction* next);
  std::string repr_impl() const { return "sqrt"; }
public:
  static instruction* create(value *arg, const std::string &name = "", instruction *next = nullptr);
};

class reduce_inst: public builtin_inst {
private:
  static type* get_res_type(value *arg, unsigned axis);

private:
  reduce_inst(value* arg, unsigned axis, const std::string& name, instruction* next);
  std::string repr_impl() const { return "reduce"; }

public:
  static instruction* create(value *arg, unsigned axis, const std::string &name = "", instruction *next = nullptr);
  unsigned get_axis() const { return axis_; }

private:
  unsigned axis_;
};

class select_inst: public builtin_inst {
private:
  select_inst(value *pred, value *if_value, value *else_value, const std::string& name, instruction* next);
  std::string repr_impl() const { return "select"; }

public:
  static instruction* create(value *pred, value *if_value, value *else_value, const std::string &name = "", instruction *next = nullptr);
};

//===----------------------------------------------------------------------===//
//                               intrinsics classes
//===----------------------------------------------------------------------===//

class copy_to_shared_inst: public unary_inst{
private:
  using unary_inst::unary_inst;
  std::string repr_impl() const { return "copy_to_shared"; }

public:
  static copy_to_shared_inst* create(value *arg, const std::string &name = "",
                                     instruction *next = nullptr);
};

class barrier_inst: public instruction{
private:
  barrier_inst(context &ctx, const std::string &name, instruction *next);
  std::string repr_impl() const { return "barrier"; }

public:
  static barrier_inst* create(context &ctx, const std::string &name = "",
                                            instruction *next = nullptr);
};

class vectorize_inst: public unary_inst{
private:
  using unary_inst::unary_inst;
  std::string repr_impl() const { return "vectorize"; }

public:
  static vectorize_inst* create(value *arg, const std::string &name = "", instruction *next = nullptr);
};

// On NVIDIA, implementation is such that
// constant_range = nv_dynamic_program_idx + nv_static_program_idx
// so as to enable re-association on nv_static_program_idx which is constant
class nv_dynamic_program_idx_inst: public instruction {
private:
  nv_dynamic_program_idx_inst(type *ty, const std::string &name, instruction *next);
  std::string repr_impl() const { return "nv_dynamic_program_idx"; }

public:
  static nv_dynamic_program_idx_inst* create(type *ty, const std::string &name = "", instruction *next = nullptr);
};

class nv_static_program_idx: public constant {
private:
  nv_static_program_idx(constant_range *range);

public:
  static nv_static_program_idx *get(constant_range* range);
  constant_range* get_range() const;

private:
  constant_range *range_;
};


}
}

#endif
