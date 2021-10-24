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
#include "triton/ir/visitor.h"

#define _TRITON_DEFINE_CLONE(name) \
  ir::instruction* clone_impl() const { return new name(*this); }

#define _TRITON_DEFINE_ACCEPT(name) \
  void accept(visitor* v) { v->visit_ ## name (this); }

namespace triton{
namespace ir{

class constant_int;
class constant;
class make_range;
class basic_block;
class context;
class visitor;

//===----------------------------------------------------------------------===//
//                               instruction classes
//===----------------------------------------------------------------------===//

class result_reference;


class instruction: public user{
public:
  virtual std::string repr_impl() const = 0;

private:
  virtual ir::instruction* clone_impl() const = 0;

protected:
  // constructors
  instruction(type *ty, value_id_t ity, unsigned num_ops,
              const std::string &name = "", instruction *next = nullptr);

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
  // metadata
  void set_metadata(ir::metadata::kind_t kind,
                    unsigned value)                           { metadatas_[kind] = value;}
  unsigned get_metadata(ir::metadata::kind_t kind)            { return metadatas_[kind];}
  // cloning
  ir::instruction* clone() {
    ir::instruction* res = clone_impl();
//    for(auto it = op_begin(); it != op_end(); it++)
//      (*it)->add_use(res);
    res->parent_ = nullptr;
    res->users_.clear();
    return res;
  }
  // instruction id
  value_id_t get_id() const { return id_; }

  void print(std::ostream &os);

private:
  basic_block *parent_;
  std::map<ir::metadata::kind_t, unsigned> metadatas_;
  value_id_t id_;
};


//===----------------------------------------------------------------------===//
//                               phi_node classes
//===----------------------------------------------------------------------===//

class phi_node: public instruction {
private:
  phi_node(type *ty, unsigned num_reserved, const std::string &name, instruction *next);
  std::string repr_impl() const { return "phi"; }

public:
  void set_incoming_value(unsigned i, value *v);
  void set_incoming_block(unsigned i, basic_block *block);
  value *get_value_for_block(basic_block *block);
  value *get_incoming_value(unsigned i) { return get_operand(i); }
  basic_block *get_incoming_block(unsigned i) { return blocks_[i]; }
  unsigned get_num_incoming() { return get_num_operands(); }
  void add_incoming(value *v, basic_block *block);

  // Type
  void set_type(type *ty) { ty_ = ty; }

  // Factory methods
  static phi_node* create(type *ty, unsigned num_reserved, const std::string &name = "", instruction *next = nullptr);

  _TRITON_DEFINE_CLONE(phi_node)
  _TRITON_DEFINE_ACCEPT(phi_node)

private:
  unsigned num_reserved_;
  std::vector<basic_block*> blocks_;
};

//===----------------------------------------------------------------------===//
//                               binary_operator classes
//===----------------------------------------------------------------------===//
class binary_operator: public instruction {
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
//  static binary_operator *create_fneg(value *arg, const std::string &name = "", instruction *next = nullptr);
//  static binary_operator *create_neg(value *arg, const std::string &name = "", instruction *next = nullptr);
//  static binary_operator *create_not(value *arg, const std::string &name = "", instruction *next = nullptr);

  _TRITON_DEFINE_CLONE(binary_operator)
  _TRITON_DEFINE_ACCEPT(binary_operator)

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
  cmp_inst(type *ty, value_id_t id, cmp_pred_t pred,
           value *lhs, value *rhs, const std::string &name, instruction *next);
  static bool is_fp_predicate(cmp_pred_t pred);
  static bool is_int_predicate(cmp_pred_t pred);
  static type* make_cmp_result_type(type *ty);

public:
  cmp_pred_t get_pred() const { return pred_; }

private:
  cmp_pred_t pred_;
};

class icmp_inst: public cmp_inst {
  icmp_inst(type *ty, cmp_pred_t pred,
            value *lhs, value *rhs, const std::string &name, instruction *next);

public:
  static icmp_inst* create(cmp_pred_t pred, value *lhs, value *rhs,
                    const std::string &name = "", instruction *next = nullptr);
  _TRITON_DEFINE_CLONE(icmp_inst)
  _TRITON_DEFINE_ACCEPT(icmp_inst)
};

class fcmp_inst: public cmp_inst {
  fcmp_inst(type *ty, cmp_pred_t pred,
            value *lhs, value *rhs, const std::string &name, instruction *next);

public:
  static fcmp_inst* create(cmp_pred_t pred, value *lhs, value *rhs,
                    const std::string &name = "", instruction *next = nullptr);
  _TRITON_DEFINE_CLONE(fcmp_inst)
  _TRITON_DEFINE_ACCEPT(fcmp_inst)
};

//===----------------------------------------------------------------------===//
//                               unary_inst classes
//===----------------------------------------------------------------------===//

class unary_inst: public instruction {
protected:
  unary_inst(type *ty, value_id_t id, value *v, const std::string &name, instruction *next);
};


//===----------------------------------------------------------------------===//
//                               cast_inst classes
//===----------------------------------------------------------------------===//

class cast_inst: public unary_inst{
private:
  std::string repr_impl() const;

protected:
  cast_inst(type *ty, value_id_t id, value *v, const std::string &name, instruction *next, cast_op_t op)
    : unary_inst(ty, id, v, name, next), op_(op) { }

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

  _TRITON_DEFINE_ACCEPT(cast_inst)

private:
  cast_op_t op_;
};

#define TRITON_IR_DECLARE_CAST_INST_SIMPL(name, id, op) \
class name : public cast_inst { \
  _TRITON_DEFINE_CLONE(name) \
  friend class cast_inst; \
  name(type *ty, value *v, const std::string &name, instruction *next) \
    : cast_inst(ty, id, v, name, next, op){ } \
};

TRITON_IR_DECLARE_CAST_INST_SIMPL(trunc_inst, INST_CAST_TRUNC, cast_op_t::Trunc)
TRITON_IR_DECLARE_CAST_INST_SIMPL(z_ext_inst, INST_CAST_ZEXT, cast_op_t::ZExt)
TRITON_IR_DECLARE_CAST_INST_SIMPL(s_ext_inst, INST_CAST_SEXT, cast_op_t::SExt)
TRITON_IR_DECLARE_CAST_INST_SIMPL(fp_trunc_inst, INST_CAST_FP_TRUNC, cast_op_t::FPTrunc)
TRITON_IR_DECLARE_CAST_INST_SIMPL(fp_ext_inst, INST_CAST_FP_EXT, cast_op_t::FPExt)
TRITON_IR_DECLARE_CAST_INST_SIMPL(ui_to_fp_inst, INST_CAST_UI_TO_FP, cast_op_t::UIToFP)
TRITON_IR_DECLARE_CAST_INST_SIMPL(si_to_fp_inst, INST_CAST_SI_TO_FP, cast_op_t::SIToFP)
TRITON_IR_DECLARE_CAST_INST_SIMPL(fp_to_ui_inst, INST_CAST_FP_TO_UI, cast_op_t::FPToUI)
TRITON_IR_DECLARE_CAST_INST_SIMPL(fp_to_si_inst, INST_CAST_FP_TO_SI, cast_op_t::FPToSI)
TRITON_IR_DECLARE_CAST_INST_SIMPL(ptr_to_int_inst, INST_CAST_PTR_TO_INT, cast_op_t::PtrToInt)
TRITON_IR_DECLARE_CAST_INST_SIMPL(int_to_ptr_inst, INST_CAST_INT_TO_PTR, cast_op_t::IntToPtr)
TRITON_IR_DECLARE_CAST_INST_SIMPL(bit_cast_inst, INST_CAST_BIT_CAST, cast_op_t::BitCast)
TRITON_IR_DECLARE_CAST_INST_SIMPL(addr_space_cast_inst, INST_CAST_ADDR_SPACE_CAST, cast_op_t::AddrSpaceCast)

//===----------------------------------------------------------------------===//
//                               terminator_inst classes
//===----------------------------------------------------------------------===//

class terminator_inst: public instruction{
  using instruction::instruction;
};

// return instruction
class return_inst: public terminator_inst {
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

  _TRITON_DEFINE_CLONE(return_inst)
  _TRITON_DEFINE_ACCEPT(return_inst)
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
  _TRITON_DEFINE_CLONE(cond_branch_inst)
  _TRITON_DEFINE_ACCEPT(cond_branch_inst)
};

// unconditional branch
class uncond_branch_inst: public branch_inst {
private:
  friend class branch_inst;
  uncond_branch_inst(basic_block *dst, instruction *next);

public:
  basic_block *get_dest()  { return (basic_block*)get_operand(0); }
  _TRITON_DEFINE_CLONE(uncond_branch_inst)
  _TRITON_DEFINE_ACCEPT(uncond_branch_inst)
};


//===----------------------------------------------------------------------===//
//                               getelementptr_inst classes
//===----------------------------------------------------------------------===//

class getelementptr_inst: public instruction {
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
  _TRITON_DEFINE_CLONE(getelementptr_inst)
  _TRITON_DEFINE_ACCEPT(getelementptr_inst)

private:
  type *source_elt_ty;
  type *res_elt_ty;
};

//===----------------------------------------------------------------------===//
//                          load_inst/store_inst classes
//===----------------------------------------------------------------------===//

class io_inst: public instruction {
protected:
  io_inst(type *ty, value_id_t id, unsigned num_ops,
          const std::string &name = "", instruction *next = nullptr);

public:
  // accessors
  value *get_pointer_operand() { return get_operand(0); }
};

// load
class load_inst: public io_inst {
public:
  enum CACHE_MODIFIER : uint32_t {
    NONE=0,
    CA,
    CG,
  }; 

  CACHE_MODIFIER get_cache_modifier() const { return cache_; }
protected:
  load_inst(value *ptr, value_id_t id, unsigned num_ops, CACHE_MODIFIER cache,
          const std::string &name = "", instruction *next = nullptr);
  std::string get_cache_modifier_repr() const {
    if (cache_ == CA) return ".ca";
    if (cache_ == CG) return ".cg";
    return ""; 
  }
  CACHE_MODIFIER cache_;

private:
  static type *get_pointee_type(type *ty);

};

// unmasked load
class unmasked_load_inst: public load_inst {
private:
  std::string repr_impl() const { return "unmasked_load" + get_cache_modifier_repr(); }
  unmasked_load_inst(value *ptr, load_inst::CACHE_MODIFIER cache, const std::string &name, instruction *next);

public:
  static unmasked_load_inst* create(value *ptr,
                                    CACHE_MODIFIER cache,
                                    const std::string &name = "",
                                    instruction *next = nullptr);
  _TRITON_DEFINE_CLONE(unmasked_load_inst)
  _TRITON_DEFINE_ACCEPT(unmasked_load_inst)
};

// masked load
class masked_load_inst: public load_inst {
private:
  std::string repr_impl() const { return "masked_load" + get_cache_modifier_repr(); }
  masked_load_inst(value *ptr, value *mask, value *false_value, load_inst::CACHE_MODIFIER cache,
                   const std::string &name, instruction *next);

public:
  // accessors
  value *get_mask_operand() { return get_operand(1); }
  value *get_false_value_operand() { return get_operand(2); }
  // factory method
  static masked_load_inst* create(value *ptr, value *mask, value *false_value,
                                  CACHE_MODIFIER cache,
                                  const std::string &name = "",
                                  instruction *next = nullptr);
  _TRITON_DEFINE_CLONE(masked_load_inst)
  _TRITON_DEFINE_ACCEPT(masked_load_inst)
};

// masked load async
class masked_load_async_inst: public load_inst {
private:
  std::string repr_impl() const { return "masked_load_async_async" + get_cache_modifier_repr(); }
  masked_load_async_inst(value *ptr, value *mask, value *false_value, load_inst::CACHE_MODIFIER cache,
                   const std::string &name, instruction *next);

public:
  // accessors
  value *get_mask_operand() { return get_operand(1); }
  value *get_false_value_operand() { return get_operand(2); }
  // factory method
  static masked_load_async_inst* create(value *ptr, value *mask, value *false_value,
                                  load_inst::CACHE_MODIFIER cache,
                                  const std::string &name = "",
                                  instruction *next = nullptr);
  _TRITON_DEFINE_CLONE(masked_load_async_inst)
  _TRITON_DEFINE_ACCEPT(masked_load_async_inst)
};



// store
class store_inst: public io_inst {
protected:
  store_inst(value *ptr, value_id_t id, unsigned num_ops,
            const std::string &name = "", instruction *next = nullptr);

public:
  value *get_value_operand() { return get_operand(1); }
};

// unmasked_store
class unmasked_store_inst: public store_inst{
private:
  std::string repr_impl() const { return "unmasked_store"; }
  unmasked_store_inst(value *ptr, value *v, const std::string &name, instruction *next);

public:
  // factory method
  static unmasked_store_inst* create(value* ptr, value *v,
                                    const std::string &name = "",
                                    instruction *next = nullptr);
  _TRITON_DEFINE_CLONE(unmasked_store_inst)
  _TRITON_DEFINE_ACCEPT(unmasked_store_inst)
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
  _TRITON_DEFINE_CLONE(masked_store_inst)
  _TRITON_DEFINE_ACCEPT(masked_store_inst)
};

//===----------------------------------------------------------------------===//
//                               retile_inst classes
//===----------------------------------------------------------------------===//

// cat

class cat_inst: public instruction {
private:
  std::string repr_impl() const { return "cat"; }
  cat_inst(value *x, value *y, const std::string &name, instruction *next);

public:
  static instruction* create(value *lhs, value *rhs,
                             const std::string &name = "",
                             instruction *next = nullptr);
  _TRITON_DEFINE_CLONE(cat_inst)
  _TRITON_DEFINE_ACCEPT(cat_inst)
};

// retile

class retile_inst: public unary_inst {
protected:
  retile_inst(value *arg, value_id_t id, const type::block_shapes_t &shapes, const std::string &name, instruction *next);
};

// reshape

class reshape_inst: public retile_inst {
private:
  using retile_inst::retile_inst;
  std::string repr_impl() const { return "reshape"; }

public:
  static instruction* create(value *arg, const type::block_shapes_t &shape_suffix,
                      const std::string &name = "", instruction *next = nullptr);
  _TRITON_DEFINE_CLONE(reshape_inst)
  _TRITON_DEFINE_ACCEPT(reshape_inst)
};

// splat

class splat_inst: public retile_inst {
private:
  using retile_inst::retile_inst;
  std::string repr_impl() const { return "splat"; }

public:
  static instruction* create(value *arg, const type::block_shapes_t &shape_suffix,
                      const std::string &name = "", instruction *next = nullptr);
  _TRITON_DEFINE_CLONE(splat_inst)
  _TRITON_DEFINE_ACCEPT(splat_inst)
};

// broadcast

class broadcast_inst: public retile_inst {
private:
  using retile_inst::retile_inst;
  std::string repr_impl() const { return "broadcast"; }

public:
  static instruction* create(value *arg, const type::block_shapes_t &shape_suffix,
                      const std::string &name = "", instruction *next = nullptr);
  _TRITON_DEFINE_CLONE(broadcast_inst)
  _TRITON_DEFINE_ACCEPT(broadcast_inst)
};


// downcast

class downcast_inst: public unary_inst {
private:
  using unary_inst::unary_inst;
  std::string repr_impl() const { return "downcast"; }

public:
  static instruction* create(value *arg, const std::string &name = "", instruction *next = nullptr);
  _TRITON_DEFINE_CLONE(downcast_inst)
  _TRITON_DEFINE_ACCEPT(downcast_inst)
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
  _TRITON_DEFINE_CLONE(get_program_id_inst)
  _TRITON_DEFINE_ACCEPT(get_program_id_inst)

private:
  unsigned axis_;
};

class get_num_programs_inst: public builtin_inst {
private:
  get_num_programs_inst(type *ty, unsigned axis, const std::string &name, instruction *next);
  std::string repr_impl() const { return "get_num_programs(" + std::to_string(axis_) + ")"; }

public:
  static instruction* create(context &ctx, unsigned axis, const std::string &name = "", instruction *next = nullptr);
  unsigned get_axis() const { return axis_; }
  _TRITON_DEFINE_CLONE(get_num_programs_inst)
  _TRITON_DEFINE_ACCEPT(get_num_programs_inst)

private:
  unsigned axis_;
};


class atomic_inst: public io_inst {
public:
  using io_inst::io_inst;
};

class atomic_rmw_inst: public atomic_inst {
private:
  atomic_rmw_inst(atomic_rmw_op_t op, value *ptr, value *val, value *msk, const std::string &name = "", instruction *next = nullptr);
  std::string repr_impl() const { return "atomic_rmw"; }
  _TRITON_DEFINE_CLONE(atomic_rmw_inst)
  _TRITON_DEFINE_ACCEPT(atomic_rmw_inst)

public:
  static instruction* create(atomic_rmw_op_t op, value *ptr, value *val, value *msk, const std::string &name = "", instruction *next = nullptr);
  atomic_rmw_op_t get_op() { return op_; }

private:
  atomic_rmw_op_t op_;
};

class atomic_cas_inst: public atomic_inst {
private:
  atomic_cas_inst(value *ptr, value *cmp, value *val, const std::string &name, instruction *next);
  std::string repr_impl() const { return "atomic_cas"; }
  _TRITON_DEFINE_CLONE(atomic_cas_inst)
  _TRITON_DEFINE_ACCEPT(atomic_cas_inst)

public:
  static instruction* create(value *ptr, value *cmp, value *val, const std::string &name = "", instruction *next = nullptr);
};

class umulhi_inst: public builtin_inst {
private:
  umulhi_inst(value *lhs, value *rhs, const std::string &name = "", instruction *next = nullptr);
  std::string repr_impl() const { return "umulhi"; }
  _TRITON_DEFINE_CLONE(umulhi_inst)
  _TRITON_DEFINE_ACCEPT(umulhi_inst)

public:
  static instruction* create(value *lhs, value *rhs, const std::string &name = "", instruction *next = nullptr);
};

class exp_inst: public builtin_inst {
private:
  exp_inst(value *val, const std::string &name = "", instruction *next = nullptr);
  std::string repr_impl() const { return "exp"; }
  _TRITON_DEFINE_CLONE(exp_inst)
  _TRITON_DEFINE_ACCEPT(exp_inst)

public:
  static instruction* create(value *val, const std::string &name = "", instruction *next = nullptr);
};

class cos_inst: public builtin_inst {
private:
  cos_inst(value *val, const std::string &name = "", instruction *next = nullptr);
  std::string repr_impl() const { return "cos"; }
  _TRITON_DEFINE_CLONE(cos_inst)
  _TRITON_DEFINE_ACCEPT(cos_inst)

public:
  static instruction* create(value *val, const std::string &name = "", instruction *next = nullptr);
};

class sin_inst: public builtin_inst {
private:
  sin_inst(value *val, const std::string &name = "", instruction *next = nullptr);
  std::string repr_impl() const { return "sin"; }
  _TRITON_DEFINE_CLONE(sin_inst)
  _TRITON_DEFINE_ACCEPT(sin_inst)

public:
  static instruction* create(value *val, const std::string &name = "", instruction *next = nullptr);
};

class log_inst: public builtin_inst {
private:
  log_inst(value *val, const std::string &name = "", instruction *next = nullptr);
  std::string repr_impl() const { return "log"; }
  _TRITON_DEFINE_CLONE(log_inst)
  _TRITON_DEFINE_ACCEPT(log_inst)

public:
  static instruction* create(value *val, const std::string &name = "", instruction *next = nullptr);
};


class dot_inst: public builtin_inst {
public:
  enum TransT { NoTrans, Trans };

private:
  dot_inst(value *A, value *B, value *C, TransT AT, TransT BT, const std::string &name, instruction *next);
  std::string repr_impl() const { return "dot"; }

  bool is_prefetched_ = false;
public:
  bool is_prefetched() const { return is_prefetched_; }
  void set_prefetched(bool is_prefetched) { is_prefetched_ = is_prefetched; }

public:
  static instruction *create(value *A, value *B, value *C, bool AT, bool BT, const std::string &name = "", instruction *next = nullptr);
  static instruction* create_nn(value *A, value *B, value *C, const std::string &name = "", instruction *next = nullptr);
  static instruction* create_nt(value *A, value *B, value *C, const std::string &name = "", instruction *next = nullptr);
  static instruction* create_tn(value *A, value *B, value *C, const std::string &name = "", instruction *next = nullptr);
  static instruction* create_tt(value *A, value *B, value *C, const std::string &name = "", instruction *next = nullptr);
  _TRITON_DEFINE_CLONE(dot_inst)
  _TRITON_DEFINE_ACCEPT(dot_inst)
};

//class outer_inst: public builtin_inst {
//private:
//  outer_inst(value *A, value *B, value *C, const std::string &name, instruction *next);
//public:
//  static instruction* create(value *A, value *B, value *C, const std::string &name = "", instruction *next = nullptr);
//};

class trans_inst: public builtin_inst {
public:
  ir::type* get_res_ty(ir::type* in, std::vector<int> perm);
  std::vector<int> init_perm(ir::type* ty, const std::vector<int>& perm);

private:
  trans_inst(value *arg, const std::vector<int>& perm, const std::string& name, instruction* next);
  std::string repr_impl() const { return "trans"; }

public:
  static instruction* create(value *arg, const std::vector<int> &perm = {}, const std::string &name = "", instruction *next = nullptr);
  const std::vector<int> get_perm() const;
  _TRITON_DEFINE_CLONE(trans_inst)
  _TRITON_DEFINE_ACCEPT(trans_inst)

private:
  std::vector<int> perm_;
};

class sqrt_inst: public builtin_inst {
private:
  sqrt_inst(value *arg, const std::string& name, instruction* next);
  std::string repr_impl() const { return "sqrt"; }
public:
  static instruction* create(value *arg, const std::string &name = "", instruction *next = nullptr);
  _TRITON_DEFINE_CLONE(sqrt_inst)
  _TRITON_DEFINE_ACCEPT(sqrt_inst)
};

class reduce_inst: public builtin_inst {
public:
  enum op_t{
    ADD, SUB, MAX, MIN,
    FADD, FSUB, FMAX, FMIN
  };

private:
  static type* get_res_type(value *arg, unsigned axis);
  static std::string to_str(op_t op);

private:
  reduce_inst(value* arg, op_t op, unsigned axis, const std::string& name, instruction* next);
  std::string repr_impl() const { return "reduce"; }
  _TRITON_DEFINE_CLONE(reduce_inst)
  _TRITON_DEFINE_ACCEPT(reduce_inst)

public:
  static instruction* create(value *arg, op_t op, unsigned axis, const std::string &name = "", instruction *next = nullptr);
  unsigned get_axis() const { return axis_; }
  op_t get_op() const { return op_; }

private:
  unsigned axis_;
  op_t op_;
};

class select_inst: public builtin_inst {
private:
  select_inst(value *pred, value *if_value, value *else_value, const std::string& name, instruction* next);
  std::string repr_impl() const { return "select"; }
  _TRITON_DEFINE_CLONE(select_inst)
  _TRITON_DEFINE_ACCEPT(select_inst)

public:
  static instruction* create(value *pred, value *if_value, value *else_value, const std::string &name = "", instruction *next = nullptr);
  value* get_pred_op() { return get_operand(0); }
  value* get_if_value_op() { return get_operand(1); }
  value* get_else_value_op() { return get_operand(2); }
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
  _TRITON_DEFINE_CLONE(copy_to_shared_inst)
  _TRITON_DEFINE_ACCEPT(copy_to_shared_inst)
};

class copy_from_shared_inst: public unary_inst{
private:
  using unary_inst::unary_inst;
  std::string repr_impl() const { return "copy_from_shared"; }

public:
  static copy_from_shared_inst* create(value *arg, const std::string &name = "",
                                     instruction *next = nullptr);
  _TRITON_DEFINE_CLONE(copy_from_shared_inst)
  _TRITON_DEFINE_ACCEPT(copy_from_shared_inst)
};

class cvt_layout_inst: public unary_inst {
private:
  using unary_inst::unary_inst;
  std::string repr_impl() const { return "cvt_layout_inst"; }

public:
  static cvt_layout_inst* create(value *arg, const std::string &name = "", instruction *next = nullptr);
  _TRITON_DEFINE_CLONE(cvt_layout_inst)
  _TRITON_DEFINE_ACCEPT(cvt_layout_inst)
};

class barrier_inst: public instruction{
private:
  barrier_inst(context &ctx, const std::string &name, instruction *next);
  std::string repr_impl() const { return "barrier"; }
  _TRITON_DEFINE_CLONE(barrier_inst)
  _TRITON_DEFINE_ACCEPT(barrier_inst)

public:
  static barrier_inst* create(context &ctx, const std::string &name = "",
                                            instruction *next = nullptr);
};

class async_wait_inst: public instruction{
private:
  async_wait_inst(context &ctx, int N, const std::string &name, instruction *next);
  std::string repr_impl() const { return "async_wait_group " + std::to_string(N_) ; }
  _TRITON_DEFINE_CLONE(async_wait_inst)
  _TRITON_DEFINE_ACCEPT(async_wait_inst)

public:
  static async_wait_inst* create(context &ctx, int N,
                                 const std::string &name = "", instruction *next = nullptr);
  int get_N() { return N_; }
  void set_N(int n) { N_ = n; }

private:
  int N_;
};

class prefetch_s_inst : public instruction {
  std::string repr_impl() const { return "prefetch_s"; }
  _TRITON_DEFINE_CLONE(prefetch_s_inst)
  _TRITON_DEFINE_ACCEPT(prefetch_s_inst)
  
  /// inc_: 0->first, 1->latch
  int inc_ = 0;
public:
  prefetch_s_inst(context &ctx, value *arg, int inc, const std::string &name, instruction *next) 
    : instruction(type::get_void_ty(ctx), INST_PREFETCH_S, 1, name, next), inc_(inc) {
    set_operand(0, arg);
  }
  int get_inc() const { return inc_; }
  static prefetch_s_inst *create(context &ctx, value *arg, int inc, const std::string &name = "",
   instruction *next=nullptr);
};

/* constant range */
class make_range: public instruction{
  make_range(type *ty, constant_int* first, constant_int* last);
  std::string repr_impl() const { return "make_range[" + first_->repr() + " : " + last_->repr() + "]"; }
  _TRITON_DEFINE_CLONE(make_range)
  _TRITON_DEFINE_ACCEPT(make_range)

public:
  static make_range *create(constant_int *first, constant_int *last);
  const constant_int* get_first() const;
  const constant_int* get_last() const;

private:
  constant_int* first_;
  constant_int* last_;
};


}
}

#endif
