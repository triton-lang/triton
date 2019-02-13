#ifndef TDL_INCLUDE_IR_INSTRUCTIONS_H
#define TDL_INCLUDE_IR_INSTRUCTIONS_H

#include <vector>
#include "value.h"
#include "llvm/IR/Instructions.h"

namespace tdl{
namespace ir{

class basic_block;
class context;

//===----------------------------------------------------------------------===//
//                               instruction classes
//===----------------------------------------------------------------------===//

class instruction: public user{
protected:
  // constructors
  instruction(type *ty, unsigned num_ops, const std::string &name = "", instruction *next = nullptr);

public:
  // parent
  void set_parent(basic_block *block)   { parent_ = block; }
  const basic_block *get_parent() const { return parent_;  }
  basic_block *get_parent()             { return parent_;  }
  void erase_from_parent();
  // helpers
  bool has_tile_result_or_op();

private:
  basic_block *parent_;
  value *pred_;
};

//===----------------------------------------------------------------------===//
//                               phi_node classes
//===----------------------------------------------------------------------===//

class phi_node: public instruction{
private:
  phi_node(type *ty, unsigned num_reserved, const std::string &name, instruction *next);

public:
  void set_incoming_value(unsigned i, value *v);
  void set_incoming_block(unsigned i, basic_block *block);
  value *get_incoming_value(unsigned i) { return get_operand(i); }
  basic_block *get_incoming_block(unsigned i) { return blocks_[i]; }
  unsigned get_num_incoming() { return get_num_operands(); }
  void add_incoming(value *v, basic_block *block);

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
  typedef llvm::BinaryOperator::BinaryOps op_t;

protected:
  // Constructors
  binary_operator(op_t op, value *lhs, value *rhs, type *ty, const std::string &name, instruction *next);

public:
  // Get operand
  op_t get_op() const { return op_; }

  // Bool
  bool is_terminator()  const;
  bool is_binary_op()   const;
  bool is_int_div_rem() const;
  bool is_shift()       const;
  bool is_cast()        const;

  // Wraps
  void set_has_no_unsigned_wrap(bool b = true) { has_no_unsigned_wrap_ = b; }
  void set_has_no_signed_wrap(bool b = true)   { has_no_signed_wrap_ = b; }

  // Factory methods
  static binary_operator *create(op_t op, value *lhs, value *rhs,
                                 const std::string &name = "", instruction *next = nullptr);
  static binary_operator *create_fneg(value *arg, const std::string &name = "", instruction *next = nullptr);
  static binary_operator *create_neg(value *arg, const std::string &name = "", instruction *next = nullptr);
  static binary_operator *create_not(value *arg, const std::string &name = "", instruction *next = nullptr);

public:
  op_t op_;
  bool has_no_unsigned_wrap_;
  bool has_no_signed_wrap_;
};


//===----------------------------------------------------------------------===//
//                               cmp_inst classes
//===----------------------------------------------------------------------===//

class cmp_inst: public instruction{
public:
  typedef llvm::CmpInst::Predicate pred_t;
  using pcmp = llvm::CmpInst;

protected:
  cmp_inst(type *ty, pred_t pred, value *lhs, value *rhs, const std::string &name, instruction *next);
  static bool is_fp_predicate(pred_t pred);
  static bool is_int_predicate(pred_t pred);
  static type* make_cmp_result_type(type *ty);

public:
  pred_t get_pred() const { return pred_; }

private:
  pred_t pred_;
};

class icmp_inst: public cmp_inst{
  using cmp_inst::cmp_inst;

public:
  static icmp_inst* create(pred_t pred, value *lhs, value *rhs,
                    const std::string &name = "", instruction *next = nullptr);
};

class fcmp_inst: public cmp_inst{
  using cmp_inst::cmp_inst;

public:
  static fcmp_inst* create(pred_t pred, value *lhs, value *rhs,
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
  using ic = llvm::Instruction::CastOps;

public:
  typedef llvm::CastInst::CastOps op_t;

protected:
  cast_inst(type *ty, value *v, const std::string &name, instruction *next, op_t op)
    : unary_inst(ty, v, name, next), op_(op) { }

private:
  static bool is_valid(op_t op, value *arg, type *ty);

public:
  // accessors
  op_t get_op() const { return op_; }

  // factory methods
  static cast_inst *create(op_t op, value *arg, type *ty,
                           const std::string &name = "", instruction *next = nullptr);
  static cast_inst *create_integer_cast(value *arg, type *ty, bool is_signed,
                           const std::string &name = "", instruction *next = nullptr);

private:
  op_t op_;
};

#define TDL_IR_DECLARE_CAST_INST_SIMPLE(name, op) \
class name : public cast_inst{ \
  friend class cast_inst; \
  name(type *ty, value *v, const std::string &name, instruction *next) \
    : cast_inst(ty, v, name, next, op){ } \
};

TDL_IR_DECLARE_CAST_INST_SIMPLE(trunc_inst, llvm::Instruction::CastOps::Trunc)
TDL_IR_DECLARE_CAST_INST_SIMPLE(z_ext_inst, llvm::Instruction::CastOps::ZExt)
TDL_IR_DECLARE_CAST_INST_SIMPLE(s_ext_inst, llvm::Instruction::CastOps::SExt)
TDL_IR_DECLARE_CAST_INST_SIMPLE(fp_trunc_inst, llvm::Instruction::CastOps::FPTrunc)
TDL_IR_DECLARE_CAST_INST_SIMPLE(fp_ext_inst, llvm::Instruction::CastOps::FPExt)
TDL_IR_DECLARE_CAST_INST_SIMPLE(ui_to_fp_inst, llvm::Instruction::CastOps::UIToFP)
TDL_IR_DECLARE_CAST_INST_SIMPLE(si_to_fp_inst, llvm::Instruction::CastOps::SIToFP)
TDL_IR_DECLARE_CAST_INST_SIMPLE(fp_to_ui_inst, llvm::Instruction::CastOps::FPToUI)
TDL_IR_DECLARE_CAST_INST_SIMPLE(fp_to_si_inst, llvm::Instruction::CastOps::FPToSI)
TDL_IR_DECLARE_CAST_INST_SIMPLE(ptr_to_int_inst, llvm::Instruction::CastOps::PtrToInt)
TDL_IR_DECLARE_CAST_INST_SIMPLE(int_to_ptr_inst, llvm::Instruction::CastOps::IntToPtr)
TDL_IR_DECLARE_CAST_INST_SIMPLE(bit_cast_inst, llvm::Instruction::CastOps::BitCast)
TDL_IR_DECLARE_CAST_INST_SIMPLE(addr_space_cast_inst, llvm::Instruction::CastOps::AddrSpaceCast)

//===----------------------------------------------------------------------===//
//                               terminator_inst classes
//===----------------------------------------------------------------------===//

class terminator_inst: public instruction{
  using instruction::instruction;
};

// return instruction
class return_inst: public terminator_inst{
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
  cond_branch_inst(basic_block *if_dst, basic_block *else_dst, value *cond, instruction *next);
  friend class branch_inst;

public:
  basic_block *get_true_dest()  { return (basic_block*)get_operand(0); }
  basic_block *get_false_dest() { return (basic_block*)get_operand(1); }
  value *get_cond()             { return get_operand(2); }
};

// unconditional branch
class uncond_branch_inst: public branch_inst {
  friend class branch_inst;
  uncond_branch_inst(basic_block *dst, instruction *next);

public:
  basic_block *get_dest()  { return (basic_block*)get_operand(0); }
};
//===----------------------------------------------------------------------===//
//                               getelementptr_inst classes
//===----------------------------------------------------------------------===//

class getelementptr_inst: public instruction{
private:
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

class load_inst: public unary_inst{
private:
  load_inst(value *ptr, const std::string &name, instruction *next);

private:
  static type *get_pointee_type(type *ty);

public:
  // accessors
  value *get_pointer_operand() { return get_operand(0); }
  // factory method
  static load_inst* create(value *ptr, const std::string &name = "",
                           instruction *next = nullptr);
};

class store_inst: public instruction{
private:
  store_inst(value *ptr, value *v, const std::string &name, instruction *next);

public:
  value *get_pointer_operand() { return get_operand(0); }
  value *get_value_operand() { return get_operand(1); }
  // factory method
  static store_inst* create(value* ptr, value *v, const std::string &name = "",
                            instruction *next = nullptr);
};

//===----------------------------------------------------------------------===//
//                               retile_inst classes
//===----------------------------------------------------------------------===//

// retile

class retile_inst: public unary_inst {
protected:
  retile_inst(value *arg, const std::vector<unsigned> &shapes, const std::string &name, instruction *next);
};

// reshape

class reshape_inst: public retile_inst {
  using retile_inst::retile_inst;

public:
  static instruction* create(value *arg, const std::vector<unsigned> &shapes,
                      const std::string &name = "", instruction *next = nullptr);
};

// splat

class splat_inst: public retile_inst {
  using retile_inst::retile_inst;

public:
  static instruction* create(value *arg, const std::vector<unsigned> &shapes,
                      const std::string &name = "", instruction *next = nullptr);
};

// broadcast

class broadcast_inst: public retile_inst {
  using retile_inst::retile_inst;

public:
  static instruction* create(value *arg, const std::vector<unsigned> &shapes,
                      const std::string &name = "", instruction *next = nullptr);
};


//===----------------------------------------------------------------------===//
//                               builtin_inst classes
//===----------------------------------------------------------------------===//

class builtin_inst: public instruction{
protected:
  using instruction::instruction;
};

class get_global_range_inst: public builtin_inst {
  get_global_range_inst(type *ty, unsigned axis, const std::string &name, instruction *next);

public:
  static instruction* create(context &ctx, unsigned axis, unsigned size,
                             const std::string &name = "",
                             instruction *next = nullptr);
  unsigned get_axis() const { return axis_; }

private:
  unsigned axis_;
};

class matmul_inst: public builtin_inst {
  matmul_inst(value *A, value *B, value *C, const std::string &name, instruction *next);

public:
  static instruction* create(value *A, value *B, value *C,
                             const std::string &name = "",
                             instruction *next = nullptr);
};


//===----------------------------------------------------------------------===//
//                               intrinsics classes
//===----------------------------------------------------------------------===//

class copy_to_shared_inst: public unary_inst{
  using unary_inst::unary_inst;

public:
  static copy_to_shared_inst* create(value *arg, const std::string &name = "",
                                     instruction *next = nullptr);
};

class barrier_inst: public instruction{
private:
  barrier_inst(context &ctx, const std::string &name, instruction *next);

public:
  static barrier_inst* create(context &ctx, const std::string &name = "",
                                            instruction *next = nullptr);
};

class vectorize_inst: public unary_inst{
  using unary_inst::unary_inst;

public:
  static vectorize_inst* create(value *arg, const std::string &name = "", instruction *next = nullptr);
};

}
}

#endif
