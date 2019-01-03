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
  const basic_block *get_parent() const { return parent_;}
  basic_block *get_parent()             { return parent_; }

private:
  basic_block *parent_;
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

  // Factory methods
  static binary_operator *create(op_t op, value *lhs, value *rhs,
                                 const std::string &name = "", instruction *next = nullptr);
  static binary_operator *create_fneg(value *arg, const std::string &name = "", instruction *next = nullptr);
  static binary_operator *create_neg(value *arg, const std::string &name = "", instruction *next = nullptr);
  static binary_operator *create_not(value *arg, const std::string &name = "", instruction *next = nullptr);

public:
  op_t op_;
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

  static type* make_cmp_result_type(type *ty);

  static bool is_fp_predicate(pred_t pred);
  static bool is_int_predicate(pred_t pred);

public:


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
  using unary_inst::unary_inst;
  using ic = llvm::Instruction::CastOps;

public:
  typedef llvm::CastInst::CastOps op_t;

private:
  bool is_valid(op_t op, value *arg, type *ty);

public:
  // Factory methods
  static cast_inst *create(op_t op, value *arg, type *ty,
                           const std::string &name = "", instruction *next = nullptr);
  static cast_inst *create_integer_cast(value *arg, type *ty, bool is_signed,
                           const std::string &name = "", instruction *next = nullptr);

private:
  op_t op_;
};

#define TDL_IR_DECLARE_CAST_INST_SIMPLE(name) \
  class name : public cast_inst{ \
    friend class cast_inst; \
    using cast_inst::cast_inst; \
  };

TDL_IR_DECLARE_CAST_INST_SIMPLE(trunc_inst)
TDL_IR_DECLARE_CAST_INST_SIMPLE(z_ext_inst)
TDL_IR_DECLARE_CAST_INST_SIMPLE(s_ext_inst)
TDL_IR_DECLARE_CAST_INST_SIMPLE(fp_trunc_inst)
TDL_IR_DECLARE_CAST_INST_SIMPLE(fp_ext_inst)
TDL_IR_DECLARE_CAST_INST_SIMPLE(ui_to_fp_inst)
TDL_IR_DECLARE_CAST_INST_SIMPLE(si_to_fp_inst)
TDL_IR_DECLARE_CAST_INST_SIMPLE(fp_to_ui_inst)
TDL_IR_DECLARE_CAST_INST_SIMPLE(fp_to_si_inst)
TDL_IR_DECLARE_CAST_INST_SIMPLE(ptr_to_int_inst)
TDL_IR_DECLARE_CAST_INST_SIMPLE(int_to_ptr_inst)
TDL_IR_DECLARE_CAST_INST_SIMPLE(bit_cast_inst)
TDL_IR_DECLARE_CAST_INST_SIMPLE(addr_space_cast_inst)

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

// conditional/unconditional branch instruction

class branch_inst: public terminator_inst{
  branch_inst(basic_block *dst, instruction *next);
  branch_inst(basic_block *if_dst, basic_block *else_dst, value *cond, instruction *next);

public:

  // factory methods
  static branch_inst* create(basic_block *dest,
                             instruction *next = nullptr);
  static branch_inst* create(value *cond, basic_block *if_dest, basic_block *else_dest,
                             instruction *next = nullptr);
};

//===----------------------------------------------------------------------===//
//                               getelementptr_inst classes
//===----------------------------------------------------------------------===//

class getelementptr_inst: public instruction{
  getelementptr_inst(type *pointee_ty, value *ptr, const std::vector<value*> &idx, const std::string &name, instruction *next);

private:
  static type *get_return_type(type *ty, value *ptr, const std::vector<value*> &idx);
  static type *get_indexed_type_impl(type *ty, const std::vector<value *> &idx);
  static type *get_indexed_type(type *ty, const std::vector<value*> &idx);

public:
  static getelementptr_inst* create(type *pointee_ty, value *ptr, const std::vector<value*> &idx,
                                    const std::string &name = "", instruction *next = nullptr);

private:
  type *source_elt_ty;
  type *res_elt_ty;
};

//===----------------------------------------------------------------------===//
//                               retile_inst classes
//===----------------------------------------------------------------------===//

// retile

class retile_inst: public instruction {
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



}
}

#endif
