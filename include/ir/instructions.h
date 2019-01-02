#ifndef TDL_INCLUDE_IR_INSTRUCTIONS_H
#define TDL_INCLUDE_IR_INSTRUCTIONS_H

#include <vector>
#include "value.h"
#include "llvm/IR/Instructions.h"

namespace tdl{
namespace ir{

class basic_block;

//===----------------------------------------------------------------------===//
//                               instruction classes
//===----------------------------------------------------------------------===//

class instruction: public user{
protected:
  // constructors
  instruction(type *ty, unsigned num_ops, instruction *next = nullptr);

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
  phi_node(type *ty, unsigned num_reserved);

public:
  void add_incoming(value *x, basic_block *bb);

  // Factory methods
  static phi_node* create(type *ty, unsigned num_reserved);

private:
  unsigned num_reserved_;
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

private:
  type* make_cmp_result_type(type *ty);

protected:
  cmp_inst(pred_t pred, value *lhs, value *rhs, const std::string &name, instruction *next);

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
//                               cast_inst classes
//===----------------------------------------------------------------------===//

class cast_inst: public instruction{
public:
  typedef llvm::CastInst::CastOps op_t;

protected:
  // Constructors
  cast_inst(op_t op, value *arg, type *ty, const std::string &name, instruction *next);

public:
  // Factory methods
  static cast_inst *create(op_t op, value *arg, type *ty,
                           const std::string &name = "", instruction *next = nullptr);
  static cast_inst *create_integer_cast(value *arg, type *ty, bool is_signed,
                           const std::string &name = "", instruction *next = nullptr);


private:
  op_t op_;
};

//===----------------------------------------------------------------------===//
//                               terminator_inst classes
//===----------------------------------------------------------------------===//

class terminator_inst: public instruction{
public:
};

class return_inst: public instruction{

};

//===----------------------------------------------------------------------===//
//                               branch_inst classes
//===----------------------------------------------------------------------===//

class branch_inst: public instruction{
public:
  static branch_inst* create(basic_block *dest,
                             const std::string &name = "", instruction *next = nullptr);
  static branch_inst* create(value *cond, basic_block *if_dest, basic_block *else_dest,
                             const std::string &name = "", instruction *next = nullptr);
};

//===----------------------------------------------------------------------===//
//                               getelementptr_inst classes
//===----------------------------------------------------------------------===//

class getelementptr_inst: public instruction{
public:
  static getelementptr_inst* create(value *ptr, const std::vector<value*> &idx,
                                    const std::string &name = "", instruction *next = nullptr);
};


}
}

#endif
