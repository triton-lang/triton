#include "ir/basic_block.h"
#include "ir/instructions.h"
#include "ir/constant.h"

namespace tdl{
namespace ir{

//===----------------------------------------------------------------------===//
//                               instruction classes
//===----------------------------------------------------------------------===//

instruction::instruction(type *ty, unsigned num_ops, instruction *next)
    : user(ty, num_ops) {
  if(next){
    basic_block *block = next->get_parent();
    assert(block && "Next instruction is not in a basic block!");
    auto it = std::find(block->begin(), block->end(), next);
    block->get_inst_list().insert(it, next);
  }
}

//===----------------------------------------------------------------------===//
//                               phi_node classes
//===----------------------------------------------------------------------===//

// Add incoming
void phi_node::add_incoming(value *x, basic_block *bb){

}

// Factory methods
phi_node* phi_node::create(type *ty, unsigned num_reserved){
  return new phi_node(ty, num_reserved);
}


//===----------------------------------------------------------------------===//
//                               binary_operator classes
//===----------------------------------------------------------------------===//

binary_operator::binary_operator(op_t op, value *lhs, value *rhs, type *ty, const std::string &name, instruction *next)
    : instruction(ty, 2, next), op_(op){
  set_operand(0, lhs);
  set_operand(1, rhs);
}

binary_operator *binary_operator::create(op_t op, value *lhs, value *rhs, const std::string &name, instruction *next){
  assert(lhs->get_type() == rhs->get_type() &&
         "Cannot create binary operator with two operands of differing type!");
  return new binary_operator(op, lhs, rhs, lhs->get_type(), name, next);
}

binary_operator *binary_operator::create_fneg(value *arg, const std::string &name, instruction *next){
  assert(arg->get_type()->is_floating_point_ty());
  value *zero = constant_fp::get_zero_value_for_negation(arg->get_type());
  return binary_operator::create(llvm::Instruction::FSub, zero, arg, name, next);
}

binary_operator *binary_operator::create_neg(value *arg, const std::string &name, instruction *next){
  assert(arg->get_type()->is_integer_ty());
  value *zero = constant_fp::get_zero_value_for_negation(arg->get_type());
  return binary_operator::create(llvm::Instruction::Sub, zero, arg, name, next);
}

binary_operator *binary_operator::create_not(value *arg, const std::string &name, instruction *next){
  assert(arg->get_type()->is_integer_ty());
  constant *mask = constant::get_all_ones_value(arg->get_type());
  return binary_operator::create(llvm::Instruction::Xor, arg, mask, name, next);
}

//===----------------------------------------------------------------------===//
//                               cmp_inst classes
//===----------------------------------------------------------------------===//

bool cmp_inst::is_fp_predicate(pred_t pred) {
  return pred >= pcmp::FIRST_FCMP_PREDICATE && pred <= pcmp::LAST_FCMP_PREDICATE;
}

bool cmp_inst::is_int_predicate(pred_t pred) {
  return pred >= pcmp::FIRST_ICMP_PREDICATE && pred <= pcmp::LAST_ICMP_PREDICATE;
}

// icmp_inst

icmp_inst* icmp_inst::create(pred_t pred, value *lhs, value *rhs, const std::string &name, instruction *next){
  assert(is_int_predicate(pred));
  return new icmp_inst(pred, lhs, rhs, name, next);
}

// fcmp_inst

fcmp_inst* fcmp_inst::create(pred_t pred, value *lhs, value *rhs, const std::string &name, instruction *next){
  assert(is_fp_predicate(pred));
  return new fcmp_inst(pred, lhs, rhs, name, next);
}

}
}
