#include <algorithm>
#include <iostream>
#include "triton/ir/context.h"
#include "triton/ir/basic_block.h"
#include "triton/ir/instructions.h"
#include "triton/ir/constant.h"
#include "triton/ir/type.h"

namespace triton{
namespace ir{

//===----------------------------------------------------------------------===//
//                               instruction classes
//===----------------------------------------------------------------------===//

instruction::instruction(type *ty, value_id_t ity, unsigned num_ops,
                         const std::string &name, instruction *next)
    : user(ty, num_ops, name), id_(ity) {
  if(next){
    basic_block *block = next->get_parent();
    assert(block && "Next instruction is not in a basic block!");
    auto it = std::find(block->begin(), block->end(), next);
    block->get_inst_list().insert(it, next);
  }
}

void instruction::erase_from_parent() {
  parent_->erase(this);
  for(ir::value* op: ops())
    op->erase_use(this);
}

bool instruction::has_tile_result_or_op() {
  bool result = get_type()->is_block_ty();
  for(unsigned i = 0; i < get_num_operands(); i++)
    result |= get_operand(i)->get_type()->is_block_ty();
  return result;
}

//===----------------------------------------------------------------------===//
//                               phi_node classes
//===----------------------------------------------------------------------===//

phi_node::phi_node(type *ty, unsigned num_reserved, std::string const &name, instruction *next)
    : instruction(ty, INST_PHI, 0, name, next) {
  blocks_.reserve(num_reserved);
}

value* phi_node::get_value_for_block(basic_block * block) {
  auto it = std::find(blocks_.begin(), blocks_.end(), block);
  size_t n = std::distance(blocks_.begin(), it);
  return get_incoming_value(n);
}

// Set incoming value
void phi_node::set_incoming_value(unsigned i, value *v){
  assert(v && "PHI node got a null value!");
  assert(get_type() == v->get_type() &&
         "All operands to PHI node must be the same type as the PHI node!");
  set_operand(i, v);
}

// Set incoming block
void phi_node::set_incoming_block(unsigned i, basic_block *block){
  assert(block && "PHI node got a null basic block!");
  blocks_[i] = block;
}

// Add incoming
void phi_node::add_incoming(value *v, basic_block *block){
  resize_ops(get_num_operands() + 1);
  blocks_.resize(get_num_operands() + 1);
  set_incoming_value(get_num_operands() - 1, v);
  set_incoming_block(get_num_operands() - 1, block);
}

// Factory methods
phi_node* phi_node::create(type *ty, unsigned num_reserved, const std::string &name, instruction *next){
  return new phi_node(ty, num_reserved, name, next);
}


//===----------------------------------------------------------------------===//
//                               binary_operator classes
//===----------------------------------------------------------------------===//

std::string binary_operator::repr_impl() const {
  switch(op_) {
  case Add  : return "add";
  case FAdd : return "fadd";
  case Sub  : return "sub";
  case FSub : return "fsub";
  case Mul  : return "mul";
  case FMul : return "fmul";
  case UDiv : return "udiv";
  case SDiv : return "sdiv";
  case FDiv : return "fdiv";
  case URem : return "urem";
  case SRem : return "srem";
  case FRem : return "frem";
  case Shl  : return "shl";
  case LShr : return "lshr";
  case AShr : return "ashr";
  case And  : return "and";
  case Or   : return "or";
  case Xor  : return "xor";
  default: throw std::runtime_error("unknown binary operator");
  }
}

bool binary_operator::is_int_div() const {
  return op_ == binary_op_t::UDiv || op_ == binary_op_t::SDiv;
}

bool binary_operator::is_int_rem() const {
  return op_ == binary_op_t::URem || op_ == binary_op_t::SRem;
}

bool binary_operator::is_shl() const {
  return op_ == binary_op_t::Shl;
}

bool binary_operator::is_shr() const {
  return op_ == binary_op_t::LShr || op_ == binary_op_t::AShr;
}

bool binary_operator::is_int_mult()    const {
  return op_ == binary_op_t::Mul;
}

bool binary_operator::is_int_add_sub() const {
  return op_ == binary_op_t::Add || op_ == binary_op_t::Sub;
}


binary_operator::binary_operator(binary_op_t op, value *lhs, value *rhs, type *ty, const std::string &name, instruction *next)
    : instruction(ty, INST_BINOP, 2, name, next), op_(op){
  set_operand(0, lhs);
  set_operand(1, rhs);
}

binary_operator *binary_operator::create(binary_op_t op, value *lhs, value *rhs, const std::string &name, instruction *next){
  assert(lhs->get_type() == rhs->get_type() &&
         "Cannot create binary operator with two operands of differing type!");
  return new binary_operator(op, lhs, rhs, lhs->get_type(), name, next);
}

//binary_operator *binary_operator::create_fneg(value *arg, const std::string &name, instruction *next){
//  assert(arg->get_type()->get_scalar_ty()->is_floating_point_ty());
//  value *zero = constant_fp::get_zero_value_for_negation(arg->get_type());
//  return binary_operator::create(binary_op_t::FSub, zero, arg, name, next);
//}

//binary_operator *binary_operator::create_neg(value *arg, const std::string &name, instruction *next){
//  assert(arg->get_type()->get_scalar_ty()->is_integer_ty());
//  value *zero = constant_fp::get_zero_value_for_negation(arg->get_type()->get_scalar_ty());
//  return binary_operator::create(binary_op_t::Sub, zero, arg, name, next);
//}

//binary_operator *binary_operator::create_not(value *arg, const std::string &name, instruction *next){
//  assert(arg->get_type()->is_integer_ty());
//  constant *mask = constant::get_all_ones_value(arg->get_type());
//  return binary_operator::create(binary_op_t::Xor, arg, mask, name, next);
//}

//===----------------------------------------------------------------------===//
//                               cmp_inst classes
//===----------------------------------------------------------------------===//



// cmp_inst
std::string cmp_inst::repr_impl() const {
  switch (pred_) {
    case FCMP_FALSE :  return "false";
    case FCMP_OEQ   :  return "fcmp_oeq";
    case FCMP_OGT   :  return "fcmp_ogt";
    case FCMP_OGE   :  return "fcmp_oge";
    case FCMP_OLT   :  return "fcmp_olt";
    case FCMP_OLE   :  return "fcmp_ole";
    case FCMP_ONE   :  return "fcmp_one";
    case FCMP_ORD   :  return "fcmp_ord";
    case FCMP_UNO   :  return "fcmp_uno";
    case FCMP_UEQ   :  return "fcmp_ueq";
    case FCMP_UGT   :  return "fcmp_ugt";
    case FCMP_UGE   :  return "fcmp_uge";
    case FCMP_ULT   :  return "fcmp_ult";
    case FCMP_ULE   :  return "fcmp_ule";
    case FCMP_UNE   :  return "fcmp_une";
    case FCMP_TRUE  :  return "true";
    case ICMP_EQ    :  return "icmp_eq";
    case ICMP_NE    :  return "icmp_ne";
    case ICMP_UGT   :  return "icmp_ugt";
    case ICMP_UGE   :  return "icmp_uge";
    case ICMP_ULT   :  return "icmp_ult";
    case ICMP_ULE   :  return "icmp_ule";
    case ICMP_SGT   :  return "icmp_sgt";
    case ICMP_SGE   :  return "icmp_sge";
    case ICMP_SLT   :  return "icmp_slt";
    case ICMP_SLE   :  return "icmp_sle";
    default: throw std::runtime_error("unreachable");
  }
}

cmp_inst::cmp_inst(type *ty, value_id_t id, cmp_pred_t pred, value *lhs, value *rhs, const std::string &name, instruction *next)
    : instruction(ty, id, 2, name, next), pred_(pred) {
  set_operand(0, lhs);
  set_operand(1, rhs);
}

type* cmp_inst::make_cmp_result_type(type *ty){
  type* int1_ty = type::get_int1_ty(ty->get_context());
  if (block_type* tile_ty = dynamic_cast<block_type*>(ty))
    return block_type::get_same_shapes(int1_ty, tile_ty);
  return int1_ty;
}


bool cmp_inst::is_fp_predicate(cmp_pred_t pred) {
  return pred >= FIRST_FCMP_PREDICATE && pred <= LAST_FCMP_PREDICATE;
}

bool cmp_inst::is_int_predicate(cmp_pred_t pred) {
  return pred >= FIRST_ICMP_PREDICATE && pred <= LAST_ICMP_PREDICATE;
}


// icmp_inst
icmp_inst::icmp_inst(type *ty, cmp_pred_t pred,
                     value *lhs, value *rhs, const std::string &name, instruction *next)
  : cmp_inst(ty, INST_ICMP, pred, lhs, rhs, name, next){ }

icmp_inst* icmp_inst::create(cmp_pred_t pred, value *lhs, value *rhs, const std::string &name, instruction *next){
  assert(is_int_predicate(pred));
  type *res_ty = make_cmp_result_type(lhs->get_type());
  return new icmp_inst(res_ty, pred, lhs, rhs, name, next);
}

// fcmp_inst
fcmp_inst::fcmp_inst(type *ty, cmp_pred_t pred,
                     value *lhs, value *rhs, const std::string &name, instruction *next)
  : cmp_inst(ty, INST_FCMP, pred, lhs, rhs, name, next){ }

fcmp_inst* fcmp_inst::create(cmp_pred_t pred, value *lhs, value *rhs, const std::string &name, instruction *next){
  assert(is_fp_predicate(pred));
  type *res_ty = make_cmp_result_type(lhs->get_type());
  return new fcmp_inst(res_ty, pred, lhs, rhs, name, next);
}

//===----------------------------------------------------------------------===//
//                               unary_inst classes
//===----------------------------------------------------------------------===//

unary_inst::unary_inst(type *ty, value_id_t id, value *v, const std::string &name, instruction *next)
    : instruction(ty, id, 1, name, next) {
  set_operand(0, v);
}

//===----------------------------------------------------------------------===//
//                               cast_inst classes
//===----------------------------------------------------------------------===//

std::string cast_inst::repr_impl() const {
  switch (op_){
  case cast_op_t::Trunc:         return "trunc";
  case cast_op_t::ZExt:          return "zext";
  case cast_op_t::SExt:          return "sext";
  case cast_op_t::FPTrunc:       return "fp_trunc";
  case cast_op_t::FPExt:         return "fp_ext";
  case cast_op_t::UIToFP:        return "ui_to_fp";
  case cast_op_t::SIToFP:        return "si_to_fp";
  case cast_op_t::FPToUI:        return "fp_to_ui";
  case cast_op_t::FPToSI:        return "fp_to_si";
  case cast_op_t::PtrToInt:      return "ptr_to_int";
  case cast_op_t::IntToPtr:      return "int_to_ptr";
  case cast_op_t::BitCast:       return "bitcast";
  case cast_op_t::AddrSpaceCast: return "addr_space_cast";
  default: throw std::runtime_error("unreachable");
  }
}
// TODO
bool cast_inst::is_valid(cast_op_t op, value *arg, type *ty) {
  assert(arg->get_type()->is_block_ty() == ty->is_block_ty());
  return true;
}

cast_inst *cast_inst::create(cast_op_t op, value *arg, type *ty, const std::string &name, instruction *next){
  assert(is_valid(op, arg, ty) && "Invalid cast!");
  // Construct and return the appropriate CastInst subclass
  switch (op) {
  case cast_op_t::Trunc:         return new trunc_inst           (ty, arg, name, next);
  case cast_op_t::ZExt:          return new z_ext_inst           (ty, arg, name, next);
  case cast_op_t::SExt:          return new s_ext_inst           (ty, arg, name, next);
  case cast_op_t::FPTrunc:       return new fp_trunc_inst        (ty, arg, name, next);
  case cast_op_t::FPExt:         return new fp_ext_inst          (ty, arg, name, next);
  case cast_op_t::UIToFP:        return new ui_to_fp_inst        (ty, arg, name, next);
  case cast_op_t::SIToFP:        return new si_to_fp_inst        (ty, arg, name, next);
  case cast_op_t::FPToUI:        return new fp_to_ui_inst        (ty, arg, name, next);
  case cast_op_t::FPToSI:        return new fp_to_si_inst        (ty, arg, name, next);
  case cast_op_t::PtrToInt:      return new ptr_to_int_inst      (ty, arg, name, next);
  case cast_op_t::IntToPtr:      return new int_to_ptr_inst      (ty, arg, name, next);
  case cast_op_t::BitCast:       return new bit_cast_inst        (ty, arg, name, next);
  case cast_op_t::AddrSpaceCast: return new addr_space_cast_inst (ty, arg, name, next);
  default: throw std::runtime_error("unreachable");
  }
}

cast_inst *cast_inst::create_integer_cast(value *arg, type *ty, bool is_signed, const std::string &name, instruction *next){
  type *arg_ty = arg->get_type();
  assert(arg_ty->is_int_or_tileint_ty() && ty->is_int_or_tileint_ty() && "Invalid integer cast!");
  unsigned arg_bits = arg_ty->get_scalar_ty()->get_integer_bitwidth();
  unsigned dst_bits = ty->get_scalar_ty()->get_integer_bitwidth();
  cast_op_t op = (arg_bits == dst_bits ? cast_op_t::BitCast :
            (arg_bits > dst_bits  ? cast_op_t::Trunc :
            (is_signed            ? cast_op_t::SExt : cast_op_t::ZExt)));
  return create(op, arg, ty, name, next);
}

//===----------------------------------------------------------------------===//
//                               terminator_inst classes
//===----------------------------------------------------------------------===//


// return_inst
return_inst::return_inst(context &ctx, value *ret_val, instruction *next)
    : terminator_inst(type::get_void_ty(ctx), INST_RETURN, ret_val!=nullptr, "", next){
  if(ret_val)
    set_operand(0, ret_val);
}

return_inst *return_inst::create(context &ctx, value *ret_val, instruction *next){
  return new return_inst(ctx, ret_val, next);
}


// branch_inst
branch_inst* branch_inst::create(basic_block *dst, instruction *next) {
  assert(dst && "Branch destination may not be null!");
  return new uncond_branch_inst(dst, next);
}

branch_inst* branch_inst::create(value *cond, basic_block *if_dst, basic_block *else_dst, instruction *next) {
  assert(cond->get_type()->is_integer_ty(1) && "May only branch on boolean predicates!");
  return new cond_branch_inst(if_dst, else_dst, cond, next);
}

// uncond_branch_inst
uncond_branch_inst::uncond_branch_inst(basic_block *dst, instruction *next)
    : branch_inst(type::get_void_ty(dst->get_context()), INST_UNCOND_BRANCH, 1, "", next){
  set_operand(0, dst);
}

// cond_branch_inst
cond_branch_inst::cond_branch_inst(basic_block *if_dst, basic_block *else_dst, value *cond, instruction *next)
    : branch_inst(type::get_void_ty(if_dst->get_context()), INST_COND_BRANCH, 3, "", next){
  assert(cond->get_type()->is_integer_ty(1) && "May only branch on boolean predicates!");
  set_operand(0, if_dst);
  set_operand(1, else_dst);
  set_operand(2, cond);
}


//===----------------------------------------------------------------------===//
//                               getelementptr_inst classes
//===----------------------------------------------------------------------===//

getelementptr_inst::getelementptr_inst(type *pointee_ty, value *ptr, const std::vector<value *> &idx, const std::string &name, instruction *next)
    : instruction(get_return_type(pointee_ty, ptr, idx), INST_GETELEMENTPTR, 1 + idx.size(), name, next),
      source_elt_ty(pointee_ty),
      res_elt_ty(get_indexed_type(pointee_ty, idx)){
  // sanity check
  type *expected_ty = get_type()->get_scalar_ty();
  expected_ty = ((pointer_type*)expected_ty)->get_element_ty();
  assert(res_elt_ty == expected_ty);
  // set operands
  set_operand(0, ptr);
  for(size_t i = 0; i < idx.size(); i++)
    set_operand(1 + i, idx[i]);
}

type *getelementptr_inst::get_return_type(type *elt_ty, value *x, const std::vector<value *> &idx_list) {
  // result pointer type
  type *ty = x->get_type();
  unsigned addr_space = ty->get_scalar_ty()->get_pointer_address_space();
  type *ptr_ty = pointer_type::get(get_indexed_type(elt_ty, idx_list), addr_space);
  // Tile GEP
  if(ty->is_block_ty())
    return block_type::get_same_shapes(ptr_ty, ty);
  for(value *idx : idx_list)
  if (idx->get_type()->is_block_ty())
    return block_type::get_same_shapes(ptr_ty, ty);
  // Scalar GEP
  return ptr_ty;
}

type *getelementptr_inst::get_indexed_type_impl(type *ty, const std::vector<value *> &idx_list) {
  if(idx_list.empty())
    return ty;
  if(!ty->is_sized())
    return nullptr;
  unsigned cur_idx = 1;
  for(; cur_idx != idx_list.size(); cur_idx++){
    composite_type *cty = dynamic_cast<composite_type*>(ty);
    if(!cty || cty->is_pointer_ty())
      break;
    value *idx = idx_list[cur_idx];
    if(!cty->index_valid(idx))
      break;
    ty = cty->get_type_at_index(idx);
  }
  return (cur_idx == idx_list.size())? ty : nullptr;
}

type *getelementptr_inst::get_indexed_type(type *ty, const std::vector<value *> &idx_list) {
  type *result = get_indexed_type_impl(ty, idx_list);
  assert(result && "invalid GEP type!");
  return result;
}

getelementptr_inst *getelementptr_inst::create(value *ptr, const std::vector<value *> &idx, const std::string &name, instruction *next) {
  type *pointee_ty = ((pointer_type*)(ptr->get_type()->get_scalar_ty()))->get_element_ty();
  return new getelementptr_inst(pointee_ty, ptr, idx, name, next);
}


//===----------------------------------------------------------------------===//
//                               load_inst/store_inst classes
//===----------------------------------------------------------------------===//

// io_inst
io_inst::io_inst(type *ty, value_id_t id, unsigned num_ops, const std::string &name, instruction *next)
  : instruction(ty, id, num_ops, name, next)
{ }

// load_inst
load_inst::load_inst(value *ptr, value_id_t id, unsigned num_ops, const std::string &name, instruction *next)
  : io_inst(get_pointee_type(ptr->get_type()), id, num_ops, name, next)
{ }

// load
type *load_inst::get_pointee_type(type *ty) {
  type *scalar_ty = ty->get_scalar_ty();
  type *pointee_ty = scalar_ty->get_pointer_element_ty();
  if(ty->is_block_ty())
    return block_type::get_same_shapes(pointee_ty, ty);
  return pointee_ty;
}

// unmasked_load
unmasked_load_inst::unmasked_load_inst(value *ptr, const std::string &name, instruction *next)
  : load_inst(ptr, INST_UNMASKED_LOAD, 1, name, next) {
  set_operand(0, ptr);
}

unmasked_load_inst* unmasked_load_inst::create(value *ptr, const std::string &name, instruction *next) {
  return new unmasked_load_inst(ptr, name, next);
}

// masked load
masked_load_inst::masked_load_inst(value *ptr, value *mask, value *false_value,
                                   const std::string &name, instruction *next)
  : load_inst(ptr, INST_MASKED_LOAD, 3, name, next) {
  set_operand(0, ptr);
  set_operand(1, mask);
  set_operand(2, false_value);
}

masked_load_inst* masked_load_inst::create(value *ptr, value *mask, value *false_value,
                                           const std::string &name, instruction *next) {
  return new masked_load_inst(ptr, mask, false_value, name, next);
}

// masked load async
masked_load_async_inst::masked_load_async_inst(value *ptr, value *mask, value *false_value,
                                   const std::string &name, instruction *next)
  : load_inst(ptr, INST_MASKED_LOAD_ASYNC, 3, name, next) {
  set_operand(0, ptr);
  set_operand(1, mask);
  set_operand(2, false_value);
}

masked_load_async_inst* masked_load_async_inst::create(value *ptr, value *mask, value *false_value,
                                           const std::string &name, instruction *next) {
  return new masked_load_async_inst(ptr, mask, false_value, name, next);
}

// store

store_inst::store_inst(value *ptr, value_id_t id, unsigned num_ops, const std::string &name, instruction *next)
  : io_inst(type::get_void_ty(ptr->get_type()->get_context()), id, num_ops, name, next)
{ }

// unmasked_store
unmasked_store_inst::unmasked_store_inst(value *ptr, value *val,
                                         const std::string &name, instruction *next)
    : store_inst(ptr, INST_UNMASKED_STORE, 2, name, next)  {
  set_operand(0, ptr);
  set_operand(1, val);
}

unmasked_store_inst* unmasked_store_inst::create(value *ptr, value *val,
                                                 const std::string &name, instruction *next) {
  return new unmasked_store_inst(ptr, val, name, next);
}

// masked store
masked_store_inst::masked_store_inst(value *ptr, value *val, value *mask,
                                     const std::string &name, instruction *next)
  : store_inst(ptr, INST_MASKED_STORE, 3, name, next) {
  set_operand(0, ptr);
  set_operand(1, val);
  set_operand(2, mask);
}

masked_store_inst* masked_store_inst::create(value *ptr, value *val, value *mask, const std::string &name, instruction *next)  {
  return new masked_store_inst(ptr, val, mask, name, next);
}
//===----------------------------------------------------------------------===//
//                               retile_inst classes
//===----------------------------------------------------------------------===//

retile_inst::retile_inst(value *arg, value_id_t id, const type::block_shapes_t &shapes,
                         const std::string &name, instruction *next)
   : unary_inst(block_type::get(arg->get_type()->get_scalar_ty(), shapes), id, arg, name, next) { }


// reshape

instruction* reshape_inst::create(value *arg, const type::block_shapes_t &shapes,
                                  const std::string &name, instruction *next) {
  return new reshape_inst(arg, INST_RESHAPE, shapes, name, next);
}


// splat

instruction* splat_inst::create(value *arg, const type::block_shapes_t &shapes,
                                  const std::string &name, instruction *next) {
  return new splat_inst(arg, INST_SPLAT, shapes, name, next);
}

// broadcast

instruction* broadcast_inst::create(value *arg, const type::block_shapes_t &shapes,
                                  const std::string &name, instruction *next) {
  return new broadcast_inst(arg, INST_BROADCAST, shapes, name, next);
}

// downcast

instruction* downcast_inst::create(value *arg, const std::string &name, instruction *next) {
  return new downcast_inst(arg->get_type()->get_scalar_ty(), INST_DOWNCAST, arg, name, next);
}

//===----------------------------------------------------------------------===//
//                               matmul_inst classes
//===----------------------------------------------------------------------===//

dot_inst::dot_inst(value *A, value *B, value *C, TransT AT, TransT BT,
                         const std::string &name, instruction *next)
    : builtin_inst(C->get_type(), INST_DOT, 3, name, next) {
  set_operand(0, A);
  set_operand(1, B);
  set_operand(2, C);
}

instruction *dot_inst::create(value *A, value *B, value *C,
                              bool AT, bool BT,
                              const std::string &name, instruction *next) {
  TransT OPA = AT ? Trans : NoTrans;
  TransT OPB = BT ? Trans : NoTrans;
  return new dot_inst(A, B, C, OPA, OPB, name, next);
}

instruction *dot_inst::create_nn(value *A, value *B, value *C,
                                 const std::string &name, instruction *next) {
  return new dot_inst(A, B, C, NoTrans, NoTrans, name, next);
}

instruction *dot_inst::create_nt(value *A, value *B, value *C,
                                 const std::string &name, instruction *next) {
  return new dot_inst(A, B, C, NoTrans, Trans, name, next);
}

instruction *dot_inst::create_tn(value *A, value *B, value *C,
                                 const std::string &name, instruction *next) {
  return new dot_inst(A, B, C, Trans, NoTrans, name, next);
}

instruction *dot_inst::create_tt(value *A, value *B, value *C,
                                 const std::string &name, instruction *next) {
  return new dot_inst(A, B, C, Trans, Trans, name, next);
}

//===----------------------------------------------------------------------===//
//                               trans instructions
//===----------------------------------------------------------------------===//

ir::type* trans_inst::get_res_ty(ir::type* ty, std::vector<int> perm) {
  // get argument shapes
  ir::block_type::block_shapes_t arg_shapes = ty->get_block_shapes();
  // permutate argument shapes
  perm = init_perm(ty, perm);
  ir::block_type::block_shapes_t res_shapes = arg_shapes;
  for(size_t i = 0; i < perm.size(); i++)
    res_shapes[i] = arg_shapes[perm[i]];
  // construct type
  return block_type::get(ty->get_scalar_ty(), res_shapes);
}

std::vector<int> trans_inst::init_perm(ir::type* ty, const std::vector<int>& perm) {
  if(!perm.empty())
    return perm;
  auto size = ty->get_block_shapes().size();
  std::vector<int> result;
  result.push_back(size - 1);
  for(size_t i = 0; i < size - 1; i++)
    result.push_back(i);
  return result;
}

trans_inst::trans_inst(value *arg, const std::vector<int> &perm, const std::string &name, instruction *next)
  : builtin_inst(get_res_ty(arg->get_type(), perm), INST_TRANS, 1, name, next) {
  // sanity check
  perm_ = init_perm(arg->get_type(), perm);
  //auto size = arg->get_type()->get_tile_shapes().size();
  //assert(perm_.size() == size);
  set_operand(0, arg);
}

instruction* trans_inst::create(value *arg, const std::vector<int> &perm, const std::string &name, instruction *next) {
  return new trans_inst(arg, perm, name, next);
}

const std::vector<int> trans_inst::get_perm() const {
  return perm_;
}

//===----------------------------------------------------------------------===//
//                               sqrt instructions
//===----------------------------------------------------------------------===//

sqrt_inst::sqrt_inst(value *arg, const std::string &name, instruction *next)
  : builtin_inst(arg->get_type(), INST_SQRT, 1, name, next){
  set_operand(0, arg);
}

instruction* sqrt_inst::create(value *arg, const std::string &name, instruction *next) {
  return new sqrt_inst(arg, name, next);
}

//===----------------------------------------------------------------------===//
//                               reduce instructions
//===----------------------------------------------------------------------===//

std::string reduce_inst::to_str(op_t op) {
  switch (op) {
    case ADD: return "+";
    case SUB: return "-";
    case MAX: return "imax";
    case MIN: return "imin";
    case FADD: return "+";
    case FSUB: return "-";
    case FMAX: return "fmax";
    case FMIN: return "fmin";
    default: break;
  }
  assert(false);
  return "";
}

type* reduce_inst::get_res_type(value *arg, unsigned axis) {
  ir::block_type::block_shapes_t shapes = arg->get_type()->get_block_shapes();
  shapes.erase(shapes.begin() + axis);
  type *scalar_ty = arg->get_type()->get_scalar_ty();
  if(shapes.empty())
//    shapes.push_back(1);
    return scalar_ty;
  return block_type::get(scalar_ty, shapes);
}

reduce_inst::reduce_inst(value *arg, op_t op, unsigned axis, const std::string &name, instruction *next)
  : builtin_inst(get_res_type(arg, axis), INST_REDUCE, 1, name, next),
    op_(op),
    axis_(axis){
  set_operand(0, arg);
}

instruction* reduce_inst::create(value *arg, op_t op, unsigned axis, const std::string &name, instruction *next) {
  return new reduce_inst(arg, op, axis, name, next);
}


//===----------------------------------------------------------------------===//
//                               select instructions
//===----------------------------------------------------------------------===//

select_inst::select_inst(value *pred, value *if_value, value *else_value, const std::string &name, instruction *next)
  : builtin_inst(if_value->get_type(), INST_SELECT, 3, name, next){
  set_operand(0, pred);
  set_operand(1, if_value);
  set_operand(2, else_value);
}

instruction* select_inst::create(value *pred, value *if_value, value *else_value, const std::string &name, instruction *next) {
  return new select_inst(pred, if_value, else_value, name, next);
}
//===----------------------------------------------------------------------===//
//                               builtin instructions
//===----------------------------------------------------------------------===//


// get_program_id
get_program_id_inst::get_program_id_inst(type *ty, unsigned axis, const std::string &name, instruction *next)
  : builtin_inst(ty, INST_GET_PROGRAM_ID, 0, name, next), axis_(axis){

}

instruction* get_program_id_inst::create(context &ctx, unsigned axis, const std::string &name, instruction *next) {
  return new get_program_id_inst(type::get_int32_ty(ctx), axis, name, next);
}

// get_num_program
get_num_programs_inst::get_num_programs_inst(type *ty, unsigned axis, const std::string &name, instruction *next)
  : builtin_inst(ty, INST_GET_NUM_PROGRAMS, 0, name, next), axis_(axis){

}

instruction* get_num_programs_inst::create(context &ctx, unsigned axis, const std::string &name, instruction *next) {
  return new get_num_programs_inst(type::get_int32_ty(ctx), axis, name, next);
}

// atomic_rmw

atomic_rmw_inst::atomic_rmw_inst(atomic_rmw_op_t op, value *ptr, value *val, value *msk, const std::string &name, instruction *next)
  : atomic_inst(ptr->get_type()->get_pointer_element_ty(), INST_ATOMIC_RMW, 3, name, next), op_(op) {
  set_operand(0, ptr);
  set_operand(1, val);
  set_operand(2, msk);
}

instruction* atomic_rmw_inst::create(atomic_rmw_op_t op, value *ptr, value *val, value *msk, const std::string &name, instruction *next) {
  return new atomic_rmw_inst(op, ptr, val, msk, name, next);
}


// atomic cas

atomic_cas_inst::atomic_cas_inst(value *ptr, value *cmp, value *val, const std::string &name, instruction *next)
  : atomic_inst(ptr->get_type()->get_pointer_element_ty(), INST_ATOMIC_CAS, 3, name, next) {
  set_operand(0, ptr);
  set_operand(1, cmp);
  set_operand(2, val);
}

instruction* atomic_cas_inst::create(value *ptr, value *cmp, value *val, const std::string &name, instruction *next) {
  return new atomic_cas_inst(ptr, cmp, val, name, next);
}


// exp

exp_inst::exp_inst(value *val, const std::string &name, instruction *next)
  : builtin_inst(val->get_type(), INST_EXP, 1, name, next) {
  set_operand(0, val);
}

instruction* exp_inst::create(value *val, const std::string& name, instruction *next) {
  return new exp_inst(val, name, next);
}

// cos
cos_inst::cos_inst(value *val, const std::string &name, instruction *next)
  : builtin_inst(val->get_type(), INST_COS, 1, name, next) {
  set_operand(0, val);
}

instruction* cos_inst::create(value *val, const std::string& name, instruction *next) {
  return new cos_inst(val, name, next);
}

// sin
sin_inst::sin_inst(value *val, const std::string &name, instruction *next)
  : builtin_inst(val->get_type(), INST_SIN, 1, name, next) {
  set_operand(0, val);
}

instruction* sin_inst::create(value *val, const std::string& name, instruction *next) {
  return new sin_inst(val, name, next);
}


// log

log_inst::log_inst(value *val, const std::string &name, instruction *next)
  : builtin_inst(val->get_type(), INST_LOG, 1, name, next) {
  set_operand(0, val);
}

instruction* log_inst::create(value *val, const std::string& name, instruction *next) {
  return new log_inst(val, name, next);
}


//===----------------------------------------------------------------------===//
//                               intrinsic instructions
//===----------------------------------------------------------------------===//

// cvt_scanline
cvt_layout_inst* cvt_layout_inst::create(value *arg, const std::string &name, instruction *next) {
  return new cvt_layout_inst(arg->get_type(), INST_CVT_LAYOUT, arg, name, next);
}

// copy to shared
copy_to_shared_inst* copy_to_shared_inst::create(value *arg, const std::string &name,
                                                 instruction *next) {
  return new copy_to_shared_inst(arg->get_type(), INST_COPY_TO_SHARED, arg, name, next);
}

// copy from shared
copy_from_shared_inst* copy_from_shared_inst::create(value *arg, const std::string &name,
                                                 instruction *next) {
  return new copy_from_shared_inst(arg->get_type(), INST_COPY_FROM_SHARED, arg, name, next);
}

// barrier
barrier_inst::barrier_inst(context &ctx, const std::string &name,
                                                       instruction *next)
  : instruction(type::get_void_ty(ctx), INST_BARRIER, 0, name, next) { }

barrier_inst* barrier_inst::create(context &ctx, const std::string &name, instruction *next) {
  return new barrier_inst(ctx, name, next);
}

async_wait_inst::async_wait_inst(context &ctx, int N, const std::string &name, instruction *next)
  : instruction(type::get_void_ty(ctx), INST_ASYNC_WAIT, 0, name, next), N_(N) { }

async_wait_inst* async_wait_inst::create(context &ctx, int N, const std::string &name, instruction *next) {
  return new async_wait_inst(ctx, N, name, next);
}

// prefetch_s
prefetch_s_inst *prefetch_s_inst::create(context &ctx, value *arg, int inc, const std::string &name, instruction *next) {
  return new prefetch_s_inst(ctx, arg, inc, name, next);
}

//// nv_dynamic_program_idx
//make_range_dyn::make_range_dyn(type *ty, const std::string &name, instruction *next)
//  : instruction(ty, INST_MAKE_RANGE_DYN, 0, name, next) { }

//make_range_dyn* make_range_dyn::create(type *ty, const std::string &name, instruction *next) {
//  return new make_range_dyn(ty, name, next);
//}

//// nv_static_program_idx
//make_range_sta::make_range_sta(make_range *range)
//  : constant(range->get_type(), 0), range_(range) { }

//make_range* make_range_sta::get_range() const
//{ return range_; }

//make_range_sta* make_range_sta::get(make_range* range) {
//  static std::map<make_range*, make_range_sta*> cache;
//  if(cache.find(range) == cache.end())
//    cache.insert({range, new make_range_sta(range)});
//  return cache.at(range);
//}


// make_range
make_range::make_range(type *ty, constant_int *first, constant_int *last)
  : instruction(ty, INST_MAKE_RANGE, 0), first_(first), last_(last){ }

make_range *make_range::create(constant_int *first, constant_int *last) {
  assert(first->get_type()->is_integer_ty());
  assert(first->get_type() == last->get_type());
  assert(((constant_int*)first)->get_value() == 0);
  type *ty = block_type::get(first->get_type(), {(unsigned)last->get_value() - (unsigned)first->get_value()});
  return new make_range(ty, first, last);
}

const constant_int* make_range::get_first() const {
  return first_;
}

const constant_int* make_range::get_last() const {
  return last_;
}



}
}
