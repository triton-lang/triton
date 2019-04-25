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

instruction::instruction(type *ty, unsigned num_ops, unsigned num_results, const std::string &name, instruction *next)
    : user(ty, num_ops, name) {
  if(next){
    basic_block *block = next->get_parent();
    assert(block && "Next instruction is not in a basic block!");
    auto it = std::find(block->begin(), block->end(), next);
    block->get_inst_list().insert(it, next);
  }
  if(num_results == 1)
    results_.push_back(this);
  else
    for(unsigned i = 0; i < num_results; i++)
      results_.push_back(new result_reference(this, i));
}

void instruction::erase_from_parent() {
  parent_->erase(this);
  for(ir::value* op: ops())
    op->erase_use(this);
}

bool instruction::has_tile_result_or_op() {
  bool result = get_type()->is_tile_ty();
  for(unsigned i = 0; i < get_num_operands(); i++)
    result |= get_operand(i)->get_type()->is_tile_ty();
  return result;
}


// result reference
result_reference::result_reference(instruction *ref, unsigned arg_id, const std::string &name)
  : value(ref->get_type(), name), arg_id_(arg_id){ }

//===----------------------------------------------------------------------===//
//                               phi_node classes
//===----------------------------------------------------------------------===//

phi_node::phi_node(type *ty, unsigned num_reserved, std::string const &name, instruction *next)
    : instruction(ty, 0, 1, name, next) {
  blocks_.reserve(num_reserved);
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
  case llop::Add  : return "add";
  case llop::FAdd : return "fadd";
  case llop::Sub  : return "sub";
  case llop::FSub : return "fsub";
  case llop::Mul  : return "mul";
  case llop::FMul : return "fmul";
  case llop::UDiv : return "udiv";
  case llop::SDiv : return "sdiv";
  case llop::FDiv : return "fdiv";
  case llop::URem : return "urem";
  case llop::SRem : return "srem";
  case llop::FRem : return "frem";
  case llop::Shl  : return "shl";
  case llop::LShr : return "lshr";
  case llop::AShr : return "ashr";
  case llop::And  : return "and";
  case llop::Or   : return "or";
  case llop::Xor  : return "xor";
  default: throw std::runtime_error("unknown binary operator");
  }
}

binary_operator::binary_operator(op_t op, value *lhs, value *rhs, type *ty, const std::string &name, instruction *next)
    : instruction(ty, 2, 1, name, next), op_(op){
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

// cmp_inst
std::string cmp_inst::repr_impl() const {
  switch (pred_) {
    case llop::FCMP_FALSE :  return "false";
    case llop::FCMP_OEQ   :  return "fcmp_oeq";
    case llop::FCMP_OGT   :  return "fcmp_ogt";
    case llop::FCMP_OGE   :  return "fcmp_oge";
    case llop::FCMP_OLT   :  return "fcmp_olt";
    case llop::FCMP_OLE   :  return "fcmp_ole";
    case llop::FCMP_ONE   :  return "fcmp_one";
    case llop::FCMP_ORD   :  return "fcmp_ord";
    case llop::FCMP_UNO   :  return "fcmp_uno";
    case llop::FCMP_UEQ   :  return "fcmp_ueq";
    case llop::FCMP_UGT   :  return "fcmp_ugt";
    case llop::FCMP_UGE   :  return "fcmp_uge";
    case llop::FCMP_ULT   :  return "fcmp_ult";
    case llop::FCMP_ULE   :  return "fcmp_ule";
    case llop::FCMP_UNE   :  return "fcmp_une";
    case llop::FCMP_TRUE  :  return "true";
    case llop::ICMP_EQ    :  return "icmp_eq";
    case llop::ICMP_NE    :  return "icmp_ne";
    case llop::ICMP_UGT   :  return "icmp_ugt";
    case llop::ICMP_UGE   :  return "icmp_uge";
    case llop::ICMP_ULT   :  return "icmp_ult";
    case llop::ICMP_ULE   :  return "icmp_ule";
    case llop::ICMP_SGT   :  return "icmp_sgt";
    case llop::ICMP_SGE   :  return "icmp_sge";
    case llop::ICMP_SLT   :  return "icmp_slt";
    case llop::ICMP_SLE   :  return "icmp_sle";
    default: throw std::runtime_error("unreachable");
  }
}

cmp_inst::cmp_inst(type *ty, cmp_inst::pred_t pred, value *lhs, value *rhs, const std::string &name, instruction *next)
    : instruction(ty, 2, 1, name, next), pred_(pred) {
  set_operand(0, lhs);
  set_operand(1, rhs);
}

type* cmp_inst::make_cmp_result_type(type *ty){
  type* int1_ty = type::get_int1_ty(ty->get_context());
  if (tile_type* tile_ty = dynamic_cast<tile_type*>(ty))
    return tile_type::get_same_shapes(int1_ty, tile_ty);
  return int1_ty;
}


bool cmp_inst::is_fp_predicate(pred_t pred) {
  return pred >= llop::FIRST_FCMP_PREDICATE && pred <= llop::LAST_FCMP_PREDICATE;
}

bool cmp_inst::is_int_predicate(pred_t pred) {
  return pred >= llop::FIRST_ICMP_PREDICATE && pred <= llop::LAST_ICMP_PREDICATE;
}

// icmp_inst
icmp_inst* icmp_inst::create(pred_t pred, value *lhs, value *rhs, const std::string &name, instruction *next){
  assert(is_int_predicate(pred));
  type *res_ty = make_cmp_result_type(lhs->get_type());
  return new icmp_inst(res_ty, pred, lhs, rhs, name, next);
}

// fcmp_inst
fcmp_inst* fcmp_inst::create(pred_t pred, value *lhs, value *rhs, const std::string &name, instruction *next){
  assert(is_fp_predicate(pred));
  type *res_ty = make_cmp_result_type(lhs->get_type());
  return new fcmp_inst(res_ty, pred, lhs, rhs, name, next);
}

//===----------------------------------------------------------------------===//
//                               unary_inst classes
//===----------------------------------------------------------------------===//

unary_inst::unary_inst(type *ty, value *v, const std::string &name, instruction *next)
    : instruction(ty, 1, 1, name, next) {
  set_operand(0, v);
}

//===----------------------------------------------------------------------===//
//                               cast_inst classes
//===----------------------------------------------------------------------===//

std::string cast_inst::repr_impl() const {
  switch (op_){
  case ic::Trunc:         return "trunc";
  case ic::ZExt:          return "zext";
  case ic::SExt:          return "sext";
  case ic::FPTrunc:       return "fp_trunc";
  case ic::FPExt:         return "fp_ext";
  case ic::UIToFP:        return "ui_to_fp";
  case ic::SIToFP:        return "si_to_fp";
  case ic::FPToUI:        return "fp_to_ui";
  case ic::FPToSI:        return "fp_to_si";
  case ic::PtrToInt:      return "ptr_to_int";
  case ic::IntToPtr:      return "int_to_ptr";
  case ic::BitCast:       return "bitcast";
  case ic::AddrSpaceCast: return "addr_space_cast";
  default: throw std::runtime_error("unreachable");
  }
}
// TODO
bool cast_inst::is_valid(op_t op, value *arg, type *ty) {
  return true;
}

cast_inst *cast_inst::create(op_t op, value *arg, type *ty, const std::string &name, instruction *next){
  assert(is_valid(op, arg, ty) && "Invalid cast!");
  // Construct and return the appropriate CastInst subclass
  switch (op) {
  case ic::Trunc:         return new trunc_inst           (ty, arg, name, next);
  case ic::ZExt:          return new z_ext_inst           (ty, arg, name, next);
  case ic::SExt:          return new s_ext_inst           (ty, arg, name, next);
  case ic::FPTrunc:       return new fp_trunc_inst        (ty, arg, name, next);
  case ic::FPExt:         return new fp_ext_inst          (ty, arg, name, next);
  case ic::UIToFP:        return new ui_to_fp_inst        (ty, arg, name, next);
  case ic::SIToFP:        return new si_to_fp_inst        (ty, arg, name, next);
  case ic::FPToUI:        return new fp_to_ui_inst        (ty, arg, name, next);
  case ic::FPToSI:        return new fp_to_si_inst        (ty, arg, name, next);
  case ic::PtrToInt:      return new ptr_to_int_inst      (ty, arg, name, next);
  case ic::IntToPtr:      return new int_to_ptr_inst      (ty, arg, name, next);
  case ic::BitCast:       return new bit_cast_inst        (ty, arg, name, next);
  case ic::AddrSpaceCast: return new addr_space_cast_inst (ty, arg, name, next);
  default: throw std::runtime_error("unreachable");
  }
}

cast_inst *cast_inst::create_integer_cast(value *arg, type *ty, bool is_signed, const std::string &name, instruction *next){
  type *arg_ty = arg->get_type();
  assert(arg_ty->is_int_or_tileint_ty() && ty->is_int_or_tileint_ty() && "Invalid integer cast!");
  unsigned arg_bits = arg_ty->get_scalar_ty()->get_integer_bitwidth();
  unsigned dst_bits = ty->get_scalar_ty()->get_integer_bitwidth();
  op_t op = (arg_bits == dst_bits ? ic::BitCast :
            (arg_bits > dst_bits  ? ic::Trunc :
            (is_signed            ? ic::SExt : ic::ZExt)));
  return create(op, arg, ty, name, next);
}

//===----------------------------------------------------------------------===//
//                               terminator_inst classes
//===----------------------------------------------------------------------===//


// return_inst
return_inst::return_inst(context &ctx, value *ret_val, instruction *next)
    : terminator_inst(type::get_void_ty(ctx), ret_val!=nullptr, 0, "", next){
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
    : branch_inst(type::get_void_ty(dst->get_context()), 1, 0, "", next){
  set_operand(0, dst);
}

// cond_branch_inst
cond_branch_inst::cond_branch_inst(basic_block *if_dst, basic_block *else_dst, value *cond, instruction *next)
    : branch_inst(type::get_void_ty(if_dst->get_context()), 3, 0, "", next){
  assert(cond->get_type()->is_integer_ty(1) && "May only branch on boolean predicates!");
  set_operand(0, if_dst);
  set_operand(1, else_dst);
  set_operand(2, cond);
}

// mask_inst
mask_inst::mask_inst(value *pred, const std::string &name, instruction *next)
  : instruction(pred->get_type(), 1, 2, name, next) {
  set_operand(0, pred);
}

mask_inst* mask_inst::create(value *pred, const std::string &name, instruction *next) {
  return new mask_inst(pred, name, next);
}

// merge_inst
merge_inst::merge_inst(value *mask_true, value *value_true,
                       value *mask_false, value *value_false,
                       const std::string &name, instruction *next)
    : instruction(value_true->get_type(), 4, 1, name, next) {
  set_operand(0, mask_true);
  set_operand(1, value_true);
  set_operand(2, mask_false);
  set_operand(3, value_false);
}

merge_inst* merge_inst::create(value *mask_true, value *value_true,
                               value *mask_false, value *value_false,
                               const std::string &name, instruction *next) {
  return new merge_inst(mask_true, value_true, mask_false, value_false, name, next);
}



//===----------------------------------------------------------------------===//
//                               getelementptr_inst classes
//===----------------------------------------------------------------------===//

getelementptr_inst::getelementptr_inst(type *pointee_ty, value *ptr, const std::vector<value *> &idx, const std::string &name, instruction *next)
    : instruction(get_return_type(pointee_ty, ptr, idx), 1 + idx.size(), 1, name, next),
      source_elt_ty(pointee_ty),
      res_elt_ty(get_indexed_type(pointee_ty, idx)){
  type *expected_ty = ((pointer_type*)(get_type()->get_scalar_ty()))->get_element_ty();
  assert(res_elt_ty == expected_ty);
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
  if(ty->is_tile_ty())
    return tile_type::get_same_shapes(ptr_ty, ty);
  for(value *idx : idx_list)
  if (idx->get_type()->is_tile_ty())
    return tile_type::get_same_shapes(ptr_ty, ty);
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
type *load_inst::get_pointee_type(type *ty) {
  type *scalar_ty = ty->get_scalar_ty();
  type *pointee_ty = scalar_ty->get_pointer_element_ty();
  if(ty->is_tile_ty())
    return tile_type::get_same_shapes(pointee_ty, ty);
  return pointee_ty;
}

load_inst::load_inst(value *ptr, const std::string &name, instruction *next)
  : unary_inst(get_pointee_type(ptr->get_type()), ptr, name, next) {
}

load_inst* load_inst::create(value *ptr, const std::string &name, instruction *next) {
  return new load_inst(ptr, name, next);
}

// store
store_inst::store_inst(value *ptr, value *v, const std::string &name, instruction *next)
    : instruction(type::get_void_ty(ptr->get_type()->get_context()), 2, 1, name, next) {
  set_operand(0, ptr);
  set_operand(1, v);
}

store_inst* store_inst::create(value *ptr, value *v, const std::string &name, instruction *next) {
  return new store_inst(ptr, v, name, next);
}

//===----------------------------------------------------------------------===//
//                               retile_inst classes
//===----------------------------------------------------------------------===//

std::string retile_inst::shape_suffix(ir::type* ty){
  std::string res = "[";
  const auto& shapes = ty->get_tile_shapes();
  for(unsigned i = 0; i < shapes.size(); i++){
    res += std::to_string(ty->get_tile_shapes()[i]->get_value());
    if(i < shapes.size() - 1)
      res += ", ";
  }
  res += "]";
  return res;
}

retile_inst::retile_inst(value *arg, const type::tile_shapes_t &shapes,
                         const std::string &name, instruction *next)
   : unary_inst(tile_type::get(arg->get_type()->get_scalar_ty(), shapes), arg, name, next) { }

// reshape

instruction* reshape_inst::create(value *arg, const type::tile_shapes_t &shapes,
                                  const std::string &name, instruction *next) {
  return new reshape_inst(arg, shapes, name, next);
}


// splat

instruction* splat_inst::create(value *arg, const type::tile_shapes_t &shapes,
                                  const std::string &name, instruction *next) {
  return new splat_inst(arg, shapes, name, next);
}

// broadcast

instruction* broadcast_inst::create(value *arg, const type::tile_shapes_t &shapes,
                                  const std::string &name, instruction *next) {
  return new broadcast_inst(arg, shapes, name, next);
}

// downcast

instruction* downcast_inst::create(value *arg, const std::string &name, instruction *next) {
  return new downcast_inst(arg->get_type()->get_scalar_ty(), arg, name, next);
}

//===----------------------------------------------------------------------===//
//                               matmul_inst classes
//===----------------------------------------------------------------------===//

dot_inst::dot_inst(value *A, value *B, value *C, TransT AT, TransT BT,
                         const std::string &name, instruction *next)
    : builtin_inst(C->get_type(), 3, 1, name, next), AT_(AT), BT_(BT) {
  set_operand(0, A);
  set_operand(1, B);
  set_operand(2, C);
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

ir::type* trans_inst::get_res_ty(ir::type* ty) {
  auto shapes = ty->get_tile_shapes();
  std::rotate(shapes.begin(), shapes.begin() + 1, shapes.end());
  return tile_type::get(ty->get_scalar_ty(), shapes);
}

trans_inst::trans_inst(value *arg, const std::string &name, instruction *next)
  : builtin_inst(get_res_ty(arg->get_type()), 1, 1, name, next) {
  set_operand(0, arg);
}

instruction* trans_inst::create(value *arg, const std::string &name, instruction *next) {
  return new trans_inst(arg, name, next);
}

//===----------------------------------------------------------------------===//
//                               select instructions
//===----------------------------------------------------------------------===//

select_inst::select_inst(value *pred, value *if_value, value *else_value, const std::string &name, instruction *next)
  : builtin_inst(if_value->get_type(), 3, 1, name, next){
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

// get_global_range
get_global_range_inst::get_global_range_inst(type *ty, unsigned axis,
                                             const std::string &name, instruction *next)
  : builtin_inst(ty, 0, 1, name, next), axis_(axis) {

}

instruction* get_global_range_inst::create(context &ctx, unsigned axis, type::tile_shapes_t::value_type size,
                                           const std::string &name, instruction *next) {
  type *int_ty = type::get_int32_ty(ctx);
  type *tile_ty = tile_type::get(int_ty, {size});
  return new get_global_range_inst(tile_ty, axis, name, next);
}

// get_range_id
get_range_id_inst::get_range_id_inst(type *ty, unsigned axis, const std::string &name, instruction *next)
  : builtin_inst(ty, 0, 1, name, next), axis_(axis){

}

instruction* get_range_id_inst::create(context &ctx, unsigned axis, const std::string &name, instruction *next) {
  return new get_range_id_inst(type::get_int32_ty(ctx), axis, name, next);
}

// atomic cas

atomic_cas_inst::atomic_cas_inst(value *ptr, value *cmp, value *val, const std::string &name, instruction *next)
  : builtin_inst(ptr->get_type()->get_pointer_element_ty(), 3, 1, name, next) {
  set_operand(0, ptr);
  set_operand(1, cmp);
  set_operand(2, val);
}

instruction* atomic_cas_inst::create(value *ptr, value *cmp, value *val, const std::string &name, instruction *next) {
  return new atomic_cas_inst(ptr, cmp, val, name, next);
}
//===----------------------------------------------------------------------===//
//                               intrinsic instructions
//===----------------------------------------------------------------------===//
copy_to_shared_inst* copy_to_shared_inst::create(value *arg, const std::string &name,
                                                 instruction *next) {
  return new copy_to_shared_inst(arg->get_type(), arg, name, next);
}

vectorize_inst* vectorize_inst::create(value *arg, const std::string &name, instruction *next) {
  return new vectorize_inst(arg->get_type(), arg, name, next);
}

barrier_inst::barrier_inst(context &ctx, const std::string &name,
                                                       instruction *next)
  : instruction(type::get_void_ty(ctx), 0, 0, name, next){ }

barrier_inst* barrier_inst::create(context &ctx, const std::string &name, instruction *next) {
  return new barrier_inst(ctx, name, next);
}

}
}
