#include <string>
#include "triton/ir/basic_block.h"
#include "triton/ir/builder.h"
#include "triton/ir/constant.h"
#include "triton/ir/instructions.h"
#include "triton/ir/type.h"
#include "llvm/IR/Instruction.h"

namespace tdl{
namespace ir{

builder::builder(context &ctx):
  ctx_(ctx), block_(nullptr), insert_point_(nullptr) {}

//===----------------------------------------------------------------------===//
//                               utilities
//===----------------------------------------------------------------------===//
void builder::set_insert_point(basic_block::iterator it){
  block_ = (*it)->get_parent();
  insert_point_ = it;
}

void builder::set_insert_point(instruction* i){
  block_ = i->get_parent();
  auto it = std::find(block_->begin(), block_->end(), i);
  set_insert_point(it);
}


void builder::set_insert_point(basic_block *block){
  block_ = block;
  insert_point_ = block->end();
}


//===----------------------------------------------------------------------===//
//                               convenience functions
//===----------------------------------------------------------------------===//

value *builder::get_int32(unsigned val) {
  return constant_int::get(type::get_int32_ty(ctx_), val);
}

type *builder::get_float_ty()
{ return type::get_float_ty(ctx_); }

type *builder::get_double_ty()
{ return type::get_double_ty(ctx_); }


//===----------------------------------------------------------------------===//
//                               terminator instructions
//===----------------------------------------------------------------------===//

value* builder::create_br(basic_block *dest){
  dest->add_predecessor(block_);
  return insert(branch_inst::create(dest));
}

value* builder::create_cond_br(value *cond, basic_block *if_dest, basic_block *else_dest){
  if_dest->add_predecessor(block_);
  else_dest->add_predecessor(block_);
  return insert(branch_inst::create(cond, if_dest, else_dest));
}

value *builder::create_ret_void() {
  return insert(return_inst::create(ctx_));
}

//===----------------------------------------------------------------------===//
//                               cast instructions
//===----------------------------------------------------------------------===//
#define DEFINE_CAST_INSTR(SUFFIX, OPCODE)\
  value *builder::create_ ## SUFFIX(value *src, type *dst_ty, std::string const &name){\
    return create_cast(OPCODE, src, dst_ty, name);\
  }

DEFINE_CAST_INSTR(si_to_fp, llvm::Instruction::SIToFP)
DEFINE_CAST_INSTR(ui_to_fp, llvm::Instruction::UIToFP)
DEFINE_CAST_INSTR(fp_to_si, llvm::Instruction::FPToSI)
DEFINE_CAST_INSTR(fp_to_ui, llvm::Instruction::FPToUI)
DEFINE_CAST_INSTR(fp_ext, llvm::Instruction::FPExt)
DEFINE_CAST_INSTR(fp_trunc, llvm::Instruction::FPTrunc)

value* builder::create_cast(cast_inst::op_t op, value *v, type *dst_ty, const std::string &name){
  return insert(cast_inst::create(op, v, dst_ty), name);
}

value* builder::create_int_cast(value *src, type *dst_ty, bool is_signed, const std::string &name){
  return insert(cast_inst::create_integer_cast(src, dst_ty, is_signed), name);
}

//===----------------------------------------------------------------------===//
//                               phi instructions
//===----------------------------------------------------------------------===//

phi_node* builder::create_phi(type *ty, unsigned num_reserved, const std::string &name){
  return insert(phi_node::create(ty, num_reserved), name);
}

//===----------------------------------------------------------------------===//
//                               binary float instructions
//===----------------------------------------------------------------------===//

#define DEFINE_BINARY_FLOAT(SUFFIX, OPCODE)\
  value *builder::create_ ## SUFFIX(value *lhs, value *rhs, const std::string &name){\
    return insert(binary_operator::create(OPCODE, lhs, rhs), name);\
  }

#define DEFINE_UNARY_FLOAT(SUFFIX)\
  value *builder::create_ ## SUFFIX(value *arg, const std::string &name){\
    return insert(binary_operator::create_ ## SUFFIX(arg), name);\
  }

// Binary
DEFINE_BINARY_FLOAT(fmul, llvm::Instruction::FMul)
DEFINE_BINARY_FLOAT(fdiv, llvm::Instruction::FDiv)
DEFINE_BINARY_FLOAT(frem, llvm::Instruction::FRem)
DEFINE_BINARY_FLOAT(fadd, llvm::Instruction::FAdd)
DEFINE_BINARY_FLOAT(fsub, llvm::Instruction::FSub)
// Unary
DEFINE_UNARY_FLOAT(fneg)


//===----------------------------------------------------------------------===//
//                               binary int instructions
//===----------------------------------------------------------------------===//


value* builder::create_insert_nuwnswb_binop(binary_operator::op_t op, value *lhs,
                                            value *rhs, const std::string &name,
                                            bool has_nuw, bool has_nsw) {
  binary_operator* result = insert(binary_operator::create(op, lhs, rhs), name);
  if (has_nuw) result->set_has_no_unsigned_wrap();
  if (has_nsw) result->set_has_no_signed_wrap();
  return result;
}

#define DEFINE_NOWRAP_BINARY(SUFFIX, OPCODE)\
  value* builder::create_ ## SUFFIX(value *lhs, value *rhs, const std::string &name, bool has_nuw, bool has_nsw){\
    return create_insert_nuwnswb_binop(OPCODE, lhs, rhs, name, has_nuw, has_nsw);\
  }\

#define DEFINE_BINARY_INT(SUFFIX, OPCODE)\
  value *builder::create_ ## SUFFIX(value *lhs, value *rhs, const std::string &name){\
    return insert(binary_operator::create(OPCODE, lhs, rhs), name);\
  }

#define DEFINE_UNARY_INT(SUFFIX)\
  value *builder::create_ ## SUFFIX(value *arg, const std::string &name){\
    return insert(binary_operator::create_ ## SUFFIX(arg), name);\
  }

// Binary
DEFINE_NOWRAP_BINARY(mul, llvm::Instruction::Mul)
DEFINE_NOWRAP_BINARY(add, llvm::Instruction::Add)
DEFINE_NOWRAP_BINARY(sub, llvm::Instruction::Sub)
DEFINE_NOWRAP_BINARY(shl, llvm::Instruction::Shl)
DEFINE_NOWRAP_BINARY(ashr, llvm::Instruction::AShr)
DEFINE_BINARY_INT(sdiv, llvm::Instruction::SDiv)
DEFINE_BINARY_INT(udiv, llvm::Instruction::UDiv)
DEFINE_BINARY_INT(srem, llvm::Instruction::SRem)
DEFINE_BINARY_INT(urem, llvm::Instruction::URem)
DEFINE_BINARY_INT(and, llvm::Instruction::And)
DEFINE_BINARY_INT(or, llvm::Instruction::Or)
DEFINE_BINARY_INT(xor, llvm::Instruction::Xor)
// Unary
DEFINE_UNARY_INT(neg)
DEFINE_UNARY_INT(not)


//===----------------------------------------------------------------------===//
//                               getelementptr instructions
//===----------------------------------------------------------------------===//

value* builder::create_gep(value *ptr, const std::vector<value*>& idx_list, const std::string &name){
  return insert(getelementptr_inst::create(ptr, idx_list), name);
}

//===----------------------------------------------------------------------===//
//                               icmp instructions
//===----------------------------------------------------------------------===//

value *builder::create_icmp(cmp_inst::pred_t pred, value *lhs, value *rhs, const std::string &name){
  return insert(icmp_inst::create(pred, lhs, rhs), name);
}

#define DEFINE_ICMP_INSTR(SUFFIX, OPCODE)\
  value *builder::create_icmp ## SUFFIX(value *lhs, value *rhs, const std::string &name){\
    return create_icmp(OPCODE, lhs, rhs, name);\
  }

// Signed
DEFINE_ICMP_INSTR(SLE, llvm::ICmpInst::ICMP_SLE)
DEFINE_ICMP_INSTR(SLT, llvm::ICmpInst::ICMP_SLT)
DEFINE_ICMP_INSTR(SGE, llvm::ICmpInst::ICMP_SGE)
DEFINE_ICMP_INSTR(SGT, llvm::ICmpInst::ICMP_SGT)
// Unsigned
DEFINE_ICMP_INSTR(ULE, llvm::ICmpInst::ICMP_ULE)
DEFINE_ICMP_INSTR(ULT, llvm::ICmpInst::ICMP_ULT)
DEFINE_ICMP_INSTR(UGE, llvm::ICmpInst::ICMP_UGE)
DEFINE_ICMP_INSTR(UGT, llvm::ICmpInst::ICMP_UGT)
// General
DEFINE_ICMP_INSTR(EQ, llvm::ICmpInst::ICMP_EQ)
DEFINE_ICMP_INSTR(NE, llvm::ICmpInst::ICMP_NE)


//===----------------------------------------------------------------------===//
//                               fcmp instructions
//===----------------------------------------------------------------------===//

value *builder::create_fcmp(cmp_inst::pred_t pred, value *lhs, value *rhs, const std::string &name){
  return insert(fcmp_inst::create(pred, lhs, rhs), name);
}

#define DEFINE_FCMP_INSTR(SUFFIX, OPCODE)\
  value *builder::create_fcmp ## SUFFIX(value *lhs, value *rhs, const std::string &name){\
    return create_fcmp(OPCODE, lhs, rhs, name);\
  }

// Ordered
DEFINE_FCMP_INSTR(OLE, llvm::FCmpInst::FCMP_OLE)
DEFINE_FCMP_INSTR(OLT, llvm::FCmpInst::FCMP_OLT)
DEFINE_FCMP_INSTR(OGE, llvm::FCmpInst::FCMP_OGE)
DEFINE_FCMP_INSTR(OGT, llvm::FCmpInst::FCMP_OGT)
DEFINE_FCMP_INSTR(OEQ, llvm::FCmpInst::FCMP_OEQ)
DEFINE_FCMP_INSTR(ONE, llvm::FCmpInst::FCMP_ONE)



//===----------------------------------------------------------------------===//
//                               load/store instructions
//===----------------------------------------------------------------------===//

value *builder::create_load(value *arg, const std::string &name){
  return insert(load_inst::create(arg, name));
}

value *builder::create_store(value *ptr, value *val, const std::string &name){
  return insert(store_inst::create(ptr, val, name));
}

//===----------------------------------------------------------------------===//
//                               tile instructions
//===----------------------------------------------------------------------===//

value *builder::create_reshape(value *arg, const type::tile_shapes_t &shapes, const std::string &name) {
  return insert(reshape_inst::create(arg, shapes, name));
}

value *builder::create_splat(value *arg, const type::tile_shapes_t &shapes, const std::string &name) {
  return insert(splat_inst::create(arg, shapes, name));
}

value *builder::create_broadcast(value *arg, const type::tile_shapes_t &shapes, const std::string &name) {
  return insert(broadcast_inst::create(arg, shapes, name));
}

//===----------------------------------------------------------------------===//
//                               built-in instructions
//===----------------------------------------------------------------------===//

value *builder::create_get_global_range(unsigned axis, type::tile_shapes_t::value_type size, const std::string &name) {
  return insert(get_global_range_inst::create(ctx_, axis, size, name));
}

value *builder::create_matmul(value *A, value *B, value *C, const std::string &name) {
  return insert(matmul_inst::create(A, B, C, name));
}

//===----------------------------------------------------------------------===//
//                               intrinsic instructions
//===----------------------------------------------------------------------===//


value *builder::create_copy_to_shared(value *arg, const std::string &name) {
  return insert(copy_to_shared_inst::create(arg, name));
}

value *builder::create_vectorize(value *arg, const std::string &name) {
  return insert(vectorize_inst::create(arg, name));
}

value *builder::create_barrier(const std::string &name) {
  return insert(barrier_inst::create(ctx_, name));
}

}
}
