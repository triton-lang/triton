#include <numeric>
#include "triton/codegen/selection/generator.h"
#include "triton/codegen/target.h"
#include "triton/codegen/analysis/axes.h"
#include "triton/codegen/analysis/allocation.h"
#include "triton/codegen/analysis/align.h"
#include "triton/codegen/analysis/swizzle.h"
#include "triton/codegen/transform/coalesce.h"
#include "triton/ir/context.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/ir/type.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

namespace triton{
namespace codegen{

using namespace llvm;

// types
#define void_ty              builder_->getVoidTy()
#define f16_ty               builder_->getHalfTy()
#define f32_ty               builder_->getFloatTy()
#define i32_ty               builder_->getInt32Ty()
#define vec_ty(type, num_el) VectorType::get(type, num_el, false)
#define ptr_ty(...)          PointerType::get(__VA_ARGS__)
// constants
#define i32(...)             builder_->getInt32(__VA_ARGS__)
// ops
#define add(...)             builder_->CreateAdd(__VA_ARGS__)
#define and_(...)            builder_->CreateAnd(__VA_ARGS__)
#define atomic_cmp_xchg(...) builder_->CreateAtomicCmpXchg(__VA_ARGS__)
#define atomic_rmw(...)      builder_->CreateAtomicRMW(__VA_ARGS__)
#define bin_op(...)          builder_->CreateBinOp(__VA_ARGS__)
#define bit_cast(...)        builder_->CreateBitCast(__VA_ARGS__)
#define br(...)              builder_->CreateBr(__VA_ARGS__)
#define call(...)            builder_->CreateCall(__VA_ARGS__)
#define cast(...)            builder_->CreateCast(__VA_ARGS__)
#define cond_br(...)         builder_->CreateCondBr(__VA_ARGS__)
#define exact_udiv(...)      builder_->CreateExactUDiv(__VA_ARGS__)
#define extract_elt(...)     builder_->CreateExtractElement(__VA_ARGS__)
#define extract_val(...)     builder_->CreateExtractValue(__VA_ARGS__)
#define fadd(...)            builder_->CreateFAdd(__VA_ARGS__)
#define fcmp(...)            builder_->CreateFCmp(__VA_ARGS__)
#define fmul(...)            builder_->CreateFMul(__VA_ARGS__)
#define fpcast(...)          builder_->CreateFPCast(__VA_ARGS__)
#define fsub(...)            builder_->CreateFSub(__VA_ARGS__)
#define gep(...)             builder_->CreateGEP(__VA_ARGS__)
#define icmp(...)            builder_->CreateICmp(__VA_ARGS__)
#define icmp_eq(...)         builder_->CreateICmpEQ(__VA_ARGS__)
#define icmp_sge(...)        builder_->CreateICmpSGE(__VA_ARGS__)
#define icmp_sle(...)        builder_->CreateICmpSLE(__VA_ARGS__)
#define icmp_ult(...)        builder_->CreateICmpULT(__VA_ARGS__)
#define insert_elt(...)      builder_->CreateInsertElement(__VA_ARGS__)
#define intrinsic(...)       builder_->CreateIntrinsic(__VA_ARGS__)
#define load(...)            builder_->CreateLoad(__VA_ARGS__)
#define max_num(...)         builder_->CreateMaxNum(__VA_ARGS__)
#define min_num(...)         builder_->CreateMinNum(__VA_ARGS__)
#define mul(...)             builder_->CreateMul(__VA_ARGS__)
#define neg(...)             builder_->CreateNeg(__VA_ARGS__)
#define phi(...)             builder_->CreatePHI(__VA_ARGS__)
#define ret(...)             builder_->CreateRet(__VA_ARGS__)
#define select(...)          builder_->CreateSelect(__VA_ARGS__)
#define store(...)           builder_->CreateStore(__VA_ARGS__)
#define sub(...)             builder_->CreateSub(__VA_ARGS__)
#define udiv(...)            builder_->CreateUDiv(__VA_ARGS__)
#define urem(...)            builder_->CreateURem(__VA_ARGS__)
#define splat(...)           builder_->CreateVectorSplat(__VA_ARGS__)
#define xor_(...)            builder_->CreateXor(__VA_ARGS__)


/**
 * \brief Convert Triton-IR Type to LLVM-IR Type
 */
Type *generator::cvt(ir::type *ty) {
  // function
  if(auto* tt = dynamic_cast<ir::function_type*>(ty)){
    Type *ret_ty = cvt(tt->get_return_ty());
    std::vector<Type*> arg_tys(tt->get_num_params());
    for(size_t i = 0; i < arg_tys.size(); i++)
      arg_tys[i] = cvt(tt->get_param_ty(i));
    return FunctionType::get(ret_ty, arg_tys, false);
  }
  // pointer
  if(ty->is_pointer_ty()){
    Type *elt_ty = cvt(ty->get_pointer_element_ty());
    unsigned addr_space = ty->get_pointer_address_space();
    return ptr_ty(elt_ty, addr_space);
  }
  // integer
  if(ty->is_integer_ty()){
    unsigned bitwidth = ty->get_integer_bitwidth();
    return IntegerType::get(*ctx_, bitwidth);
  }
  // primitive types
  switch(ty->get_type_id()){
    case ir::type::VoidTyID:      return Type::getVoidTy(*ctx_);
    case ir::type::HalfTyID:      return Type::getHalfTy(*ctx_);
    case ir::type::FloatTyID:     return Type::getFloatTy(*ctx_);
    case ir::type::DoubleTyID:    return Type::getDoubleTy(*ctx_);
    case ir::type::X86_FP80TyID:  return Type::getX86_FP80Ty(*ctx_);
    case ir::type::PPC_FP128TyID: return Type::getPPC_FP128Ty(*ctx_);
    case ir::type::LabelTyID:     return Type::getLabelTy(*ctx_);
    case ir::type::MetadataTyID:  return Type::getMetadataTy(*ctx_);
    case ir::type::TokenTyID:     return Type::getTokenTy(*ctx_);
    default: break;
  }
  // unknown type
  throw std::runtime_error("unknown conversion from ir::type to Type");
}

/**
 * \brief Convert Triton-IR Attribute to LLVM-IR Attribute
 */
llvm::Attribute generator::cvt(ir::attribute attr) {
  switch(attr.get_kind()){
    case ir::noalias: return llvm::Attribute::get(*ctx_, llvm::Attribute::NoAlias);
    case ir::readonly: return llvm::Attribute::get(*ctx_, llvm::Attribute::ReadOnly);
    case ir::writeonly: return llvm::Attribute::get(*ctx_, llvm::Attribute::WriteOnly);
    case ir::aligned: return llvm::Attribute::get(*ctx_, llvm::Attribute::Alignment, attr.get_value());
    case ir::retune: return llvm::Attribute::get(*ctx_, llvm::Attribute::None);
    default: throw std::runtime_error("cannot convert ir::attribute_t to llvm::Attribute");
  }
}

/**
 * \brief Constructor of LLVM code generator
 */
generator::generator(analysis::axes *a_axes,
                    analysis::layouts *layouts,
                    analysis::align *alignment,
                    analysis::allocation *alloc,
                    analysis::swizzle *swizzle,
                    target *tgt,
                    unsigned num_warps)
  : a_axes_(a_axes), layouts_(layouts), alignment_(alignment), alloc_(alloc), swizzle_(swizzle),
    tgt_(tgt), num_warps_(num_warps) {

}

/**
 * \brief Code Generation for `value`
 */
void generator::visit_value(ir::value* v) {
  if(!seen_.insert(v).second)
    return;
  if(v->get_type()->is_block_ty()){
    if(analysis::shared_layout* layout = layouts_->get(v)->to_shared()){
      auto double_buffer = layout->get_double_buffer();
      // offset
      Value *offset = nullptr;
      if(double_buffer && v == double_buffer->phi)
        offset = shared_off_[layout];
      // base pointer
      Value *ptr = shared_ptr_[layout];
      if(double_buffer && v == double_buffer->latch)
        ptr = shared_next_ptr_[layout];
      else if(double_buffer && v == double_buffer->first)
        ptr = shared_pre_ptr_[layout];
      shmems_[v] = ptr;
      shoffs_[v] = offset;
    }
  }
  // visit operands
  BasicBlock *current = builder_->GetInsertBlock();
  auto *inst = dynamic_cast<ir::instruction*>(v);
  if(inst)
    for(ir::value *op: inst->ops()){
      if(dynamic_cast<ir::constant*>(op) || !dynamic_cast<ir::phi_node*>(v))
        visit_value(op);
    }
  init_idx(v);
  // change insert point for phi node
  builder_->SetInsertPoint(current);
  auto *phi = dynamic_cast<ir::phi_node*>(v);
  if(phi && !current->empty() && current->getFirstNonPHI())
    builder_->SetInsertPoint(&*current->getFirstNonPHI());
  // visit user
  if(auto *usr = dynamic_cast<ir::user*>(v)){
    usr->accept(this);
  }
  // revert insert point
  if(phi && !current->empty() && current->getFirstNonPHI())
    builder_->SetInsertPoint(current);
}

/**
 * \brief Code Generation for `phi`
 */
void generator::visit_phi_node(ir::phi_node* x) {
  Type *ty = cvt(x->get_type()->get_scalar_ty());
  for(indices_t idx: idxs_.at(x))
    vals_[x][idx] = phi(ty, x->get_num_operands());
}

/**
 * \brief Code Generation for `binary_operator`
 */
void generator::visit_binary_operator(ir::binary_operator*x) {
  auto cvt = [](ir::binary_op_t op){
    using ll = llvm::Instruction::BinaryOps;
    using tt = ir::binary_op_t;
    switch(op) {
      case tt::Add: return ll::Add;
      case tt::FAdd: return ll::FAdd;
      case tt::Sub: return ll::Sub;
      case tt::FSub: return ll::FSub;
      case tt::Mul: return ll::Mul;
      case tt::FMul: return ll::FMul;
      case tt::UDiv: return ll::UDiv;
      case tt::SDiv: return ll::SDiv;
      case tt::FDiv: return ll::FDiv;
      case tt::URem: return ll::URem;
      case tt::SRem: return ll::SRem;
      case tt::FRem: return ll::FRem;
      case tt::Shl: return ll::Shl;
      case tt::LShr: return ll::LShr;
      case tt::AShr: return ll::AShr;
      case tt::And: return ll::And;
      case tt::Or: return ll::Or;
      case tt::Xor: return ll::Xor;
      default: throw std::runtime_error("unreachable switch");
    }
  };
  for(indices_t idx: idxs_.at(x)){
    Value *lhs = vals_[x->get_operand(0)][idx];
    Value *rhs = vals_[x->get_operand(1)][idx];
    vals_[x][idx] = bin_op(cvt(x->get_op()), lhs, rhs);
  }
}

/**
 * \brief Code Generation for `getelementptr`
 */
void generator::visit_getelementptr_inst(ir::getelementptr_inst* x) {
  for(indices_t idx: idxs_.at(x)){
    Value *ptr = vals_[x->get_pointer_operand()][idx];
    std::vector<Value*> vals;
    for(auto it= x->idx_begin(); it != x->idx_end(); it++)
      vals.push_back(vals_[*it][idx]);
    Type *ty = cvt(x->get_source_elt_ty()->get_scalar_ty());
    vals_[x][idx] = gep(ty, ptr, vals);
  }
}

/**
 * \brief Code Generation for `icmp`
 */
void generator::visit_icmp_inst(ir::icmp_inst* x) {
  auto cvt = [](ir::cmp_pred_t pred) {
    using ll = llvm::CmpInst::Predicate;
    using tt = ir::cmp_pred_t;
    switch(pred){
      case tt::FIRST_ICMP_PREDICATE: return ll::FIRST_ICMP_PREDICATE;
      case tt::ICMP_EQ: return ll::ICMP_EQ;
      case tt::ICMP_NE: return ll::ICMP_NE;
      case tt::ICMP_UGT: return ll::ICMP_UGT;
      case tt::ICMP_UGE: return ll::ICMP_UGE;
      case tt::ICMP_ULT: return ll::ICMP_ULT;
      case tt::ICMP_ULE: return ll::ICMP_ULE;
      case tt::ICMP_SGT: return ll::ICMP_SGT;
      case tt::ICMP_SGE: return ll::ICMP_SGE;
      case tt::ICMP_SLT: return ll::ICMP_SLT;
      case tt::ICMP_SLE: return ll::ICMP_SLE;
      case tt::LAST_ICMP_PREDICATE: return ll::LAST_ICMP_PREDICATE;
      default: throw std::runtime_error("unreachable switch");
    }
  };

  for(indices_t idx: idxs_.at(x)){
    Value *lhs = vals_[x->get_operand(0)][idx];
    Value *rhs = vals_[x->get_operand(1)][idx];
    vals_[x][idx] = icmp(cvt(x->get_pred()), lhs, rhs);
  }
}

/**
 * \brief Code Generation for `fcmp`
 */
void generator::visit_fcmp_inst(ir::fcmp_inst* x) {
  auto cvt = [](ir::cmp_pred_t pred) {
    using ll = llvm::CmpInst::Predicate;
    using tt = ir::cmp_pred_t;
    switch(pred){
      case tt::FIRST_FCMP_PREDICATE: return ll::FIRST_FCMP_PREDICATE;
      case tt::FCMP_FALSE: return ll::FCMP_FALSE;
      case tt::FCMP_OEQ: return ll::FCMP_OEQ;
      case tt::FCMP_OGT: return ll::FCMP_OGT;
      case tt::FCMP_OGE: return ll::FCMP_OGE;
      case tt::FCMP_OLT: return ll::FCMP_OLT;
      case tt::FCMP_OLE: return ll::FCMP_OLE;
      case tt::FCMP_ONE: return ll::FCMP_ONE;
      case tt::FCMP_ORD: return ll::FCMP_ORD;
      case tt::FCMP_UNO: return ll::FCMP_UNO;
      case tt::FCMP_UEQ: return ll::FCMP_UEQ;
      case tt::FCMP_UGT: return ll::FCMP_UGT;
      case tt::FCMP_UGE: return ll::FCMP_UGE;
      case tt::FCMP_ULT: return ll::FCMP_ULT;
      case tt::FCMP_ULE: return ll::FCMP_ULE;
      case tt::FCMP_UNE: return ll::FCMP_UNE;
      case tt::FCMP_TRUE: return ll::FCMP_TRUE;
      case tt::LAST_FCMP_PREDICATE: return ll::LAST_FCMP_PREDICATE;
      default: throw std::runtime_error("unreachable switch");
    }
  };
  for(indices_t idx: idxs_.at(x)){
    Value *lhs = vals_[x->get_operand(0)][idx];
    Value *rhs = vals_[x->get_operand(1)][idx];
    vals_[x][idx] = fcmp(cvt(x->get_pred()), lhs, rhs);
  }
}

/**
 * \brief Code Generation for `cast`
 */
void generator::visit_cast_inst(ir::cast_inst* x) {
  Type *ty = cvt(x->get_type()->get_scalar_ty());
  auto cvt = [](ir::cast_op_t op){
    using ll = llvm::Instruction::CastOps;
    using tt = ir::cast_op_t;
    switch(op){
      case tt::Trunc: return ll::Trunc;
      case tt::ZExt: return ll::ZExt;
      case tt::SExt: return ll::SExt;
      case tt::FPTrunc: return ll::FPTrunc;
      case tt::FPExt: return ll::FPExt;
      case tt::UIToFP: return ll::UIToFP;
      case tt::SIToFP: return ll::SIToFP;
      case tt::FPToUI: return ll::FPToUI;
      case tt::FPToSI: return ll::FPToSI;
      case tt::PtrToInt: return ll::PtrToInt;
      case tt::IntToPtr: return ll::IntToPtr;
      case tt::BitCast: return ll::BitCast;
      case tt::AddrSpaceCast: return ll::AddrSpaceCast;
      default: throw std::runtime_error("unreachable switch");
    }
  };
  for(indices_t idx: idxs_.at(x)){
    Value *arg = vals_[x->get_operand(0)][idx];
    vals_[x][idx] = cast(cvt(x->get_op()), arg, ty);
  }
}

/**
 * \brief Code Generation for `return`
 */
void generator::visit_return_inst(ir::return_inst* rr) {
  ir::value *ret_val = rr->get_return_value();
  ret(ret_val ? vals_[ret_val][{}] : nullptr);
}

/**
 * \brief Code Generation for `cond_branch`
 */
void generator::visit_cond_branch_inst(ir::cond_branch_inst* br) {
  BasicBlock *true_dest  = bbs_.at(br->get_true_dest());
  BasicBlock *false_dest = bbs_.at(br->get_false_dest());
  Value *cond = vals_[br->get_cond()][{}];
  cond_br(cond, true_dest, false_dest);
}

/**
 * \brief Code Generation for `uncond_branch`
 */
void generator::visit_uncond_branch_inst(ir::uncond_branch_inst* br) {
  BasicBlock *dest = bbs_.at(br->get_dest());
  br(dest);
}

/**
 * \brief Code Generation for a (synchronous) `load`
 */
void generator::visit_load_inst(ir::load_inst* x){
  ir::value *op = x->get_pointer_operand();
  ir::masked_load_inst *mx = dynamic_cast<ir::masked_load_inst*>(x);
  Type* ty  = cvt(op->get_type()->get_scalar_ty()->get_pointer_element_ty());
  int space = op->get_type()->get_scalar_ty()->get_pointer_address_space();

  // compute vector width
  size_t vec = 1;
  if(op->get_type()->is_block_ty()){
    auto   ord = ords_.at(op);
    size_t aln = alignment_->get(op, ord[0]);
    size_t nts = layouts_->get(x)->to_scanline()->nts(ord[0]);
    vec = std::min(nts, aln);
  }

  // code generation
  auto idxs = idxs_.at(x);
  for(size_t i = 0; i < idxs.size(); i += vec){
    indices_t idx = idxs[i];
    // pointer value
    Value *ptr = bit_cast(vals_[op][idx], ptr_ty(vec_ty(ty, vec), space));
    // masked load
    Value *ret = nullptr;
    if(mx){
      // if mask:
      //   ret = load(ptr)
      // else:
      //   ret = false_value
      PHINode *_ret = phi(ptr->getType()->getPointerElementType(), 2);
      Instruction *then_term;
            Instruction *else_term;
            builder_->SetInsertPoint(_ret->getParent());
            Instruction* dummy = builder_->CreateRet(nullptr);
            llvm::SplitBlockAndInsertIfThenElse(vals_[mx->get_mask_operand()][idx], _ret, &then_term, &else_term);
            dummy->removeFromParent();
      builder_->SetInsertPoint(then_term);
      Value* then_ret = load(ptr);
      builder_->SetInsertPoint(else_term);
      Value* else_ret = splat(vec, vals_[mx->get_false_value_operand()][idx]);
      builder_->SetInsertPoint(_ret->getParent());
      _ret->addIncoming(then_ret, then_term->getParent());
      _ret->addIncoming(else_ret, else_term->getParent());
      ret = (Value*)_ret;
    }
    else
      ret = load(ptr);
    // write back
    for(size_t ii = 0; ii < vec; ii++)
      vals_[x][idxs[i+ii]] = extract_elt(ret, ii);
  }
}
void generator::visit_unmasked_load_inst(ir::unmasked_load_inst* x) {
  visit_load_inst(x);
}
void generator::visit_masked_load_inst(ir::masked_load_inst* x) {
  visit_load_inst(x);
}

/**
 * \brief Code Generation for a (synchronous) `store`
 */
void generator::visit_store_inst(ir::store_inst * x){
  ir::masked_store_inst *mx = dynamic_cast<ir::masked_store_inst*>(x);
  // operands
  ir::value *ptr_op = x->get_pointer_operand();
  ir::value *val_op = x->get_value_operand();
  // vector size
  size_t vec = 1;
  if(val_op->get_type()->is_block_ty()){
    auto ord = ords_.at(x->get_pointer_operand());
    size_t aln = alignment_->get(ptr_op, ord[0]);
    size_t nts = axes_.at(a_axes_->get(x->get_pointer_operand(), ord[0])).contiguous;
    vec  = std::min(nts, aln);
//    std::cout << aln << " " << nts << std::endl;
  }
  auto idxs    = idxs_.at(val_op);
  Type *ty = cvt(val_op->get_type()->get_scalar_ty());
  for(size_t i = 0; i < idxs.size(); i += vec){
    auto idx = idxs[i];
    // pointer
    Value *ptr = vals_[ptr_op][idx];
    ptr = bit_cast(ptr, vec_ty(ty, vec)->getPointerTo(1));
    // value
    Value* val = UndefValue::get(vec_ty(ty, vec));
    for(size_t ii = 0; ii < vec; ii++)
      val = insert_elt(val, vals_.at(val_op)[idxs[i + ii]], ii);
    if(mx){
      Value *msk = vals_[mx->get_mask_operand()][idx];
      Instruction *no_op = intrinsic(Intrinsic::donothing, {}, {});
      Instruction *term = llvm::SplitBlockAndInsertIfThen(msk, no_op, false);
      builder_->SetInsertPoint(term);
      store(val, ptr);
      builder_->SetInsertPoint(no_op);
    }
    else
      store(val, ptr);
  }
}
void generator::visit_unmasked_store_inst(ir::unmasked_store_inst* x) {
  visit_store_inst(x);
}
void generator::visit_masked_store_inst(ir::masked_store_inst* x) {
  visit_store_inst(x);
}

/**
 * \brief Code Generation for `reshape`
 */
void generator::visit_reshape_inst(ir::reshape_inst* x) {
  auto idxs = idxs_.at(x);
  for(size_t i = 0; i < idxs_.at(x).size(); i ++){
    ir::value* op = x->get_operand(0);
    vals_[x][idxs_[x][i]] = vals_[op][idxs_[op][i]];
  };
}

/**
 * \brief Code Generation for `splat`
 */
void generator::visit_splat_inst(ir::splat_inst* x) {
  for(auto idx: idxs_.at(x))
    vals_[x][idx] = vals_[x->get_operand(0)][{}];
}

/**
 * \brief Code Generation for `broadcast`
 */
void generator::visit_broadcast_inst(ir::broadcast_inst* x) {
  ir::value* op = x->get_operand(0);
  const auto& shape = op->get_type()->get_block_shapes();
  for(auto out_idx: idxs_.at(x)){
    indices_t in_idx = out_idx;
    for(size_t k = 0; k < in_idx.size(); k++)
      in_idx[k] = shape[k] == 1 ? i32(0) : in_idx[k];
    vals_[x][out_idx] = vals_[op][in_idx];
  }
}

/**
 * \brief Code Generation for `downcast`
 */
void generator::visit_downcast_inst(ir::downcast_inst* x) {
  vals_[x][{}] = vals_[x->get_operand(0)][{i32(0)}];
}

/**
 * \brief Code Generation for `get_program_id`
 */
void generator::visit_get_program_id_inst(ir::get_program_id_inst* pid) {
  Module *module = builder_->GetInsertBlock()->getModule();
  Value *ret = tgt_->get_block_id(module, *builder_, pid->get_axis());
  vals_[pid][{}] = ret;
}

/**
 * \brief Code Generation for `get_num_program`
 */
void generator::visit_get_num_program_inst(ir::get_num_program_inst* np) {
  Module *module = builder_->GetInsertBlock()->getModule();
  Value *ret = tgt_->get_num_blocks(module, *builder_, np->get_axis());
  vals_[np][{}] = ret;
}

/**
 * \brief Code Generation for `exp`
 */
void generator::visit_exp_inst(ir::exp_inst* x){
  Constant *log2e = ConstantFP::get(f32_ty, 1.4426950408889634);
  std::vector<llvm::Type*> tys = {f32_ty};
  FunctionType *fn_ty = FunctionType::get(f32_ty, tys, false);
  InlineAsm *ex2 = InlineAsm::get(fn_ty, "ex2.approx.f32 $0, $1;", "=f,f", false);
  for(auto idx: idxs_.at(x)){
    Value *ex2arg = fmul(vals_[x->get_operand(0)][idx], log2e);
    vals_[x][idx] = call(ex2, std::vector<llvm::Value*>{ex2arg});
  }
}

/**
 * \brief Code Generation for `log`
 */
void generator::visit_log_inst(ir::log_inst* x){
  Constant *rcplog2e = ConstantFP::get(f32_ty, 0.6931471805599453);
  std::vector<llvm::Type*> tys = {f32_ty};
  FunctionType *fn_ty = FunctionType::get(f32_ty, tys, false);
  InlineAsm *lg2 = InlineAsm::get(fn_ty, "lg2.approx.f32 $0, $1;", "=f,f", false);
  for(auto idx: idxs_.at(x)){
    Value *lg2arg = call(lg2, std::vector<llvm::Value*>{vals_[x->get_operand(0)][idx]});
    vals_[x][idx] = fmul(lg2arg, rcplog2e);
  }
}

/**
 * \brief Code Generation for `atomic_cas`
 */
void generator::visit_atomic_cas_inst(ir::atomic_cas_inst* cas) {
  BasicBlock *current = builder_->GetInsertBlock();
  Module *module = current->getModule();
  Value *tid = tgt_->get_local_id(module, *builder_, 0);
  Value *pred = icmp_eq(tid, i32(0));
  BasicBlock *tid_0_bb = BasicBlock::Create(*ctx_, "tid_0", current->getParent());
  BasicBlock *tid_0_done_bb = BasicBlock::Create(*ctx_, "tid_0_done", current->getParent());
  add_barrier();
  tgt_->add_memfence(module, *builder_);
  cond_br(pred, tid_0_bb, tid_0_done_bb);
  builder_->SetInsertPoint(tid_0_bb);
  Value *cas_ptr = vals_[cas->get_operand(0)][{}];
  Value *cas_cmp = vals_[cas->get_operand(1)][{}];
  Value *cas_val = vals_[cas->get_operand(2)][{}];
  Value *old = atomic_cmp_xchg(cas_ptr, cas_cmp, cas_val, AtomicOrdering::Monotonic, AtomicOrdering::Monotonic);
  old = extract_val(old, std::vector<unsigned>{0});
  Value *atom_ptr;
  atom_ptr = gep(shmem_, i32(alloc_->offset(layouts_->get(layouts_->tmp(cas)))), "");
  atom_ptr = bit_cast(atom_ptr, ptr_ty(old->getType(), 3));
  store(old, atom_ptr);
  br(tid_0_done_bb);
  builder_->SetInsertPoint(tid_0_done_bb);
  tgt_->add_memfence(module, *builder_);
  add_barrier();
  vals_[cas][{}] = load(atom_ptr);
}

/**
 * \brief Code Generation for `atomic_exch`
 */
void generator::visit_atomic_exch_inst(ir::atomic_exch_inst* xchg) {
  BasicBlock *current = builder_->GetInsertBlock();
  Module *module = current->getModule();
  Value *rmw_ptr = vals_[xchg->get_operand(0)][{}];
  Value *rmw_val = vals_[xchg->get_operand(1)][{}];
  Value *tid = tgt_->get_local_id(module, *builder_, 0);
  Value *pred = icmp_eq(tid, i32(0));
  BasicBlock *tid_0_bb = BasicBlock::Create(*ctx_, "tid_0", current->getParent());
  BasicBlock *tid_0_done_bb = BasicBlock::Create(*ctx_, "tid_0_done", current->getParent());
  tgt_->add_memfence(module, *builder_);
  add_barrier();
  cond_br(pred, tid_0_bb, tid_0_done_bb);
  builder_->SetInsertPoint(tid_0_bb);
  atomic_rmw(AtomicRMWInst::Xchg, rmw_ptr, rmw_val, AtomicOrdering::Monotonic, SyncScope::System);
  br(tid_0_done_bb);
  builder_->SetInsertPoint(tid_0_done_bb);
  tgt_->add_memfence(module, *builder_);
}

/**
 * \brief Code Generation for `atomic_add`
 */
//TODO: clean-up
void generator::visit_atomic_add_inst(ir::atomic_add_inst* add) {

  if(add->get_type()->is_block_ty()){
    ir::value* ptr = add->get_operand(0);
    ir::value* val = add->get_operand(1);
    ir::value* msk = add->get_operand(2);

    // vector size
    int vec = 1;
    int ld = ords_.at(ptr)[0];
    unsigned alignment = alignment_->get(ptr, ld);
    vec = std::min<int>(layouts_->get(ptr)->to_scanline()->nts(ld), alignment);
    vec = std::min(vec, val->get_type()->get_tile_element_ty()->is_half_ty() ? 2 : 1);

    for(int i = 0; i < idxs_.at(val).size(); i += vec){
      auto idx = idxs_[val][i];
      Value *rmw_val = UndefValue::get(vec_ty(vals_[val][idx]->getType(), vec));
      for(int ii = 0; ii < vec; ii++)
        rmw_val = insert_elt(rmw_val, vals_[val][idxs_[val][i+ii]], ii);
      Value *rmw_ptr = vals_[ptr][idx];
      Value *rmw_msk = vals_[msk][idx];
      if(vec == 1)
        rmw_val = extract_elt(rmw_val, i32(0));
      Type* ty = rmw_val->getType();
      size_t nbits = ty->getScalarSizeInBits();
      // extract pointer offset
      std::string offset = "";
      if(GetElementPtrInst *gep = dyn_cast<GetElementPtrInst>(rmw_ptr))
      if(gep->getNumIndices() == 1)
      if(ConstantInt *cst = dyn_cast<ConstantInt>(gep->idx_begin())){
        offset = " + " + std::to_string(cst->getValue().getSExtValue()*nbits/8);
        rmw_ptr = gep->getPointerOperand();
      }
      rmw_ptr = bit_cast(rmw_ptr, ty->getPointerTo(1));
      // asm argument type
      std::vector<Type*> arg_ty = {rmw_msk->getType(), rmw_ptr->getType(), rmw_val->getType()};
      // asm function type
      FunctionType *fn_ty = FunctionType::get(ty, arg_ty, false);
      // asm string
      std::string suffix = vec == 2 ? "x2" : "";
      std::string mod = nbits == 32 ? "" : ".noftz";
      std::string asm_str = "@$0 atom.global.gpu.add" + mod + ".f" + std::to_string(nbits) + suffix + " $1, [$2" + offset + "], $3;";
      std::string ty_id = nbits == 32 ? "f" : (vec == 1 ? "h" : "r");
      std::string constraint = "b,=" + ty_id + ",l," + ty_id;
      // create inline asm
      InlineAsm *iasm = InlineAsm::get(fn_ty, asm_str, constraint, true);
      // call asm
      call(iasm, (ArrayRef<Value*>{rmw_msk, rmw_ptr, rmw_val}));
    }
  }
  else{
    Value *rmw_ptr = vals_[add->get_operand(0)][{}];
    Value *rmw_val = vals_[add->get_operand(1)][{}];
    Value *rmw_msk = vals_[add->get_operand(2)][{}];
    Type* ty = rmw_val->getType();
    size_t nbits = ty->getScalarSizeInBits();
    std::vector<Type*> arg_ty = {rmw_msk->getType(), rmw_ptr->getType(), rmw_val->getType()};
    FunctionType *fn_ty = FunctionType::get(ty, arg_ty, false);
    std::string mod = nbits == 32 ? "" : ".noftz";
    std::string asm_str = "@$0 atom.global.gpu.add" + mod + ".f" + std::to_string(nbits) + " $1, [$2], $3;";
    std::string ty_id = nbits == 32 ? "f" : "h";
    InlineAsm *iasm = InlineAsm::get(fn_ty, asm_str, "b,="+ty_id+",l,"+ty_id, true);

    BasicBlock *current = builder_->GetInsertBlock();
    Module *module = current->getModule();

    Value *tid = tgt_->get_local_id(module, *builder_, 0);
    Value *pred = icmp_eq(tid, i32(0));
    BasicBlock *tid_0_bb = BasicBlock::Create(*ctx_, "tid_0", current->getParent());
    BasicBlock *tid_0_done_bb = BasicBlock::Create(*ctx_, "tid_0_done", current->getParent());
    tgt_->add_memfence(module, *builder_);
    add_barrier();
    cond_br(pred, tid_0_bb, tid_0_done_bb);
    builder_->SetInsertPoint(tid_0_bb);
    call(iasm, (ArrayRef<Value*>{rmw_msk, rmw_ptr, rmw_val}));
    br(tid_0_done_bb);
    builder_->SetInsertPoint(tid_0_done_bb);
    tgt_->add_memfence(module, *builder_);
  }
}

/**
 * \brief Code Generation for `mma.884` (V100)
 */
//TODO: clean-up
void generator::visit_mma884(ir::dot_inst* C, ir::value *A, ir::value *B, ir::value *D, unsigned NK) {
  // shapes
  auto shape_c = C->get_type()->get_block_shapes();
  auto shape_a = A->get_type()->get_block_shapes();
  auto shape_b = B->get_type()->get_block_shapes();
  // order
  auto ord_a = layouts_->get(A)->get_order();
  auto ord_b = layouts_->get(B)->get_order();
  // layouts
  analysis::mma_layout*    layout_c = layouts_->get(C)->to_mma();
  analysis::shared_layout* layout_a = layouts_->get(A)->to_shared();
  analysis::shared_layout* layout_b = layouts_->get(B)->to_shared();
  // vectorization
  int vec_a = swizzle_->get_vec(layout_a);
  int vec_b = swizzle_->get_vec(layout_b);
  // strides
  bool is_a_row = ord_a[0] != 0;
  bool is_b_row = ord_b[0] != 0;
  int stride_am = is_a_row ? shape_a[1] : 1;
  int stride_ak = is_a_row ? 1 : shape_a[0];
  int stride_a0 = is_a_row ? stride_ak : stride_am;
  int stride_a1 = is_a_row ? stride_am : stride_ak;
  int stride_bn = is_b_row ? 1 : shape_b[0];
  int stride_bk = is_b_row ? shape_b[1] : 1;
  int stride_b0 = is_b_row ? stride_bn : stride_bk;
  int stride_b1 = is_b_row ? stride_bk : stride_bn;
  int stride_rep_m = layout_c->wpt(0) * layout_c->fpw(0) * 8;
  int stride_rep_n = layout_c->wpt(1) * layout_c->fpw(1) * 8;
  int stride_rep_k = 1;
  // swizzling
  int per_phase_a = swizzle_->get_per_phase(layout_a);
  int max_phase_a = swizzle_->get_max_phase(layout_a);
  int step_a0   = is_a_row ? stride_rep_k : stride_rep_m;
  int num_ptr_a = std::max(2 * per_phase_a * max_phase_a / step_a0, 1);
  int per_phase_b = swizzle_->get_per_phase(layout_b);
  int max_phase_b = swizzle_->get_max_phase(layout_b);
  int step_b0   = is_b_row ? stride_rep_n : stride_rep_k;
  int num_ptr_b = std::max(2 * per_phase_b * max_phase_b / step_b0, 1);

  /* --------------------------------- */
  /* --- pre-compute pointer lanes --- */
  /* --------------------------------- */
  BasicBlock* curr_bb = builder_->GetInsertBlock();
  BasicBlock* entry = &curr_bb->getParent()->getEntryBlock();
  builder_->SetInsertPoint(entry->getTerminator());
  Value* off_a0 = is_a_row ? offset_a_k_[layout_c] : offset_a_m_[layout_c];
  Value* off_a1 = is_a_row ? offset_a_m_[layout_c] : offset_a_k_[layout_c];
  Value* phase_a = urem(udiv(off_a1, i32(per_phase_a)), i32(max_phase_a));
  std::vector<Value*> off_a(num_ptr_a);
  for(int i = 0; i < num_ptr_a; i++){
    Value* off_a0i = add(off_a0, i32(i*(is_a_row?4:stride_rep_m)));
    off_a0i = exact_udiv(off_a0i, i32(vec_a));
    off_a0i = xor_(off_a0i, phase_a);
    off_a0i = mul(off_a0i, i32(vec_a));
    off_a[i] = add(mul(off_a0i, i32(stride_a0)), mul(off_a1, i32(stride_a1)));
  }
  Value* off_b0 = is_b_row ? offset_b_n_[layout_c] : offset_b_k_[layout_c];
  Value* off_b1 = is_b_row ? offset_b_k_[layout_c] : offset_b_n_[layout_c];
  Value* phase_b = urem(udiv(off_b1, i32(per_phase_b)), i32(max_phase_b));
  std::vector<Value*> off_b(num_ptr_b);
  for(int i = 0; i < num_ptr_b; i++){
    Value* off_b0i = add(off_b0, i32(i*(is_b_row?stride_rep_n:4)));
    off_b0i = udiv(off_b0i, i32(vec_b));
    off_b0i = xor_(off_b0i, phase_b);
    off_b0i = mul(off_b0i, i32(vec_b));
    off_b[i] = add(mul(off_b0i, i32(stride_b0)), mul(off_b1, i32(stride_b1)));
  }
  builder_->SetInsertPoint(curr_bb);

  /* --------------------------------- */
  /* ---       MMA intrinsic       --- */
  /* --------------------------------- */
  Type *f16x2_ty = vec_ty(f16_ty, 2);
  Type *ret_ty = StructType::get(*ctx_, {f32_ty, f32_ty, f32_ty, f32_ty, f32_ty, f32_ty, f32_ty, f32_ty});
  std::vector<Type*> arg_ty = {f16x2_ty, f16x2_ty, f16x2_ty, f16x2_ty,
                               f32_ty, f32_ty, f32_ty, f32_ty, f32_ty, f32_ty, f32_ty, f32_ty};
  InlineAsm *mma = InlineAsm::get(FunctionType::get(ret_ty, arg_ty, false),
                                             " mma.sync.aligned.m8n8k4."
                                             + std::string(is_a_row ? "row" : "col")
                                             + "."
                                             + std::string(is_b_row ? "row" : "col")
                                             + ".f32.f16.f16.f32 "
                                             "{$0, $1, $2, $3, $4, $5, $6, $7}, "
                                             "{$8, $9}, "
                                             "{$10, $11}, "
                                             "{$0, $1, $2, $3, $4, $5, $6, $7};", "=f,=f,=f,=f,=f,=f,=f,=f,r,r,r,r,0,1,2,3,4,5,6,7", false);


  std::vector<Value*> ptr_a(num_ptr_a);
  std::vector<Value*> ptr_b(num_ptr_b);
  std::map<std::pair<int, int>, std::pair<Value*, Value*>> has, hbs;
  for(int i = 0; i < num_ptr_a; i++)
    ptr_a[i] = gep(shmems_[A], off_a[i]);
  for(int i = 0; i < num_ptr_b; i++)
    ptr_b[i] = gep(shmems_[B], off_b[i]);


  // initialize accumulators
  std::vector<Value*> acc;
  for(indices_t idx: idxs_.at(C))
    acc.push_back(vals_[D][idx]);

  // update accumulators
  unsigned num_m = layout_c->rep(0) * shape_c[0] / layout_c->spt(0);
  unsigned num_n = layout_c->rep(1) * shape_c[1] / layout_c->spt(1);
  for(unsigned m = 0; m < num_m/2; m++)
  for(unsigned n = 0; n < num_n/2; n++)
  for(unsigned K = 0; K < NK; K += 4){
    if(has.find({m, K}) == has.end()){
      Value* ptra = ptr_a[(is_a_row ? K/4 : m) % num_ptr_a];
      int step_am = is_a_row ? m : m / (num_ptr_a)*(num_ptr_a);
      int step_ak = is_a_row ? K / (num_ptr_a*vec_a)*(num_ptr_a*vec_a) : K;
      Value* pa =  gep(ptra, i32(step_am*stride_rep_m*stride_am + step_ak*stride_ak));
      Value* ha = load(bit_cast(pa, ptr_ty(vec_ty(i32_ty, vec_a/2), 3)));
      Value *ha00 = bit_cast(extract_elt(ha, i32(0)), f16x2_ty);
      Value *ha01 = bit_cast(extract_elt(ha, i32(1)), f16x2_ty);
      has[{m, K}]   = {ha00, ha01};
      if(vec_a > 4){
        Value *ha10 = bit_cast(extract_elt(ha, i32(2)), f16x2_ty);
        Value *ha11 = bit_cast(extract_elt(ha, i32(3)), f16x2_ty);
        if(is_a_row)
          has[{m, K+4}] = {ha10, ha11};
        else
          has[{m+1, K}] = {ha10, ha11};
      }
    }
    if(hbs.find({n, K}) == hbs.end()){
      Value* ptrb = ptr_b[(is_b_row? n : K/4) % num_ptr_b];
      int stepbn = is_b_row ? n / (num_ptr_b)*(num_ptr_b) : n;
      int stepbk = is_b_row ? K : K / (num_ptr_b*vec_b)*(num_ptr_b*vec_b);
      Value* pb =   gep(ptrb, i32(stepbn*stride_rep_n*stride_bn + stepbk*stride_bk));
      Value* hb =   load(bit_cast(pb, ptr_ty(vec_ty(i32_ty, vec_b/2), 3)));
      Value *hb00 = bit_cast(extract_elt(hb, i32(0)), f16x2_ty);
      Value *hb01 = bit_cast(extract_elt(hb, i32(1)), f16x2_ty);
      hbs[{n, K}]   = {hb00, hb01};
      if(vec_b > 4){
        Value *hb10 = bit_cast(extract_elt(hb, i32(2)), f16x2_ty);
        Value *hb11 = bit_cast(extract_elt(hb, i32(3)), f16x2_ty);
        if(is_b_row)
          hbs[{n+1, K}] = {hb10, hb11};
        else
          hbs[{n, K+4}] = {hb10, hb11};
      }
    }
    auto ha = has[{m, K}];
    auto hb = hbs[{n, K}];
    // arguments
    std::vector<size_t> idx = {
      (m*2 + 0) + (n*4 + 0)*num_m, (m*2 + 0) + (n*4 + 1)*num_m,
      (m*2 + 1) + (n*4 + 0)*num_m, (m*2 + 1) + (n*4 + 1)*num_m,
      (m*2 + 0) + (n*4 + 2)*num_m, (m*2 + 0) + (n*4 + 3)*num_m,
      (m*2 + 1) + (n*4 + 2)*num_m, (m*2 + 1) + (n*4 + 3)*num_m
    };
    std::vector<Value*> args = {ha.first, ha.second, hb.first, hb.second};
    for(unsigned i = 0; i < 8; i++)
      args.push_back(acc[idx[i]]);
    // execute mma
    Value *nc = call(mma, args);
    // unpack
    for(unsigned i = 0; i < 8; i++)
      acc[idx[i]] = extract_val(nc, {i});
  }

  // write back accumulators
  for(size_t i = 0; i < idxs_.at(C).size(); i++)
    vals_[C][idxs_[C][i]] = acc[i];
}

/**
 * \brief Code Generation for `mma.16816` (A100)
 */
//TODO: clean-up
void generator::visit_mma16816(ir::dot_inst* dot, ir::value *A, ir::value *B, ir::value *D, unsigned NK) {
  const auto& shapes = dot->get_type()->get_block_shapes();

  std::map<std::vector<Value*>, std::vector<Value*>> fcs;

  for(indices_t idx: idxs_.at(dot)){
    std::vector<Value*> key(idx.size() - 2);
    std::copy(idx.begin() + 2, idx.end(), key.begin());
    fcs[key].push_back(vals_[D][idx]);
  };

  auto shape_a = A->get_type()->get_block_shapes();
  auto shape_b = B->get_type()->get_block_shapes();
  auto ord_a = layouts_->get(A)->get_order();
  auto ord_b = layouts_->get(B)->get_order();
  analysis::mma_layout* layout = layouts_->get(dot)->to_mma();
  analysis::shared_layout* layout_a = (analysis::shared_layout*)layouts_->get(dot->get_operand(0));
  analysis::shared_layout* layout_b = (analysis::shared_layout*)layouts_->get(dot->get_operand(1));
  bool is_a_row = ord_a[0] == 1;
  bool is_b_row = ord_b[0] == 1;
  std::string a_trans = is_a_row ? "" : ".trans";
  std::string b_trans = is_b_row ? ".trans" : "";
  int stride_a_m = is_a_row ? shape_a[1] : 1;
  int stride_a_k = is_a_row ? 1 : shape_a[0];
  int stride_b_n = is_b_row ? 1 : shape_b[0];
  int stride_b_k = is_b_row ? shape_b[1] : 1;
  int stride_a0 = is_a_row ? stride_a_k : stride_a_m;
  int stride_a1 = is_a_row ? stride_a_m : stride_a_k;
  int stride_b0 = is_b_row ? stride_b_n : stride_b_k;
  int stride_b1 = is_b_row ? stride_b_k : stride_b_n;
  int lda = is_a_row ? stride_a_m : stride_a_k;
  int ldb = is_b_row ? stride_b_k : stride_b_n;
  int per_phase_a = swizzle_->get_per_phase(layout_a);
  int max_phase_a = swizzle_->get_max_phase(layout_a);
  int per_phase_b = swizzle_->get_per_phase(layout_b);
  int max_phase_b = swizzle_->get_max_phase(layout_b);
  int num_ptr_a   = 8;
  int num_ptr_b   = 8;
  int vec_a = 8;
  int vec_b = 8;



  Type *fp32_ty = f32_ty;
  Type *fp16x2_ty = vec_ty(f16_ty, 2);
  Type *fp16x2_pack4_ty = StructType::get(*ctx_, std::vector<llvm::Type*>{fp16x2_ty, fp16x2_ty, fp16x2_ty, fp16x2_ty});
  Type *fp32_pack4_ty = StructType::get(*ctx_, std::vector<llvm::Type*>{fp32_ty, fp32_ty, fp32_ty, fp32_ty});
  FunctionType *ld_x4_ty = FunctionType::get(fp16x2_pack4_ty, std::vector<llvm::Type*>{ptr_ty(f16_ty, 3)}, false);

  // left-hand-side values
  std::map<std::pair<unsigned, unsigned>, std::pair<Value*, Value*>> ha;
  std::map<std::pair<unsigned, unsigned>, Value*> hb;


  BasicBlock* CurrBB = builder_->GetInsertBlock();
  BasicBlock* FirstBB = &CurrBB->getParent()->getEntryBlock();
  if(FirstBB != CurrBB)
    builder_->SetInsertPoint(FirstBB->getTerminator());

  Value* thread = tgt_->get_local_id(mod_, *builder_, 0);
  Value *lane   = urem(thread, i32(32));
  Value *warp   = udiv(thread, i32(32));
  Value *warp12 = udiv(warp, i32(layout->wpt(0)));
  Value *warp0  = urem(warp, i32(layout->wpt(0)));
  Value *warp1  = urem(warp12, i32(layout->wpt(1)));
  std::vector<Value *>& fc = fcs.begin()->second;

  Value *tidr8  = urem(lane, i32(8));
  Value *phase_a = urem(udiv(tidr8, i32(per_phase_a)), i32(max_phase_a));
  Value* off_a0   = mul(tidr8, i32(lda));
  Value *off_am  = mul(add(urem(udiv(lane, i32(8)), i32(2)), mul(warp0, i32(2))), i32(8));
  Value *off_ak  = mul(udiv(lane, i32(16)), i32(8));
  off_am = urem(off_am, i32(shape_a[0]));
  off_ak = urem(off_ak, i32(shape_a[1]));
  off_a0 = add(off_a0, is_a_row ? off_ak : off_am);
  Value* off_a1 = is_a_row ? off_am : off_ak;
  std::vector<Value*> off_a(num_ptr_a);
  for(int i = 0; i < num_ptr_a; i++){
    Value* off_a0i = add(off_a0, i32(i*16*(is_a_row?1:layout->wpt(0))));
    off_a0i = exact_udiv(off_a0i, i32(vec_a));
    off_a0i = xor_(off_a0i, phase_a);
    off_a0i = mul(off_a0i, i32(vec_a));
    off_a[i] = add(mul(off_a0i, i32(stride_a0)), mul(off_a1, i32(stride_a1)));
  }

  Value *phase_b = urem(udiv(tidr8, i32(per_phase_b)), i32(max_phase_b));
  Value* off_b0   = mul(tidr8, i32(ldb));
  Value *off_bn  = mul(add(mul(udiv(lane, i32(16)), i32(layout->wpt(1))), mul(warp1, i32(1))), i32(8));
  Value *off_bk  = mul(urem(udiv(lane, i32(8)), i32(2)), i32(8));
  off_bn = urem(off_bn, i32(shape_b[1]));
  off_bk = urem(off_bk, i32(shape_b[0]));
  off_b0 = add(off_b0, is_b_row ? off_bn : off_bk);
  Value* off_b1 = is_b_row ? off_bk : off_bn;
  std::vector<Value*> off_b(num_ptr_b);
  for(int i = 0; i < num_ptr_b; i++){
    Value* off_b0i = add(off_b0, i32(i*(is_b_row?8*layout->wpt(1):16)));
    off_b0i = exact_udiv(off_b0i, i32(vec_b));
    off_b0i = xor_(off_b0i, phase_b);
    off_b0i = mul(off_b0i, i32(vec_b));
    off_b[i] = add(mul(off_b0i, i32(stride_b0)), mul(off_b1, i32(stride_b1)));
  }

  builder_->SetInsertPoint(CurrBB);
  // A pointer
  std::vector<Value*> ptrs_a(num_ptr_a);
  for(int i = 0; i < num_ptr_a; i++)
    ptrs_a[i] = gep(shmems_[A], {off_a[i]});
  // B pointer
  std::vector<Value*> ptrs_b(num_ptr_b);
  for(int i = 0; i < num_ptr_b; i++)
    ptrs_b[i] = gep(shmems_[B], {off_b[i]});

  FunctionType *mma_ty = FunctionType::get(fp32_pack4_ty, std::vector<llvm::Type*>{fp16x2_ty, fp16x2_ty, fp16x2_ty, fp16x2_ty, fp16x2_ty, fp16x2_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty}, false);
  InlineAsm *mma_fn = InlineAsm::get(mma_ty, "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                                             "{$0, $1, $2, $3}, "
                                             "{$4, $5, $6, $7}, "
                                             "{$8, $9}, "
                                             "{$10, $11, $12, $13};", "=f,=f,=f,=f,r,r,r,r,r,r,0,1,2,3", false);
  unsigned num_rep_0 = shapes[0] / layout->spt(0);
  unsigned num_rep_1 = shapes[1] / layout->spt(1);
  for(unsigned K = 0; K < NK; K += 16)
  for(unsigned m = 0; m < num_rep_0; m++)
  for(unsigned n = 0; n < num_rep_1; n++){
    if(ha.find({m, K}) == ha.end()){
      Value* ptra = ptrs_a[(is_a_row ? K/16 : m) % num_ptr_a];
      int step_am = is_a_row ? m : m / (num_ptr_a)*(num_ptr_a);
      int step_ak = is_a_row ? K / (num_ptr_a*16)*(num_ptr_a*16) : K;
      InlineAsm *ld_a0_fn = InlineAsm::get(ld_x4_ty, "ldmatrix.sync.aligned.m8n8.x4" + a_trans + ".shared.b16 "
                                                "{$0, $1, $2, $3}, [$4 + " + std::to_string(2*step_am*16*layout->wpt(0)*stride_a_m + 2*step_ak*stride_a_k) + "];", "=r,=r,=r,=r,r", false);
      Value *haa = call(ld_x4_ty, ld_a0_fn, {ptra});
      Value *ha0 = extract_val(haa, std::vector<unsigned>{0});
      Value *ha1 = extract_val(haa, std::vector<unsigned>{1});
      Value *ha2 = extract_val(haa, std::vector<unsigned>{2});
      Value *ha3 = extract_val(haa, std::vector<unsigned>{3});
      ha[{m, K}] = std::make_pair(ha0, ha1);
      ha[{m, K+8}] = std::make_pair(ha2, ha3);
    }
    if(hb.find({n, K})==hb.end()){
      Value* ptrb = ptrs_b[(is_b_row ? n : K/16) % num_ptr_b];
      int step_bn = is_b_row ? n / (num_ptr_b)*(num_ptr_b) : n;
      int step_bk = is_b_row ? K : K / (num_ptr_b*8)*(num_ptr_b*8);
      InlineAsm *ld_b_fn = InlineAsm::get(ld_x4_ty, "ldmatrix.sync.aligned.m8n8.x4" + b_trans + ".shared.b16 "
                                                "{$0, $1, $2, $3}, [$4 + " + std::to_string(2*step_bn*8*layout->wpt(1)*stride_b_n + 2*step_bk*stride_b_k) + "];", "=r,=r,=r,=r,r", false);
      Value *hbb = call(ld_x4_ty, ld_b_fn, {ptrb});
      Value *hb0 = extract_val(hbb, std::vector<unsigned>{0});
      Value *hb1 = extract_val(hbb, std::vector<unsigned>{1});
      Value *hb2 = extract_val(hbb, std::vector<unsigned>{2});
      Value *hb3 = extract_val(hbb, std::vector<unsigned>{3});
      hb[{n, K}] = hb0;
      hb[{n+1, K}] = hb2;
      hb[{n, K+8}] = hb1;
      hb[{n+1, K+8}] = hb3;
    }
    unsigned cols_per_thread = num_rep_0 * 2;
    std::vector<size_t> idx = {
      (m*2 + 0) + (n*2 + 0)*cols_per_thread,
      (m*2 + 0) + (n*2 + 1)*cols_per_thread,
      (m*2 + 1) + (n*2 + 0)*cols_per_thread,
      (m*2 + 1) + (n*2 + 1)*cols_per_thread
    };
    Value *nc = call(mma_ty, mma_fn, {ha[{m, K}].first, ha[{m, K}].second,ha[{m, K+8}].first, ha[{m, K+8}].second,
                                                      hb[{n, K}], hb[{n, K+8}],
                                                      fc[idx[0]], fc[idx[1]], fc[idx[2]], fc[idx[3]]});
    fc[idx[0]] = extract_val(nc, std::vector<unsigned>{0});
    fc[idx[1]] = extract_val(nc, std::vector<unsigned>{1});
    fc[idx[2]] = extract_val(nc, std::vector<unsigned>{2});
    fc[idx[3]] = extract_val(nc, std::vector<unsigned>{3});
  }

  // write back
  unsigned i = 0;
  for(indices_t idx: idxs_.at(dot)){
    std::vector<Value*> key(idx.size() - 2);
    std::copy(idx.begin() + 2, idx.end(), key.begin());
    if(i >= fcs.at(key).size())
      i = 0;
    vals_[dot][idx] = fcs.at(key)[i++];
  };
}

/**
 * \brief Code Generation for FMA-based `dot` (FP32, FP64, Default)
 */
void generator::visit_fmadot(ir::dot_inst* C, ir::value* A, ir::value* B, ir::value* D, unsigned NK, Type *c_ty, Function *f_mul_add) {
  auto shape_c = C->get_type()->get_block_shapes();
  auto shape_a = A->get_type()->get_block_shapes();
  auto shape_b = B->get_type()->get_block_shapes();
  auto ord_a = layouts_->get(A)->get_order();
  auto ord_b = layouts_->get(B)->get_order();
  analysis::scanline_layout* layout_c = layouts_->get(C)->to_scanline();
  analysis::shared_layout* layout_a = (analysis::shared_layout*)layouts_->get(C->get_operand(0));
  analysis::shared_layout* layout_b = (analysis::shared_layout*)layouts_->get(C->get_operand(1));
  bool is_a_row = ord_a[0] == 1;
  bool is_b_row = ord_b[0] == 1;
  std::string a_trans = is_a_row ? "" : ".trans";
  std::string b_trans = is_b_row ? ".trans" : "";
  int stride_a_m = is_a_row ? shape_a[1] : 1;
  int stride_a_k = is_a_row ? 1 : shape_a[0];
  int stride_b_n = is_b_row ? 1 : shape_b[0];
  int stride_b_k = is_b_row ? shape_b[1] : 1;
  int stride_a0 = is_a_row ? stride_a_k : stride_a_m;
  int stride_a1 = is_a_row ? stride_a_m : stride_a_k;
  int stride_b0 = is_b_row ? stride_b_n : stride_b_k;
  int stride_b1 = is_b_row ? stride_b_k : stride_b_n;
  int lda = is_a_row ? stride_a_m : stride_a_k;
  int ldb = is_b_row ? stride_b_k : stride_b_n;
  int per_phase_a = swizzle_->get_per_phase(layout_a);
  int max_phase_a = swizzle_->get_max_phase(layout_a);
  int per_phase_b = swizzle_->get_per_phase(layout_b);
  int max_phase_b = swizzle_->get_max_phase(layout_b);
  int num_ptr_a   = 8;
  int num_ptr_b   = 8;
  int vec_a = 2;
  int vec_b = 4;
  distributed_axis ax_m = axes_.at(a_axes_->get(C, 0));
  distributed_axis ax_n = axes_.at(a_axes_->get(C, 1));
//  Value* thread = tgt_->get_local_id(mod_, *builder_, 0);

  Value* off_a0 = is_a_row ? i32(0) : mul(ax_m.thread_id, i32(ax_m.contiguous));
  Value* off_a1 = is_a_row ? mul(ax_m.thread_id, i32(ax_m.contiguous)): i32(0);
  std::vector<Value*> off_a(num_ptr_a);
  for(int i = 0; i < num_ptr_a; i++){
//    Value* off_a0i = add(off_a0, i32(is_a_row ? vec_a : layout_c->mts(0)*vec_a));
//    off_a0i = exact_udiv(off_a0i, i32(vec_a));
//    off_a0i = xor_(off_a0i, phase_a);
//    off_a0i = mul(off_a0i, i32(vec_a));
    off_a[i] = add(mul(off_a0, i32(stride_a0)), mul(off_a1, i32(stride_a1)));
  }
  Value* off_b0 = is_b_row ? mul(ax_n.thread_id, i32(ax_n.contiguous)): i32(0);
  Value* off_b1 = is_b_row ? i32(0) : mul(ax_n.thread_id, i32(ax_n.contiguous));
  std::vector<Value*> off_b(num_ptr_b);
  for(int i = 0; i < num_ptr_b; i++){
//    Value* off_b0i = add(off_b0, i32(is_b_row ? layout_c->mts(1)*vec_b : vec_b));
//    off_b0i = exact_udiv(off_b0i, i32(vec_b));
//    off_b0i = xor_(off_b0i, phase_b);
//    off_b0i = mul(off_b0i, i32(vec_b));
    off_b[i] = add(mul(off_b0, i32(stride_b0)), mul(off_b1, i32(stride_b1)));
  }
  std::vector<Value*> ptrs_a(num_ptr_a);
  for(int i = 0; i < num_ptr_a; i++)
    ptrs_a[i] = gep(shmems_[A], off_a[i]);
  std::vector<Value*> ptrs_b(num_ptr_b);
  for(int i = 0; i < num_ptr_b; i++)
    ptrs_b[i] = gep(shmems_[B], off_b[i]);

  std::map<indices_t, Value*> ret = vals_[D];
  std::map<std::pair<int, int>, Value*> has, hbs;
  for(unsigned k = 0; k < NK; k++){
    int z = 0;
    for(unsigned m = 0; m < shape_c[0]; m+=layout_c->mts(0)*layout_c->nts(0))
    for(unsigned n = 0; n < shape_c[1]; n+=layout_c->mts(1)*layout_c->nts(1))
    for(unsigned mm = 0; mm < layout_c->nts(0); mm++)
    for(unsigned nn = 0; nn < layout_c->nts(1); nn++)
    {
      if(has.find({m + mm, k}) == has.end()){
        Value* pa = gep(ptrs_a[0], i32((m + mm)*stride_a_m + k*stride_a_k));
        Value* va = load(pa);
        has[{m + mm, k}] = va;
      }
      if(hbs.find({n + nn, k}) == hbs.end()){
        Value* pb = gep(ptrs_b[0], i32((n + nn)*stride_b_n + k*stride_b_k));
        Value* vb = load(pb);
        hbs[{n + nn, k}] = vb;
      }
      ret[idxs_[C].at(z)] = call(f_mul_add, {has[{m+mm,k}], hbs[{n+nn, k}], ret[idxs_[C].at(z)]});
      z++;
    }
  }

  for(indices_t idx: idxs_.at(C)){
    vals_[C][idx] = ret[idx];
  }
}

/**
 * \brief Code Generation for `dot`
 * Dispatches to appropriate specialized function
 */
void generator::visit_dot_inst(ir::dot_inst* dot) {
  Function *fn = builder_->GetInsertBlock()->getParent();
  Module *module = fn->getParent();
  ir::value *A = dot->get_operand(0);
  ir::value *B = dot->get_operand(1);
  ir::value *D = dot->get_operand(2);
  Type *c_ty = cvt(D->get_type()->get_scalar_ty());
  Function *f_mul_add = Intrinsic::getDeclaration(module, Intrinsic::fmuladd, std::vector<llvm::Type*>{c_ty});
  auto A_shapes = A->get_type()->get_block_shapes();
  size_t red_axis = 1;
  unsigned NK = A_shapes[red_axis];
  bool is_outer = NK == 1;
  bool is_mma = layouts_->get(dot)->to_mma();
  if(!is_outer && is_mma && tgt_->as_nvidia()->sm() < 80)
    return visit_mma884(dot, A, B, D, NK);
  if(!is_outer && is_mma && tgt_->as_nvidia()->sm() >= 80)
    return visit_mma16816(dot, A, B, D, NK);
  return visit_fmadot(dot, A, B, D, NK, c_ty, f_mul_add);
}

void generator::visit_trans_inst(ir::trans_inst* trans) {
  throw std::runtime_error("not supported");
}

/**
 * \brief Code Generation for `sqrt`
 */
void generator::visit_sqrt_inst(ir::sqrt_inst* x) {
  for(indices_t idx: idxs_.at(x)){
    Value *val = vals_[x->get_operand(0)][idx];
    Value *ret = intrinsic(Intrinsic::sqrt, {val->getType()}, {val});
    vals_[x][idx] = ret;
  }
}

Value* generator::shared_off(const std::vector<unsigned>& shapes, const std::vector<int>& order, indices_t idx){
  // strides
  std::vector<Value*> strides(shapes.size(), builder_->getInt32(0));
  strides[order[0]] = builder_->getInt32(1);
  for(size_t i = 1; i < idx.size(); i++)
    strides[order[i]] = builder_->CreateMul(strides[order[i-1]], builder_->getInt32(shapes[order[i-1]]));
  // result
  Value *result = builder_->getInt32(0);
  for(size_t i = 0; i < idx.size(); i++)
    result = builder_->CreateAdd(result, builder_->CreateMul(idx[i], strides[i]));
  return result;
}

/**
 * \brief Code Generation for `reduce` (1D case)
 */
void generator::visit_reduce1d_inst(ir::reduce_inst* x, std::function<Value*(Value*,Value*)> do_acc, Value *neutral) {
  std::map<indices_t, Value*> partial;
  ir::value *arg = x->get_operand(0);
  Type *ty = cvt(x->get_type()->get_scalar_ty());
  Value *acc = nullptr;

  // reduce within thread
  for(indices_t idx: idxs_.at(arg)){
    Value *val = vals_[arg][idx];
    acc = !acc ? val : do_acc(acc, val);
  }
  // reduce within wrap
  InlineAsm *shfl = InlineAsm::get(FunctionType::get(ty, {ty, i32_ty}, false),
                                   "shfl.sync.bfly.b32 $0, $1, $2, 0x1f, 0xffffffff;", "=f,f,r", false);
  for(int i = 16; i > 0; i >>= 1)
    acc = do_acc(acc, call(shfl, {acc, i32(i)}));
  // pointers
  unsigned addr_space = shmem_->getType()->getPointerAddressSpace();
  Value *base = bit_cast(shmem_, ptr_ty(ty, addr_space));
  Value* thread = tgt_->get_local_id(mod_, *builder_, 0);
  Value* warp = udiv(thread, i32(32));
  Value* lane = urem(thread, i32(32));
  // store warp result in shared memory
  add_barrier();
  store(neutral, gep(base, lane));
  add_barrier();
  store(acc, gep(base, warp));
  add_barrier();

  // reduce across warps
  Value *cond = icmp_eq(warp, i32(0));
  Instruction *barrier = add_barrier();
  Instruction *term = llvm::SplitBlockAndInsertIfThen(cond, barrier, false);
  builder_->SetInsertPoint(term);
  Value* ret = load(gep(base, thread));
  for(int i = (num_warps_+1)/2; i > 0; i >>= 1){
    Value *current = call(shfl, {ret, i32(i)});
    ret = do_acc(ret, current);
  }
  store(ret, gep(base, thread));

  // store first warp done
  builder_->SetInsertPoint(barrier->getParent());
  ret = load(base);
  for(indices_t idx: idxs_.at(x))
    vals_[x][idx] = ret;
}

/**
 * \brief Code Generation for `reduce` (ND case)
 */
void generator::visit_reducend_inst(ir::reduce_inst* x, std::function<Value*(Value*,Value*)> do_acc, Value *neutral) {
  ir::value *arg = x->get_operand(0);
  Type *ty = cvt(x->get_type()->get_scalar_ty());
  unsigned axis = x->get_axis();

  // reduce within thread
  std::map<indices_t, Value*> accs;
  for(indices_t idx: idxs_.at(arg)){
    indices_t pidx = idx;
    pidx[axis] = i32(0);
    Value *current = vals_[arg][idx];
    bool is_first = accs.find(pidx) == accs.end();
    accs[pidx] = is_first ? current : do_acc(accs[pidx], current);
  };

  // reduce within blocks
  analysis::data_layout* layout = layouts_->get(layouts_->tmp(x));
  Value *base = shared_ptr_.at(layout);
  auto shape  = layout->get_shape();
  auto order  = layout->get_order();
  int  space = base->getType()->getPointerAddressSpace();
  Value *ptr = bit_cast(base, ptr_ty(ty, space));
  Value *lane = axes_.at(a_axes_->get(arg, axis)).thread_id;
  for(auto& x: accs) {
    // current element being computed
    Value *&acc = x.second;
    indices_t write_idx = x.first;
    write_idx[axis] = lane;
    // shared memory write  pointer
    Value *write_off = shared_off(shape, order, write_idx);
    Value *write_ptr = gep(ptr, write_off);
    // initialize shared memory
    add_barrier();
    store(acc, write_ptr);
    // build result
    indices_t idx(write_idx.size(), i32(0));
    for(size_t i = shape[axis]/2; i > 0; i >>= 1){
      idx[axis] = i32(i);
      // read pointer
      Value *read_msk = icmp_ult(lane, i32(i));
      Value *read_off = select(read_msk, shared_off(shape, order, idx), i32(0));
      Value *read_ptr = gep(write_ptr, read_off);
      add_barrier();
      // update accumulator
      acc = do_acc(acc, load(read_ptr));
      store(acc, write_ptr);
    }
  }
  add_barrier();

  // write back
  for(indices_t idx: idxs_.at(x)){
    indices_t read_idx = idx;
    read_idx.insert(read_idx.begin() + axis, i32(0));
    Value *read_off = shared_off(shape, order, read_idx);
    Value *read_ptr = gep(ptr, read_off);
    vals_[x][idx] = load(read_ptr);
  };
}

/**
 * \brief Code Generation for `reduce` (generic case)
 */
void generator::visit_reduce_inst(ir::reduce_inst* x) {
  Type *ty = cvt(x->get_type()->get_scalar_ty());
  // accumulation function
  ir::reduce_inst::op_t op = x->get_op();
  auto do_acc = [&](Value *x, Value *y) -> Value* {
    switch(op){
    case ir::reduce_inst::ADD: return add(x, y);
    case ir::reduce_inst::SUB: return sub(x, y);
    case ir::reduce_inst::MAX: return select(icmp_sge(x, y), x, y);
    case ir::reduce_inst::MIN: return select(icmp_sle(x, y), x, y);
    case ir::reduce_inst::FADD: return fadd(x, y);
    case ir::reduce_inst::FSUB: return fsub(x, y);
    case ir::reduce_inst::FMAX: return max_num(x, y);
    case ir::reduce_inst::FMIN: return min_num(x, y);
    default: throw std::runtime_error("unreachable");
    }
  };
  // neutral element
  Value *neutral;
  switch(op) {
    case ir::reduce_inst::ADD: neutral = i32(0); break;
    case ir::reduce_inst::SUB:  neutral = i32(0); break;
    case ir::reduce_inst::MAX:  neutral = i32(INT32_MIN); break;
    case ir::reduce_inst::MIN:  neutral = i32(INT32_MAX); break;
    case ir::reduce_inst::FADD: neutral = ConstantFP::get(ty, 0); break;
    case ir::reduce_inst::FSUB: neutral = ConstantFP::get(ty, 0); break;
    case ir::reduce_inst::FMAX: neutral = ConstantFP::get(ty, -INFINITY); break;
    case ir::reduce_inst::FMIN: neutral = ConstantFP::get(ty, INFINITY); break;
    default: throw std::runtime_error("unreachable");
  }
  ir::value *arg = x->get_operand(0);
  if(arg->get_type()->get_tile_rank() == 1)
    visit_reduce1d_inst(x, do_acc, neutral);
  else
    visit_reducend_inst(x, do_acc, neutral);
}

/**
 * \brief Code Generation for `select`
 */
void generator::visit_select_inst(ir::select_inst* x) {
  for(indices_t idx: idxs_.at(x))
    vals_[x][idx] = select(vals_[x->get_operand(0)][idx],
                           vals_[x->get_operand(1)][idx],
                           vals_[x->get_operand(2)][idx]);
}

/**
 * \brief Code Generation for `recoalesce`
 */
void generator::visit_recoalesce_inst(ir::recoalesce_inst* rc) {
  ir::value *op = rc->get_operand(0);
  ir::block_type::block_shapes_t shape = rc->get_type()->get_block_shapes();
  // pointer to temporary shared memory
  Type *ty = cvt(rc->get_type()->get_scalar_ty());
  // layout
  analysis::mma_layout* in_layout = layouts_->get(op)->to_mma();
  analysis::scanline_layout* out_layout = layouts_->get(rc)->to_scanline();
  // Orders
  auto ord = layouts_->get(rc)->to_scanline()->get_order();
  Value *base;
  base = gep(shmem_, i32(alloc_->offset(layouts_->get(layouts_->tmp(rc)))));
  base = bit_cast(base, ptr_ty(ty, 3));
  Value *ld = i32(shape[ord[0]]);
  auto in_ord0 = axes_.at(a_axes_->get(op, ord[0])).values;
  auto in_ord1 = axes_.at(a_axes_->get(op, ord[1])).values;
  auto out_ord0 = axes_.at(a_axes_->get(rc, ord[0])).values;
  auto out_ord1 = axes_.at(a_axes_->get(rc, ord[1])).values;
  int in_spt0  = in_layout->spt(ord[0]);
  int in_spt1  = in_layout->spt(ord[1]);
  int out_spt0 = out_layout->mts(ord[0])*out_layout->nts(ord[0]);
  int out_spt1 = out_layout->mts(ord[1])*out_layout->nts(ord[1]);
  int max_spt1 = std::max(in_spt1, out_spt1);
  indices_t idx(2);
  int num_packs = shape[ord[1]]/max_spt1;
  for(size_t j = 0; j < num_packs; j++){
    add_barrier();
    for(size_t k = 0; k < in_ord1.size()/num_packs; k++)
    for(size_t i = 0; i < in_ord0.size(); i++){
      idx[ord[0]] = in_ord0[i];
      idx[ord[1]] = in_ord1[j*in_ord1.size()/num_packs + k];
      Value *off = add(idx[ord[0]], mul(in_ord1[k], ld));
      Value *ptr = gep(base, off);
      store(vals_[op][idx], ptr);
    }
    add_barrier();
    for(size_t k = 0; k < out_ord1.size()/num_packs; k++)
    for(size_t i = 0; i < out_ord0.size(); i++){
      idx[ord[0]] = out_ord0[i];
      idx[ord[1]] = out_ord1[j*out_ord1.size()/num_packs + k];
      Value *off = add(idx[ord[0]], mul(out_ord1[k], ld));
      Value *ptr  = gep(base, off);
      vals_[rc][idx] = load(ptr);
    }
  }
}

void generator::visit_masked_load_async_inst(ir::masked_load_async_inst* x){
  unsigned in_vec = 1;
  ir::value *arg = x->get_pointer_operand();
  analysis::shared_layout* out_layout = layouts_->get(x)->to_shared();
  analysis::scanline_layout* in_layout = layouts_->get(arg)->to_scanline();
  auto out_order = out_layout->get_order();
  auto in_order = in_layout->get_order();
  // tiles
  if(out_order == in_order)
    in_vec = in_layout->nts(in_order[0]);
  int out_vec = swizzle_->get_vec(out_layout);
  int min_vec = std::min<int>(out_vec, in_vec);
  int s = std::max<int>(out_vec / in_vec, 1);
  //
  int per_phase = swizzle_->get_per_phase(out_layout);
  int max_phase = swizzle_->get_max_phase(out_layout);
  //
  int in_ld = in_layout->get_shape()[in_order[0]] / in_layout->mts(in_order[0]);
  int n_shared_1 = std::max<int>(per_phase*max_phase / in_layout->mts(in_order[1]), 1);
  int n_shared_0 = std::max<int>(in_vec    / out_vec, 1);
  auto shapes = x->get_type()->get_block_shapes();
  BasicBlock* CurrBB = builder_->GetInsertBlock();
  BasicBlock* FirstBB = &CurrBB->getParent()->getEntryBlock();
  std::map<std::pair<int, int>, Value*> tmp;
  std::vector<std::pair<Value*, int>> shared;
  for(int i = 0; i < idxs_.at(arg).size(); i++){
    unsigned id = i / min_vec;
    // input ptr info
    int id_0 = id % (in_ld/min_vec);
    int id_1 = id / (in_ld/min_vec);
    int off_0 = id_0 / n_shared_0 * n_shared_0 * in_layout->mts(in_order[0]);
    int off_1 = id_1 / n_shared_1 * n_shared_1 * in_layout->mts(in_order[1]);
    int off = (off_1*shapes[in_order[0]] + off_0);
    std::pair<int, int> key = {id_1  % n_shared_1, id_0 % n_shared_0};
    if(tmp.find(key) == tmp.end()){
      if(CurrBB != FirstBB)
        builder_->SetInsertPoint(FirstBB->getTerminator());
      indices_t idx = idxs_.at(arg).at(key.first*in_ld);
      Value* phase = udiv(idx[in_order[1]], i32(per_phase));
      phase = urem(phase, i32(max_phase));
      Value* off_1 = mul(idx[in_order[1]], i32(shapes[in_order[0]]));
      Value* off_0  = add(idx[in_order[0]], i32(key.second*out_vec));
      off_0 = udiv(off_0, i32(min_vec));
      off_0 = add(mul(xor_(udiv(off_0, i32(s)), phase),i32(s)), urem(off_0, i32(s)));
      off_0 = mul(off_0 , i32(min_vec));
      Value* off = add(off_0, off_1);
      if(CurrBB != FirstBB)
        builder_->SetInsertPoint(CurrBB);
      tmp[key] = gep(shmems_[x], {off});
    }
    shared.push_back({tmp[key], off});
  }
  size_t dtsize = x->get_type()->get_scalar_ty()->get_primitive_size_in_bits() / 8;
  for(size_t i = 0; i < idxs_.at(arg).size(); i += in_vec){
    auto idx = idxs_[arg][i];
    // input ptr info
    GetElementPtrInst *in_gep = dyn_cast<GetElementPtrInst>(vals_[arg][idx]);
    Value *in_base = in_gep->getPointerOperand();
    ConstantInt* cst = dyn_cast<ConstantInt>(in_gep->idx_begin());
    size_t in_off = cst ? cst->getValue().getSExtValue()*dtsize*in_vec : 0;
    in_base = cst ? in_base : in_gep;
    // output ptr info
    Value* out_base = shared[i].first;
    int out_off = shared[i].second*dtsize;
    // asm
    std::string mod = (in_vec*dtsize == 16) ? ".cg" : ".ca";
//    Value* false_value = vals_[x->get_false_value_operand()][idx];
//    bool is_zero_false_value = false;
//    if(Constant* cst = dyn_cast<Constant>(false_value))
//      is_zero_false_value = cst->isZeroValue();
    Value* src_size = builder_->CreateSelect(vals_[x->get_mask_operand()][idx], i32(in_vec*dtsize), i32(0));
    std::string asm_str = "cp.async" + mod + ".shared.global [$0 + " + std::to_string(out_off) + "], [$1 + " + std::to_string(in_off) + "], " + std::to_string(in_vec*dtsize) + ", $2;";
    FunctionType *ty = FunctionType::get(void_ty, {out_base->getType(), in_base->getType(), builder_->getInt32Ty()}, false);
    InlineAsm *iasm = InlineAsm::get(ty, asm_str, "r,l,r", true);
    call(iasm, {out_base, in_base, src_size});
  }

  std::string asm_str = "cp.async.commit_group;";
  InlineAsm *iasm = InlineAsm::get(FunctionType::get(void_ty, {}), asm_str, "", true);
  call(iasm);
}

void generator::visit_copy_to_shared_inst(ir::copy_to_shared_inst* cts) {
  unsigned in_vec = 1;
  ir::value *arg = cts->get_operand(0);
  analysis::shared_layout* out_layout = layouts_->get(cts)->to_shared();
  analysis::scanline_layout* in_layout = layouts_->get(arg)->to_scanline();
  auto out_order = out_layout->get_order();
  auto in_order = in_layout->get_order();
  // tiles
  if(out_order == in_order)
    in_vec = in_layout->nts(in_order[0]);
  int out_vec = swizzle_->get_vec(out_layout);
  int min_vec = std::min<int>(out_vec, in_vec);
  int s = std::max<int>(out_vec / in_vec, 1);
  //
  int per_phase = swizzle_->get_per_phase(out_layout);
  int max_phase = swizzle_->get_max_phase(out_layout);
  //
  int in_ld = in_layout->get_shape()[in_order[0]] / in_layout->mts(in_order[0]);
  int n_shared_1 = std::max<int>(per_phase*max_phase / in_layout->mts(in_order[1]), 1);
  int n_shared_0 = std::max<int>(in_vec    / out_vec, 1);

  BasicBlock* CurrBB = builder_->GetInsertBlock();
  BasicBlock* FirstBB = &CurrBB->getParent()->getEntryBlock();
  auto shapes = cts->get_type()->get_block_shapes();

  // store to shared
  Value *current = nullptr;
  std::map<std::pair<int, int>, Value*> ptrs;
  for(int i = 0; i < idxs_.at(arg).size(); i++){
    auto idx = idxs_[arg][i];
    Value *in_value = vals_[arg][idx];
    if(i % min_vec == 0)
      current = UndefValue::get(vec_ty(in_value->getType(), min_vec));
    current = insert_elt(current, in_value, i % min_vec);
    if(i % min_vec == min_vec - 1){
      unsigned id = i / min_vec;
      // input ptr info
      int id_0 = id % (in_ld/min_vec);
      int id_1 = id / (in_ld/min_vec);
      int off_0 = id_0 / n_shared_0 * n_shared_0 * in_layout->mts(in_order[0]);
      int off_1 = id_1 / n_shared_1 * n_shared_1 * in_layout->mts(in_order[1]);
      int off = (off_1*shapes[in_order[0]] + off_0);
      std::pair<int, int> key = {id_1  % n_shared_1, id_0 % n_shared_0};
      if(ptrs.find(key) == ptrs.end()){
        if(FirstBB != CurrBB)
          builder_->SetInsertPoint(FirstBB->getTerminator());
        indices_t idx = idxs_.at(arg).at(key.first*in_ld);
        Value* phase = udiv(idx[in_order[1]], i32(per_phase));
        phase = urem(phase, i32(max_phase));
        Value* off_1 = mul(idx[in_order[1]], i32(shapes[in_order[0]]));
        Value* off_0  = add(idx[in_order[0]], i32(key.second*out_vec));
        off_0 = udiv(off_0, i32(min_vec));
        off_0 = add(mul(xor_(udiv(off_0, i32(s)), phase),i32(s)), urem(off_0, i32(s)));
        off_0 = mul(off_0 , i32(min_vec));
        Value* off = add(off_0, off_1);
        builder_->SetInsertPoint(CurrBB);
        ptrs[key] = gep(shmems_.at(cts), {off});
      }
      Value* ptr = gep(ptrs[key], {i32(off)});
      ptr = bit_cast(ptr, current->getType()->getPointerTo(3));
      // asm
      store(current, ptr);
    }
  };
}

void generator::visit_copy_from_shared_inst(ir::copy_from_shared_inst*) {
  throw std::runtime_error("TODO");
}

Instruction* generator::add_barrier() {
  Module *module = builder_->GetInsertBlock()->getModule();
  return tgt_->add_barrier(module, *builder_);
}

void generator::visit_barrier_inst(ir::barrier_inst*) {
  add_barrier();
}

void generator::visit_async_wait_inst(ir::async_wait_inst* i) {
  std::string asm_str = "cp.async.wait_group " + std::to_string(i->get_N()) + ";";
  InlineAsm *iasm = InlineAsm::get(FunctionType::get(void_ty, {}), asm_str, "", true);
  call(iasm);
}

void generator::visit_make_range_dyn(ir::make_range_dyn* x) {
  for(indices_t idx: idxs_.at(x)){
    assert(idx.size() == 1);
    if(idx[0] == i32(0))
      vals_[x][idx] = idx[0];
    else{
      BinaryOperator *bin_add = dyn_cast<BinaryOperator>(idx[0]);
      assert(bin_add);
      vals_[x][idx] = bin_add->getOperand(0);
    }
  }
}

void generator::visit_make_range_sta(ir::make_range_sta* x) {
  for(indices_t idx: idxs_.at(x)){
    assert(idx.size() == 1);
    if(idx[0] == i32(0)){
      vals_[x][idx] = idx[0];
    }
    else{
      BinaryOperator *bin_add = dyn_cast<BinaryOperator>(idx[0]);
      assert(bin_add);
      Value *cst = bin_add->getOperand(1);
      assert(isa<Constant>(cst));
      vals_[x][idx] = cst;
    }
  };
}

void generator::visit_make_range(ir::make_range* x) {
  for(indices_t idx: idxs_.at(x)){
    vals_[x][idx] = idx[0];
  }
}

void generator::visit_undef_value(ir::undef_value *x) {
  Type* ty = cvt(x->get_type()->get_scalar_ty());
  for(indices_t idx: idxs_.at(x))
    vals_[x][idx] = llvm::UndefValue::get(ty);
}

void generator::visit_constant_int(ir::constant_int *x){
  Type *ty = cvt(x->get_type()->get_scalar_ty());
  for(indices_t idx: idxs_.at(x))
    vals_[x][idx] = ConstantInt::get(ty, x->get_value());
}

void generator::visit_constant_fp(ir::constant_fp *x){
  Type *ty = cvt(x->get_type()->get_scalar_ty());
  for(indices_t idx: idxs_.at(x))
    vals_[x][idx] = ConstantFP::get(ty, x->get_value());
}

void generator::visit_alloc_const(ir::alloc_const *alloc) {
  unsigned size = ((ir::constant_int*)alloc->get_operand(0))->get_value();
  Type *element_ty = cvt(alloc->get_type()->get_pointer_element_ty());
  Type *array_ty = llvm::ArrayType::get(element_ty, size);
  Value *array = new llvm::GlobalVariable(*mod_, array_ty, false, llvm::GlobalVariable::ExternalLinkage,
                                            nullptr, alloc->get_name(), nullptr, llvm::GlobalVariable::NotThreadLocal, 4);
  vals_[alloc][{}] = bit_cast(array, element_ty->getPointerTo(4));
}


void generator::visit_function(ir::function* fn) {
  LLVMContext &ctx = builder_->getContext();
  FunctionType *fn_ty = (FunctionType*)cvt(fn->get_fn_type());
  if(!tgt_->is_gpu()){
    Type *fn_ret_ty = fn_ty->getReturnType();
    std::vector<Type*> fn_args_ty;
    for(unsigned i = 0; i < fn_ty->getNumParams(); i++)
      fn_args_ty.push_back(fn_ty->getParamType(i));
    fn_args_ty.push_back(i32_ty);
    fn_args_ty.push_back(i32_ty);
    fn_args_ty.push_back(i32_ty);
    fn_ty = FunctionType::get(fn_ret_ty, fn_args_ty, false);
  }
  Function *ret = Function::Create(fn_ty, Function::ExternalLinkage, fn->get_name(), mod_);
  // set attributes
  for(auto attr_pair: fn->attrs()){
    unsigned id = attr_pair.first;
    for(ir::attribute attr: attr_pair.second)
    if(attr.is_llvm_attr()){
      llvm::Attribute llattr = cvt(attr);
      if(llattr.getKindAsEnum() != llvm::Attribute::None)
        ret->addAttribute(id, cvt(attr));
    }
  }
  // set metadata
  if(tgt_->is_gpu()){
      tgt_->set_kernel(*builder_, ctx, mod_, ret);
      Metadata *md_args[] = {
        ValueAsMetadata::get(ret),
        MDString::get(ctx, "maxntidx"),
        ValueAsMetadata::get(i32(num_warps_*32))
      };
      mod_->getOrInsertNamedMetadata("nvvm.annotations")->addOperand(MDNode::get(ctx, md_args));
  }
  // set arguments
  for(unsigned i = 0; i < fn->args().size(); i++)
    vals_[fn->args()[i]][{}] = &*(ret->arg_begin() + i);
  // create blocks
  for(ir::basic_block *block: fn->blocks()) {
    BasicBlock *dst_block = BasicBlock::Create(ctx, block->get_name(), ret);
    bbs_[block] = dst_block;
  }
  builder_->SetInsertPoint(bbs_[fn->blocks()[0]]);
  // initialize layouts
  for(auto x: layouts_->get_all()){
    visit_layout(x.second);
  }
  // generate LLVM-IR code
  for(ir::basic_block *block: fn->blocks())
    visit_basic_block(block);
  // finalize
  finalize_function(fn);
}



void generator::visit_layout_mma(analysis::mma_layout* layout) {
  ir::value *a = nullptr;
  ir::value *b = nullptr;
  for(ir::value* v: layout->get_values())
    if(ir::dot_inst* dot = dynamic_cast<ir::dot_inst*>(v)){
      a = dot->get_operand(0);
      b = dot->get_operand(1);
    }
  analysis::data_layout* layout_a = layouts_->get(a);
  analysis::data_layout* layout_b = layouts_->get(b);

  const auto& shape = layout->get_shape();
  Value *_1 = i32(1);
  Value *_2 = i32(2);
  Value *_3 = i32(3);
  Value *_4 = i32(4);
  Value *_8 = i32(8);
  Value *_16 = i32(16);
  Value *_32 = i32(32);
  int cc = tgt_->as_nvidia()->sm();
  std::vector<Value*> idx_m;
  std::vector<Value*> idx_n;
  std::vector<Value*> idx_z;
  //
  Value* thread = tgt_->get_local_id(mod_, *builder_, 0);
  Value *lane = urem(thread, _32);
  Value *warp = udiv(thread, _32);
  /* lane offset */
  if(cc < 80){
    auto ord_a = layout_a->get_order();
    auto ord_b = layout_b->get_order();
    bool is_a_row = ord_a[0] != 0;
    bool is_b_row = ord_b[0] != 0;
    /* warp offset */
    Value *warp_0 = urem(warp, i32(layout->wpt(0)));
    Value *warp_12 = udiv(warp, i32(layout->wpt(0)));
    Value *warp_1 = urem(warp_12, i32(layout->wpt(1)));
    Value *off_warp_m = mul(warp_0, i32(layout->spw(0)));
    Value *off_warp_n = mul(warp_1, i32(layout->spw(1)));
    // Quad offset
    Value *off_quad_m = mul(udiv(and_(lane, _16), _4), i32(layout->fpw(0)));
    Value *off_quad_n = mul(udiv(and_(lane, _16), _4), i32(layout->fpw(1)));
    // Pair offset
    Value *off_pair_m = udiv(urem(lane, _16), _4);
    off_pair_m = urem(off_pair_m, i32(layout->fpw(0)));
    off_pair_m = mul(off_pair_m, i32(4));
    Value *off_pair_n = udiv(urem(lane, _16), _4);
    off_pair_n = udiv(off_pair_n, i32(layout->fpw(0)));
    off_pair_n = urem(off_pair_n, i32(layout->fpw(1)));
    off_pair_n = mul(off_pair_n, i32(4));
    // scale
    off_pair_m = mul(off_pair_m, i32(layout->rep(0)/2));
    off_quad_m = mul(off_quad_m, i32(layout->rep(0)/2));
    off_pair_n = mul(off_pair_n, i32(layout->rep(1)/2));
    off_quad_n = mul(off_quad_n, i32(layout->rep(1)/2));
    // Quad pair offset
    Value *off_lane_m = add(off_pair_m, off_quad_m);
    Value *off_lane_n = add(off_pair_n, off_quad_n);
    // a offset
    offset_a_m_[layout] = add(off_warp_m, off_lane_m);
    offset_a_k_[layout] = and_(lane, _3);
    // b offsets
    offset_b_n_[layout] = add(off_warp_n, off_lane_n);
    offset_b_k_[layout] = and_(lane, _3);
    // i indices
    Value *offset_c_m = add(and_(lane, _1), offset_a_m_[layout]);
    for(unsigned m = 0; m < shape[0]; m+=layout->spt(0))
    for(unsigned mm = 0; mm < layout->rep(0); mm++)
      idx_m.push_back(add(offset_c_m, i32(m + mm*2)));
    // j indices
    Value *offset_c_n = add(and_(lane, _2), add(off_warp_n, off_pair_n));
    for(unsigned n = 0; n < shape[1]; n+=layout->spt(1))
    for(unsigned nn = 0; nn < layout->rep(1); nn++){
      idx_n.push_back(add(offset_c_n, i32(n + nn/2*4 + (nn%2)*2*layout->fpw(1)*layout->rep(1))));
      idx_n.push_back(add(offset_c_n, i32(n + nn/2*4 + (nn%2)*2*layout->fpw(1)*layout->rep(1) + 1)));
    }
    if(is_a_row){
      offset_a_m_[layout] = add(offset_a_m_[layout], urem(thread, i32(4)));
      offset_a_k_[layout] = i32(0);
    }
    if(!is_b_row){
      offset_b_n_[layout] = add(offset_b_n_[layout], urem(thread, i32(4)));
      offset_b_k_[layout] = i32(0);
    }
    /* axes */
    axes_[layout->get_axis(0)] = distributed_axis{1, idx_m, warp_0};
    axes_[layout->get_axis(1)] = distributed_axis{1, idx_n, warp_1};
  }
  else{
    /* warp offset */
    Value *warp_0 = urem(warp, i32(layout->wpt(0)));
    Value *warp_12 = udiv(warp, i32(layout->wpt(0)));
    Value *warp_1 = urem(warp_12, i32(layout->wpt(1)));
    Value *off_warp_m = mul(warp_0, i32(layout->spw(0)));
    Value *off_warp_n = mul(warp_1, i32(layout->spw(1)));
    Value *off_lane_m = urem(lane, _16);
    Value *off_lane_n = urem(lane, _8);
    /* offsets */
    // a offset
    offset_a_m_[layout] = add(off_warp_m, off_lane_m);
    offset_a_k_[layout] = i32(0);
    // b offsets
    offset_b_n_[layout] = add(off_warp_n, off_lane_n);
    offset_b_k_[layout] = i32(0);
    // c offset
    Value *off_c_m = add(udiv(lane, _4), off_warp_m);
    Value *off_c_n = add(mul(_2, urem(lane, _4)), off_warp_n);
    for(unsigned m = 0; m < shape[0]; m+=layout->spt(0)){
      idx_m.push_back(add(off_c_m, i32(m)));
      idx_m.push_back(add(off_c_m, i32(m + 8)));
    }
    for(unsigned n = 0; n < shape[1]; n+=layout->spt(1)){
      idx_n.push_back(add(off_c_n, i32(n)));
      idx_n.push_back(add(off_c_n, i32(n + 1)));
    }
    /* axes */
    axes_[layout->get_axis(0)] = distributed_axis{1, idx_m, warp_0};
    axes_[layout->get_axis(1)] = distributed_axis{1, idx_n, warp_1};
  }
}

void generator::visit_layout_scanline(analysis::scanline_layout* layout) {
  Value *warp_size = i32(32);
  Value* u_thread_id_0 = tgt_->get_local_id(mod_, *builder_, 0);
  Value *u_thread_id = urem(u_thread_id_0, warp_size);
  Value *u_warp_id = udiv(u_thread_id_0, warp_size);

  auto order = layout->get_order();
  const auto& shape = layout->get_shape();
  Value* full_thread_id = add(mul(u_warp_id, i32(32)), u_thread_id);
  // Delinearize
  size_t dim = shape.size();
  std::vector<Value*> thread_id(dim);
  for(unsigned k = 0; k < dim - 1; k++){
    Constant *dim_k = i32(layout->mts(order[k]));
    Value *rem = urem(full_thread_id, dim_k);
    full_thread_id = udiv(full_thread_id, dim_k);
    thread_id[order[k]] = rem;
  }
  thread_id[order[dim - 1]] = full_thread_id;
  // Create axes
  for(unsigned k = 0; k < dim; k++) {
    int nts = layout->nts(k);
    int mts = layout->mts(k);
    std::string str_k = std::to_string(k);
    Value *contiguous_k = i32(nts);
    Value *scaled_thread_id = mul(thread_id[k], contiguous_k);
    unsigned per_block  = nts * mts;
    unsigned per_thread = nts * shape[k] / per_block;
    std::vector<Value*> idx_list(per_thread);
    for(unsigned n = 0 ; n < per_thread; n++){
      unsigned offset = n / nts * per_block + n % nts;
      idx_list[n] = add(scaled_thread_id, i32(offset), "idx_" + str_k + "_" + std::to_string(n));
    }
    axes_[layout->get_axis(k)] = distributed_axis{nts, idx_list, thread_id[k]};
  }
}

void generator::visit_layout_shared(analysis::shared_layout* layout) {
  Type* ty = cvt(layout->get_type());
  PointerType *ptr_ty = ty->getPointerTo(shmem_->getType()->getPointerAddressSpace());
  // double-buffered
  if(layout->get_double_buffer()) {
    BasicBlock *current = builder_->GetInsertBlock();
    auto info = *layout->get_double_buffer();
    ir::phi_node *phi = info.phi;
    BasicBlock *parent = bbs_.at(phi->get_parent());
    if(parent->empty())
      builder_->SetInsertPoint(parent);
    else
      builder_->SetInsertPoint(&*parent->getFirstNonPHI());
    // create pointers
    shared_ptr_[layout] = phi(ptr_ty, 2);
    shared_pre_ptr_[layout] = gep(shmem_, i32(alloc_->offset(layout)));
    shared_pre_ptr_[layout] = bit_cast(shared_pre_ptr_[layout], shared_ptr_[layout]->getType());
    shared_off_[layout] = phi(i32_ty, 2);
    shared_next_ptr_[layout] = gep(shared_ptr_[layout], shared_off_[layout], "next_ptr");
    builder_->SetInsertPoint(current);
  }
  else{
    size_t offset = alloc_->offset(layout);
    shared_ptr_[layout] = gep(shmem_, i32(offset));
    shared_ptr_[layout] = bit_cast(shared_ptr_[layout], ptr_ty);
  }
}

void generator::visit_basic_block(ir::basic_block * block) {
  BasicBlock *parent = bbs_[block];
  builder_->SetInsertPoint(parent);
  for(ir::instruction *i: block->get_inst_list()){
    visit_value(i);
  }
  bbs_[block] = builder_->GetInsertBlock();
}

void generator::visit_argument(ir::argument* arg) {

}

void generator::init_idx(ir::value *v) {
  idxs_[v].clear();
  if(!v->get_type()->is_block_ty()){
    idxs_[v].push_back({});
    return;
  }
  if(layouts_->get(v)->to_shared())
    return;
  const auto &shapes = v->get_type()->get_block_shapes();
  size_t rank = shapes.size();
  std::vector<distributed_axis> axes(rank);
  std::vector<int> ord(rank);
  // compute axes
  for(size_t d = 0; d < shapes.size(); d++){
    if(shapes[d] > 1){
      unsigned x = a_axes_->get(v, d);
      axes[d] = axes_.at(x);
    }
    else{
      axes[d].contiguous = 1;
      axes[d].values = {i32(0)};
    }
  }
  // compute order
  analysis::data_layout* layout = layouts_->get(v);
  std::iota(ord.begin(), ord.end(), 0);
  auto cmp = [&](int x, int y) {
    unsigned axx = a_axes_->get(v, x);
    unsigned axy = a_axes_->get(v, y);
    size_t posx = layout->find_axis(axx);
    size_t posy = layout->find_axis(axy);
    if(posx < rank && posy < rank)
      return layout->get_order(posx) < layout->get_order(posy);
    return false;
  };
  std::sort(ord.begin(), ord.end(), cmp);
  ords_[v] = ord;
  // indices
  if(axes.size() == 1)
    for(Value* x0: axes[ord[0]].values){
      idxs_[v].push_back({x0});
    }
  if(axes.size() == 2)
    for(Value* x1: axes[ord[1]].values)
    for(Value* x0: axes[ord[0]].values){
      indices_t idx(2);
      idx[ord[0]] = x0;
      idx[ord[1]] = x1;
      idxs_[v].push_back(idx);
    }
  if(axes.size() == 3)
    for(Value* x2: axes[ord[2]].values)
    for(Value* x1: axes[ord[1]].values)
    for(Value* x0: axes[ord[0]].values){
      indices_t idx(3);
      idx[ord[0]] = x0;
      idx[ord[1]] = x1;
      idx[ord[2]] = x2;
      idxs_[v].push_back(idx);
    }
}

void generator::finalize_shared_layout(analysis::shared_layout *shared) {
  if(shared->get_double_buffer()) {
    auto info = *shared->get_double_buffer();
    ir::phi_node *phi = info.phi;
    PHINode *ptr = (PHINode*)shmems_[phi];
    PHINode *offset = (PHINode*)shoffs_[phi];
    for(unsigned n = 0; n < phi->get_num_incoming(); n++){
      ir::basic_block* inc_block = phi->get_incoming_block(n);
      ir::value* inc_val = phi->get_incoming_value(n);
      BasicBlock *llvm_inc_block = bbs_.at(inc_block);
      if(inc_val == info.latch){
        builder_->SetInsertPoint(llvm_inc_block->getTerminator());
        Value *next_offset = neg(offset);
        offset->addIncoming(next_offset, llvm_inc_block);
      }
      else {
        unsigned num_bytes = shared->get_type()->get_primitive_size_in_bits() / 8;
        offset->addIncoming(i32(shared->get_size() / (2*num_bytes)), llvm_inc_block);
      }
      ptr->addIncoming(shmems_[inc_val], llvm_inc_block);
    }
  }
}

void generator::finalize_function(ir::function *fn) {
  // finalize double-buffering
  for(const auto& x: layouts_->get_all())
  if(auto *shared = dynamic_cast<analysis::shared_layout*>(x.second))
    finalize_shared_layout(shared);
  // finalize phi
  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction *inst: block->get_inst_list())
    if(auto *phi = dynamic_cast<ir::phi_node*>(inst))
      finalize_phi_node(phi);
}

void generator::finalize_phi_node(ir::phi_node *x) {
  if(shmems_.find(x) != shmems_.end())
    return;
  for(unsigned n = 0; n < x->get_num_incoming(); n++){
    ir::basic_block *_block = x->get_incoming_block(n);
    BasicBlock *block = bbs_.at(_block);
    for(indices_t idx: idxs_.at(x)){
      PHINode *phi = (PHINode*)vals_[x][idx];
      Value *inc = vals_[x->get_incoming_value(n)][idx];
      phi->addIncoming(inc, block);
    }
  }
}

void generator::visit(ir::module &src, llvm::Module &dst) {
  mod_ = &dst;
  ctx_ = &dst.getContext();
  builder_ = new Builder(*ctx_);
  // allocate shared memory
  if(tgt_->is_gpu())
  if(unsigned alloc_size = alloc_->allocated_size()){
    Type *int_8_ty = Type::getInt8Ty(*ctx_);
    Type *int_32_ty = Type::getInt32Ty(*ctx_);
    ArrayType *array_ty = ArrayType::get(int_32_ty, 0);
    Type *ptr_ty = ptr_ty(int_8_ty, 3);
    GlobalVariable *sh_mem_array =
      new GlobalVariable(*mod_, array_ty, false, GlobalVariable::ExternalLinkage,
                         nullptr, "__shared_ptr", nullptr, GlobalVariable::NotThreadLocal, 3);
    shmem_ = bit_cast(sh_mem_array, ptr_ty);
  }
  // visit functions
  for(ir::function *fn: src.get_function_list())
    visit_function(fn);
}


}
}
