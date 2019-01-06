#ifndef TDL_INCLUDE_IR_CODEGEN_LOWERING_H
#define TDL_INCLUDE_IR_CODEGEN_LOWERING_H

#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include "ir/context.h"
#include "ir/module.h"
#include "ir/function.h"
#include "ir/type.h"


namespace tdl{
namespace codegen{

using namespace llvm;

/* convert ir::type to Type */
Type *llvm_type(ir::type *ty, LLVMContext &ctx) {
  // function
  if(auto* tt = dynamic_cast<ir::function_type*>(ty)){
    Type *return_ty = llvm_type(tt->get_return_ty(), ctx);
    std::vector<Type*> param_tys;
    std::transform(tt->params_begin(), tt->params_end(), std::back_inserter(param_tys),
                   [&ctx](ir::type* t){ return llvm_type(t, ctx);});
    return FunctionType::get(return_ty, param_tys, false);
  }
  // pointer
  if(ty->is_pointer_ty()){
    Type *elt_ty = llvm_type(ty->get_pointer_element_ty(), ctx);
    unsigned addr_space = ty->get_pointer_address_space();
    return PointerType::get(elt_ty, addr_space);
  }
  // integer
  if(ty->is_integer_ty()){
    unsigned bitwidth = ty->get_integer_bitwidth();
    return IntegerType::get(ctx, bitwidth);
  }
  // primitive types
  switch(ty->get_type_id()){
    case ir::type::VoidTyID:      return Type::getVoidTy(ctx);
    case ir::type::HalfTyID:      return Type::getHalfTy(ctx);
    case ir::type::FloatTyID:     return Type::getFloatTy(ctx);
    case ir::type::DoubleTyID:    return Type::getDoubleTy(ctx);
    case ir::type::X86_FP80TyID:  return Type::getX86_FP80Ty(ctx);
    case ir::type::PPC_FP128TyID: return Type::getPPC_FP128Ty(ctx);
    case ir::type::LabelTyID:     return Type::getLabelTy(ctx);
    case ir::type::MetadataTyID:  return Type::getMetadataTy(ctx);
    case ir::type::TokenTyID:     return Type::getTokenTy(ctx);
    default: break;
  }
  // unknown type
  throw std::runtime_error("unknown conversion from ir::type to Type");
}

Value* llvm_value(ir::value *v, LLVMContext &ctx,
                        std::map<ir::value*, Value*> &vmap,
                        std::map<ir::basic_block*, BasicBlock*> &bmap);

/* convert ir::constant to Constant */
Constant *llvm_constant(ir::constant *cst, LLVMContext &ctx) {
  Type *dst_ty = llvm_type(cst->get_type(), ctx);
  if(auto* cc = dynamic_cast<ir::constant_int*>(cst))
    return ConstantInt::get(dst_ty, cc->get_value());
  if(auto* cc = dynamic_cast<ir::constant_fp*>(cst))
    return ConstantFP::get(dst_ty, cc->get_value());
  // unknown constant
  throw std::runtime_error("unknown conversion from ir::constant to Constant");
}


/* convert ir::instruction to Instruction */
Instruction *llvm_inst(ir::instruction *inst, LLVMContext & ctx,
                             std::map<ir::value*, Value*> &vmap,
                             std::map<ir::basic_block*, BasicBlock*> &bmap) {
  auto value = [&](ir::value *x) { return llvm_value(x, ctx, vmap, bmap); };
  auto block = [&](ir::basic_block *x) { return bmap.at(x); };
  auto type = [&](ir::type *x) { return llvm_type(x, ctx); };
  if(auto* ii = dynamic_cast<ir::cond_branch_inst*>(inst)){
    BasicBlock *true_dest  = block(ii->get_true_dest());
    BasicBlock *false_dest = block(ii->get_false_dest());
    Value *cond = value(ii->get_cond());
    return BranchInst::Create(true_dest, false_dest, cond);
  }
  if(auto* ii = dynamic_cast<ir::uncond_branch_inst*>(inst)){
    BasicBlock *dest = block(ii->get_dest());
    return BranchInst::Create(dest);
  }
  if(auto* ii = dynamic_cast<ir::phi_node*>(inst)){
    Type *ty = type(ii->get_type());
    unsigned num_ops = ii->get_num_operands();
    return PHINode::Create(ty, num_ops, ii->get_name());
  }
  if(auto* ii = dynamic_cast<ir::return_inst*>(inst)){
    ir::value *ret_val = ii->get_return_value();
    return ReturnInst::Create(ctx, ret_val?value(ret_val):nullptr);
  }
  if(auto* ii = dynamic_cast<ir::binary_operator*>(inst)){
    Value *lhs = value(ii->get_operand(0));
    Value *rhs = value(ii->get_operand(1));
    return BinaryOperator::Create(ii->get_op(), lhs, rhs, ii->get_name());
  }
  if(auto* ii = dynamic_cast<ir::icmp_inst*>(inst)){
    CmpInst::Predicate pred = ii->get_pred();
    Value *lhs = value(ii->get_operand(0));
    Value *rhs = value(ii->get_operand(1));
    return CmpInst::Create(Instruction::ICmp, pred, lhs, rhs, ii->get_name());
  }
  if(auto* ii = dynamic_cast<ir::fcmp_inst*>(inst)){
    CmpInst::Predicate pred = ii->get_pred();
    Value *lhs = value(ii->get_operand(0));
    Value *rhs = value(ii->get_operand(1));
    return FCmpInst::Create(Instruction::FCmp, pred, lhs, rhs, ii->get_name());
  }
  if(auto* ii = dynamic_cast<ir::cast_inst*>(inst)){
    Value *arg = value(ii->get_operand(0));
    Type *dst_ty = type(ii->get_type());
    return CastInst::Create(ii->get_op(), arg, dst_ty, ii->get_name());
  }
  if(auto* ii = dynamic_cast<ir::getelementptr_inst*>(inst)){
    std::vector<Value*> idx_vals;
    std::transform(ii->idx_begin(), ii->idx_end(), std::back_inserter(idx_vals),
                   [&value](ir::value* x){ return value(x);});
    Type *source_ty = type(ii->get_source_elt_ty());
    Value *arg = value(ii->get_operand(0));
    return GetElementPtrInst::Create(source_ty, arg, idx_vals, ii->get_name());
  }
  if(ir::load_inst* ii = dynamic_cast<ir::load_inst*>(inst)){
    Value *ptr = value(ii->get_pointer_operand());
    return new LoadInst(ptr, ii->get_name());
  }
  // unknown instruction
  throw std::runtime_error("unknown conversion from ir::type to Type");
}

Value* llvm_value(ir::value *v, LLVMContext &ctx,
                        std::map<ir::value*, Value*> &vmap,
                        std::map<ir::basic_block*, BasicBlock*> &bmap) {
  if(vmap.find(v) != vmap.end())
    return vmap.at(v);
  // create operands
  if(auto *uu = dynamic_cast<ir::user*>(v))
  for(ir::use u: uu->ops()){
    vmap[u.get()] = llvm_value(u, ctx, vmap, bmap);
  }
  if(auto *cc = dynamic_cast<ir::constant*>(v))
    return llvm_constant(cc, ctx);
  // instruction
  if(auto *ii = dynamic_cast<ir::instruction*>(v))
    return llvm_inst(ii, ctx, vmap, bmap);
  // unknown value
  throw std::runtime_error("unknown conversion from ir::value to Value");
}

void lowering(ir::module &src, Module &dst){
  std::map<ir::value*, Value*> vmap;
  std::map<ir::basic_block*, BasicBlock*> bmap;
  LLVMContext &dst_ctx = dst.getContext();
  IRBuilder<> dst_builder(dst_ctx);
  // iterate over functions
  for(ir::function *fn: src.get_function_list()) {
    // create LLVM function
    FunctionType *fn_ty = (FunctionType*)llvm_type(fn->get_fn_type(), dst_ctx);
    Function *dst_fn = Function::Create(fn_ty, Function::ExternalLinkage, "kernel", &dst);
//    std::cout << ((FunctionType*)fn_ty)->getNumParams() << std::endl;
    // map parameters
    for(unsigned i = 0; i < fn->args().size(); i++)
      vmap[fn->args()[i]] = &*(dst_fn->arg_begin() + i);
    // create blocks
    for(ir::basic_block *block: fn->blocks()) {
      BasicBlock *dst_block = BasicBlock::Create(dst_ctx, block->get_name(), dst_fn);
      bmap[block] = dst_block;
    }
    // iterate through block
    for(ir::basic_block *block: fn->blocks()) {
      dst_builder.SetInsertPoint(bmap[block]);
      for(ir::instruction *inst: block->get_inst_list()) {
        Instruction *dst_inst = llvm_inst(inst, dst_ctx, vmap, bmap);
        vmap[inst] = dst_inst;
        dst_builder.Insert(dst_inst);
      }
    }
    // add phi operands
    for(ir::basic_block *block: fn->blocks())
    for(ir::instruction *inst: block->get_inst_list())
    if(auto *phi = dynamic_cast<ir::phi_node*>(inst)){
      PHINode *dst_phi = (PHINode*)vmap.at(phi);
      for(unsigned i = 0; i < phi->get_num_incoming(); i++){
        ir::value *inc_val = phi->get_incoming_value(i);
        ir::basic_block *inc_block = phi->get_incoming_block(i);
        dst_phi->addIncoming(vmap[inc_val], bmap[inc_block]);
      }
    }
  }
}


}
}

#endif
