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

/* convert ir::type to llvm::Type */

llvm::Type *llvm_type(ir::type *ty, llvm::LLVMContext &ctx) {
  // function
  if(auto* tt = dynamic_cast<ir::function_type*>(ty)){
    llvm::Type *return_ty = llvm_type(tt->get_return_ty(), ctx);
    std::vector<llvm::Type*> param_tys;
    std::transform(tt->params_begin(), tt->params_end(), std::back_inserter(param_tys),
                   [&ctx](ir::type* t){ return llvm_type(t, ctx);});
    return llvm::FunctionType::get(return_ty, param_tys, false);
  }
  // pointer
  if(ty->is_pointer_ty()){
    llvm::Type *elt_ty = llvm_type(ty->get_pointer_element_ty(), ctx);
    unsigned addr_space = ty->get_pointer_address_space();
    return llvm::PointerType::get(elt_ty, addr_space);
  }
  // integer
  if(ty->is_integer_ty()){
    unsigned bitwidth = ty->get_integer_bitwidth();
    return llvm::IntegerType::get(ctx, bitwidth);
  }
  // primitive types
  switch(ty->get_type_id()){
    case ir::type::VoidTyID:      return llvm::Type::getVoidTy(ctx);
    case ir::type::HalfTyID:      return llvm::Type::getHalfTy(ctx);
    case ir::type::FloatTyID:     return llvm::Type::getFloatTy(ctx);
    case ir::type::DoubleTyID:    return llvm::Type::getDoubleTy(ctx);
    case ir::type::X86_FP80TyID:  return llvm::Type::getX86_FP80Ty(ctx);
    case ir::type::PPC_FP128TyID: return llvm::Type::getPPC_FP128Ty(ctx);
    case ir::type::LabelTyID:     return llvm::Type::getLabelTy(ctx);
    case ir::type::MetadataTyID:  return llvm::Type::getMetadataTy(ctx);
    case ir::type::TokenTyID:     return llvm::Type::getTokenTy(ctx);
    default: break;
  }
  // unknown type
  throw std::runtime_error("unknown conversion from ir::type to llvm::Type");
}

/* convert ir::instruction to llvm::Instruction */
llvm::Instruction *llvm_inst(ir::instruction *inst, llvm::LLVMContext & ctx,
                             std::map<ir::value*, llvm::Value*> &v,
                             std::map<ir::basic_block*, llvm::BasicBlock*> &b) {
  if(auto* ii = dynamic_cast<ir::cond_branch_inst*>(inst))
    return llvm::BranchInst::Create(b[ii->get_true_dest()], b[ii->get_false_dest()], v[ii->get_cond()]);
  if(auto* ii = dynamic_cast<ir::uncond_branch_inst*>(inst))
    return llvm::BranchInst::Create(b[ii->get_dest()]);
  if(auto* ii = dynamic_cast<ir::phi_node*>(inst))
    return llvm::PHINode::Create(llvm_type(ii->get_type(), ctx), ii->get_num_operands(), ii->get_name());
  if(auto* ii = dynamic_cast<ir::return_inst*>(inst))
    return llvm::ReturnInst::Create(ctx, v[ii->get_return_value()]);
  if(auto* ii = dynamic_cast<ir::binary_operator*>(inst))
    return llvm::BinaryOperator::Create(ii->get_op(), v[ii->get_operand(0)], v[ii->get_operand(1)], ii->get_name());
  if(auto* ii = dynamic_cast<ir::icmp_inst*>(inst))
    return llvm::CmpInst::Create(llvm::Instruction::ICmp, ii->get_pred(), v[ii->get_operand(0)], v[ii->get_operand(1)], ii->get_name());
  if(auto* ii = dynamic_cast<ir::fcmp_inst*>(inst))
    return llvm::FCmpInst::Create(llvm::Instruction::FCmp, ii->get_pred(), v[ii->get_operand(0)], v[ii->get_operand(1)], ii->get_name());
  if(auto* ii = dynamic_cast<ir::cast_inst*>(inst))
    return llvm::CastInst::Create(ii->get_op(), v[ii->get_operand(0)], llvm_type(ii->get_type(), ctx), ii->get_name());
  if(auto* ii = dynamic_cast<ir::getelementptr_inst*>(inst)){
    std::vector<llvm::Value*> idx_vals;
    std::transform(ii->idx_begin(), ii->idx_end(), std::back_inserter(idx_vals),
                   [&v](ir::value* x){ return v[x];});
    return llvm::GetElementPtrInst::Create(llvm_type(ii->get_source_elt_ty(), ctx), v[ii->get_operand(0)], idx_vals, ii->get_name());
  }
  if(ir::load_inst* ii = dynamic_cast<ir::load_inst*>(inst))
    return new llvm::LoadInst(v[ii->get_pointer_operand()], ii->get_name());
  // unknown instruction
  throw std::runtime_error("unknown conversion from ir::type to llvm::Type");
}

void lowering(ir::module &src, llvm::Module &dst){
  using namespace llvm;
  std::map<ir::value*, Value*> vmap;
  std::map<ir::basic_block*, BasicBlock*> bmap;
  LLVMContext &dst_ctx = dst.getContext();
  IRBuilder<> dst_builder(dst_ctx);
  // iterate over functions
  for(ir::function *fn: src.get_function_list()) {
    // create LLVM function
    Type *fn_ty = llvm_type(fn->get_type(), dst_ctx);
    Function *dst_function = (Function*)dst.getOrInsertFunction(fn->get_name(), fn_ty);
    // create blocks
    for(ir::basic_block *block: fn->blocks()) {
      BasicBlock *dst_block = BasicBlock::Create(dst_ctx, block->get_name(), dst_function);
      bmap[block] = dst_block;
    }
    // iterate through block
    for(ir::basic_block *block: fn->blocks()) {
      dst_builder.SetInsertPoint(bmap[block]);
      for(ir::instruction *inst: block->get_inst_list()) {
        Instruction *dst_inst = llvm_inst(inst, dst_ctx, vmap, bmap);
        vmap[inst] = dst_inst;
      }
    }
    // add phi operands
  }
}


}
}

#endif
