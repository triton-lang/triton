#include "common.h"

namespace triton::codegen {

using namespace llvm;

Value* adder::operator()(Value *x, Value *y, const std::string& name) {
  // (x + cst) + y -> (x + y) + cst
  if(auto* bin = dyn_cast<BinaryOperator>(x))
  if(bin->getOpcode() == llvm::BinaryOperator::BinaryOps::Add)
  if(dyn_cast<Constant>(bin->getOperand(1))){
    return (*builder_)->CreateAdd((*builder_)->CreateAdd(bin->getOperand(0), y),
                                  bin->getOperand(1));
  }
  // (x + (y + cst)) -> (x + y) + cst
  if(auto* bin = dyn_cast<BinaryOperator>(y))
  if(bin->getOpcode() == llvm::BinaryOperator::BinaryOps::Add)
  if(dyn_cast<Constant>(bin->getOperand(1))){
    return (*builder_)->CreateAdd((*builder_)->CreateAdd(x, bin->getOperand(0)),
                                  bin->getOperand(1));
  }

  // default
  return (*builder_)->CreateAdd(x, y, name);
}

Value* multiplier::operator()(Value *x, Value *y, const std::string &name) {
  // (x + cst1) * cst2 -> (x * cst2) + (cst1 * cst2)
  if(auto* bin = dyn_cast<BinaryOperator>(x))
  if(bin->getOpcode() == llvm::BinaryOperator::BinaryOps::Add)
  if(dyn_cast<Constant>(bin->getOperand(1)))
  if(dyn_cast<Constant>(y)){
    return (*builder_)->CreateAdd((*builder_)->CreateMul(bin->getOperand(0), y),
                                  (*builder_)->CreateMul(bin->getOperand(1), y));
  }
  // default
  return (*builder_)->CreateMul(x, y, name);
}

Value* geper::operator()(Value *ptr, Value* off, const std::string& name){
  // (ptr + cst1) + (cst2) -> ptr + (cst1 + cst2)
  if(auto* gep = dyn_cast<GetElementPtrInst>(ptr))
  if(ConstantInt* cst1 = dyn_cast<ConstantInt>(gep->idx_begin()))
  if(ConstantInt* cst2 = dyn_cast<ConstantInt>(off)){
    return (*builder_)->CreateGEP(gep->getPointerOperand(),
                                  (*builder_)->CreateAdd(cst1, cst2));
  }
  // ptr + (off + cst) -> (ptr + off) + cst
  if(auto* bin = dyn_cast<BinaryOperator>(off))
  if(bin->getOpcode() == llvm::BinaryOperator::BinaryOps::Add)
  if(ConstantInt* cst = dyn_cast<ConstantInt>(bin->getOperand(1))){
    return (*builder_)->CreateGEP((*builder_)->CreateGEP(ptr, bin->getOperand(0)),
                                  bin->getOperand(1));
  }
  // default
 return (*builder_)->CreateGEP(ptr, off, name);
}

//Value* geper::operator()(Type *ty, Value *ptr, std::vector<Value *> vals, const std::string &name) {
//  return (*builder_)->CreateGEP(ty, ptr, vals, name);
//}

} // namespace triton::codegen