#ifndef TDL_INCLUDE_CODEGEN_SELECTION_H
#define TDL_INCLUDE_CODEGEN_SELECTION_H

#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include "ir/context.h"
#include "ir/module.h"
#include "ir/function.h"
#include "ir/type.h"


namespace llvm{
  class Type;
  class Value;
  class Instruction;
  class Constant;
  class LLVMContext;
}

namespace tdl{
namespace codegen{

class allocation;

struct distributed_axis {

};



class selection{
  typedef std::map<ir::value *, llvm::Value *> vmap_t;
  typedef std::map<ir::basic_block *, llvm::BasicBlock *> bmap_t;

private:
  llvm::Type*        llvm_type(ir::type *ty, llvm::LLVMContext &ctx);
  llvm::Value*       llvm_value(ir::value *v,llvm:: LLVMContext &ctx);
  llvm::Instruction* llvm_inst(ir::instruction *inst, llvm::LLVMContext &ctx);
  llvm::Constant*    llvm_constant(ir::constant *cst, llvm::LLVMContext &ctx);

public:
  selection(allocation *alloc): alloc_(alloc){ }
  void run(ir::module &src, llvm::Module &dst);

private:
  vmap_t vmap_;
  bmap_t bmap_;
  allocation *alloc_;

};

}
}

#endif
