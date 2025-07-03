#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "triton/Target/LLVMIR/Passes.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"

// #include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
//===----------------------------------------------------------------------===//
// This file implements a pass to add ... to LLVM operations, and ...
//===----------------------------------------------------------------------===//

namespace mlir {

#define DEBUG_TYPE "name-preservation"

#define GEN_PASS_DEF_LLVMDILOCALVARIABLE
#include "triton/Target/LLVMIR/Passes.h.inc"

struct LLVMDILocalVariablePass : public impl::LLVMDILocalVariableBase<LLVMDILocalVariablePass> {


  void fuseDILocalVariable(Operation* op){
    if(op->getNumResults() == 0){
      return;
    }

    MLIRContext* context = op->getContext();
    OpBuilder builder(context);
    Location loc = op->getLoc();

    // if the location is a NameLoc, a.k.a it defines a value, then insert a dbg-value intrinsic after the op
    if(auto nameLoc = dyn_cast<NameLoc>(loc)){
      Location childLoc = nameLoc.getChildLoc();
      StringAttr nameAttr = nameLoc.getName();

      // also see reference of operation construction from mlir/lib/Target/LLVMIR/ModuleImport.cpp
      // which translated llvm::Module into mlir::LLVM::Operation

      // TODO: Those instantiation using defult is necessary for first viable result, but no meaning for now
      mlir::LLVM::DIFileAttr diFileAttr = LLVM::DIFileAttr::get(context, "<unknown>", "<unknown>");
      // FIXME NOW: construct a valid scope to enable llvmdialect-llvmir translation!!!
      mlir::LLVM::DIScopeAttr diScopeAttr = diFileAttr;
      // TODO: getting type info requires improvement on higher level ast walking
      mlir::LLVM::DITypeAttr diTypeAttr= LLVM::DINullTypeAttr::get(context);
      mlir::LLVM::DIFlags diFlags = LLVM::DIFlags::Zero;

      // TODO: current parameter only for first viable result for now
      LLVM::DILexicalBlockFileAttr lexicalBlockFileAttr = LLVM::DILexicalBlockFileAttr::get(
        context, diScopeAttr, diFileAttr, /*discriminator=*/0);

      // LLVM Dialect to LLVM translation requires DILocalScope when DILocalVariable is present
      LLVM::DILocalScopeAttr diLocalScopeAttr = dyn_cast<LLVM::DILocalScopeAttr>(lexicalBlockFileAttr);

      // DILocalVariable of LLVM Dialect, which will be translated to LLVM IR's llvm::DILocalVariable
      LLVM::DILocalVariableAttr diLocalVarAttr;

      diLocalVarAttr = LLVM::DILocalVariableAttr::get(context, diLocalScopeAttr, nameAttr, diFileAttr, 0, 0, 0, diTypeAttr, diFlags);

      LLVM::DIExpressionAttr diExprAttr = LLVM::DIExpressionAttr::get(context);
      // Note: must set insertion point before calling create since it will automatically insert the op
      builder.setInsertionPointAfter(op);
      // a subclass of mlir::Value, which is the value defined by this operation
      OpResult opResult = op->getResult(0);
      // create and insert this call-dbg-value intrinsic after the op
      Operation* dbgOp = builder.create<LLVM::DbgValueOp>(childLoc, opResult, diLocalVarAttr, diExprAttr);

    }
  }

  void runOnOperation() override {
    Operation* op = getOperation();

    getOperation()->walk<WalkOrder::PreOrder>([&](Operation *op) -> void {
      fuseDILocalVariable(op);
    });

  }
};

} // namespace mlir
