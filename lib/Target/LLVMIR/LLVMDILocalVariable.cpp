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
      // TODO: getting type info requires improvement on higher level ast walking
      mlir::LLVM::DITypeAttr diTypeAttr= LLVM::DINullTypeAttr::get(context);
      mlir::LLVM::DIFlags diFlags = LLVM::DIFlags::Zero;


      // LLVM Dialect to LLVM translation requires DILocalScope when DILocalVariable is present
      LLVM::DILocalScopeAttr diLocalScopeAttr = dyn_cast<LLVM::DILocalScopeAttr>(diSubprogramAttr);

      // DILocalVariable of LLVM Dialect, which will be translated to LLVM IR's llvm::DILocalVariable
      LLVM::DILocalVariableAttr diLocalVarAttr;

      // TODO: current parameter only for first viable result for now
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

  /// Attempt to extract a filename for the given loc.
  FileLineColLoc extractFileLoc(Location loc) {
    if (auto fileLoc = dyn_cast<FileLineColLoc>(loc))
      return fileLoc;
    if (auto nameLoc = dyn_cast<NameLoc>(loc))
      return extractFileLoc(nameLoc.getChildLoc());
    if (auto opaqueLoc = dyn_cast<OpaqueLoc>(loc))
      return extractFileLoc(opaqueLoc.getFallbackLocation());
    if (auto fusedLoc = dyn_cast<FusedLoc>(loc))
      return extractFileLoc(fusedLoc.getLocations().front());
    if (auto callerLoc = dyn_cast<CallSiteLoc>(loc))
      return extractFileLoc(callerLoc.getCaller());
    StringAttr unknownFile = mlir::StringAttr::get(loc.getContext(), "<unknown>");
    return mlir::FileLineColLoc::get(unknownFile, 0, 0);
  }

  // Follow the same logic as LLVMDIScopePass to construct a subprogram scope
  LLVM::DISubprogramAttr getDISubprogramAttr(LLVM::LLVMFuncOp funcOp){
    Location loc = funcOp.getLoc();
    if (auto fusedSubprogramAttr = loc->findInstanceOf<mlir::FusedLocWith<LLVM::DISubprogramAttr>>())
      return fusedSubprogramAttr.getMetadata();

    MLIRContext *context = &getContext();

    // To find a DICompileUnitAttr attached to a parent (the module for
    // example), otherwise create a default one.
    LLVM::DICompileUnitAttr compileUnitAttr;
    if (ModuleOp module = funcOp->getParentOfType<ModuleOp>()) {
      auto fusedCompileUnitAttr =
          module->getLoc()
              ->findInstanceOf<mlir::FusedLocWith<LLVM::DICompileUnitAttr>>();
      if (fusedCompileUnitAttr)
        compileUnitAttr = fusedCompileUnitAttr.getMetadata();
    }

    // Filename, line and colmun to associate to the function.
    LLVM::DIFileAttr fileAttr;
    int64_t line = 1, col = 1;
    FileLineColLoc fileLoc = extractFileLoc(loc);
    if (!fileLoc && compileUnitAttr) {
      fileAttr = compileUnitAttr.getFile();
    } else if (!fileLoc) {
      fileAttr = LLVM::DIFileAttr::get(context, "<unknown>", "");
    } else {
      line = fileLoc.getLine();
      col = fileLoc.getColumn();
      StringRef inputFilePath = fileLoc.getFilename().getValue();
      fileAttr = LLVM::DIFileAttr::get(
          context, llvm::sys::path::filename(inputFilePath),
          llvm::sys::path::parent_path(inputFilePath));
    }

    auto subroutineTypeAttr =
      LLVM::DISubroutineTypeAttr::get(context, llvm::dwarf::DW_CC_normal, {});

    DistinctAttr distinctId;
    auto subprogramFlags = LLVM::DISubprogramFlags::Optimized;
    if (!funcOp.isExternal()) {
      distinctId = mlir::DistinctAttr::create(mlir::UnitAttr::get(context));
      if (!compileUnitAttr) {
        compileUnitAttr = LLVM::DICompileUnitAttr::get(
            distinctId, llvm::dwarf::DW_LANG_C, fileAttr,
            StringAttr::get(context, "triton"),
            /*isOptimized=*/true, LLVM::DIEmissionKind::LineTablesOnly);
      }
      subprogramFlags = subprogramFlags | LLVM::DISubprogramFlags::Definition;
    } else {
      compileUnitAttr = {};
    }


    StringAttr funcNameAttr = funcOp.getNameAttr();
    // Note that scopeline is set differently from LLVM's
    // DIScopeForLLVMFuncOpPass. I don't find reasons why scopeline should be
    // the column offset
    auto subprogramAttr = LLVM::DISubprogramAttr::get(
        context, distinctId, compileUnitAttr, fileAttr, funcNameAttr,
        funcNameAttr, fileAttr, /*line=*/line, /*scopeline=*/line,
        subprogramFlags, subroutineTypeAttr, /*retainNodes=*/{},
        /*annotations=*/{});

    return subprogramAttr;
  }

  // construct a subprogram of an operation by using its parent function's DISubprogramAttr construction
  LLVM::DISubprogramAttr getDISubprogramAttr(Operation op){
    auto funcOp = op.getParentOfType<LLVM::LLVMFuncOp>();
    return getDISubprogramAttr(funcOp);
  }

  // set it while traversing into a function
  LLVM::DISubprogramAttr diSubprogramAttr;

  void runOnOperation() override {
    Operation* op = getOperation();

    getOperation()->walk<WalkOrder::PreOrder>([&](Operation *op) -> void {
      if (isa<LLVM::LLVMFuncOp>(op)){
        diSubprogramAttr = getDISubprogramAttr(cast<LLVM::LLVMFuncOp>(op));
      } else{
        fuseDILocalVariable(op);
      }
    });

  }
};

} // namespace mlir
