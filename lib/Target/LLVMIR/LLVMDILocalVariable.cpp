#include "lib/Target/LLVMIR/LLVMDIUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "triton/Target/LLVMIR/Passes.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"

// #include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
//===----------------------------------------------------------------------===//
// This file implements a pass to add ... to LLVM operations, and ...
//===----------------------------------------------------------------------===//

namespace mlir {
using namespace LLVMDIUtils;

#define DEBUG_TYPE "name-preservation"

#define GEN_PASS_DEF_LLVMDILOCALVARIABLE
#include "triton/Target/LLVMIR/Passes.h.inc"

struct LLVMDILocalVariablePass
    : public impl::LLVMDILocalVariableBase<LLVMDILocalVariablePass> {

  void fuseDILocalVariable(Operation *op) {
    if (op->getNumResults() == 0) {
      return;
    }

    MLIRContext *context = op->getContext();
    OpBuilder builder(context);
    Location loc = op->getLoc();

    // if the location is a NameLoc, a.k.a it defines a value, then insert a
    // dbg-value intrinsic after the op
    if (auto nameLoc = dyn_cast<NameLoc>(loc)) {
      Location childLoc = nameLoc.getChildLoc();
      StringAttr nameAttr = nameLoc.getName();

      // also see reference of operation construction from
      // mlir/lib/Target/LLVMIR/ModuleImport.cpp which translated llvm::Module
      // into mlir::LLVM::Operation

      // TODO: Those instantiation using defult is necessary for first viable
      // result, but no meaning for now
      LLVM::DIFileAttr diFileAttr =
          LLVM::DIFileAttr::get(context, "<unknown>", "<unknown>");

      // Extracting type info into DITypeAttr
      mlir::Type resultType = op->getResult(0).getType();
      if (isa<LLVM::LLVMVoidType>(resultType)) {
        // we cannot allow void type to be noted as data type, otherwise trigger
        // later assertion fault
        return;
      }
      LLVM::DITypeAttr diTypeAttr = convertType(context, resultType);
      LLVM::DIFlags diFlags = LLVM::DIFlags::Zero;

      // LLVM Dialect to LLVM translation requires DILocalScope when
      // DILocalVariable is present
      LLVM::DILocalScopeAttr diLocalScopeAttr =
          dyn_cast<LLVM::DILocalScopeAttr>(diSubprogramAttr);

      // DILocalVariable of LLVM Dialect, which will be translated to LLVM IR's
      // llvm::DILocalVariable
      LLVM::DILocalVariableAttr diLocalVarAttr;

      // TODO: current parameter only for first viable result for now
      diLocalVarAttr = LLVM::DILocalVariableAttr::get(
          context, diLocalScopeAttr, nameAttr, diFileAttr, 0, 0, 0, diTypeAttr,
          diFlags);

      LLVM::DIExpressionAttr diExprAttr = LLVM::DIExpressionAttr::get(context);
      // Note: must set insertion point before calling create since it will
      // automatically insert the op
      builder.setInsertionPointAfter(op);
      // a subclass of mlir::Value, which is the value defined by this operation
      OpResult opResult = op->getResult(0);
      // create and insert this call-dbg-value intrinsic after the op
      Operation *dbgOp = LLVM::DbgValueOp::create(builder, childLoc, opResult,
                                                  diLocalVarAttr, diExprAttr);
    }
  }

  // Follow the same logic as LLVMDIScopePass to construct a subprogram scope
  LLVM::DISubprogramAttr getDISubprogramAttr(LLVM::LLVMFuncOp funcOp) {
    Location loc = funcOp.getLoc();
    if (auto fusedSubprogramAttr =
            loc->findInstanceOf<mlir::FusedLocWith<LLVM::DISubprogramAttr>>())
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

    DistinctAttr distinctId;
    auto subprogramFlags = LLVM::DISubprogramFlags::Optimized;
    if (!funcOp.isExternal()) {
      distinctId = mlir::DistinctAttr::create(mlir::UnitAttr::get(context));
      if (!compileUnitAttr) {
        compileUnitAttr = LLVM::DICompileUnitAttr::get(
            distinctId, llvm::dwarf::DW_LANG_C, fileAttr,
            StringAttr::get(context, "triton"),
            /*isOptimized=*/true, LLVM::DIEmissionKind::Full);
      }
      subprogramFlags = subprogramFlags | LLVM::DISubprogramFlags::Definition;
    } else {
      compileUnitAttr = {};
    }

    llvm::SmallVector<mlir::LLVM::DITypeAttr> types;
    mlir::DataLayout dl(
        funcOp.getOperation()->getParentOfType<mlir::ModuleOp>());
    for (auto resTy : funcOp.getResultTypes()) {
      LLVM::DITypeAttr tyAttr = convertType(context, resTy);
      types.push_back(tyAttr);
    }
    // If no return type then add a null type as a place holder for that.
    if (types.empty())
      types.push_back(mlir::LLVM::DINullTypeAttr::get(context));

    // Only pointer type and scalar types are supported for now
    OpBuilder builder(context);
    for (auto [idx, inTy] : llvm::enumerate(funcOp.getArgumentTypes())) {
      if (auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(inTy)) {
        auto pointeeTy =
            funcOp.getArgAttrOfType<TypeAttr>(idx, "tt.pointee_type");
        auto sizeInBits = dl.getTypeSizeInBits(ptrTy);
        // If no valid pointee type for this function argument, skip it.
        mlir::Type elTy =
            pointeeTy ? pointeeTy.getValue() : builder.getNoneType();
        LLVM::DITypeAttr tyAttr = convertPtrType(context, ptrTy, elTy, dl);
        types.push_back(tyAttr);
      } else if (auto structTy = dyn_cast<LLVM::LLVMStructType>(inTy)) {
        LLVM::DITypeAttr tyAttr =
            convertStructType(context, structTy, fileAttr, dl, line);
        types.push_back(tyAttr);
      } else if (auto arrayTy = dyn_cast<LLVM::LLVMArrayType>(inTy)) {
        LLVM::DITypeAttr tyAttr =
            convertArrayType(context, arrayTy, fileAttr, dl, line);
        types.push_back(tyAttr);
      } else {
        // Here assume remaining inTys are only scalar types
        assert(inTy.isIntOrFloat() && "Expected scalar types");
        LLVM::DITypeAttr tyAttr = convertType(context, inTy);
        types.push_back(tyAttr);
      }
    }

    auto subroutineTypeAttr = LLVM::DISubroutineTypeAttr::get(
        context, llvm::dwarf::DW_CC_normal, types);

    StringAttr funcNameAttr = funcOp.getNameAttr();
    // Note that scopeline is set differently from LLVM's
    // DIScopeForLLVMFuncOpPass. I don't find reasons why scopeline should be
    // the column offset

    auto recId = mlir::DistinctAttr::create(mlir::UnitAttr::get(context));
    auto id = mlir::DistinctAttr::create(mlir::UnitAttr::get(context));
    auto subprogramAttr = LLVM::DISubprogramAttr::get(
        context, recId, /*isRecSelf=*/true, id, compileUnitAttr, fileAttr,
        funcNameAttr, funcNameAttr, fileAttr, /*line=*/line, /*scopeline=*/line,
        subprogramFlags, subroutineTypeAttr, /*retainNodes=*/{},
        /*annotations=*/{});

    return subprogramAttr;
  }

  // construct a subprogram of an operation by using its parent function's
  // DISubprogramAttr construction
  LLVM::DISubprogramAttr getDISubprogramAttr(Operation op) {
    auto funcOp = op.getParentOfType<LLVM::LLVMFuncOp>();
    return getDISubprogramAttr(funcOp);
  }

  LLVM::DISubprogramAttr
  fuseFuncArgVariables(LLVM::LLVMFuncOp funcOp,
                       LLVM::DISubprogramAttr subprogramAttr) {

    MLIRContext *context = &getContext();
    OpBuilder builder(context);
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    llvm::SmallVector<mlir::LLVM::DINodeAttr> retainedNodes;

    LLVM::DIFileAttr fileAttr = subprogramAttr.getFile();
    LLVM::DISubroutineTypeAttr subroutineTypeAttr = subprogramAttr.getType();
    int64_t line = subprogramAttr.getLine();
    auto localScopeAttr = dyn_cast<LLVM::DILocalScopeAttr>(subprogramAttr);
    auto diFlag = LLVM::DIFlags::Zero;

    // Extract function arguments and add them to retainedNodes:
    // 0. Extract function argument types from subroutineTypeAttr
    // 1. Create DILocalVariable and DebugValueOp for each arg
    // 2. Add each arg as DILocalVariableAttr to retainedNodes
    auto argTypeAttrs = subroutineTypeAttr.getTypes();
    unsigned resNum = funcOp.getNumResults() ? funcOp.getNumResults() : 1;
    for (unsigned idx = resNum; idx < argTypeAttrs.size(); idx++) {
      LLVM::DITypeAttr argTypeAttr = argTypeAttrs[idx];
      unsigned argIdx = idx - resNum;
      BlockArgument arg = funcOp.getArgument(argIdx);

      Location argLoc = arg.getLoc();
      auto nameLoc = dyn_cast<NameLoc>(argLoc);
      if (!nameLoc)
        continue;
      Location childLoc = nameLoc.getChildLoc();
      StringAttr nameAttr = nameLoc.getName();

      auto argVarAttr = LLVM::DILocalVariableAttr::get(
          context, localScopeAttr, nameAttr, fileAttr, line, argIdx + 1, 0,
          argTypeAttr, diFlag);

      auto exprAttr = LLVM::DIExpressionAttr::get(context);
      (void)LLVM::DbgValueOp::create(builder, childLoc, arg, argVarAttr,
                                     exprAttr);

      retainedNodes.push_back(argVarAttr);
    }

    mlir::DistinctAttr recId = subprogramAttr.getRecId();
    mlir::DistinctAttr id =
        mlir::DistinctAttr::create(mlir::UnitAttr::get(context));
    LLVM::DICompileUnitAttr compileUnitAttr = subprogramAttr.getCompileUnit();
    StringAttr funcNameAttr = subprogramAttr.getName();
    LLVM::DISubprogramFlags subprogramFlags =
        subprogramAttr.getSubprogramFlags();
    subprogramAttr = LLVM::DISubprogramAttr::get(
        context, recId, /*isRecSelf=*/false, id, compileUnitAttr, fileAttr,
        funcNameAttr, funcNameAttr, fileAttr, line, line, subprogramFlags,
        subroutineTypeAttr, retainedNodes, /*annotations=*/{});

    Location loc = funcOp.getLoc();
    // Reset the subprogramAttr with retainedNodes to the funcOp
    funcOp->setLoc(mlir::FusedLoc::get(context, {loc}, subprogramAttr));
    return subprogramAttr;
  }

  // set it while traversing into a function
  LLVM::DISubprogramAttr diSubprogramAttr;

  void runOnOperation() override {
    Operation *op = getOperation();

    getOperation()->walk<WalkOrder::PreOrder>([&](Operation *op) -> void {
      if (isa<LLVM::LLVMFuncOp>(op)) {
        auto funcOp = cast<LLVM::LLVMFuncOp>(op);
        diSubprogramAttr = getDISubprogramAttr(funcOp);
        diSubprogramAttr = fuseFuncArgVariables(funcOp, diSubprogramAttr);
      } else {
        fuseDILocalVariable(op);
      }
    });
  }
};

} // namespace mlir
