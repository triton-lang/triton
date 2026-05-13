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

    // Skip ops outside of a function (e.g. module-level globals) where
    // diSubprogramAttr may not be valid for this op's scope.
    if (!diSubprogramAttr || !op->getParentOfType<LLVM::LLVMFuncOp>()) {
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
      LLVM::DbgValueOp::create(builder, childLoc, opResult, diLocalVarAttr,
                               diExprAttr);
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
    int64_t line = 1;
    FileLineColLoc fileLoc = extractFileLoc(loc);
    if (!fileLoc && compileUnitAttr) {
      fileAttr = compileUnitAttr.getFile();
    } else if (!fileLoc) {
      fileAttr = LLVM::DIFileAttr::get(context, "<unknown>", "");
    } else {
      line = fileLoc.getLine();
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
        // Remaining types are scalar (int/float) or vector types
        // (e.g., from external function declarations for GPU builtins).
        assert((inTy.isIntOrFloat() || isa<mlir::VectorType>(inTy)) &&
               "Expected scalar or vector types");
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
  LLVM::DISubprogramAttr getDISubprogramAttr(Operation &op) {
    auto funcOp = op.getParentOfType<LLVM::LLVMFuncOp>();
    return getDISubprogramAttr(funcOp);
  }

  LLVM::DISubprogramAttr
  fuseFuncArgVariables(LLVM::LLVMFuncOp funcOp,
                       LLVM::DISubprogramAttr subprogramAttr) {

    MLIRContext *context = &getContext();
    OpBuilder builder(context);
    builder.setInsertionPointToStart(&funcOp.getBody().front());

    LLVM::DIFileAttr fileAttr = subprogramAttr.getFile();
    LLVM::DISubroutineTypeAttr subroutineTypeAttr = subprogramAttr.getType();
    int64_t line = subprogramAttr.getLine();
    // The input subprogramAttr has isRecSelf=true. Use it as the scope for
    // retainedNodes variables (proper recursive self-reference pattern).
    auto selfRefScopeAttr = dyn_cast<LLVM::DILocalScopeAttr>(subprogramAttr);
    auto diFlag = LLVM::DIFlags::Zero;

    // Collect argument info and create retainedNodes variables scoped to the
    // isRecSelf=true placeholder (these are nested inside the definition and
    // resolved during translation).
    auto argTypeAttrs = subroutineTypeAttr.getTypes();
    unsigned resNum = funcOp.getNumResults() ? funcOp.getNumResults() : 1;

    struct ArgInfo {
      unsigned argIdx;
      BlockArgument arg;
      Location childLoc;
      StringAttr nameAttr;
      LLVM::DITypeAttr typeAttr;
    };
    llvm::SmallVector<ArgInfo> argInfos;
    llvm::SmallVector<mlir::LLVM::DINodeAttr> retainedNodes;

    for (unsigned idx = resNum; idx < argTypeAttrs.size(); idx++) {
      LLVM::DITypeAttr argTypeAttr = argTypeAttrs[idx];
      unsigned argIdx = idx - resNum;
      BlockArgument arg = funcOp.getArgument(argIdx);

      Location argLoc = arg.getLoc();
      auto nameLoc = dyn_cast<NameLoc>(argLoc);
      if (!nameLoc)
        continue;

      // Create variable with isRecSelf=true scope for retainedNodes
      auto argVarAttr = LLVM::DILocalVariableAttr::get(
          context, selfRefScopeAttr, nameLoc.getName(), fileAttr, line,
          argIdx + 1, 0, argTypeAttr, diFlag);
      retainedNodes.push_back(argVarAttr);

      argInfos.push_back(
          {argIdx, arg, nameLoc.getChildLoc(), nameLoc.getName(), argTypeAttr});
    }

    // Create the resolved subprogram (isRecSelf=false) with retainedNodes.
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

    // Now create DbgValueOps with variables scoped to the RESOLVED subprogram.
    // The isRecSelf=true scope must NOT appear in DbgValueOps because the
    // translator's recursiveNodeMap entry is only valid during translation of
    // the isRecSelf=false definition and is popped afterwards.
    auto resolvedScopeAttr = dyn_cast<LLVM::DILocalScopeAttr>(subprogramAttr);
    for (auto &info : argInfos) {
      auto argVarAttr = LLVM::DILocalVariableAttr::get(
          context, resolvedScopeAttr, info.nameAttr, fileAttr, line,
          info.argIdx + 1, 0, info.typeAttr, diFlag);
      auto exprAttr = LLVM::DIExpressionAttr::get(context);
      (void)LLVM::DbgValueOp::create(builder, info.childLoc, info.arg,
                                     argVarAttr, exprAttr);
    }

    Location loc = funcOp.getLoc();
    // Unwrap the old FusedLoc to avoid nesting a stale isRecSelf=true
    // subprogram in the location chain.
    if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
      if (fusedLoc.getMetadata() &&
          isa<LLVM::DISubprogramAttr>(fusedLoc.getMetadata())) {
        loc = FusedLoc::get(context, fusedLoc.getLocations(), Attribute());
      }
    }
    funcOp->setLoc(mlir::FusedLoc::get(context, {loc}, subprogramAttr));
    return subprogramAttr;
  }

  // After creating the resolved subprogram, fix any DILexicalBlockFileAttr
  // locations in the function that still reference the old isRecSelf=true
  // subprogram. These were created by add_di_scope before this pass ran.
  void fixLexicalBlockScopes(LLVM::LLVMFuncOp funcOp,
                             LLVM::DISubprogramAttr oldSubprogram,
                             LLVM::DISubprogramAttr newSubprogram) {
    MLIRContext *context = &getContext();
    funcOp.walk([&](Operation *op) {
      Location loc = op->getLoc();
      Location newLoc =
          replaceLexicalBlockScope(context, loc, oldSubprogram, newSubprogram);
      if (newLoc != loc)
        op->setLoc(newLoc);
    });
  }

  Location replaceLexicalBlockScope(MLIRContext *context, Location loc,
                                    LLVM::DISubprogramAttr oldSP,
                                    LLVM::DISubprogramAttr newSP) {
    if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
      // Recursively fix inner locations
      SmallVector<Location> newLocs;
      bool changed = false;
      for (Location inner : fusedLoc.getLocations()) {
        Location fixed = replaceLexicalBlockScope(context, inner, oldSP, newSP);
        newLocs.push_back(fixed);
        if (fixed != inner)
          changed = true;
      }

      auto metadata = fusedLoc.getMetadata();
      if (auto lexBlock =
              dyn_cast_or_null<LLVM::DILexicalBlockFileAttr>(metadata)) {
        if (auto scope =
                dyn_cast_or_null<LLVM::DISubprogramAttr>(lexBlock.getScope())) {
          if (scope == oldSP) {
            auto newBlock = LLVM::DILexicalBlockFileAttr::get(
                context, newSP, lexBlock.getFile(),
                lexBlock.getDiscriminator());
            return FusedLoc::get(context, newLocs, newBlock);
          }
        }
      }

      if (changed)
        return FusedLoc::get(context, newLocs, metadata);
    } else if (auto callSiteLoc = dyn_cast<CallSiteLoc>(loc)) {
      Location newCallee = replaceLexicalBlockScope(
          context, callSiteLoc.getCallee(), oldSP, newSP);
      Location newCaller = replaceLexicalBlockScope(
          context, callSiteLoc.getCaller(), oldSP, newSP);
      if (newCallee != callSiteLoc.getCallee() ||
          newCaller != callSiteLoc.getCaller())
        return CallSiteLoc::get(newCallee, newCaller);
    }
    return loc;
  }

  // set it while traversing into a function
  LLVM::DISubprogramAttr diSubprogramAttr = {};

  void runOnOperation() override {
    getOperation()->walk<WalkOrder::PreOrder>([&](Operation *op) -> void {
      if (isa<LLVM::LLVMFuncOp>(op)) {
        auto funcOp = cast<LLVM::LLVMFuncOp>(op);
        auto oldSubprogram = getDISubprogramAttr(funcOp);
        // External declarations (e.g., runtime builtins like vprintf) have no
        // body, so we cannot insert debug value intrinsics into them.
        if (!funcOp.isExternal()) {
          diSubprogramAttr = fuseFuncArgVariables(funcOp, oldSubprogram);
          // Fix DILexicalBlockFileAttr locations that still reference the old
          // isRecSelf=true subprogram from add_di_scope.
          if (oldSubprogram.getIsRecSelf())
            fixLexicalBlockScopes(funcOp, oldSubprogram, diSubprogramAttr);
        } else {
          diSubprogramAttr = oldSubprogram;
        }
      } else {
        fuseDILocalVariable(op);
      }
    });
  }
};

} // namespace mlir
