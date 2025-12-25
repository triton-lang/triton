#include "lib/Target/LLVMIR/LLVMDIUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "triton/Target/LLVMIR/Passes.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"

//===----------------------------------------------------------------------===//
// This file implements a pass to add debug info scope to LLVM operations, and
// is inspired by the DIScopeForLLVMFuncOpPass in LLVM/MLIR. Different from the
// DIScopeForLLVMFuncOpPass, this pass also handles inlined functions.
//===----------------------------------------------------------------------===//

namespace mlir {

#define GEN_PASS_DEF_LLVMDISCOPE
#include "triton/Target/LLVMIR/Passes.h.inc"

using namespace LLVMDIUtils;

/// Add a debug info scope to LLVMFuncOp that are missing it.
struct LLVMDIScopePass : public impl::LLVMDIScopeBase<LLVMDIScopePass> {
  void setSubprogramAttr(LLVM::LLVMFuncOp funcOp) {
    Location loc = funcOp.getLoc();
    if (loc->findInstanceOf<mlir::FusedLocWith<LLVM::DISubprogramAttr>>())
      return;

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

    // Figure out debug information (`subprogramFlags` and `compileUnitAttr`) to
    // attach to the function definition / declaration. External functions are
    // declarations only, and are defined in a different compile unit, so mark
    // them appropriately in `subprogramFlags`, and set an empty
    // `compileUnitAttr`.
    bool extractDILocalVar =
        triton::tools::getBoolEnv("LLVM_EXTRACT_DI_LOCAL_VARIABLES");
    bool disableLineInfo =
        triton::tools::getBoolEnv("TRITON_DISABLE_LINE_INFO");
    DistinctAttr recId; // Recursive ID to mark the DICompileUnitAttr and
                        // DISubprogramAttr that are recursively defined
    auto subprogramFlags = LLVM::DISubprogramFlags::Optimized;
    if (!funcOp.isExternal()) {
      recId = mlir::DistinctAttr::create(mlir::UnitAttr::get(context));
      if (!compileUnitAttr) {
        compileUnitAttr = LLVM::DICompileUnitAttr::get(
            recId, llvm::dwarf::DW_LANG_C, fileAttr,
            StringAttr::get(context, "triton"),
            /*isOptimized=*/true,
            extractDILocalVar
                ? LLVM::DIEmissionKind::Full
                : LLVM::DIEmissionKind::
                      LineTablesOnly); // DIEmissionKind::Full is required by
                                       // emitting ptx with dbg-metadata
                                       // (otherwise assertion fail)
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
        // If no valid pointee type for this function argument, use null type as
        // unknown type.
        mlir::Type elTy =
            pointeeTy ? pointeeTy.getValue() : builder.getNoneType();
        LLVM::DITypeAttr tyAttr =
            convertPtrType(context, ptrTy, elTy, sizeInBits);
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

    bool isRecSelf = !disableLineInfo && extractDILocalVar;
    auto id = mlir::DistinctAttr::create(mlir::UnitAttr::get(context));
    auto subprogramAttr = LLVM::DISubprogramAttr::get(
        context, recId, isRecSelf, id, compileUnitAttr, fileAttr, funcNameAttr,
        funcNameAttr, fileAttr,
        /*line=*/line, /*scopeline=*/line, subprogramFlags, subroutineTypeAttr,
        /*retainNodes=*/{}, /*annotations=*/{});

    funcOp->setLoc(FusedLoc::get(context, {loc}, subprogramAttr));
  }

  void setLexicalBlockFileAttr(Operation *op) {
    Location opLoc = op->getLoc();
    if (!isa<CallSiteLoc>(opLoc))
      return;

    auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    auto funcOpLoc = mlir::cast<FusedLoc>(funcOp.getLoc());
    auto scopeAttr =
        mlir::cast<LLVM::DISubprogramAttr>(funcOpLoc.getMetadata());

    MLIRContext *ctx = op->getContext();
    std::function<Location(Location)> makeScoped =
        [&](Location loc) -> Location {
      if (auto cs = dyn_cast<CallSiteLoc>(loc)) {
        Location newCallee = makeScoped(cs.getCallee());
        Location newCaller = makeScoped(cs.getCaller());
        return CallSiteLoc::get(newCallee, newCaller);
      }

      // Build a DIFile for this leaf location
      FileLineColLoc fileLine = extractFileLoc(loc, /*getCaller=*/false);
      StringRef inputFilePath = fileLine.getFilename().getValue();
      LLVM::DIFileAttr fileAttr =
          LLVM::DIFileAttr::get(ctx, llvm::sys::path::filename(inputFilePath),
                                llvm::sys::path::parent_path(inputFilePath));

      auto lexicalBlock =
          LLVM::DILexicalBlockFileAttr::get(ctx, scopeAttr, fileAttr,
                                            /*discriminator=*/0);
      return FusedLoc::get(ctx, {loc}, lexicalBlock);
    };

    op->setLoc(makeScoped(opLoc));
  }

  void runOnOperation() override {
    getOperation()->walk<WalkOrder::PreOrder>([&](Operation *op) -> void {
      if (isa<LLVM::LLVMFuncOp>(op))
        setSubprogramAttr(cast<LLVM::LLVMFuncOp>(op));
      else
        setLexicalBlockFileAttr(op);
    });
  }
};

} // namespace mlir
