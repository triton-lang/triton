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

namespace {

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
  // Prefer the innermost callee for callsite locations.
  if (auto csLoc = dyn_cast<CallSiteLoc>(loc))
    return extractFileLoc(csLoc.getCallee());
  StringAttr unknownFile = mlir::StringAttr::get(loc.getContext(), "<unknown>");
  return mlir::FileLineColLoc::get(unknownFile, 0, 0);
}

} // anonymous namespace

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
    auto subroutineTypeAttr =
        LLVM::DISubroutineTypeAttr::get(context, llvm::dwarf::DW_CC_normal, {});

    // Figure out debug information (`subprogramFlags` and `compileUnitAttr`) to
    // attach to the function definition / declaration. External functions are
    // declarations only, and are defined in a different compile unit, so mark
    // them appropriately in `subprogramFlags`, and set an empty
    // `compileUnitAttr`.
    DistinctAttr distinctId;
    auto subprogramFlags = LLVM::DISubprogramFlags::Optimized;
    if (!funcOp.isExternal()) {
      distinctId = mlir::DistinctAttr::create(mlir::UnitAttr::get(context));
      if (!compileUnitAttr) {
        compileUnitAttr = LLVM::DICompileUnitAttr::get(
            distinctId, llvm::dwarf::DW_LANG_C, fileAttr,
            StringAttr::get(context, "triton"),
            /*isOptimized=*/true,
            triton::tools::getBoolEnv("LLVM_EXTRACT_DI_LOCAL_VARIABLES")
                ? LLVM::DIEmissionKind::Full
                : LLVM::DIEmissionKind::
                      LineTablesOnly); // DIEmissionKind::Full is required by
                                       // emiting ptx with dbg-metadata
                                       // (otherwise assertion fail)
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
      FileLineColLoc fileLine = extractFileLoc(loc);
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
