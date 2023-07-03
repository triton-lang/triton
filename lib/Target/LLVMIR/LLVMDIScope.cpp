#include "triton/Target/LLVMIR/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"

//===----------------------------------------------------------------------===//
// This file implements a pass to add debug info scope to LLVM operations, and
// is inspired by the DIScopeForLLVMFuncOpPass in LLVM. Different from the
// DIScopeForLLVMFuncOpPass, this pass also handles inlined functions.
//===----------------------------------------------------------------------===//

using namespace mlir;

#define GEN_PASS_CLASSES
#include "triton/Target/LLVMIR/Passes.h.inc"

namespace {

/// Attempt to extract a filename for the given loc.
static FileLineColLoc extractFileLoc(Location loc) {
  if (auto fileLoc = loc.dyn_cast<FileLineColLoc>())
    return fileLoc;
  if (auto nameLoc = loc.dyn_cast<NameLoc>())
    return extractFileLoc(nameLoc.getChildLoc());
  if (auto opaqueLoc = loc.dyn_cast<OpaqueLoc>())
    return extractFileLoc(opaqueLoc.getFallbackLocation());
  return FileLineColLoc();
}

/// Add a debug info scope to LLVMFuncOp that are missing it.
struct LLVMDIScopePass : public LLVMDIScopeBase<LLVMDIScopePass> {
  LLVMDIScopePass() = default;

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
    if (!compileUnitAttr) {
      compileUnitAttr = LLVM::DICompileUnitAttr::get(
          context, llvm::dwarf::DW_LANG_C, fileAttr,
          StringAttr::get(context, "triton"), /*isOptimized=*/true,
          LLVM::DIEmissionKind::LineTablesOnly);
    }
    auto subroutineTypeAttr =
        LLVM::DISubroutineTypeAttr::get(context, llvm::dwarf::DW_CC_normal, {});

    StringAttr funcNameAttr = funcOp.getNameAttr();
    // Note that scopeline is set differently from LLVM's
    // DIScopeForLLVMFuncOpPass. I don't find reasons why scopeline should be
    // the column offset
    auto subprogramAttr =
        LLVM::DISubprogramAttr::get(context, compileUnitAttr, fileAttr,
                                    funcNameAttr, funcNameAttr, fileAttr,
                                    /*line=*/line,
                                    /*scopeline=*/line,
                                    LLVM::DISubprogramFlags::Definition |
                                        LLVM::DISubprogramFlags::Optimized,
                                    subroutineTypeAttr);
    funcOp->setLoc(FusedLoc::get(context, {loc}, subprogramAttr));
  }

  void setLexicalBlockFileAttr(Operation *op) {
    auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!funcOp)
      return;

    auto funcOpLoc = funcOp.getLoc().cast<FusedLoc>();
    auto funcFileName = extractFileLoc(funcOpLoc).getFilename();
    auto subProgramAttr =
        funcOpLoc.getMetadata().cast<LLVM::DISubprogramAttr>();
    auto opLoc = op->getLoc();
    if (auto callSiteLoc = dyn_cast<CallSiteLoc>(opLoc)) {
      auto calleeLoc = callSiteLoc.getCallee();
      auto calleeFileName = extractFileLoc(calleeLoc).getFilename();
      if (calleeFileName != funcFileName) {
        // Create a DILexicalBlockFile for the second file, using the
        // DISubprogram as its scope
        auto context = op->getContext();
        LLVM::DIFileAttr calleeFileAttr =
            LLVM::DIFileAttr::get(context, calleeFileName, "");
        auto lexicalBlockFileAttr = LLVM::DILexicalBlockFileAttr::get(
            context, subProgramAttr, calleeFileAttr, /*discriminator=*/0);
        op->setLoc(FusedLoc::get(context, {opLoc}, lexicalBlockFileAttr));
      }
    }
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

} // end anonymous namespace

std::unique_ptr<Pass> mlir::createLLVMDIScopePass() {
  return std::make_unique<LLVMDIScopePass>();
}