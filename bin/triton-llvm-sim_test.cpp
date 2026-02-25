#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/SourceMgr.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace mlir;

/// Command line options
static cl::opt<std::string> inputFilename(
    cl::Positional,
    cl::desc("<input MLIR file>"),
    cl::Required
);

static cl::opt<bool> verifyDiagnostics(
    "verify-diagnostics",
    cl::desc("Check that emitted diagnostics match expected ones"),
    cl::init(false)
);

static cl::opt<bool> showAST(
    "show-ast",
    cl::desc("Print the MLIR AST after parsing"),
    cl::init(false)
);

namespace mlir {
namespace triton {

OwningOpRef<ModuleOp> loadMLIRModule(StringRef inputFilename, MLIRContext &context) {
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    WithColor::error(errs()) << "Failed to open input file: " << errorMessage << "\n";
    return nullptr;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*file), SMLoc());

  return parseSourceFile<ModuleOp>(sourceMgr, &context);
}

} // namespace triton
} // namespace mlir

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "MLIR Module Loader\n");

  // Create an MLIR context
  mlir::MLIRContext context;

  // Load MLIR module
  auto module = mlir::triton::loadMLIRModule(inputFilename, context);
  if (!module) {
    WithColor::error(errs()) << "Failed to parse the input file as an MLIR module.\n";
    return 1;
  }

  if (verifyDiagnostics) {
    // Optionally verify diagnostics (this is placeholder logic)
    // Real diagnostic checking would require a diagnostic handler setup.
    llvm::outs() << "Diagnostic verification is not implemented yet.\n";
  }

  if (showAST) {
    module->dump();
  }

  return 0;
}
