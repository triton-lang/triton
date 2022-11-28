#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "triton/Conversion/TritonGPUToLLVM/TritonGPUToLLVM.h"
#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPU.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Target/LLVMIR/LLVMIRTranslation.h"
#include "triton/driver/llvm.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include <iostream>

namespace mlir {
namespace triton {

OwningOpRef<ModuleOp> loadMLIRModule(llvm::StringRef inputFilename,
                                     MLIRContext &context) {
  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return nullptr;
  }

  mlir::DialectRegistry registry;
  registry.insert<TritonDialect, triton::gpu::TritonGPUDialect,
                  mlir::math::MathDialect, arith::ArithmeticDialect,
                  StandardOpsDialect, scf::SCFDialect>();

  context.appendDialectRegistry(registry);

  auto processBuffer = [&](std::unique_ptr<llvm::MemoryBuffer> ownedBuffer)
      -> OwningOpRef<ModuleOp> {
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(ownedBuffer), SMLoc());

    context.loadAllAvailableDialects();
    context.allowUnregisteredDialects();

    OwningOpRef<ModuleOp> module(parseSourceFile(sourceMgr, &context));
    if (!module) {
      llvm::errs() << "Parse MLIR file failed.";
      return nullptr;
    }

    return module;
  };

  auto module = processBuffer(std::move(input));
  if (!module) {
    return nullptr;
  }

  return module;
}

LogicalResult tritonTranslateMain(int argc, char **argv,
                                  llvm::StringRef toolName) {
  static llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"),
      llvm::cl::init("-"));

  static llvm::cl::opt<std::string> outputFilename(
      "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
      llvm::cl::init("-"));

  static llvm::cl::opt<std::string> targetKind(
      "target", llvm::cl::desc("<translation target, options: llvmir/ptx>"),
      llvm::cl::value_desc("target"), llvm::cl::init("llvmir"));

  static llvm::cl::opt<int> SMArch("sm", llvm::cl::desc("sm arch"),
                                   llvm::cl::init(80));

  static llvm::cl::opt<int> ptxVersion(
      "ptx-version", llvm::cl::desc("PTX version"), llvm::cl::init(10000));

  llvm::InitLLVM y(argc, argv);

  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, toolName);

  mlir::MLIRContext context;
  auto module = loadMLIRModule(inputFilename, context);
  if (!module) {
    return failure();
  }

  std::string errorMessage;
  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  llvm::LLVMContext llvmContext;
  auto llvmir =
      translateTritonGPUToLLVMIR(&llvmContext, *module, SMArch.getValue());
  if (!llvmir) {
    llvm::errs() << "Translate to LLVM IR failed";
  }

  if (targetKind == "llvmir")
    llvm::outs() << *llvmir << '\n';
  else if (targetKind == "ptx")
    llvm::outs() << ::triton::driver::llir_to_ptx(
        llvmir.get(), SMArch.getValue(), ptxVersion.getValue());

  return success();
}

} // namespace triton
} // namespace mlir

int main(int argc, char **argv) {
  return failed(mlir::triton::tritonTranslateMain(
      argc, argv, "Triton Translate Testing Tool."));
}
