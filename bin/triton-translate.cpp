#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "triton/Conversion/TritonGPUToLLVM/TritonGPUToLLVMPass.h"
#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Target/LLVMIR/LLVMIRTranslation.h"
#include "triton/Target/PTX/PTXTranslation.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

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
  registry
      .insert<TritonDialect, triton::gpu::TritonGPUDialect,
              triton::nvidia_gpu::TritonNvidiaGPUDialect,
              mlir::math::MathDialect, arith::ArithDialect, scf::SCFDialect>();

  context.appendDialectRegistry(registry);

  auto processBuffer = [&](std::unique_ptr<llvm::MemoryBuffer> ownedBuffer)
      -> OwningOpRef<ModuleOp> {
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(ownedBuffer), SMLoc());

    context.loadAllAvailableDialects();
    context.allowUnregisteredDialects();

    OwningOpRef<ModuleOp> module =
        parseSourceFile<ModuleOp>(sourceMgr, &context);
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
      "target",
      llvm::cl::desc("<translation target, options: llvmir/ptx/hsaco>"),
      llvm::cl::value_desc("target"), llvm::cl::init("llvmir"));

  static llvm::cl::opt<int> SMArch("sm", llvm::cl::desc("sm arch"),
                                   llvm::cl::init(80));

  static llvm::cl::opt<int> ptxVersion(
      "ptx-version", llvm::cl::desc("PTX version"), llvm::cl::init(10000));

  static llvm::cl::opt<std::string> GCNArch(
      "gfx", llvm::cl::desc("AMDGCN target. e.g. '90a'"),
      llvm::cl::value_desc("architecture"), llvm::cl::init("90a"));

  static llvm::cl::opt<std::string> GCNTriple(
      "amdgcn", llvm::cl::desc("AMDGCN triple. e.g. '-amd-amdhsa'"),
      llvm::cl::value_desc("target triple"), llvm::cl::init("-amd-amdhsa"));

  static llvm::cl::opt<std::string> GCNFeatures(
      "", llvm::cl::desc("AMDGCN features. e.g. '+sramecc,-xnack'"),
      llvm::cl::value_desc("features"), llvm::cl::init("+sramecc,-xnack"));

  static llvm::cl::opt<bool> enableFpFusion(
      "enable-fp-fusion", llvm::cl::desc("Enables fusion of fadd/fmul"),
      llvm::cl::init(true));

  llvm::InitLLVM y(argc, argv);

  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
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
  mlir::triton::gpu::TMAMetadataTy tmaInfos;
  auto llvmir = translateTritonGPUToLLVMIR(
      &llvmContext, *module, SMArch.getValue(), tmaInfos, Target::Default);

  if (!llvmir) {
    llvm::errs() << "Translate to LLVM IR failed";
  }

  if (targetKind == "llvmir") {
    llvm::outs() << *llvmir << '\n';
  } else if (targetKind == "ptx") {
    llvm::outs() << ::triton::translateLLVMIRToPTX(*llvmir, SMArch.getValue(),
                                                   ptxVersion.getValue(),
                                                   enableFpFusion.getValue());
  } else {
    llvm::errs() << "Error: Unknown target specified: " << targetKind << "\n";
    return failure();
  }

  return success();
}

} // namespace triton
} // namespace mlir

int main(int argc, char **argv) {
  return failed(mlir::triton::tritonTranslateMain(
      argc, argv, "Triton Translate Testing Tool."));
}
