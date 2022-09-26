#include "triton/Target/PTX/PTXTranslation.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "triton/Target/LLVMIR/LLVMIRTranslation.h"

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <regex>

namespace triton {

extern "C" {
int set_curterm(char *nterm) { return 0; }
int del_curterm(char *nterm) { return 0; }
int tigetnum(char *capname) { return 0; }
int setupterm(char *term, int fildes, int *errret) { return 0; }
}

static void init_llvm() {
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();
}

static bool find_and_replace(std::string &str, const std::string &begin,
                             const std::string &end,
                             const std::string &target) {
  size_t start_replace = str.find(begin);
  if (start_replace == std::string::npos)
    return false;
  size_t end_replace = str.find(end, start_replace);
  if (end_replace == std::string::npos)
    return false;
  str.replace(start_replace, end_replace + 1 - start_replace, target);
  return true;
}

static std::string llir_to_ptx(llvm::Module *module, int capability, int ptx) {
  // LLVM version in use may not officially support target hardware
  int max_nvvm_cc = 75;
  int max_nvvm_ptx = 74;
  // options
  auto options = llvm::cl::getRegisteredOptions();
  auto *short_ptr =
      static_cast<llvm::cl::opt<bool> *>(options["nvptx-short-ptr"]);
  assert(short_ptr);
  short_ptr->setValue(true);
  // compute capability
  std::string sm = "sm_" + std::to_string(capability);
  // max PTX version
  int ptx_major = ptx / 10;
  int ptx_minor = ptx % 10;
  // create
  llvm::SmallVector<char, 0> buffer;
  std::string triple = "nvptx64-nvidia-cuda";
  std::string proc = "sm_" + std::to_string(std::min(capability, max_nvvm_cc));
  std::string layout = "";
  std::string features = "";
  // std::string features = "+ptx" + std::to_string(std::min(ptx,
  // max_nvvm_ptx));
  init_llvm();
  // verify and store llvm
  llvm::legacy::PassManager pm;
  pm.add(llvm::createVerifierPass());
  pm.run(*module);
  // module->print(llvm::outs(), nullptr);

  // create machine
  module->setTargetTriple(triple);
  std::string error;
  auto target =
      llvm::TargetRegistry::lookupTarget(module->getTargetTriple(), error);
  llvm::TargetOptions opt;
  opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  opt.UnsafeFPMath = false;
  opt.NoInfsFPMath = false;
  opt.NoNaNsFPMath = true;
  llvm::TargetMachine *machine = target->createTargetMachine(
      module->getTargetTriple(), proc, features, opt, llvm::Reloc::PIC_,
      llvm::None, llvm::CodeGenOpt::Aggressive);
  // set data layout
  if (layout.empty())
    module->setDataLayout(machine->createDataLayout());
  else
    module->setDataLayout(layout);
  // emit machine code
  for (llvm::Function &f : module->functions())
    f.addFnAttr(llvm::Attribute::AlwaysInline);
  llvm::legacy::PassManager pass;
  llvm::raw_svector_ostream stream(buffer);
  // emit
  machine->addPassesToEmitFile(pass, stream, nullptr,
                               llvm::CodeGenFileType::CGFT_AssemblyFile);
  pass.run(*module);

  // post-process
  std::string result(buffer.begin(), buffer.end());
  find_and_replace(result, ".version", "\n",
                   ".version " + std::to_string(ptx_major) + "." +
                       std::to_string(ptx_minor) + "\n");
  find_and_replace(result, ".target", "\n", ".target " + sm + "\n");
  while (find_and_replace(result, "\t// begin inline asm", "\n", ""))
    ;
  while (find_and_replace(result, "\t// end inline asm", "\n", ""))
    ;
  return result;
}

std::string translateLLVMIRToPTX(llvm::Module &module, int cc, int version) {
  auto ptxCode = llir_to_ptx(&module, cc, version);
  return ptxCode;
}

} // namespace triton
