#include "triton/Target/PTX/PTXTranslation.h"
#include "triton/Target/LLVMIR/LLVMIRTranslation.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include <filesystem>

namespace triton {

static void initLLVM() {
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();
}

static bool findAndReplace(std::string &str, const std::string &begin,
                           const std::string &end, const std::string &target) {
  size_t startReplace = str.find(begin);
  if (startReplace == std::string::npos)
    return false;
  size_t endReplace = str.find(end, startReplace);
  if (endReplace == std::string::npos)
    return false;
  str.replace(startReplace, endReplace + 1 - startReplace, target);
  return true;
}

// When using the Math Dialect, it is possible that some ops (e.g., log) are
// lowered to a function call. In this case, we need to link the libdevice
// TODO(Keren): handle external linkage other than libdevice?
static void linkExternal(llvm::Module &module) {
  bool hasExternal = false;
  for (auto &func : module) {
    if (func.hasExternalLinkage()) {
      hasExternal = true;
      break;
    }
  }

  if (hasExternal) {
    namespace fs = std::filesystem;
    // Using the default libdevice path:
    // [triton root dir]/python/triton/language/libdevice.10.bc
    static const fs::path libdevice = fs::path(__FILE__)
                                          .parent_path()
                                          .parent_path()
                                          .parent_path()
                                          .parent_path() /
                                      "python" / "triton" / "language" /
                                      "libdevice.10.bc";
    if (mlir::triton::linkExternLib(module, "libdevice", libdevice.string()))
      llvm::errs() << "link failed for libdevice at path: "
                   << libdevice.string();
  }
}

std::string translateLLVMIRToPTX(llvm::Module &module, int cc, int version) {
  linkExternal(module);
  // LLVM version in use may not officially support target hardware
  int maxNNVMCC = 75;
  // options
  auto options = llvm::cl::getRegisteredOptions();
  auto *shortPtr =
      static_cast<llvm::cl::opt<bool> *>(options["nvptx-short-ptr"]);
  assert(shortPtr);
  shortPtr->setValue(true);
  // compute capability
  std::string sm = "sm_" + std::to_string(cc);
  // max PTX version
  int ptxMajor = version / 10;
  int ptxMinor = version % 10;
  // create
  llvm::SmallVector<char, 0> buffer;
  std::string triple = "nvptx64-nvidia-cuda";
  std::string proc = "sm_" + std::to_string(std::min(cc, maxNNVMCC));
  std::string layout = "";
  std::string features = "";
  // std::string features = "+ptx" + std::to_string(std::min(ptx,
  // max_nvvm_ptx));
  initLLVM();
  // verify and store llvm
  llvm::legacy::PassManager pm;
  pm.add(llvm::createVerifierPass());
  pm.run(module);
  // module->print(llvm::outs(), nullptr);

  // create machine
  module.setTargetTriple(triple);
  std::string error;
  auto target =
      llvm::TargetRegistry::lookupTarget(module.getTargetTriple(), error);
  llvm::TargetOptions opt;
  opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  opt.UnsafeFPMath = false;
  opt.NoInfsFPMath = false;
  opt.NoNaNsFPMath = true;
  llvm::TargetMachine *machine = target->createTargetMachine(
      module.getTargetTriple(), proc, features, opt, llvm::Reloc::PIC_,
      llvm::None, llvm::CodeGenOpt::Aggressive);
  // set data layout
  if (layout.empty())
    module.setDataLayout(machine->createDataLayout());
  else
    module.setDataLayout(layout);
  // emit machine code
  for (llvm::Function &f : module.functions())
    f.addFnAttr(llvm::Attribute::AlwaysInline);
  llvm::legacy::PassManager pass;
  llvm::raw_svector_ostream stream(buffer);
  // emit
  machine->addPassesToEmitFile(pass, stream, nullptr,
                               llvm::CodeGenFileType::CGFT_AssemblyFile);
  pass.run(module);

  // post-process
  std::string result(buffer.begin(), buffer.end());
  findAndReplace(result, ".version", "\n",
                 ".version " + std::to_string(ptxMajor) + "." +
                     std::to_string(ptxMinor) + "\n");
  findAndReplace(result, ".target", "\n", ".target " + sm + "\n");
  while (findAndReplace(result, "\t// begin inline asm", "\n", ""))
    ;
  while (findAndReplace(result, "\t// end inline asm", "\n", ""))
    ;
  return result;
}

} // namespace triton
