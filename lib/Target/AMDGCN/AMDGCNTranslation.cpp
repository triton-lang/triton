#include "triton/Target/AMDGCN/AMDGCNTranslation.h"
#include "triton/tools/sys/getenv.hpp"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

namespace {

void init_llvm() {
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUAsmPrinter();
  LLVMInitializeAMDGPUAsmParser();
}

}

namespace triton {

std::tuple<std::string, std::string>
translateLLVMIRToAMDGCN(llvm::Module &module, std::string cc) {
  // create
  llvm::SmallVector<char, 0> buffer;
  std::string triple = "amdgcn-amd-amdhsa";
  std::string proc = cc;
  std::string layout = "";
  std::string features = "+sramecc,-xnack";

  std::string kernel_name =
      std::string(std::tmpnam(nullptr)) + "_" + module.getModuleIdentifier();

  init_llvm();

  // verify and store llvm
  llvm::legacy::PassManager pm;
  pm.add(llvm::createVerifierPass());
  pm.run(module);

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

  std::string amdgcn(buffer.begin(), buffer.end());
  if (tools::getBoolEnv("AMDGCN_ENABLE_DUMP")) {
    llvm::outs() << amdgcn << "\n";
  }

  // create dump files
  std::error_code ec;
  // Save GCN ISA binary.
  std::string isabin_path = kernel_name + std::string(".o");
  std::unique_ptr<llvm::raw_fd_ostream> isabin_fs(
      new llvm::raw_fd_ostream(isabin_path, ec, llvm::sys::fs::OF_Text));
  if (ec) {
    llvm::errs() << "Error: '" << isabin_path << "' was not created. \n";
    llvm::errs() << "\tError reported: " << ec.message() << "\n";
    return std::make_tuple("", "");
  }
  // emit
  machine->addPassesToEmitFile(pass, *isabin_fs, nullptr,
                               llvm::CGFT_ObjectFile);
  pass.run(module);

  // generate HASCO file
  std::string hsaco_path = kernel_name + std::string(".hsaco");

  std::string error_message;
  int lld_result =
      llvm::sys::ExecuteAndWait("/opt/rocm/llvm/bin/ld.lld",
                                {"/opt/rocm/llvm/bin/ld.lld", "-flavor", "gnu",
                                 "-shared", "-o", hsaco_path, isabin_path},
                                llvm::None, {}, 0, 0, &error_message);
  if (lld_result) {
    llvm::errs() << "Error: ld.lld execute fail. Error code: " << lld_result
                 << "\n";
    llvm::errs() << "\tError reported: " << error_message << "\n";
    return std::make_tuple("", "");
  }

  const std::string hsaco_dump_path = tools::getenv("AMDGCN_HSACO_DUMP_PATH");
  if (!hsaco_dump_path.empty()) {
    if (std::error_code copy_result =
            llvm::sys::fs::copy_file(hsaco_path, hsaco_dump_path)) {
      llvm::errs() << "Error: cannot copy to hsaco dump file from '"
                   << hsaco_path << "' to '" << hsaco_dump_path << "'\n";
      llvm::errs() << "\tError reported: " << copy_result.message() << "\n";
      return std::make_tuple("", "");
    }
  }

  return std::make_tuple(amdgcn, hsaco_path);
}

} // namespace triton