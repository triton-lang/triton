#include "TritonMTGPUToLLVM/MUSATranslation.h"
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
#include "triton/Tools/Sys/GetEnv.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <regex>

namespace {

std::string readStringFromEnv(const std::string &env_name,
                              const std::string &default_value) {
  std::string env_path = mlir::triton::tools::getStrEnv(env_name);
  return (!env_path.empty()) ? env_path : default_value;
}

void execute_llc(const std::string &mtcc_path,
                 llvm::ArrayRef<llvm::StringRef> args) {
  auto llc_program = llvm::sys::findProgramByName("llc", {mtcc_path});
  if (!llc_program) {
    llvm::errs() << "llc program not found in path: " << mtcc_path << "\n";
    assert("llc program not found in path!");
  }
  std::string err_msg;
  int ret = llvm::sys::ExecuteAndWait(*llc_program, args, std::nullopt, {}, 0,
                                      0, &err_msg);
  if (ret) {
    llvm::errs() << "llc execute fail: " << err_msg << "\n";
    assert("using llc to generate asm or obj failed!");
  }
}

// convert latest llvm ir to mtcc compatible llvm ir.
// see llvm/docs/ReleaseNotes.rst
void convertLLVMIR(const std::string &filename) {
  // LLVM compatible. mtcc dependencies on llvm-14, convert llvm ir to mtcc
  // compatible format.
  auto make_llvm_compatible = [](std::string &ll_str) {
    // clang-format off
    std::vector<std::string> old_format = {
      "readnone",
      "readonly",
      "writeonly",
      "argmemonly",
      "argmemonly readonly",
      "argmemonly writeonly",
      "inaccessiblememonly",
      "inaccessiblememonly readonly",
      "inaccessiblememonly writeonly",
      "inaccessiblemem_or_argmemonly",
      "inaccessiblemem_or_argmemonly readonly",
      "inaccessiblemem_or_argmemonly writeonly"
    };
    std::vector<std::string> new_format = {
      "memory\\(none\\)",
      "memory\\(read\\)",
      "memory\\(write\\)",
      "memory\\(argmem: readwrite\\)",
      "memory\\(argmem: read\\)",
      "memory\\(argmem: write\\)",
      "memory\\(inaccessiblemem: readwrite\\)",
      "memory\\(inaccessiblemem: read\\)",
      "memory\\(inaccessiblemem: write\\)",
      "memory\\(argmem: readwrite, inaccessiblemem: readwrite\\)",
      "memory\\(argmem: read, inaccessiblemem: read\\)",
      "memory\\(argmem: write, inaccessiblemem: write\\)"
    };
    // clang-format on
    for (int i = 0; i < old_format.size(); ++i) {
      ll_str =
          std::regex_replace(ll_str, std::regex(new_format[i]), old_format[i]);
    }
  };

  // convert latest llvm ir to mtcc compatible llvm ir.
  std::ifstream is(filename);
  std::string ll_str((std::istreambuf_iterator<char>(is)),
                     std::istreambuf_iterator<char>());
  is.close();
  make_llvm_compatible(ll_str);

  // save the mtcc compatible llvm ir to ll file.
  std::ofstream os(filename);
  os << ll_str;
  os.close();

  if (mlir::triton::tools::getBoolEnv("MUSA_LLVMIR_ENABLE_DUMP")) {
    std::cout << "// -----// MUSA LLVMIR Dump //----- //\n"
              << ll_str << std::endl;
  }
}

std::string generate_muasm(const llvm::Module &llvmModule,
                           const std::string &opt_option, const int capability,
                           const int version, std::string &ll_file_name) {
  std::string function_name;
  std::string ll_file;
  std::string asm_file;

  llvm::SmallString<128> kernel;
  llvm::sys::fs::createTemporaryFile("mt_triton_kernel", /*suffix*/ "ll",
                                     kernel);
  ll_file = llvm::StringRef(kernel).str();
  ll_file_name = ll_file;
  llvm::sys::path::replace_extension(kernel, "s");
  asm_file = llvm::StringRef(kernel).str();

  std::error_code ec;
  llvm::raw_fd_ostream os(ll_file, ec, llvm::sys::fs::OF_None);
  llvmModule.print(os, nullptr);
  os.close();

  // get the name of mtgpu kernel.
  for (auto &F : llvmModule.getFunctionList()) {
    if (!F.isDeclaration() &&
        F.getCallingConv() == llvm::CallingConv::MTGPU_KERNEL) {
      function_name = F.getName().str();
      break;
    }
  }

  // convert latest llvm ir to mtcc compatible llvm ir.
  convertLLVMIR(ll_file);

  // because mtcc's building script has an option --disable_asm (default:
  // False), which can control mtcc's llc whether can support -filetype=asm or
  // not. so here we use an ENV: MTCC_ENABLE_ASM_BIN_PATH to indicate that this
  // path's llc can support -filetype=asm.
  //
  // by default, we use /usr/local/musa/bin/llc, which can't support
  // -filetype=asm, so we return the name of mtgpu kernel. otherwise, if we set
  // the ENV: MTCC_ENABLE_ASM_BIN_PATH, we will return the generated asm code.
  std::string mtcc_enable_asm_bin_path =
      readStringFromEnv("MTCC_ENABLE_ASM_BIN_PATH", "");

  if (!mtcc_enable_asm_bin_path.empty()) {
    // set ENV: MTCC_ENABLE_ASM_BIN_PATH, so return the generated asm code.
    // llc out.ll -march=mtgpu -O2 -filetype=asm -o out.asm
    std::string assign_subtarget = "-mcpu=mp_" + std::to_string(capability);
    llvm::SmallVector<llvm::StringRef> args{
        llvm::StringRef("llc"),
        llvm::StringRef(ll_file),
        llvm::StringRef("-march=mtgpu"),
        llvm::StringRef(assign_subtarget),
        llvm::StringRef("--opaque-pointers"),
        llvm::StringRef("-filetype=asm"),
        llvm::StringRef("-o"),
        llvm::StringRef(asm_file),
        llvm::StringRef("-O2"),
        llvm::StringRef(opt_option)};

    // use the mtcc_enable_asm_bin_path's llc to generate asm code.
    execute_llc(mtcc_enable_asm_bin_path, args);

    // get the muasm code.
    std::ifstream is(asm_file);
    std::string muasm((std::istreambuf_iterator<char>(is)),
                      std::istreambuf_iterator<char>());
    is.close();

    if (mlir::triton::tools::getBoolEnv("MUASM_ENABLE_DUMP")) {
      std::cout << "// -----// MUASM Dump //----- //\n" << muasm << std::endl;
    }

    return muasm;
  } else {
    // by default, /usr/local/musa/bin/llc can't support -filetype=asm,
    // so return the name of mtgpu kernel.
    return ".globl\t" + function_name;
  }
}

std::string generate_mubin(const std::string &ll_file_name,
                           const std::string &opt_option, const int capability,
                           const int version) {
  int pos = ll_file_name.find_last_of('.');
  std::string obj_file = ll_file_name.substr(0, pos + 1) + "o";
  std::string lld_obj_file = ll_file_name.substr(0, pos + 1) + "lld.o.mubin";

  // llc out.ll -march=mtgpu -O2 -filetype=obj -o out.o
  std::string assign_subtarget = "-mcpu=mp_" + std::to_string(capability);
  llvm::SmallVector<llvm::StringRef> args{llvm::StringRef("llc"),
                                          llvm::StringRef(ll_file_name),
                                          llvm::StringRef("-march=mtgpu"),
                                          llvm::StringRef(assign_subtarget),
                                          llvm::StringRef("--opaque-pointers"),
                                          llvm::StringRef("-filetype=obj"),
                                          llvm::StringRef("-o"),
                                          llvm::StringRef(obj_file),
                                          llvm::StringRef("-O2"),
                                          llvm::StringRef(opt_option)};

  // by default, we use the /usr/local/musa/bin/llc.
  // if we set the ENV: MTCC_ENABLE_ASM_BIN_PATH,
  // we should keep using the same llc tool with function: generate_muasm
  std::string mtcc_path =
      readStringFromEnv("MTCC_BIN_PATH", "/usr/local/musa/bin");
  std::string mtcc_enable_asm_bin_path =
      readStringFromEnv("MTCC_ENABLE_ASM_BIN_PATH", "");

  if (!mtcc_enable_asm_bin_path.empty()) {
    execute_llc(mtcc_enable_asm_bin_path, args);
  } else {
    // TODO: pre-install MTCC in docker or build bin in third_party
    execute_llc(mtcc_path, args);
  }

  // lld -flavor gnu -shared %bin -o %obj
  // clang-format off
  llvm::SmallVector<llvm::StringRef> lld_args{
    llvm::StringRef("ld.lld"),
    llvm::StringRef("-flavor"),
    llvm::StringRef("gnu"),
    llvm::StringRef("-shared"),
    llvm::StringRef(obj_file),
    llvm::StringRef("-o"),
    llvm::StringRef(lld_obj_file)
  };
  // clang-format on
  auto lld_program = llvm::sys::findProgramByName("ld.lld", {mtcc_path});
  if (!lld_program) {
    llvm::errs() << "lld program not found in path: " << mtcc_path << "\n";
    assert("using llc to generate obj failed!");
  }

  std::string err_msg;
  int lld_ret = llvm::sys::ExecuteAndWait(*lld_program, lld_args, std::nullopt,
                                          {}, 0, 0, &err_msg);
  if (lld_ret) {
    llvm::errs() << "lld execute fail: " << err_msg << "\n";
    assert("using llc to generate obj failed!");
  }

  return lld_obj_file;
}

std::tuple<std::string, std::string>
llir_to_muasm_and_mubin(llvm::Module *module, const std::string &opt_option,
                        int capability, int version) {
  std::string ll_file_name;
  auto muasm =
      generate_muasm(*module, opt_option, capability, version, ll_file_name);
  auto mubin_path =
      generate_mubin(ll_file_name, opt_option, capability, version);

  return std::make_tuple(muasm, mubin_path);
}

} // namespace

namespace mlir::triton {

std::tuple<std::string, std::string>
translateLLVMIRToMUBIN(llvm::Module &module, const std::string &opt_option,
                       int capability, int version) {
  auto muCode =
      llir_to_muasm_and_mubin(&module, opt_option, capability, version);
  return muCode;
}

} // namespace mlir::triton
