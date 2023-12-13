/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include "triton/Target/HSACO/HSACOTranslation.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/TritonGPUToLLVMPass.h"
#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
// #include "triton/Target/LLVMIR/LLVMIRTranslation.h"
#include "triton/Tools/Sys/GetEnv.hpp"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <dlfcn.h>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <vector>

namespace {

void init_llvm() {
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUAsmParser();
  LLVMInitializeAMDGPUAsmPrinter();
}

} // namespace

namespace mlir {
namespace triton {

std::string translateLLVMIRToHSACO(llvm::Module &module, std::string proc,
                                   std::string triple, std::string features) {
  // std::cout << "translateLLVMIRToHSACO" << std::endl;
  init_llvm();

  // verify and store llvm
  auto module_obj = llvm::CloneModule(module);
  if (!module_obj) {
    llvm::errs() << "Error: cloning LLIR failed"
                 << "\n";
  }

  // verify and store llvm
  llvm::legacy::PassManager pm;
  pm.add(llvm::createVerifierPass());
  pm.run(module);

  module.setTargetTriple(triple);

  std::string error;
  auto target =
      llvm::TargetRegistry::lookupTarget(module.getTargetTriple(), error);
  llvm::TargetOptions opt;
  opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  opt.UnsafeFPMath = false;
  opt.NoInfsFPMath = false;
  opt.NoNaNsFPMath = true;
  llvm::TargetMachine *_machine = target->createTargetMachine(
      module.getTargetTriple(), proc, features, opt, llvm::Reloc::PIC_,
      std::nullopt, llvm::CodeGenOptLevel::Aggressive);

  module.setDataLayout(_machine->createDataLayout());

  for (llvm::Function &f : module.functions())
    f.addFnAttr(llvm::Attribute::AlwaysInline);

  auto machine = std::unique_ptr<llvm::TargetMachine>(_machine);

  std::string dump_path = ::triton::tools::getenv("AMDGCN_DUMP_PATH");

  // create unique dir for kernel's binary and hsaco
  std::error_code ec;
  std::string kernel_name_base = "amd_triton_kernel";
  std::filesystem::path tmp = std::filesystem::temp_directory_path();
  std::filesystem::path kernel_dir_base(kernel_name_base);
  llvm::SmallString<256> unique_dir;
  ec = llvm::sys::fs::createUniqueDirectory((tmp / kernel_dir_base).string(),
                                            unique_dir);
  if (ec) {
    std::cerr << "Directory for " << kernel_name_base
              << " was not created. error code: " << ec << std::endl;
  }
  std::filesystem::path kernel_dir(unique_dir.data());
  std::string kernel_name = kernel_dir.stem();

  // Save GCN ISA binary.
  std::filesystem::path isa_binary(kernel_name + ".o");
  std::string isabin_path;
  if (!dump_path.empty())
    isabin_path = (dump_path / isa_binary).string();
  else
    isabin_path = (kernel_dir / isa_binary).string();
  std::unique_ptr<llvm::raw_fd_ostream> isabin_fs(
      new llvm::raw_fd_ostream(isabin_path, ec, llvm::sys::fs::OF_Text));
  if (ec) {
    llvm::errs() << isabin_path
                 << " was not created. error code: " << ec.category().name()
                 << ':' << ec.value() << '\n';
  }

  // Write out bitcode
  std::filesystem::path bitcode_filename(kernel_name + ".bc");
  std::string bitcode_path;
  if (!dump_path.empty())
    bitcode_path = (dump_path / bitcode_filename).string();
  else
    bitcode_path = (kernel_dir / bitcode_filename).string();
  std::unique_ptr<llvm::raw_fd_ostream> bitecode_fs(
      new llvm::raw_fd_ostream(bitcode_path, ec, llvm::sys::fs::OF_Text));
  if (ec) {
    llvm::errs() << bitcode_path
                 << " was not created. error code: " << ec.category().name()
                 << ':' << ec.value() << '\n';
  }

  llvm::WriteBitcodeToFile(module, *bitecode_fs);

  // emit
  llvm::legacy::PassManager pass;
  machine->addPassesToEmitFile(pass, *isabin_fs, nullptr,
                               llvm::CodeGenFileType::ObjectFile);
  pass.run(module);

  // generate HASCO file
  std::filesystem::path hsaco(kernel_name + ".hsaco");
  std::string hsaco_path = (kernel_dir / hsaco).string();
  std::string error_message;

  // Check in triton/third_party/rocm/llvm/bin first.  For whls this will be the
  // correct location. If not found, go back to using ROCM_PATH or /opt/rocm
  // static const auto this_library_path = [] {
  // Dl_info fileinfo;
  // if (dladdr(reinterpret_cast<void *>(generate_hsaco), &fileinfo) == 0) {
  //   return std::filesystem::path();
  // }
  // return std::filesystem::path(fileinfo.dli_fname);
  // }();

  // static const auto compiletime_path = this_library_path.parent_path()
  //                                             .parent_path()
  //                                             .parent_path() /
  //                                             "triton" / "third_party" /
  //                                             "rocm" / "llvm" / "bin" /
  //                                             "ld.lld";
  // std::string lld_path = compiletime_path.string();
  // // TODO: this should come from the runtime
  // if (!std::filesystem::exists(lld_path)) {
  static const std::string ROCM_DEFAULT_DIR = "/opt/rocm/";
  std::string rocm_path = ::triton::tools::getenv("ROCM_PATH");
  std::string lld_path = (rocm_path.empty()) ? ROCM_DEFAULT_DIR : rocm_path;
  lld_path += "/llvm/bin/ld.lld";
  // }

  int lld_result = llvm::sys::ExecuteAndWait(
      lld_path,
      {lld_path, "-flavor", "gnu", "-shared", "-o", hsaco_path, isabin_path},
      std::nullopt, {}, 0, 0, &error_message);
  if (lld_result) {
    llvm::errs() << "ld.lld execute fail: " << '\n'
                 << error_message << "Code: " << lld_result << '\n';
  }

  return hsaco_path;
}

} // namespace triton
} // namespace mlir
