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
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
// #include "triton/Target/LLVMIR/LLVMIRTranslation.h"
#include "triton/Tools/Sys/GetEnv.hpp"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
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
#include <vector>
#include <dlfcn.h>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <iterator>

namespace {

void init_llvm() {
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUAsmParser();
  LLVMInitializeAMDGPUAsmPrinter();
}

std::unique_ptr<llvm::TargetMachine>
initialize_module(llvm::Module *module, const std::string &triple,
                  const std::string &proc, const std::string &features) {
  // verify and store llvm
  llvm::legacy::PassManager pm;
  pm.add(llvm::createVerifierPass());
  pm.run(*module);

  module->setTargetTriple(triple);

  std::string error;
  auto target =
      llvm::TargetRegistry::lookupTarget(module->getTargetTriple(), error);
  if (target == nullptr) {
    llvm::errs() << "LookupTarget fail: " << error << '\n';
    return nullptr;
  }
  llvm::TargetOptions opt;
  opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  opt.UnsafeFPMath = false;
  opt.NoInfsFPMath = false;
  opt.NoNaNsFPMath = true;
  llvm::TargetMachine *machine = target->createTargetMachine(
      module->getTargetTriple(), proc, features, opt, llvm::Reloc::PIC_,
      std::nullopt, llvm::CodeGenOptLevel::Aggressive);

  module->setDataLayout(machine->createDataLayout());

  for (llvm::Function &f : module->functions())
    f.addFnAttr(llvm::Attribute::AlwaysInline);

  return std::unique_ptr<llvm::TargetMachine>(machine);
}

std::string generate_amdgcn_assembly(llvm::Module *module,
                                     const std::string &triple,
                                     const std::string &proc,
                                     const std::string &features) {
  auto machine = initialize_module(module, triple, proc, features);

  if (machine == nullptr)
    return "";

  llvm::SmallVector<char, 0> buffer;
  llvm::legacy::PassManager pass;
  llvm::raw_svector_ostream stream(buffer);

  // emit
  machine->addPassesToEmitFile(pass, stream, nullptr,
                               llvm::CodeGenFileType::AssemblyFile);
  pass.run(*module);

  std::string amdgcn(buffer.begin(), buffer.end());
  if (::triton::tools::getBoolEnv("AMDGCN_ENABLE_DUMP")) {
    llvm::dbgs() << "// -----// AMDGCN Dump //----- //\n" << amdgcn << '\n';
  }

  return amdgcn;
}

std::string generate_hsaco(llvm::Module *module, const std::string &triple,
                           const std::string &proc,
                           const std::string &features) {
  auto machine = initialize_module(module, triple, proc, features);
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

  //Write out bitcode
  std::filesystem::path bitcode_filename (kernel_name + ".bc");
  std::string bitcode_path;
  if (!dump_path.empty())
    bitcode_path = (dump_path / bitcode_filename ).string();
  else
    bitcode_path = (kernel_dir / bitcode_filename ).string();
  std::unique_ptr<llvm::raw_fd_ostream> bitecode_fs(
      new llvm::raw_fd_ostream(bitcode_path, ec, llvm::sys::fs::OF_Text));  
  if (ec) {
    llvm::errs() << bitcode_path
                 << " was not created. error code: " << ec.category().name()
                 << ':' << ec.value() << '\n';
  }
  
  llvm::WriteBitcodeToFile(*module, *bitecode_fs);  

  // emit
  llvm::legacy::PassManager pass;
  machine->addPassesToEmitFile(pass, *isabin_fs, nullptr,
                               llvm::CodeGenFileType::ObjectFile);
  pass.run(*module);

  // generate HASCO file
  std::filesystem::path hsaco(kernel_name + ".hsaco");
  std::string hsaco_path = (kernel_dir / hsaco).string();
  std::string error_message;

  // Check in triton/third_party/rocm/llvm/bin first.  For whls this will be the
  // correct location. If not found, go back to using ROCM_PATH or /opt/rocm
  static const auto this_library_path = [] {
  Dl_info fileinfo;
  if (dladdr(reinterpret_cast<void *>(generate_hsaco), &fileinfo) == 0) {
    return std::filesystem::path();
  }
  return std::filesystem::path(fileinfo.dli_fname);
  }();

  static const auto compiletime_path = this_library_path.parent_path()
                                              .parent_path()
                                              .parent_path() /
                                              "triton" / "third_party" /
                                              "hip" / "llvm" / "bin" / "ld.lld";
  std::string lld_path = compiletime_path.string();
  if (!std::filesystem::exists(lld_path)) {
    std::string rocm_path = ::triton::tools::getenv("ROCM_PATH");
    lld_path = (rocm_path.empty()) ? ROCM_DEFAULT_DIR : rocm_path;
    lld_path += "/llvm/bin/ld.lld";
  }

  int lld_result =
      llvm::sys::ExecuteAndWait(lld_path,
                                {lld_path, "-flavor", "gnu",
                                 "-shared", "-o", hsaco_path, isabin_path},
                                std::nullopt, {}, 0, 0, &error_message);
  if (lld_result) {
    llvm::errs() << "ld.lld execute fail: " << '\n'
                 << error_message << "Code: " << lld_result << '\n';
  }

  return hsaco_path;
}

std::tuple<std::string, std::string>
llir_to_amdgcn_and_hsaco(llvm::Module *module, std::string gfx_arch,
                         std::string gfx_triple, std::string gfx_features) {

  init_llvm();

  // verify and store llvm
  auto module_obj = llvm::CloneModule(*module);
  if (!module_obj) {
    llvm::errs() << "Error: cloning LLIR failed"
                 << "\n";
  }
  auto amdgcn =
      generate_amdgcn_assembly(module, gfx_triple, gfx_arch, gfx_features);
  auto hsaco_path =
      generate_hsaco(module_obj.get(), gfx_triple, gfx_arch, gfx_features);

  return std::make_tuple(amdgcn, hsaco_path);
}

} // namespace

namespace mlir {
namespace triton {

// Describes NVVM Metadata. It is used to record the nvvm related meta
// information from mlir module.
struct NVVMMetadata {
  SmallVector<int, 3> maxntid;
  bool isKernel{};
  // Free to extend with other information.
};

void translateTritonToTritonGPU(mlir::ModuleOp &module,
                                    int computeCapability, int numWarps,
                                    int numStages) {
  // std::cout << "translateTritonToTritonGPU" << std::endl;

  // triton to triton gpu
  mlir::PassManager pm(module.getContext());
  pm.addPass(
      mlir::triton::createConvertTritonToTritonGPUPass(numWarps));

  // create file for IRs
  std::error_code EC;
  llvm::raw_fd_ostream fileOS("log/translateTritonToTritonGPU.log", EC,
                              llvm::sys::fs::OF_Text);

  if (EC) {
    llvm::errs() << "Could not open file: " << EC.message() << "\n";
  }

  // optimize triton gpu
  auto printingFlags = mlir::OpPrintingFlags();
  printingFlags.elideLargeElementsAttrs(16);
  pm.enableIRPrinting(
      /*shouldPrintBeforePass=*/nullptr,
      /*shouldPrintAfterPass=*/
      [](mlir::Pass *pass, mlir::Operation *) {
        return ::triton::tools::getBoolEnv("MLIR_ENABLE_DUMP");
      },
      /*printModuleScope=*/false,
      /*printAfterOnlyOnChange=*/true,
      /*printAfterOnlyOnFailure*/ false, fileOS, printingFlags);
  pm.addPass(mlir::createTritonGPUCoalescePass());
  pm.addPass(mlir::createTritonGPURemoveLayoutConversionsPass());
  pm.addPass(mlir::createTritonGPUAccelerateMatmulPass(computeCapability));
  pm.addPass(mlir::createTritonGPURemoveLayoutConversionsPass());
  pm.addPass(mlir::createTritonGPUOptimizeDotOperandsPass());
  pm.addPass(mlir::createTritonGPUPipelinePass(numStages));
  pm.addPass(mlir::createTritonGPUPrefetchPass());
  pm.addPass(mlir::createTritonGPUOptimizeDotOperandsPass());
  pm.addPass(mlir::createTritonGPURemoveLayoutConversionsPass());
  pm.addPass(mlir::createTritonGPUDecomposeConversionsPass());
  pm.addPass(mlir::createTritonGPUReorderInstructionsPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());

  auto ret = pm.run(module);
}

// Add the nvvm related metadata to LLVM IR.
static void amendLLVMFunc(llvm::Function *func, const NVVMMetadata &metadata,
                          bool isROCM, const int threadsPerCTA) {
  auto *module = func->getParent();
  auto &ctx = func->getContext();

  if (!metadata.maxntid.empty()) {
    auto maxntid =
        llvm::to_vector(llvm::map_range(metadata.maxntid, [&](int value) {
          return llvm::ConstantInt::get(llvm::IntegerType::get(ctx, 32),
                                        llvm::APInt(32, value));
        }));

    SmallVector<llvm::Metadata *> md_args = {llvm::ValueAsMetadata::get(func)};
    if (maxntid.size() > 0) {
      md_args.push_back(llvm::MDString::get(ctx, "maxntidx"));
      md_args.push_back(llvm::ValueAsMetadata::get(maxntid[0]));
    }
    if (maxntid.size() > 1) {
      md_args.push_back(llvm::MDString::get(ctx, "maxntidy"));
      md_args.push_back(llvm::ValueAsMetadata::get(maxntid[1]));
    }
    if (maxntid.size() > 2) {
      md_args.push_back(llvm::MDString::get(ctx, "maxntidz"));
      md_args.push_back(llvm::ValueAsMetadata::get(maxntid[2]));
    }

    module->getOrInsertNamedMetadata("nvvm.annotations")
        ->addOperand(llvm::MDNode::get(ctx, md_args));
  }

  if (metadata.isKernel) {
    if (isROCM) {
      func->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
      func->addFnAttr("amdgpu-flat-work-group-size",
                      "1, " + std::to_string(threadsPerCTA));
      func->addFnAttr("denormal-fp-math-f32", "preserve-sign");
      func->addFnAttr("amdgpu-unsafe-fp-atomics", "true");
    } else {
      llvm::Metadata *mdArgs[] = {
          llvm::ValueAsMetadata::get(func), llvm::MDString::get(ctx, "kernel"),
          llvm::ValueAsMetadata::get(
              llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 1))};
      module->getOrInsertNamedMetadata("nvvm.annotations")
          ->addOperand(llvm::MDNode::get(ctx, mdArgs));
    }
  }
}

static void
extractNVVMMetadata(mlir::ModuleOp module,
                    llvm::DenseMap<llvm::StringRef, NVVMMetadata> *dic) {
  // std::cout << "extractNVVMMetadata" << std::endl;
  for (auto op : module.getOps<LLVM::LLVMFuncOp>()) {
    NVVMMetadata meta;

    bool hasMetadata{};

    // maxntid
    if (auto attr = op->getAttrOfType<ArrayAttr>("nvvm.maxntid")) {
      llvm::transform(attr.getAsValueRange<IntegerAttr>(),
                      std::back_inserter(meta.maxntid),
                      [](llvm::APInt value) { return value.getZExtValue(); });
      hasMetadata = true;
    }

    // kernel
    if (op->hasAttr("nvvm.kernel")) {
      meta.isKernel = true;
      hasMetadata = true;
    }

    if (hasMetadata)
      dic->try_emplace(op.getNameAttr().strref(), std::move(meta));
  }
}

static std::map<std::string, std::string> getExternLibs(mlir::ModuleOp module) {
  // std::cout << "getExternLibs" << std::endl;
  std::map<std::string, std::string> externLibs;
  SmallVector<LLVM::LLVMFuncOp> funcs;
  module.walk([&](LLVM::LLVMFuncOp func) {
    if (func.isExternal())
      funcs.push_back(func);
  });

  for (auto &func : funcs) {
    if (func.getOperation()->hasAttr("libname")) {
      auto name =
          func.getOperation()->getAttr("libname").dyn_cast<StringAttr>();
      auto path =
          func.getOperation()->getAttr("libpath").dyn_cast<StringAttr>();
      if (name) {
        std::string libName = name.str();
        externLibs[libName] = path.str();
      }
    }
  }

  if (module.getOperation()->hasAttr("triton_gpu.externs")) {
    auto dict = module.getOperation()
                    ->getAttr("triton_gpu.externs")
                    .dyn_cast<DictionaryAttr>();
    for (auto &attr : dict) {
      externLibs[attr.getName().strref().trim().str()] =
          attr.getValue().dyn_cast<StringAttr>().strref().trim().str();
    }
  }

  if (!funcs.empty()) {
    static const std::string libdevice = "libdevice";
    // first search for environmental path
    std::string env_path = ::triton::tools::getenv("TRITON_LIBDEVICE_PATH");
    if (!env_path.empty()) {
      externLibs.try_emplace(libdevice, env_path);
      return externLibs;
    }
    namespace fs = std::filesystem;
    // Search for libdevice relative to its library path if used from Python
    // Then native code is in `triton/_C/libtriton.so` and libdevice in
    // `triton/third_party/cuda/lib/libdevice.10.bc`
    static const auto this_library_path = [] {
      Dl_info fileinfo;
      if (dladdr(reinterpret_cast<void *>(&getExternLibs), &fileinfo) == 0) {
        return std::filesystem::path();
      }
      return std::filesystem::path(fileinfo.dli_fname);
    }();
    static const auto runtime_path =
        this_library_path.parent_path().parent_path() / "third_party" / "cuda" /
        "lib" / "libdevice.10.bc";
    if (fs::exists(runtime_path)) {
      externLibs.try_emplace(libdevice, runtime_path.string());
    } else {
      // When using the Math Dialect, it is possible that some ops (e.g., log)
      // are lowered to a function call. In this case, we need to link libdevice
      // using its default path:
      // [triton root dir]/python/triton/language/libdevice.10.bc
      // TODO(Keren): handle external linkage other than libdevice?
      static const auto this_file_path = std::filesystem::path(__FILE__);
      static const auto compiletime_path = this_file_path.parent_path()
                                               .parent_path()
                                               .parent_path()
                                               .parent_path() /
                                           "python" / "triton" / "third_party" /
                                           "cuda" / "lib" / "libdevice.10.bc";
      if (!fs::exists(compiletime_path)) {
        std::string error_msg = "Can't find libdevice at neither " +
                                runtime_path.string() + " nor " +
                                compiletime_path.string();
        llvm::report_fatal_error(error_msg.c_str());
      }
      externLibs.try_emplace(libdevice, compiletime_path.string());
    }
  }

  return externLibs;
}

static void linkLibdevice(llvm::Module &module) {
  // std::cout << "linkLibdevice" << std::endl;
  // please check https://llvm.org/docs/NVPTXUsage.html#reflection-parameters
  // this will enable fast math path in libdevice
  // for example, when enable nvvm-reflect-ftz, sqrt.approx.f32 will change to
  // sqrt.approx.ftz.f32
  auto &ctx = module.getContext();
  llvm::Type *i32 = llvm::Type::getInt32Ty(ctx);
  llvm::Metadata *mdFour =
      llvm::ConstantAsMetadata::get(llvm::ConstantInt::getSigned(i32, 4));
  llvm::Metadata *mdName = llvm::MDString::get(ctx, "nvvm-reflect-ftz");
  llvm::Metadata *mdOne =
      llvm::ConstantAsMetadata::get(llvm::ConstantInt::getSigned(i32, 1));
  llvm::MDNode *reflect = llvm::MDNode::get(ctx, {mdFour, mdName, mdOne});
  module.addModuleFlag(reflect);
}

static bool linkExternLib(llvm::Module &module, llvm::StringRef name,
                          llvm::StringRef path, bool isROCM) {
  // std::cout << "linkExternLib" << std::endl;
  llvm::SMDiagnostic err;
  auto &ctx = module.getContext();

  auto extMod = llvm::parseIRFile(path, err, ctx);
  if (!extMod) {
    llvm::errs() << "Failed to load " << path;
    return true;
  }

  extMod->setTargetTriple(module.getTargetTriple());
  extMod->setDataLayout(module.getDataLayout());

  if (llvm::Linker::linkModules(module, std::move(extMod),
                                llvm::Linker::Flags::LinkOnlyNeeded)) {
    llvm::errs() << "Failed to link " << path;
    return true;
  }

  // check if ROCM
  if (!isROCM) {
    if (name == "libdevice") {
      linkLibdevice(module);
    }
    // else {
    //   assert(false && "unknown extern lib: ");
    // }
  }

  return false;
}

std::unique_ptr<llvm::Module>
translateLLVMDialectToLLVMIR(llvm::LLVMContext *llvmContext,
                             mlir::ModuleOp module, bool isROCM) {
  // std::cout << "translateLLVMDialectToLLVMIR" << std::endl;
  DialectRegistry registry;
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerROCDLDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
  module->getContext()->appendDialectRegistry(registry);

  llvm::DenseMap<llvm::StringRef, NVVMMetadata> nvvmMetadata;
  extractNVVMMetadata(module, &nvvmMetadata);

  auto llvmModule = mlir::translateModuleToLLVMIR(module, *llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return nullptr;
  }

  // Link external libraries before perform optimizations
  // Note from libdevice users guide:
  // https://docs.nvidia.com/cuda/libdevice-users-guide/basic-usage.html
  // The standard process for linking with libdevice is to first link it with
  // the target module, then run the standard LLVM optimization and code
  // generation passes. This allows the optimizers to inline and perform
  // analyses on the used library functions, and eliminate any used functions as
  // dead code.
  auto externLibs = getExternLibs(module);
  for (auto &lib : externLibs) {
    if (linkExternLib(*llvmModule, lib.first, lib.second, isROCM))
      return nullptr;
  }

  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/3, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return nullptr;
  }

  const int numWarps = mlir::triton::gpu::TritonGPUDialect::getNumWarps(module);
  const int warpSize = mlir::triton::gpu::TritonGPUDialect::getThreadsPerWarp(module);
  const int threadsPerCTA = numWarps * warpSize;

  for (auto &func : llvmModule->functions()) {
    auto it = nvvmMetadata.find(func.getName());
    if (it != nvvmMetadata.end())
      amendLLVMFunc(&func, it->second, isROCM, threadsPerCTA);
  }

  // create file for IRs
  std::error_code EC;
  llvm::raw_fd_ostream fileOS("log/translateLLVMDialectToLLVMIR.log", EC,
                              llvm::sys::fs::OF_Text);

  if (EC) {
    llvm::errs() << "Could not open file: " << EC.message() << "\n";
  }

  if (::triton::tools::getBoolEnv("LLVM_IR_ENABLE_DUMP")) {
    std::string mod_string;
    std::unique_ptr<llvm::raw_string_ostream> ir_ss(
        new llvm::raw_string_ostream(mod_string));
    llvmModule->print(fileOS, nullptr);
    // std::cout << "// -----// LLVM IR Dump //----- //\n" << mod_string <<
    // std::endl;
  }

  return llvmModule;
}

void translateTritonGPUToLLVMDialect(mlir::ModuleOp &module,
                                         int computeCapability, bool isROCM) {
  // std::cout << "translateTritonGPUToLLVMDialect" << std::endl;
  mlir::PassManager pm(module.getContext());
  mlir::registerPassManagerCLOptions();
  if (failed(applyPassManagerCLOptions(pm))) {
    llvm::errs() << "failed to apply pass manager CL options\n";
    return;
  }

  // create file for IRs
  std::error_code EC;
  llvm::raw_fd_ostream fileOS("log/translateTritonGPUToLLVMDialect.log", EC,
                              llvm::sys::fs::OF_Text);

  if (EC) {
    llvm::errs() << "Could not open file: " << EC.message() << "\n";
  }

  auto printingFlags = mlir::OpPrintingFlags();
  printingFlags.elideLargeElementsAttrs(16);
  pm.enableIRPrinting(
      /*shouldPrintBeforePass=*/nullptr,
      /*shouldPrintAfterPass=*/
      [](mlir::Pass *pass, mlir::Operation *) {
        return ::triton::tools::getBoolEnv("MLIR_ENABLE_DUMP");
      },
      /*printModuleScope=*/false,
      /*printAfterOnlyOnChange=*/true,
      /*printAfterOnlyOnFailure*/ false, fileOS, printingFlags);

  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::createConvertIndexToLLVMPass());
  pm.addPass(mlir::triton::createConvertTritonGPUToLLVMPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createCanonicalizerPass());
  // Simplify the IR
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
#if 1
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
#endif

  if (failed(pm.run(module))) {
    llvm::errs() << "Pass execution failed";
    return;
  }
}

std::tuple<std::string, std::string>
translateLLVMIRToHSACO(llvm::Module &module, std::string gfx_arch,
                       std::string gfx_triple, std::string gfx_features) {
  // std::cout << "translateLLVMIRToHSACO" << std::endl;
  auto hsacoCode =
      llir_to_amdgcn_and_hsaco(&module, gfx_arch, gfx_triple, gfx_features);
  return hsacoCode;
}

void addExternalLibsROCM(mlir::ModuleOp &module,
                     const std::vector<std::string> &names,
                     const std::vector<std::string> &paths) {
  // std::cout << "addExternalLibsROCM" << std::endl;
  if (names.empty() || names.size() != paths.size())
    return;

  llvm::SmallVector<NamedAttribute, 2> attrs;

  for (size_t i = 0; i < names.size(); ++i) {
    auto name = StringAttr::get(module->getContext(), names[i]);
    auto path = StringAttr::get(module->getContext(), paths[i]);
    NamedAttribute attr(name, path);
    attrs.push_back(attr);
  }

  DictionaryAttr dict = DictionaryAttr::get(module->getContext(), attrs);
  module.getOperation()->setAttr("triton_gpu.externs", dict);
}

// void printVector(const std::vector<std::string>& vec, std::string name = "")
// {
//   std::cout << name << ": ";
//   for (const auto &str : vec) {
//     std::cout << str << " ";
//   }
//   std::cout << std::endl;
// }

std::tuple<std::string, std::string>
translateTritonIRToHSACO(mlir::ModuleOp module, std::string gfx_arch,
                         std::string gfx_triple, std::string gfx_features,
                         int numWarps, int numStages,
                         const std::vector<std::string> &names,
                         const std::vector<std::string> &paths) {
  // std::cout << "translateTritonIRToHSACO" << std::endl;
  // std::cout << "module: " << module << std::endl;
  // std::cout << "gfx_arch: " << gfx_arch << std::endl;
  // std::cout << "gfx_triple: " << gfx_triple << std::endl;
  // std::cout << "gfx_features: " << gfx_features << std::endl;
  // std::cout << "numWarps: " << numWarps << std::endl;
  // std::cout << "numStages: " << numStages << std::endl;
  // printVector(names, "names");
  // printVector(paths, "paths");

  // sourceModule.dump();

  // auto targetModule = sourceModule.clone();
  auto targetModule = module;
  // std::cout << "targetModule: " << targetModule << std::endl;

  // print source dialects
  // auto sourceContext = sourceModule->getContext();
  // for (mlir::Dialect* dialect : sourceContext->getLoadedDialects()) {
  //   std::cout << dialect->getNamespace().str() << std::endl;
  // }

  // copy module into new rocm context
  // auto targetContext = std::make_unique<mlir::MLIRContext>();
  // mlir::OpBuilder builder(targetContext.get());
  // mlir::ModuleOp targetModule =
  // mlir::ModuleOp::create(builder.getUnknownLoc());

  // Clone each operation from the source module into the target module.
  // for (mlir::Operation& op : sourceModule.getBody()->without_terminator()) {
  //   targetModule.push_back(op.clone());
  // }

  // set params
  bool isROCM = true;
  int computeCapability = 0;

  // triton to triton gpu
  mlir::triton::translateTritonToTritonGPU(
      targetModule, computeCapability, numWarps, numStages);

  // add external libs
  // mlir::triton::addExternalLibs(targetModule, names, paths);

  // triton gpu to llvm mlir dialect
  mlir::triton::translateTritonGPUToLLVMDialect(
      targetModule, computeCapability, isROCM);

  // llvm mlir module to llvm ir
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::triton::translateLLVMDialectToLLVMIR(&llvmContext,
                                                       targetModule, isROCM);
  if (!llvmModule) {
    llvm::errs() << "Translate to LLVM IR failed"
                 << "\n";
    llvm::report_fatal_error("Failed to translate TritonGPU to LLVM IR.");
  }

  // llvm module to string
  std::string llvmIR;
  llvm::raw_string_ostream os(llvmIR);
  llvmModule->print(os, nullptr);
  os.flush();

  // create LLVM module from C++
  llvm::LLVMContext context;
  std::unique_ptr<llvm::MemoryBuffer> buffer =
      llvm::MemoryBuffer::getMemBuffer(llvmIR.c_str());
  llvm::SMDiagnostic error;
  std::unique_ptr<llvm::Module> module2 =
      llvm::parseIR(buffer->getMemBufferRef(), error, context);

  // translate module to HSACO
  auto hsacoCode = mlir::triton::translateLLVMIRToHSACO(
      *module2, gfx_arch, gfx_triple, gfx_features);
  return hsacoCode;
}

} // namespace triton
} // namespace mlir
