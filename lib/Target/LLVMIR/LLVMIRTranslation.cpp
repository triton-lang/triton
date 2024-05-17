#include "triton/Target/LLVMIR/LLVMIRTranslation.h"
#include "LLVMPasses.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Conversion/NVGPUToLLVM/NVGPUToLLVMPass.h"
#include "triton/Conversion/TritonGPUToLLVM/TritonGPUToLLVMPass.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Target/LLVMIR/Passes.h"
#include "triton/Target/PTX/TmaMetadata.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/IR/CallingConv.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include <optional>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <filesystem>
#include <iterator>

namespace fs = std::filesystem;

namespace {
using namespace llvm;

static std::optional<OptimizationLevel> mapToLevel(unsigned optLevel,
                                                   unsigned sizeLevel) {
  switch (optLevel) {
  case 0:
    return OptimizationLevel::O0;

  case 1:
    return OptimizationLevel::O1;

  case 2:
    switch (sizeLevel) {
    case 0:
      return OptimizationLevel::O2;

    case 1:
      return OptimizationLevel::Os;

    case 2:
      return OptimizationLevel::Oz;
    }
    break;
  case 3:
    return OptimizationLevel::O3;
  }
  return std::nullopt;
}

// Create and return a lambda that uses LLVM pass manager builder to set up
// optimizations based on the given level.
static std::function<Error(Module *)>
makeOptimizingPipeline(unsigned optLevel, unsigned sizeLevel,
                       TargetMachine *targetMachine) {
  return [optLevel, sizeLevel, targetMachine](Module *m) -> Error {
    std::optional<OptimizationLevel> ol = mapToLevel(optLevel, sizeLevel);
    if (!ol) {
      return make_error<StringError>(
          formatv("invalid optimization/size level {0}/{1}", optLevel,
                  sizeLevel)
              .str(),
          inconvertibleErrorCode());
    }
    LoopAnalysisManager lam;
    FunctionAnalysisManager fam;
    CGSCCAnalysisManager cgam;
    ModuleAnalysisManager mam;

    PipelineTuningOptions tuningOptions;
    tuningOptions.LoopUnrolling = true;
    tuningOptions.LoopInterleaving = true;
    tuningOptions.LoopVectorization = true;
    // TODO: currently we run SLP vectorizer with an empty target machine. This
    // cause the vectorizer to create larger vector which could be bad.
    // Disabling it would currently cause regressions as this pass also applies
    // some scheduling that helps performance in some cases. We should work on
    // using NVPTX target instead and address the performance regressions with
    // some scheduling solution.
    tuningOptions.SLPVectorization = true;

    PassBuilder pb(targetMachine, tuningOptions);

    std::string pluginFile = mlir::triton::tools::getenv("AMDGCN_INSTRUMENTATION_LIB");
    if (!pluginFile.empty()) {
        llvm::errs() << "Adding AMDGCN instrumentation pass to Triton pipeline" << "\n";
        auto passPlugin = llvm::PassPlugin::Load(pluginFile);
        if (!passPlugin) {
                llvm::Error Err = passPlugin.takeError();
                llvm::errs() << "ERROR: " << Err << "\n";
                consumeError(std::move(Err));
        }
        passPlugin->registerPassBuilderCallbacks(pb);
    }
    
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    ModulePassManager mpm;
    pb.registerVectorizerStartEPCallback(
        [&](llvm::FunctionPassManager &fpm, llvm::OptimizationLevel level) {
          // Triton generates large structure of scalars which may pessimise
          // optimizations, we run a pass to break up phi of struct to make sure
          // all the struct are removed for the following passes.
          fpm.addPass(BreakStructPhiNodesPass());
          fpm.addPass(InstCombinePass());
        });
    mpm.addPass(pb.buildPerModuleDefaultPipeline(*ol));
    mpm.run(*m, mam);
    return Error::success();
  };
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

// Add the nvvm related metadata to LLVM IR.
static void amendLLVMFunc(llvm::Function *func, const NVVMMetadata &metadata,
                          Target target, const int threadsPerCTA,
                          const int wavesPerEU) {
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
    switch (target) {
    case Target::NVVM: {
      llvm::Metadata *mdArgs[] = {
          llvm::ValueAsMetadata::get(func), llvm::MDString::get(ctx, "kernel"),
          llvm::ValueAsMetadata::get(
              llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 1))};
      module->getOrInsertNamedMetadata("nvvm.annotations")
          ->addOperand(llvm::MDNode::get(ctx, mdArgs));
    } break;
    case Target::ROCDL: {
      func->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
      func->addFnAttr("amdgpu-flat-work-group-size",
                      "1, " + std::to_string(threadsPerCTA));
      if (wavesPerEU > 0)
        func->addFnAttr("amdgpu-waves-per-eu", std::to_string(wavesPerEU));
      func->addFnAttr("denormal-fp-math-f32", "preserve-sign");
      func->addFnAttr("amdgpu-unsafe-fp-atomics", "true");
      for (unsigned I = 0; I < func->arg_size(); ++I) {
        Argument &Arg = *func->getArg(I);
        // Check for incompatible attributes.
        if (Arg.hasByRefAttr() || Arg.hasNestAttr())
          break;

        Arg.addAttr(llvm::Attribute::InReg);
      }
    } break;
    }
  }
}

static void
extractNVVMMetadata(mlir::ModuleOp module,
                    llvm::DenseMap<llvm::StringRef, NVVMMetadata> *dic) {
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

static std::filesystem::path getThisLibraryPath() {
#ifdef _WIN32
  /* Get module of the specified address */
  HMODULE hModule;
  GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                         GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                     reinterpret_cast<LPCSTR>(&getThisLibraryPath), &hModule);
  if (NULL == hModule) {
    return std::filesystem::path();
  }

  char fileName[1024]; // this is way beyond Windows MAX_PATH limit.
  DWORD dwSize = GetModuleFileNameA(hModule, fileName, sizeof(fileName));
  if (0 == dwSize || sizeof(fileName) == dwSize) {
    return std::filesystem::path();
  }
  return std::filesystem::path(fileName);
#else
  Dl_info fileinfo;
  if (dladdr(reinterpret_cast<void *>(&getThisLibraryPath), &fileinfo) == 0) {
    return std::filesystem::path();
  }
  return std::filesystem::path(fileinfo.dli_fname);
#endif
}

static std::map<std::string, std::string> getExternLibs(mlir::ModuleOp module) {
  std::map<std::string, std::string> externLibs;
  SmallVector<LLVM::LLVMFuncOp> funcs;
  module.walk([&](LLVM::LLVMFuncOp func) {
    if (func.isExternal())
      funcs.push_back(func);
  });

  for (LLVM::LLVMFuncOp func : funcs) {
    if (auto libnameAttr = func->getDiscardableAttr("libname")) {
      auto name = libnameAttr.dyn_cast<StringAttr>();
      auto path = func.getOperation()
                      ->getDiscardableAttr("libpath")
                      .dyn_cast<StringAttr>();
      if (name) {
        std::string libName = name.str();
        externLibs[libName] = path.str();
      }
    }
  }

  if (auto externsAttr = module->getDiscardableAttr("triton_gpu.externs")) {
    for (auto &attr : externsAttr.cast<DictionaryAttr>()) {
      externLibs[attr.getName().strref().trim().str()] =
          attr.getValue().dyn_cast<StringAttr>().strref().trim().str();
    }
  }

  if (!funcs.empty()) {
    static const std::string libdevice = "libdevice";
    // first search for environmental path
    std::string env_path = mlir::triton::tools::getenv("TRITON_LIBDEVICE_PATH");
    if (!env_path.empty()) {
      externLibs.try_emplace(libdevice, env_path);
      return externLibs;
    }
    // Search for libdevice relative to its library path if used from Python
    // Then native code is in `triton/_C/libtriton.so` and libdevice in
    // `triton/third_party/cuda/lib/libdevice.10.bc`
    static const auto this_library_path = getThisLibraryPath();
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

bool linkExternLib(llvm::Module &module, llvm::StringRef name,
                   llvm::StringRef path, Target target) {
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

  if (target == Target::NVVM) {
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
translateLLVMToLLVMIR(llvm::LLVMContext *llvmContext, mlir::ModuleOp module,
                      Target target, int wavesPerEU) {
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
    if (linkExternLib(*llvmModule, lib.first, lib.second, target))
      return nullptr;
  }

  const int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(module);
  const int warpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(module);
  const int threadsPerCTA = numWarps * warpSize;

  for (auto &func : llvmModule->functions()) {
    auto it = nvvmMetadata.find(func.getName());
    if (it != nvvmMetadata.end())
      amendLLVMFunc(&func, it->second, target, threadsPerCTA, wavesPerEU);
  }

  auto optPipeline = makeOptimizingPipeline(
      /*optLevel=*/3, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return nullptr;
  }  

  return llvmModule;
}

std::unique_ptr<llvm::Module>
translateTritonGPUToLLVMIR(llvm::LLVMContext *llvmContext,
                           mlir::ModuleOp module, int computeCapability,
                           mlir::triton::gpu::TMAMetadataTy &tmaInfos,
                           Target target, int wavesPerEU) {
  mlir::PassManager pm(module->getContext());
  mlir::registerPassManagerCLOptions();
  if (failed(applyPassManagerCLOptions(pm))) {
    llvm::errs() << "failed to apply pass manager CL options\n";
    return nullptr;
  }
  auto printingFlags = mlir::OpPrintingFlags();
  printingFlags.elideLargeElementsAttrs(16);
  printingFlags.enableDebugInfo();
  pm.enableIRPrinting(
      /*shouldPrintBeforePass=*/nullptr,
      /*shouldPrintAfterPass=*/
      [](mlir::Pass *pass, mlir::Operation *) {
        return mlir::triton::tools::getBoolEnv("MLIR_ENABLE_DUMP");
      },
      /*printModuleScope=*/false,
      /*printAfterOnlyOnChange=*/true,
      /*printAfterOnlyOnFailure*/ false, llvm::dbgs(), printingFlags);

  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::createConvertIndexToLLVMPass());
  pm.addPass(
      createConvertTritonGPUToLLVMPass(computeCapability, target, &tmaInfos));
#ifndef USE_ROCM
  pm.addPass(createConvertNVGPUToLLVMPass());
#endif
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createCanonicalizerPass());
  // Simplify the IR
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
#ifdef USE_ROCM
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
#endif
  if (!mlir::triton::tools::getBoolEnv("TRITON_DISABLE_LINE_INFO"))
    pm.addPass(mlir::createLLVMDIScopePass());

  if (failed(pm.run(module))) {
    llvm::errs() << "Pass execution failed";
    return nullptr;
  }

  auto llvmIR = translateLLVMToLLVMIR(llvmContext, module, target, wavesPerEU);
  if (!llvmIR) {
    llvm::errs() << "Translate to LLVM IR failed";
    return nullptr;
  }

  if (mlir::triton::tools::getBoolEnv("LLVM_IR_ENABLE_DUMP")) {
    std::string mod_string;
    std::unique_ptr<llvm::raw_string_ostream> ir_ss(
        new llvm::raw_string_ostream(mod_string));
    llvmIR->print(*ir_ss, nullptr);
    llvm::dbgs() << "// -----// LLVM IR Dump //----- //\n" << mod_string << '\n';
  }

  return llvmIR;
}

void addExternalLibs(mlir::ModuleOp &module,
                     const std::vector<std::string> &names,
                     const std::vector<std::string> &paths) {
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

} // namespace triton
} // namespace mlir
