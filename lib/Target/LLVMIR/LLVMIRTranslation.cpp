#include "triton/Target/LLVMIR/LLVMIRTranslation.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/TritonGPUToLLVMPass.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/IR/Constants.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/SourceMgr.h"
#include <dlfcn.h>
#include <filesystem>

namespace mlir {
namespace triton {

// Describes NVVM Metadata. It is used to record the nvvm related meta
// information from mlir module.
struct NVVMMetadata {
  int maxntidx{-1};
  bool isKernel{};
  // Free to extend with other information.
};

// Add the nvvm related metadata to LLVM IR.
static void amendLLVMFunc(llvm::Function *func, const NVVMMetadata &metadata) {
  auto *module = func->getParent();
  auto &ctx = func->getContext();

  if (metadata.maxntidx > 0) {
    auto warps = llvm::ConstantInt::get(llvm::IntegerType::get(ctx, 32),
                                        llvm::APInt(32, metadata.maxntidx));

    llvm::Metadata *md_args[] = {llvm::ValueAsMetadata::get(func),
                                 llvm::MDString::get(ctx, "maxntidx"),
                                 llvm::ValueAsMetadata::get(warps)};

    module->getOrInsertNamedMetadata("nvvm.annotations")
        ->addOperand(llvm::MDNode::get(ctx, md_args));
  }

  if (metadata.isKernel) {
    llvm::Metadata *mdArgs[] = {
        llvm::ValueAsMetadata::get(func), llvm::MDString::get(ctx, "kernel"),
        llvm::ValueAsMetadata::get(
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 1))};
    module->getOrInsertNamedMetadata("nvvm.annotations")
        ->addOperand(llvm::MDNode::get(ctx, mdArgs));
  }
}

static void
extractNVVMMetadata(mlir::ModuleOp module,
                    llvm::DenseMap<llvm::StringRef, NVVMMetadata> *dic) {
  for (auto op : module.getOps<LLVM::LLVMFuncOp>()) {
    NVVMMetadata meta;

    bool hasMetadata{};

    // maxntid
    if (op->hasAttr("nvvm.maxntid")) {
      auto attr = op->getAttr("nvvm.maxntid");
      meta.maxntidx = attr.dyn_cast<IntegerAttr>().getInt();
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
                          llvm::StringRef path) {
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

  if (name == "libdevice") {
    linkLibdevice(module);
  } else {
    assert(false && "unknown extern lib: ");
  }

  return false;
}

std::unique_ptr<llvm::Module>
translateLLVMToLLVMIR(llvm::LLVMContext *llvmContext, mlir::ModuleOp module) {
  DialectRegistry registry;
  mlir::registerLLVMDialectTranslation(registry);
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
    if (linkExternLib(*llvmModule, lib.first, lib.second))
      return nullptr;
  }

  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/3, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return nullptr;
  }

  for (auto &func : llvmModule->functions()) {
    auto it = nvvmMetadata.find(func.getName());
    if (it != nvvmMetadata.end())
      amendLLVMFunc(&func, it->second);
  }

  return llvmModule;
}

std::unique_ptr<llvm::Module>
translateTritonGPUToLLVMIR(llvm::LLVMContext *llvmContext,
                           mlir::ModuleOp module, int computeCapability) {
  mlir::PassManager pm(module->getContext());
  applyPassManagerCLOptions(pm);
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
      /*printAfterOnlyOnFailure*/ false, llvm::dbgs(), printingFlags);

  pm.addPass(createConvertTritonGPUToLLVMPass(computeCapability));
  // Canonicalize to eliminate the remaining UnrealizedConversionCastOp
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass()); // Simplify the IR to improve readability.
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  if (failed(pm.run(module))) {
    llvm::errs() << "Pass execution failed";
    return nullptr;
  }

  auto llvmIR = translateLLVMToLLVMIR(llvmContext, module);
  if (!llvmIR) {
    llvm::errs() << "Translate to LLVM IR failed";
    return nullptr;
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
