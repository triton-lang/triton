#include "mlir/IR/BuiltinOps.h" // mlir::ModuleOp
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Pass.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Linker/Linker.h"
#include <filesystem>
#include <iterator>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"

namespace py = pybind11;

namespace llvm {
struct BreakStructPhiNodesPass : PassInfoMixin<BreakStructPhiNodesPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static StringRef name() { return "BreakStructPhiNodesPass"; }
};
} // namespace llvm

using namespace llvm;
//
// TODO: move to python
static void initLLVM() {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() {
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();

    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUAsmParser();
    LLVMInitializeAMDGPUAsmPrinter();
  });
}

std::string translateLLVMIRToASM(llvm::Module &module,
                                 const std::string &triple,
                                 const std::string &proc,
                                 const std::string &features,
                                 const std::vector<std::string> &flags,
                                 bool enable_fp_fusion, bool isObject) {
  using namespace mlir;
  initLLVM();
  // options
  auto options = llvm::cl::getRegisteredOptions();
  for (std::string flag : flags) {
    auto *shortPtr = static_cast<llvm::cl::opt<bool> *>(options[flag]);
    assert(shortPtr);
    shortPtr->setValue(true);
  }
  // inline everything
  for (llvm::Function &f : module.functions())
    if (!f.hasFnAttribute(llvm::Attribute::NoInline))
      f.addFnAttr(llvm::Attribute::AlwaysInline);
  // verify and store llvm
  llvm::legacy::PassManager pm;
  pm.add(llvm::createAlwaysInlinerLegacyPass());
  pm.add(llvm::createVerifierPass());
  pm.run(module);
  // module->print(llvm::outs(), nullptr);

  // create machine
  module.setTargetTriple(triple);
  std::string error;
  auto target =
      llvm::TargetRegistry::lookupTarget(module.getTargetTriple(), error);
  llvm::TargetOptions opt;
  if (enable_fp_fusion)
    opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  opt.UnsafeFPMath = false;
  opt.NoInfsFPMath = false;
  opt.NoNaNsFPMath = true;
  opt.TrapUnreachable = true;
  std::unique_ptr<llvm::TargetMachine> machine{target->createTargetMachine(
      module.getTargetTriple(), proc, features, opt, llvm::Reloc::PIC_,
      std::nullopt, llvm::CodeGenOptLevel::Aggressive)};
  // set data layout
  module.setDataLayout(machine->createDataLayout());
  // emit machine code
  std::string result;
  {
    llvm::raw_string_ostream stream(result);
    llvm::buffer_ostream pstream(stream);
    for (llvm::Function &f : module.functions())
      f.addFnAttr(llvm::Attribute::AlwaysInline);
    llvm::legacy::PassManager pass;
    // emit
    auto fileType = isObject ? llvm::CodeGenFileType::ObjectFile
                             : llvm::CodeGenFileType::AssemblyFile;
    machine->addPassesToEmitFile(pass, pstream, nullptr, fileType);
    pass.run(module);
  }
  return result;
}

struct NVVMMetadata {
  llvm::SmallVector<int, 3> maxntid;
  bool isKernel{};
  // Free to extend with other information.
};

static void
extractNVVMMetadata(mlir::ModuleOp module,
                    llvm::DenseMap<llvm::StringRef, NVVMMetadata> *dic) {
  for (auto op : module.getOps<mlir::LLVM::LLVMFuncOp>()) {
    NVVMMetadata meta;
    bool hasMetadata{};
    // maxntid
    if (auto attr = op->getAttrOfType<mlir::ArrayAttr>("nvvm.maxntid")) {
      llvm::transform(attr.getAsValueRange<mlir::IntegerAttr>(),
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

// Add the nvvm related metadata to LLVM IR.
static void amendLLVMFunc(llvm::Function *func, const NVVMMetadata &metadata) {
  auto *module = func->getParent();
  auto &ctx = func->getContext();

  if (!metadata.maxntid.empty()) {
    auto maxntid =
        llvm::to_vector(llvm::map_range(metadata.maxntid, [&](int value) {
          return llvm::ConstantInt::get(llvm::IntegerType::get(ctx, 32),
                                        llvm::APInt(32, value));
        }));

    llvm::SmallVector<llvm::Metadata *> md_args = {
        llvm::ValueAsMetadata::get(func)};
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
    // switch (target) {
    // case Target::NVVM: {
    llvm::Metadata *mdArgs[] = {
        llvm::ValueAsMetadata::get(func), llvm::MDString::get(ctx, "kernel"),
        llvm::ValueAsMetadata::get(
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 1))};
    module->getOrInsertNamedMetadata("nvvm.annotations")
        ->addOperand(llvm::MDNode::get(ctx, mdArgs));
    // } break;
    // case Target::ROCDL: {
    //   func->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
    //   func->addFnAttr("amdgpu-flat-work-group-size", "1, 1024");
    // } break;
    // }
  }
}

using ret = py::return_value_policy;

void findKernels(llvm::Module &M, std::set<llvm::Function *> &functions) {
  llvm::NamedMDNode *annotations = M.getNamedMetadata("nvvm.annotations");
  assert(annotations);
  for (auto *Node : annotations->operands()) {
    if (Node->getNumOperands() < 3)
      continue;
    llvm::Metadata *Op = Node->getOperand(0).get();
    auto *ValueAsMetadata = llvm::dyn_cast<llvm::ValueAsMetadata>(Op);
    if (!ValueAsMetadata)
      continue;
    auto *F = llvm::dyn_cast<llvm::Function>(ValueAsMetadata->getValue());
    if (!F)
      continue;
    llvm::Metadata *Property = Node->getOperand(1).get();
    if (auto *MDString = llvm::dyn_cast<llvm::MDString>(Property))
      if (MDString->getString() == "kernel")
        functions.insert(F);
  }
}

void init_triton_llvm(py::module &&m) {

  py::class_<llvm::LLVMContext>(m, "context", py::module_local())
      .def(py::init<>());
  py::class_<llvm::Module>(m, "module", py::module_local())
      .def(
          "__str__",
          [](llvm::Module *self) {
            std::string str;
            llvm::raw_string_ostream os(str);
            os << *self;
            return os.str();
          },
          ret::take_ownership);
  ;

  // optimization levels
  py::class_<llvm::OptimizationLevel>(m, "optimization_level",
                                      py::module_local());
  m.attr("OPTIMIZE_O0") = (llvm::OptimizationLevel::O0);
  m.attr("OPTIMIZE_O1") = (llvm::OptimizationLevel::O1);
  m.attr("OPTIMIZE_O2") = (llvm::OptimizationLevel::O2);
  m.attr("OPTIMIZE_O3") = (llvm::OptimizationLevel::O3);
  m.attr("OPTIMIZE_Os") = (llvm::OptimizationLevel::Os);
  m.attr("OPTIMIZE_Oz") = (llvm::OptimizationLevel::Oz);

  m.def("to_module",
        [](mlir::ModuleOp &mod, llvm::LLVMContext &ctx, std::string name) {
          // TODO: dialects can be registered earlier...
          // This shouldn't depend on ROCDL or NVVM
          mlir::DialectRegistry registry;
          mlir::registerBuiltinDialectTranslation(registry);
          mlir::registerLLVMDialectTranslation(registry);
          mlir::registerROCDLDialectTranslation(registry);
          mlir::registerNVVMDialectTranslation(registry);
          mod->getContext()->appendDialectRegistry(registry);
          return mlir::translateModuleToLLVMIR(mod, ctx);
        });

  m.def("optimize_module", [](llvm::Module *mod,
                              const llvm::OptimizationLevel &opt) {
    using namespace llvm;
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

    PassBuilder pb(nullptr /*targetMachine*/, tuningOptions);

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
    mpm.addPass(pb.buildPerModuleDefaultPipeline(opt));
    mpm.run(*mod, mam);
  });

  m.def(
      "translate_to_asm",
      [](std::string llvmIR, std::string triple, std::string proc,
         std::string features, std::vector<std::string> flags,
         bool enable_fp_fusion,
         bool isObject) -> std::tuple<py::object, std::string> {
        py::gil_scoped_release allow_threads;
        // create LLVM module from C++
        llvm::LLVMContext context;
        std::unique_ptr<llvm::MemoryBuffer> buffer =
            llvm::MemoryBuffer::getMemBuffer(llvmIR.c_str());
        llvm::SMDiagnostic error;
        std::unique_ptr<llvm::Module> module =
            llvm::parseIR(buffer->getMemBufferRef(), error, context);
        if (!module) {
          llvm::report_fatal_error(
              "failed to parse IR: " + error.getMessage() +
              "lineno: " + std::to_string(error.getLineNo()));
        }
        // Get name of kernel in the module
        std::set<llvm::Function *> kernels;
        findKernels(*module, kernels);
        assert(kernels.size() == 1);
        std::string name = (*kernels.begin())->getName().str();
        std::string obj = translateLLVMIRToASM(
            *module, triple, proc, features, flags, enable_fp_fusion, isObject);
        if (isObject)
          return std::make_tuple(py::bytes(obj), name);
        else
          return std::make_tuple(py::str(obj), name);
      },
      ret::take_ownership);

  m.def("fix_attributes", [](mlir::ModuleOp module, llvm::Module *llvmModule) {
    llvm::DenseMap<llvm::StringRef, NVVMMetadata> nvvmMetadata;
    extractNVVMMetadata(module, &nvvmMetadata);
    for (auto &func : llvmModule->functions()) {
      auto it = nvvmMetadata.find(func.getName());
      if (it != nvvmMetadata.end())
        amendLLVMFunc(&func, it->second); //, target);
    }
  });

  m.def("set_nvvm_reflect_ftz", [](llvm::Module *mod) {
    // please check https://llvm.org/docs/NVPTXUsage.html#reflection-parameters
    // this will enable fast math path in libdevice
    // for example, when enable nvvm-reflect-ftz, sqrt.approx.f32 will change to
    // sqrt.approx.ftz.f32
    using namespace llvm;
    auto &ctx = mod->getContext();
    Type *i32 = Type::getInt32Ty(ctx);
    Metadata *mdFour = ConstantAsMetadata::get(ConstantInt::getSigned(i32, 4));
    Metadata *mdName = MDString::get(ctx, "nvvm-reflect-ftz");
    Metadata *mdOne = ConstantAsMetadata::get(ConstantInt::getSigned(i32, 1));
    MDNode *reflect = MDNode::get(ctx, {mdFour, mdName, mdOne});
    mod->addModuleFlag(reflect);
  });

  m.def("link_extern_lib",
        [](llvm::Module *mod, std::string name, std::string path) {
          llvm::SMDiagnostic err;
          auto &ctx = mod->getContext();
          auto extMod = llvm::parseIRFile(path, err, ctx);
          if (!extMod) {
            llvm::errs() << "Failed to load " << path;
            return;
          }
          extMod->setTargetTriple(mod->getTargetTriple());
          extMod->setDataLayout(mod->getDataLayout());
          if (llvm::Linker::linkModules(*mod, std::move(extMod),
                                        llvm::Linker::Flags::LinkOnlyNeeded)) {
            llvm::errs() << "Failed to link " << path;
            return;
          }
        });
}
