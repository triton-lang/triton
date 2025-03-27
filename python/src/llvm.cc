#include "mlir/IR/BuiltinOps.h" // mlir::ModuleOp
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizer.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizerOptions.h"
#include <csignal>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;

namespace llvm {
struct BreakStructPhiNodesPass : PassInfoMixin<BreakStructPhiNodesPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static StringRef name() { return "BreakStructPhiNodesPass"; }
};
} // namespace llvm

using namespace llvm;

std::unique_ptr<TargetMachine>
createTargetMachine(llvm::Module *module, std::string proc,
                    bool enable_fp_fusion, const std::string &features) {
  std::string error;
  auto target = llvm::TargetRegistry::lookupTarget(
      module->getTargetTriple().str(), error);
  llvm::TargetOptions opt;
  bool disableLLVMOpt = mlir::triton::tools::getBoolEnv("DISABLE_LLVM_OPT");
  if (enable_fp_fusion)
    opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  opt.UnsafeFPMath = false;
  opt.NoInfsFPMath = false;
  opt.NoNaNsFPMath = true;
  opt.TrapUnreachable = true;
  opt.MCOptions.AsmVerbose = true;
  opt.MCOptions.PreserveAsmComments = true;
  std::unique_ptr<llvm::TargetMachine> machine{target->createTargetMachine(
      module->getTargetTriple(), proc, features, opt, llvm::Reloc::PIC_,
      std::nullopt,
      disableLLVMOpt ? llvm::CodeGenOptLevel::None
                     : llvm::CodeGenOptLevel::Aggressive)};
  return machine;
}

std::string translateLLVMIRToASM(llvm::Module &module,
                                 const std::string &triple,
                                 const std::string &proc,
                                 const std::string &features,
                                 const std::vector<std::string> &flags,
                                 bool enable_fp_fusion, bool isObject) {
  using namespace mlir;
  // options
  auto options = llvm::cl::getRegisteredOptions();
  for (std::string flag : flags) {
    auto *shortPtr = static_cast<llvm::cl::opt<bool> *>(options[flag]);
    assert(shortPtr);
    shortPtr->setValue(true);
  }
  if (triton::tools::getBoolEnv("LLVM_IR_ENABLE_DUMP")) {
    auto optIt = options.find("print-after-all");
    if (optIt != options.end()) {
      auto optPtr = static_cast<llvm::cl::opt<bool> *>(optIt->second);
      *optPtr = true;
    }
  }
  bool disableLLVMOpt = triton::tools::getBoolEnv("DISABLE_LLVM_OPT");
  if (!disableLLVMOpt) {
    // Check to see if we are passing a list of flags to disable optimizations.
    auto flagList = triton::tools::getStrEnv("DISABLE_LLVM_OPT");
    if (!flagList.empty()) {
      llvm::SmallVector<StringRef, 3> split;
      StringRef(flagList.c_str()).split(split, ',');
      for (auto flag : split) {
        auto optIt = options.find(flag);
        if (optIt != options.end()) {
          auto optPtr = static_cast<llvm::cl::opt<bool> *>(optIt->second);
          *optPtr = true;
        }
      }
    }
  }

  // inline everything
  for (llvm::Function &f : module.functions())
    if (!f.hasFnAttribute(llvm::Attribute::NoInline))
      f.addFnAttr(llvm::Attribute::AlwaysInline);
  // verify and store llvm
  llvm::legacy::PassManager pm;
  pm.add(llvm::createAlwaysInlinerLegacyPass());
  pm.add(llvm::createVerifierPass());

  const bool enabledTiming = triton::tools::getBoolEnv("LLVM_ENABLE_TIMING");
  if (enabledTiming) {
    llvm::TimePassesIsEnabled = true;
    llvm::TimePassesPerRun = true;
  }

  pm.run(module);

  SmallString<0> timePassesStr;
  raw_svector_ostream reportStream(timePassesStr);

  if (enabledTiming) {
    reportAndResetTimings(&reportStream);
    llvm::dbgs() << reportStream.str();
    timePassesStr.clear();
  }
  // module->print(llvm::outs(), nullptr);

  // create machine
  module.setTargetTriple(Triple(triple));
  auto machine = createTargetMachine(&module, proc, enable_fp_fusion, features);
  // set data layout
  module.setDataLayout(machine->createDataLayout());
  // emit machine code
  std::string result;
  {
    llvm::raw_string_ostream stream(result);
    llvm::buffer_ostream pstream(stream);
    llvm::legacy::PassManager pass;
    // emit
    auto fileType = isObject ? llvm::CodeGenFileType::ObjectFile
                             : llvm::CodeGenFileType::AssemblyFile;
    machine->addPassesToEmitFile(pass, pstream, nullptr, fileType);
    pass.run(module);

    if (enabledTiming) {
      reportAndResetTimings(&reportStream);
      llvm::dbgs() << reportStream.str();
      timePassesStr.clear();
    }
  }
  return result;
}

using ret = py::return_value_policy;

void init_triton_llvm(py::module &&m) {

  py::class_<llvm::LLVMContext>(m, "context", py::module_local())
      .def(py::init<>());
  py::class_<llvm::SourceMgr>(m, "source_mgr", py::module_local())
      .def(py::init<>());

  py::class_<llvm::Module::FunctionListType>(m, "function_list")
      .def(
          "__iter__",
          [](llvm::Module::FunctionListType &s) {
            return py::make_iterator(s.begin(), s.end());
          },
          py::keep_alive<0, 1>());

  // Module Flag behavior. See
  // https://llvm.org/doxygen/classllvm_1_1Module.html#a0a5c55e12c97b80021330fe82b642293
  // for details.
  py::class_<llvm::Module::ModFlagBehavior>(m, "module_flag_behavior",
                                            py::module_local());
  m.attr("MODULE_FLAG_BEHAVIOR_ERROR") = llvm::Module::Error;
  m.attr("MODULE_FLAG_BEHAVIOR_WARNING") = llvm::Module::Warning;
  m.attr("MODULE_FLAG_BEHAVIOR_REQUIRE") = llvm::Module::Require;
  m.attr("MODULE_FLAG_BEHAVIOR_OVERRIDE") = llvm::Module::Override;
  m.attr("MODULE_FLAG_BEHAVIOR_APPEND") = llvm::Module::Append;
  m.attr("MODULE_FLAG_BEHAVIOR_APPEND_UNIQUE") = llvm::Module::AppendUnique;
  m.attr("MODULE_FLAG_BEHAVIOR_MAX") = llvm::Module::Max;
  m.attr("MODULE_FLAG_BEHAVIOR_MIN") = llvm::Module::Min;

  py::class_<llvm::Module>(m, "module", py::module_local())
      .def(
          "__str__",
          [](llvm::Module *self) {
            std::string str;
            llvm::raw_string_ostream os(str);
            os << *self;
            return os.str();
          },
          ret::take_ownership)
      .def(
          "get_functions",
          [](llvm::Module *mod) -> llvm::Module::FunctionListType & {
            // Note: Backends assume that we are compiling exactly one kernel
            // (i.e. one function that's that's called by the CPU) and that it's
            // the first function in this list.
            return mod->getFunctionList();
          },
          ret::reference_internal)
      .def("add_flag",
           [](llvm::Module *mod, llvm::Module::ModFlagBehavior behavior,
              std::string &key, uint32_t value) {
             return mod->addModuleFlag(behavior, key, value);
           });

  py::class_<llvm::Function>(m, "function", py::module_local())
      .def_property_readonly(
          "name", [](llvm::Function *fn) { return fn->getName().str(); })
      .def("set_calling_conv", &llvm::Function::setCallingConv)
      .def("add_fn_attr", [](llvm::Function *fn, std::string &name,
                             std::string &val) { fn->addFnAttr(name, val); })
      .def("add_fn_asan_attr",
           [](llvm::Function *fn) {
             fn->addFnAttr(llvm::Attribute::SanitizeAddress);
           })
      .def("add_fn_target_feature",
           [](llvm::Function *fn, std::string &val) {
             fn->addFnAttr("target-features", val);
           })
      // Sets the nvvm.maxreg property on the given function.
      .def("set_nvvm_maxnreg",
           [](llvm::Function *fn, int maxnreg) {
             auto op = MDNode::get(
                 fn->getContext(),
                 {
                     ValueAsMetadata::get(fn),
                     MDString::get(fn->getContext(), "maxnreg"),
                     ConstantAsMetadata::get(ConstantInt::get(
                         Type::getInt32Ty(fn->getContext()), maxnreg)),
                 });
             fn->getParent()
                 ->getOrInsertNamedMetadata("nvvm.annotations")
                 ->addOperand(op);
           })
      // External functions that are definitions (i.e. not declarations) are
      // kernel functions.
      .def("is_declaration", &llvm::Function::isDeclaration)
      .def("is_external_linkage", [](llvm::Function *fn) {
        return fn->getLinkage() == llvm::GlobalValue::ExternalLinkage;
      });

  // optimization levels
  py::class_<llvm::OptimizationLevel>(m, "optimization_level",
                                      py::module_local());
  m.attr("OPTIMIZE_O0") = llvm::OptimizationLevel::O0;
  m.attr("OPTIMIZE_O1") = llvm::OptimizationLevel::O1;
  m.attr("OPTIMIZE_O2") = llvm::OptimizationLevel::O2;
  m.attr("OPTIMIZE_O3") = llvm::OptimizationLevel::O3;
  m.attr("OPTIMIZE_Os") = llvm::OptimizationLevel::Os;
  m.attr("OPTIMIZE_Oz") = llvm::OptimizationLevel::Oz;

  m.def(
      "to_module",
      [](mlir::ModuleOp &mod, llvm::LLVMContext &ctx) {
        std::unique_ptr<llvm::Module> llvmMod =
            mlir::translateModuleToLLVMIR(mod, ctx);
        if (!llvmMod) {
          throw std::runtime_error("failed to translate module to LLVM IR");
        }
        return llvmMod;
      },
      py::keep_alive<0, 2>());

  m.def("attach_datalayout", [](llvm::Module *mod, const std::string triple,
                                const std::string proc,
                                const std::string features) {
    std::string error;
    auto target = llvm::TargetRegistry::lookupTarget(triple, error);
    if (!target) {
      throw std::runtime_error("target lookup error: " + error);
    }
    llvm::TargetOptions opt;
    // Target machine is only used to create the data layout.
    std::unique_ptr<llvm::TargetMachine> machine{target->createTargetMachine(
        llvm::Triple(triple), proc, features, opt, llvm::Reloc::PIC_,
        std::nullopt, llvm::CodeGenOptLevel::None)};
    // set data layout
    mod->setDataLayout(machine->createDataLayout());
  });

  m.def(
      "optimize_module",
      [](llvm::Module *mod, const llvm::OptimizationLevel &opt,
         std::string arch, std::string features, std::vector<std::string> flags,
         bool enable_fp_fusion) {
        if (mlir::triton::tools::getBoolEnv("DISABLE_LLVM_OPT"))
          return;
        // Check to see if we are passing a list of flags to disable
        // optimizations.
        auto flagList = mlir::triton::tools::getStrEnv("DISABLE_LLVM_OPT");
        if (!flagList.empty()) {
          auto options = llvm::cl::getRegisteredOptions();
          llvm::SmallVector<StringRef, 3> split;
          StringRef(flagList.c_str()).split(split, ',');
          for (auto flag : split) {
            auto optIt = options.find(flag);
            if (optIt != options.end()) {
              auto optPtr = static_cast<llvm::cl::opt<bool> *>(optIt->second);
              *optPtr = true;
            }
          }
        }
        using namespace llvm;
        LoopAnalysisManager lam;
        FunctionAnalysisManager fam;
        CGSCCAnalysisManager cgam;
        ModuleAnalysisManager mam;

        PassInstrumentationCallbacks *instrCbPtr = nullptr;
        PassInstrumentationCallbacks passInstrCb;
        StandardInstrumentations standardInstr(mod->getContext(),
                                               /*DebugLogging*/ true);
        if (mlir::triton::tools::getBoolEnv("LLVM_IR_ENABLE_DUMP")) {
          auto optMap = llvm::cl::getRegisteredOptions();
          auto optIt = optMap.find("print-after-all");
          if (optIt != optMap.end()) {
            auto optPtr = static_cast<llvm::cl::opt<bool> *>(optIt->second);
            *optPtr = true;
          }
          standardInstr.registerCallbacks(passInstrCb, &mam);
          instrCbPtr = &passInstrCb;
        }

        PipelineTuningOptions tuningOptions;
        tuningOptions.LoopUnrolling = true;
        tuningOptions.LoopInterleaving = true;
        tuningOptions.LoopVectorization = true;
        // TODO: currently we run SLP vectorizer with an empty target machine.
        // This cause the vectorizer to create larger vector which could be bad.
        // Disabling it would currently cause regressions as this pass also
        // applies some scheduling that helps performance in some cases. We
        // should work on using NVPTX target instead and address the performance
        // regressions with some scheduling solution.
        tuningOptions.SLPVectorization = true;

        std::string pluginFile =
            mlir::triton::tools::getStrEnv("LLVM_PASS_PLUGIN_PATH");

        // We don't pass the targetMachine to the LLVM-IR pass builder, unless
        // `arch` is specified.
        //
        // Don't set target machine in LLVM pass builder when using LLVM IR
        // level plugins. LLVM IR level plugin passes typically want to insert
        // calls to externally generated code (i.e. precompile a Cuda/Hip kernel
        // with Clang and then insert a call to it within an instrumentation
        // pass) setting the targetMachine value here can can cause a mismatch
        // in the target machine between the MLIR and Clang generated kernels
        // and break the lowering of some target specific intrinsics.
        std::unique_ptr<TargetMachine> targetMachine = nullptr;
        if (!arch.empty() && pluginFile.empty())
          targetMachine =
              createTargetMachine(mod, arch, enable_fp_fusion, features);
        PassBuilder pb(/*targetMachine=*/targetMachine.get(), tuningOptions,
                       std::nullopt, instrCbPtr);

        if (!pluginFile.empty()) {
          // TODO: Add some logging here that we inserted a pass into the LLVM
          // pass pipeline
          auto passPlugin = llvm::PassPlugin::Load(pluginFile);
          if (!passPlugin) {
            llvm::Error Err = passPlugin.takeError();
            std::string ErrMsg =
                "Pass Plugin Error: " + llvm::toString(std::move(Err));
            throw std::runtime_error(ErrMsg);
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
              // optimizations, we run a pass to break up phi of struct to make
              // sure all the struct are removed for the following passes.
              fpm.addPass(BreakStructPhiNodesPass());
              fpm.addPass(InstCombinePass());
            });
        bool enableAddressSanitizer =
            mlir::triton::tools::getBoolEnv("TRITON_ENABLE_ASAN");
        if (enableAddressSanitizer) {
          AddressSanitizerOptions Opts;
          mpm.addPass(AddressSanitizerPass(Opts));
        }
        mpm.addPass(pb.buildPerModuleDefaultPipeline(opt));
        mpm.run(*mod, mam);
      },
      // Mandatory parameters
      py::arg("mod"), py::arg("opt"),
      // If we want to specify the target machine, we require additional
      // (optional) parameters
      py::arg("arch") = "", py::arg("features") = "",
      py::arg("flags") = std::vector<std::string>{},
      py::arg("enable_fp_fusion") = false);

  m.def(
      "translate_to_asm",
      [](std::string llvmIR, std::string triple, std::string proc,
         std::string features, std::vector<std::string> flags,
         bool enable_fp_fusion, bool isObject) -> py::object {
        std::string obj;
        {
          // when allow_threads goes out of scope, gil will be released
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
          obj = translateLLVMIRToASM(*module, triple, proc, features, flags,
                                     enable_fp_fusion, isObject);
        }
        if (isObject)
          return py::bytes(obj);
        else
          return py::str(obj);
      },
      ret::take_ownership);

  m.def("init_targets", []() {
    static std::once_flag init_flag;
    std::call_once(init_flag, []() {
      llvm::InitializeAllTargetInfos();
      llvm::InitializeAllTargets();
      llvm::InitializeAllTargetMCs();
      llvm::InitializeAllAsmParsers();
      llvm::InitializeAllAsmPrinters();
    });
  });

  m.def("link_extern_libs", [](llvm::Module *dstMod,
                               const std::vector<std::string> &paths) {
    if (paths.empty())
      return;

    LLVMContext &ctx = dstMod->getContext();
    llvm::Linker linker(*dstMod);
    for (const std::string &path : paths) {
      llvm::SMDiagnostic err;
      std::unique_ptr<llvm::Module> libMod = llvm::parseIRFile(path, err, ctx);
      if (!libMod) {
        std::string message = "Failed to parse library at " + path;
        throw std::invalid_argument(message);
      }
      libMod->setTargetTriple(Triple(dstMod->getTargetTriple()));
      libMod->setDataLayout(dstMod->getDataLayout());

      std::unordered_set<std::string> externalFns;
      for (llvm::Function &fn : libMod->functions()) {
        if (!fn.isDeclaration())
          externalFns.insert(fn.getName().str());
      }

      if (linker.linkInModule(std::move(libMod),
                              llvm::Linker::Flags::LinkOnlyNeeded)) {
        std::string message = "Failed to link library at " + path;
        throw std::invalid_argument(message);
      }

      // Mark linked-in functions as internal because backends use external
      // linkage as a signifier of kernel functions.
      for (llvm::Function &fn : dstMod->functions()) {
        if (externalFns.count(fn.getName().str())) {
          fn.setLinkage(llvm::GlobalValue::InternalLinkage);
        }
      }
    }
  });
}

void triton_stacktrace_signal_handler(void *) {
  llvm::sys::PrintStackTrace(llvm::errs());
  raise(SIGABRT);
}

void init_triton_stacktrace_hook(pybind11::module &m) {
  if (mlir::triton::tools::getBoolEnv("TRITON_ENABLE_PYTHON_STACKTRACE")) {
    llvm::sys::AddSignalHandler(triton_stacktrace_signal_handler, nullptr);
  }
}
