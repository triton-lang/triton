#include "TritonMTGPUToLLVM/MUSATranslation.h"
#include "TritonMTGPUToLLVM/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/MTGPU/MTGPUToLLVMIRTranslation.h"
#include "passes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#ifdef _WIN32
#define PLUGIN_EXPORT __declspec(dllexport)
#else
#define PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

namespace py = pybind11;

using namespace mlir;

PLUGIN_EXPORT void init_triton_mthreads_passes_ttgpuir(py::module &&m) {
  using namespace mlir::triton;

  // ttgir -> llvm dialect
  m.def("add_to_llvmir", [](mlir::PassManager &pm, int32_t capability) {
    pm.addPass(mlir::triton::createConvertTritonMTGPUToLLVMPass(capability));
  });
  m.def("add_mtgpu_builtin_func_to_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createConvertMTGPUBuiltinFuncToLLVMPass());
  });
}

PLUGIN_EXPORT void init_triton_mthreads(py::module &&m) {
  using ret = py::return_value_policy;

  auto passes = m.def_submodule("passes");
  init_triton_mthreads_passes_ttgpuir(passes.def_submodule("ttgpuir"));

  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    mlir::registerMTGPUDialectTranslation(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });
  m.def(
      "translate_llvmir_to_mubin",
      [](const std::string llvmIR, const std::string opt_option, int capability,
         int version) -> std::tuple<std::string, std::string> {
        // create LLVM module from C++
        llvm::LLVMContext context;
        std::unique_ptr<llvm::MemoryBuffer> buffer =
            llvm::MemoryBuffer::getMemBuffer(llvmIR.c_str());
        llvm::SMDiagnostic error;
        std::unique_ptr<llvm::Module> module =
            llvm::parseIR(buffer->getMemBufferRef(), error, context);
        // translate module to mubin
        if (!module) {
          llvm::report_fatal_error(
              "failed to parse IR: " + error.getMessage() +
              "lineno: " + std::to_string(error.getLineNo()));
        }
        auto mubinCode = triton::translateLLVMIRToMUBIN(*module, opt_option,
                                                        capability, version);
        return mubinCode;
      },
      ret::take_ownership);
  m.def("attach_datalayout", [](llvm::Module &module) {
    const std::string dataLayout = "e-p:64:64:64:64-"
                                   "p1:64:64:64:64-"
                                   "p2:64:64:64:64-"
                                   "p3:32:32-"
                                   "p4:32:32-"
                                   "p5:64:64-"
                                   "i64:64-"
                                   "v16:16-"
                                   "v24:32-"
                                   "v32:32-"
                                   "v48:64-"
                                   "v96:128";
    module.setDataLayout(dataLayout);
  });
}

extern "C" {
PLUGIN_EXPORT void registerMthreadsPasses() {
  mlir::triton::registerConvertTritonMTGPUToLLVM();
  mlir::triton::registerConvertMTGPUBuiltinFuncToLLVM();
}
}
