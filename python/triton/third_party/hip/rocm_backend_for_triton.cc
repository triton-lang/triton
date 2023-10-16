#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "triton/AnalysisROCM/Allocation.h"
#include "triton/Conversion/TritonGPUROCMToLLVM/TritonGPUToLLVMPass.h"
#include "triton/Conversion/TritonToTritonGPUROCM/TritonToTritonGPUPass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPUROCM/Transforms/Passes.h"
#include "triton/Target/HSACO/HSACOTranslation.h"
// #include "triton/Target/LLVMIR/LLVMIRTranslation.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "triton/Tools/Sys/GetPlatform.hpp"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Support/SourceMgr.h"

#include <Python.h>
#include <cctype>
#include <fstream>
#include <optional>
#include <pybind11/buffer_info.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <regex>
#include <signal.h>
#include <sstream>
#include <stdexcept>
#include <string>

namespace py = pybind11;

mlir::ModuleOp parse_mlir_module(std::string &module_str,
                                 mlir::MLIRContext &context) {
  // std::cout << "module_str:" << module_str << std::endl;
  // initialize registry
  // note: we initialize llvm for undef
  mlir::DialectRegistry registry;
  registry.insert<mlir::triton::TritonDialect,
                  mlir::triton::gpu_rocm::TritonGPUROCMDialect,
                  mlir::math::MathDialect, mlir::arith::ArithDialect,
                  mlir::index::IndexDialect, mlir::scf::SCFDialect,
                  mlir::cf::ControlFlowDialect, mlir::LLVM::LLVMDialect,
                  mlir::ROCDL::ROCDLDialect, mlir::BuiltinDialect>();
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();

  // parse module
  mlir::OwningOpRef<mlir::ModuleOp> module_op =
      mlir::parseSourceString<mlir::ModuleOp>(module_str, &context);
  if (!module_op)
    throw std::runtime_error("Parse MLIR file failed.");

  // return by value is fine
  return module_op->clone();
}

std::unique_ptr<llvm::Module> parse_llir_module(const std::string &module_str,
                                                llvm::LLVMContext &context) {
  // parse string
  std::unique_ptr<llvm::MemoryBuffer> buffer =
      llvm::MemoryBuffer::getMemBuffer(module_str);
  llvm::SMDiagnostic error;
  std::unique_ptr<llvm::Module> llvmMod =
      llvm::parseIR(buffer->getMemBufferRef(), error, context);

  // Check for parsing errors
  if (!llvmMod) {
    std::string errorMsg = "Error parsing LLIR: " + error.getMessage().str();
    throw std::runtime_error(errorMsg);
  }

  // Return the module by value
  return llvmMod;
}

void init_triton_rocm_translation(py::module &m) {
  // std::cout << "init_triton_rocm_translation" << std::endl;
  using ret = py::return_value_policy;

  m.def("get_shared_memory_size", [](std::string ttgir) -> int {
    // std::cout << "get_shared_memory_size" << std::endl;
    mlir::MLIRContext context;
    mlir::ModuleOp ttgir_module = parse_mlir_module(ttgir, context);

    // triton gpu to llvm mlir dialect
    mlir::triton::translateTritonGPUROCMToLLVMDialect(
        ttgir_module, 0, true);

    auto shared = ttgir_module->getAttrOfType<mlir::IntegerAttr>(
        "triton_gpu_rocm.shared");
    if (!ttgir_module->hasAttr("triton_gpu_rocm.shared")) {
      std::cerr << "Attribute triton_gpu_rocm.shared does not exist"
                << std::endl;
      return -1;
    }

    int ret = shared.getInt();
    return ret;
  });

  m.def("get_num_warps", [](std::string ttgir) -> int {
    // parse module
    mlir::MLIRContext context;
    mlir::ModuleOp ttgir_module = parse_mlir_module(ttgir, context);

    // check attribute
    auto num_warps_str = "triton_gpu_rocm.num-warps";
    auto num_warps = ttgir_module->getAttrOfType<mlir::IntegerAttr>(
        num_warps_str);
    if (!ttgir_module->hasAttr(num_warps_str)) {
      std::cerr << "Attribute" << num_warps_str << "does not exist"
                << std::endl;
      return -1;
    }

    int ret = num_warps.getInt();
    return ret;
  });

  m.def(
      "translate_ttir_to_ttgir_rocm",
      [](std::string ttir, int computeCapability, int numWarps,
         int numStages) -> std::string {
        // std::cout << "translate_ttir_to_ttgir_rocm" << std::endl;
        mlir::MLIRContext context;
        mlir::ModuleOp ttir_module = parse_mlir_module(ttir, context);

        // triton to triton gpu
        mlir::triton::translateTritonToTritonGPUROCM(
            ttir_module, computeCapability, numWarps, numStages);

        // write to str
        std::string moduleStr;
        llvm::raw_string_ostream os(moduleStr);
        ttir_module.print(os);
        os.flush();

        return moduleStr;
      },
      ret::take_ownership);

  m.def(
      "translate_ttgir_to_llvmir",
      [](std::string ttgir, const std::vector<std::string> &names,
         const std::vector<std::string> &paths) -> std::string {
        // std::cout << "translate_ttgir_to_llvmir" << std::endl;
        // params
        bool isROCM = true;
        int computeCapability = 0;

        // load ttgir
        mlir::MLIRContext context;
        mlir::ModuleOp ttgir_module = parse_mlir_module(ttgir, context);

        // add external libs
        mlir::triton::addExternalLibs(ttgir_module, names, paths);

        // triton gpu to llvm mlir dialect
        mlir::triton::translateTritonGPUROCMToLLVMDialect(
            ttgir_module, computeCapability, isROCM);

        // llvm mlir module to llvm ir
        llvm::LLVMContext llvmContext;
        std::unique_ptr<llvm::Module> llvmModule =
            mlir::triton::translateLLVMDialectToLLVMIR(
                &llvmContext, ttgir_module, isROCM);
        if (!llvmModule) {
          llvm::report_fatal_error(
              "Failed to translate TritonGPUROCM to LLVM IR.");
        }

        std::string llvmIR;
        llvm::raw_string_ostream os(llvmIR);
        llvmModule->print(os, nullptr);
        os.flush();

        return llvmIR;
      },
      ret::take_ownership);

  m.def(
      "translate_llvmir_to_hsaco",
      [](std::string llvmIR, std::string gfx_arch, std::string gfx_triple,
         std::string gfx_features) -> std::tuple<std::string, std::string> {
        // std::cout << "translate_llvmir_to_hsaco" << std::endl;

        llvm::LLVMContext llvmContext;
        std::unique_ptr<llvm::Module> llvmModule =
            parse_llir_module(llvmIR, llvmContext);

        // translate module to HSACO
        auto hsacoCode = mlir::triton::translateLLVMIRToHSACO(
            *llvmModule, gfx_arch, gfx_triple, gfx_features);

        return hsacoCode;
      },
      ret::take_ownership);

  m.def("add_external_libs_rocm",
        [](mlir::ModuleOp &op, const std::vector<std::string> &names,
           const std::vector<std::string> &paths) {
          // std::cout << "add_external_libs_rocm" << std::endl;
          ::mlir::triton::addExternalLibs(op, names, paths);
        });

  m.def(
      "translate_triton_ir_to_amdgcn_and_hsaco",
      [](std::string ttir, std::string gfx_arch, std::string gfx_triple,
         std::string gfx_features, int numWarps, int numStages,
         const std::vector<std::string> &names,
         const std::vector<std::string> &paths)
          -> std::tuple<std::string, std::string> {
        // std::cout << "translate_triton_ir_to_amdgcn_and_hsaco" << std::endl;

        mlir::MLIRContext context;
        mlir::ModuleOp ttir_module = parse_mlir_module(ttir, context);

        // triton to hsaco Code
        auto hsacoCode = mlir::triton::translateTritonIRToHSACO(
            ttir_module, gfx_arch, gfx_triple, gfx_features, numWarps,
            numStages, names, paths);

        return hsacoCode;
      },
      ret::take_ownership);
  //  std::cout << "init_triton_rocm_translation: done!" << std::endl;
}

void init_rocm_backend_for_triton(py::module &m) {
  // std::cout << "init_rocm_backend_for_triton" << std::endl;
  py::module subm = m.def_submodule("triton");

  init_triton_rocm_translation(subm);
  // std::cout << "init_rocm_backend_for_triton: done!" << std::endl;
}

PYBIND11_MODULE(librocm_backend_for_triton, m) {
  m.doc() = "Python bindings to the ROCM Backend for Triton API";
  init_rocm_backend_for_triton(m);
}
