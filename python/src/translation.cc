#include "triton/Conversion/TritonGPUToLLVM/TritonGPUToLLVMPass.h"
#include "triton/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Target/LLVMIR/LLVMIRTranslation.h"
#include "triton/Target/PTX/TmaMetadata.h"
#include "triton/Tools/Sys/GetEnv.hpp"

#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

#include <fstream>
#include <sstream>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <signal.h>

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(mlir::triton::gpu::TMAMetadataTy);

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

void init_triton_translation(py::module &&m) {
  using ret = py::return_value_policy;

  py::class_<mlir::triton::nvidia_gpu::ClusterInfo>(m, "ClusterInfo")
      .def(py::init<>())
      .def_readwrite("clusterDimX",
                     &mlir::triton::nvidia_gpu::ClusterInfo::clusterDimX)
      .def_readwrite("clusterDimY",
                     &mlir::triton::nvidia_gpu::ClusterInfo::clusterDimY)
      .def_readwrite("clusterDimZ",
                     &mlir::triton::nvidia_gpu::ClusterInfo::clusterDimZ)
      .def("__repr__", [](mlir::triton::nvidia_gpu::ClusterInfo &self) {
        std::ostringstream oss;
        oss << "(" << self.clusterDimX << ", " << self.clusterDimY << ", "
            << self.clusterDimZ << ")";
        return oss.str();
      });

  py::class_<mlir::triton::gpu::TMAInfo>(m, "TMAInfo")
      .def(py::init<>())
      .def_readwrite("tensorDataType",
                     &mlir::triton::gpu::TMAInfo::tensorDataType)
      .def_readwrite("tensorRank", &mlir::triton::gpu::TMAInfo::tensorRank)
      .def_readwrite("globalAddressArgIdx",
                     &mlir::triton::gpu::TMAInfo::globalAddressArgIdx)
      .def_readwrite("globalStridesArgIdx",
                     &mlir::triton::gpu::TMAInfo::globalStridesArgIdx)
      .def_readwrite("globalDimsArgIdx",
                     &mlir::triton::gpu::TMAInfo::globalDimsArgIdx)
      .def_readwrite("boxDims", &mlir::triton::gpu::TMAInfo::boxDims)
      .def_readwrite("elementStrides",
                     &mlir::triton::gpu::TMAInfo::elementStrides)
      .def_readwrite("interleave", &mlir::triton::gpu::TMAInfo::interleave)
      .def_readwrite("swizzle", &mlir::triton::gpu::TMAInfo::swizzle)
      .def_readwrite("l2Promotion", &mlir::triton::gpu::TMAInfo::l2Promotion)
      .def_readwrite("oobFill", &mlir::triton::gpu::TMAInfo::oobFill)
      .def_readwrite("TMADescArgIdx",
                     &mlir::triton::gpu::TMAInfo::TMADescArgIdx);
  py::bind_vector<std::vector<mlir::triton::gpu::TMAInfo>>(m, "TMAInfos");

  m.def("get_shared_memory_size", [](mlir::ModuleOp mod) {
    auto shared = mod->getAttrOfType<mlir::IntegerAttr>("triton_gpu.shared");
    return shared.getInt();
  });
  m.def("get_num_warps", [](mlir::ModuleOp mod) {
    auto shared = mod->getAttrOfType<mlir::IntegerAttr>("triton_gpu.num-warps");
    assert(shared);
    int num_warps = shared.getInt();

    if (auto attr = mod->getAttrOfType<mlir::IntegerAttr>(
            "triton_gpu.num-warp-groups-per-cta")) {
      num_warps *= attr.getInt();
    }

    return num_warps;
  });

  m.def(
      "translate_triton_gpu_to_llvmir",
      [](mlir::ModuleOp op, int computeCapability,
         mlir::triton::gpu::TMAMetadataTy &tmaInfos) {
        auto target = mlir::triton::NVVM;
        py::gil_scoped_release allow_threads;
        llvm::LLVMContext llvmContext;
        auto llvmModule = ::mlir::triton::translateTritonGPUToLLVMIR(
            &llvmContext, op, computeCapability, tmaInfos, target);
        if (!llvmModule)
          llvm::report_fatal_error("Failed to translate TritonGPU to LLVM IR.");

        std::string str;
        llvm::raw_string_ostream os(str);
        llvmModule->print(os, nullptr);
        os.flush();
        return str;
      },
      ret::take_ownership);

  m.def(
      "translate_llvmir_to_asm",
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
        // TODO: noinline stuff; only consider kernels
        std::set<llvm::Function *> kernels;
        findKernels(*module, kernels);
        assert(kernels.size() == 1);
        std::string name = (*kernels.begin())->getName().str();
        std::string obj = mlir::triton::translateLLVMIRToASM(
            *module, triple, proc, features, flags, enable_fp_fusion, isObject);
        if (isObject)
          return std::make_tuple(py::bytes(obj), name);
        else
          return std::make_tuple(py::str(obj), name);
      },
      ret::take_ownership);

  m.def("compile_ptx_to_cubin",
        [](const std::string &ptxCode, const std::string &ptxasPath,
           int capability, bool enable_fp_fusion) -> py::object {
          std::string cubin;
          {
            py::gil_scoped_release allow_threads;

            // compile ptx with ptxas
            llvm::SmallString<64> fsrc;
            llvm::SmallString<64> flog;
            llvm::sys::fs::createTemporaryFile("compile-ptx-src", "", fsrc);
            llvm::sys::fs::createTemporaryFile("compile-ptx-log", "", flog);
            std::string fbin = std::string(fsrc) + ".o";
            llvm::FileRemover logRemover(flog);
            llvm::FileRemover binRemover(fbin);
            const char *_fsrc = fsrc.c_str();
            const char *_flog = flog.c_str();
            const char *_fbin = fbin.c_str();
            std::ofstream ofs(_fsrc);
            ofs << ptxCode << std::endl;
            ofs.close();

            auto lineInfoOption =
                triton::tools::getBoolEnv("TRITON_DISABLE_LINE_INFO")
                    ? ""
                    : " -lineinfo";
            auto fmadOption = enable_fp_fusion ? "" : " --fmad=false";
            auto capabilitySuffix = (capability == 90) ? "a " : " ";
            auto outputFileName = std::string(_fsrc) + ".o";
            auto logRedirect = " 2> " + std::string(_flog);
            std::string cmd = ptxasPath + lineInfoOption + fmadOption +
                              " -v --gpu-name=sm_" +
                              std::to_string(capability) + capabilitySuffix +
                              _fsrc + " -o " + outputFileName + logRedirect;

            int err = system(cmd.c_str());
            if (err != 0) {
              err >>= 8;
              std::ifstream _log(_flog);
              std::string log(std::istreambuf_iterator<char>(_log), {});
              if (err == 255) {
                throw std::runtime_error(
                    "Internal Triton PTX codegen error: \n" + log);
              } else if (err == 128 + SIGSEGV) {
                throw std::runtime_error("Please run `ptxas " +
                                         fsrc.str().str() +
                                         "` to confirm that this is a "
                                         "bug in `ptxas`\n" +
                                         log);
              } else {
                throw std::runtime_error("`ptxas` failed with error code " +
                                         std::to_string(err) + ": \n" + log);
              }
              return {};
            } else {
              llvm::FileRemover srcRemover(fsrc);
              std::ifstream _cubin(_fbin, std::ios::binary);
              cubin = std::string(std::istreambuf_iterator<char>(_cubin), {});
              _cubin.close();
              // Do not return here, exit the gil scope and return below
            }
          }
          py::bytes bytes(cubin);
          return std::move(bytes);
        });

  m.def("add_external_libs",
        [](mlir::ModuleOp &op, const std::vector<std::string> &names,
           const std::vector<std::string> &paths) {
          ::mlir::triton::addExternalLibs(op, names, paths);
        });
}
