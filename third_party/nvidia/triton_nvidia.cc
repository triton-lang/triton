#include "Dialect/NVGPU/IR/Dialect.h"
#include "Dialect/NVWS/IR/Dialect.h"
#include "NVGPUToLLVM/Passes.h"
#include "TritonNVIDIAGPUToLLVM/Passes.h"
#include "cublas_instance.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "passes.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/IR/Constants.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
namespace ttng = mlir::triton::nvidia_gpu;

void init_triton_nvidia_passes_ttgpuir(py::module &&m) {
  using namespace mlir::triton;
  // TODO: it is weird to pass mlir::triton::NVVM here since the conversion is
  // nvidia-specificontext
  m.def("add_to_llvmir",
        [](mlir::PassManager &pm, int32_t capability, int32_t ptxVersion) {
          pm.addPass(mlir::triton::createConvertTritonGPUToLLVMPass(
              capability, ptxVersion));
        });
}

static std::unique_ptr<mlir::Pass>
createTritonGPUFenceInsertionWrapper(int32_t capability) {
  ttng::TritonGPUFenceInsertionOptions options;
  options.computeCapability = capability;
  return ttng::createTritonGPUFenceInsertion(options);
}

static std::unique_ptr<mlir::Pass>
createTritonGPUProxyFenceInsertionWrapper(int32_t capability) {
  ttng::TritonGPUProxyFenceInsertionOptions options;
  options.computeCapability = capability;
  return ttng::createTritonGPUProxyFenceInsertion(options);
}

void init_triton_nvidia_passes_ttnvgpuir(py::module &&m) {
  ADD_PASS_WRAPPER_1("add_plan_cta", ttng::createTritonNvidiaGPUPlanCTAPass,
                     mlir::triton::nvidia_gpu::ClusterInfo *);
  ADD_PASS_WRAPPER_1("add_fence_insertion",
                     createTritonGPUFenceInsertionWrapper, int32_t);
  ADD_PASS_WRAPPER_1("add_proxy_fence_insertion",
                     createTritonGPUProxyFenceInsertionWrapper, int32_t);
  ADD_PASS_WRAPPER_0("add_tma_lowering",
                     ttng::createTritonNvidiaGPUTMALoweringPass);
  ADD_PASS_WRAPPER_0("add_promote_lhs_to_tmem",
                     ttng::createTritonNvidiaGPUPromoteLHSToTMemPass);
  ADD_PASS_WRAPPER_0("add_remove_tmem_tokens",
                     ttng::createTritonNvidiaGPURemoveTMEMTokensPass);
  ADD_PASS_WRAPPER_0("add_nvgpu_to_llvm",
                     mlir::triton::createConvertNVGPUToLLVM);
  ADD_PASS_WRAPPER_0("add_warp_specialize_to_llvm",
                     mlir::triton::createConvertWarpSpecializeToLLVM);
  ADD_PASS_WRAPPER_0("add_allocate_tensor_memory",
                     ttng::createTritonTensorMemoryAllocationPass);
  ADD_PASS_WRAPPER_0("add_lower_mma",
                     ttng::createTritonNvidiaGPUMMALoweringPass);
  ADD_PASS_WRAPPER_0("add_optimize_descriptor_encoding",
                     ttng::createTritonNvidiaGPUOptimizeDescriptorEncodingPass);
  ADD_PASS_WRAPPER_0("add_optimize_tmem_layouts",
                     ttng::createTritonNvidiaGPUOptimizeTMemLayoutsPass);
  ADD_PASS_WRAPPER_0("add_interleave_tmem",
                     ttng::createTritonNvidiaGPUInterleaveTMemPass);
}

void init_triton_nvidia_passes_nvws(py::module &&m) {
  ADD_PASS_WRAPPER_0("add_lower_warp_group",
                     mlir::triton::createNVWSLowerWarpGroup);
  ADD_PASS_WRAPPER_0("add_lower_aref", mlir::triton::createNVWSLowerAref);
}

void init_triton_hopper_passes(py::module &&m) {
  // Meta's autoWS
  ADD_PASS_OPTION_WRAPPER_2("add_hopper_warpspec",
                            mlir::createNVGPUWarpSpecialization, int, bool);
}

void init_triton_nvidia(py::module &&m) {
  auto passes = m.def_submodule("passes");
  init_triton_nvidia_passes_nvws(passes.def_submodule("nvws"));
  init_triton_nvidia_passes_ttgpuir(passes.def_submodule("ttgpuir"));
  init_triton_nvidia_passes_ttnvgpuir(passes.def_submodule("ttnvgpuir"));
  init_triton_hopper_passes(passes.def_submodule("hopper"));

  // cluster info
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

  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,
                    mlir::triton::nvgpu::NVGPUDialect,
                    mlir::triton::nvws::NVWSDialect>();
    mlir::registerNVVMDialectTranslation(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  // Set short point option, this needs to be set before setting the data
  // layout.
  m.def("set_short_ptr", []() {
    auto options = llvm::cl::getRegisteredOptions();
    const char *flag = "nvptx-short-ptr";
    auto *shortPtr = static_cast<llvm::cl::opt<bool> *>(options[flag]);
    assert(shortPtr);
    shortPtr->setValue(true);
  });

  // TODO: could be done in python if we had a generic interface to set metadata
  m.def("set_nvvm_reflect_ftz", [](llvm::Module *mod) {
    // please check https://llvm.org/docs/NVPTXUsage.html#reflection-parameters
    // this will enable fast math path in libdevice
    // for example, when enable nvvm-reflect-ftz, sqrt.approx.f32 will change to
    // sqrt.approx.ftz.f32
    using namespace llvm;
    auto &ctx = mod->getContext();
    Type *i32 = Type::getInt32Ty(ctx);
    auto *mdFour = ConstantAsMetadata::get(ConstantInt::getSigned(i32, 4));
    auto *mdName = MDString::get(ctx, "nvvm-reflect-ftz");
    auto *mdOne = ConstantAsMetadata::get(ConstantInt::getSigned(i32, 1));
    auto *reflect = MDNode::get(ctx, {mdFour, mdName, mdOne});
    mod->addModuleFlag(reflect);
  });

  // cublas
  auto cublas = m.def_submodule("cublas");

  py::class_<CublasLtInstance>(cublas, "CublasLt")
      .def(py::init<>([&](py::object &workspace) {
        auto wrk_ptr = workspace.attr("data_ptr")().cast<uint64_t>();
        auto wrk_size = workspace.attr("numel")().cast<size_t>() *
                        workspace.attr("element_size")().cast<size_t>();
        return new CublasLtInstance(wrk_ptr, wrk_size);
      }))
      .def("matmul", [](CublasLtInstance &self, py::object &A, py::object &B,
                        py::object &C) {
        auto A_ptr = A.attr("data_ptr")().cast<uint64_t>();
        auto B_ptr = B.attr("data_ptr")().cast<uint64_t>();
        auto C_ptr = C.attr("data_ptr")().cast<uint64_t>();

        auto A_shape = A.attr("shape").cast<std::vector<int>>();
        auto B_shape = B.attr("shape").cast<std::vector<int>>();
        auto C_shape = C.attr("shape").cast<std::vector<int>>();

        auto A_dtype = A.attr("dtype").attr("__str__")().cast<std::string>();
        auto B_dtype = B.attr("dtype").attr("__str__")().cast<std::string>();
        auto C_dtype = C.attr("dtype").attr("__str__")().cast<std::string>();

        assert(A_dtype == B_dtype && A_dtype == C_dtype);
        assert(A_dtype == "torch.float8_e4m3fn" || A_dtype == "torch.float16");

        std::string dtype_str = A_dtype.substr(A_dtype.find_last_of('.') + 1);
        cudaDataType_t dtype;
        if (dtype_str == "float8_e4m3fn") {
          dtype = CUDA_R_8F_E4M3;
        } else if (dtype_str == "float16") {
          dtype = CUDA_R_16F;
        }

        if (A_shape.size() != 2 || B_shape.size() != 2 || C_shape.size() != 2) {
          throw std::runtime_error("Only 2D matrices are supported.");
        }

        int k = A_shape[1];
        if (k != B_shape[1]) {
          throw std::runtime_error("Matrix dimensions do not match. A is [" +
                                   std::to_string(A_shape[0]) + ", " +
                                   std::to_string(A_shape[1]) + "], B is [" +
                                   std::to_string(B_shape[0]) + ", " +
                                   std::to_string(B_shape[1]) +
                                   "]. Expected A.shape[1] == B.shape[1]. Note "
                                   "that B needs to be transposed.");
        }

        int m = A_shape[0];
        if (m != C_shape[0]) {
          throw std::runtime_error("Matrix dimensions do not match. A is [" +
                                   std::to_string(A_shape[0]) + ", " +
                                   std::to_string(A_shape[1]) + "], C is [" +
                                   std::to_string(C_shape[0]) + ", " +
                                   std::to_string(C_shape[1]) +
                                   "]. Expected A.shape[0] == C.shape[0].");
        }

        int n = B_shape[0];
        if (n != C_shape[1]) {
          throw std::runtime_error("Matrix dimensions do not match. B is [" +
                                   std::to_string(B_shape[0]) + ", " +
                                   std::to_string(B_shape[1]) + "], C is [" +
                                   std::to_string(C_shape[0]) + ", " +
                                   std::to_string(C_shape[1]) +
                                   "]. Expected B.shape[0] == C.shape[1]. Note "
                                   "that B needs to be transposed.");
        }

        self.matmul(A_shape[0], B_shape[0], A_shape[1], A_ptr, B_ptr, C_ptr,
                    dtype);
      });
}
