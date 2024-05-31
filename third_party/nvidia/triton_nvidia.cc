#include "Dialect/NVGPU/IR/Dialect.h"
#include "NVGPUToLLVM/NVGPUToLLVMPass.h"
#include "TritonNVIDIAGPUToLLVM/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "passes.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/IR/Constants.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "backend/include/cuda.h"
#include <dlfcn.h>

namespace py = pybind11;

void init_triton_nvidia_passes_ttgpuir(py::module &&m) {
  using namespace mlir::triton;
  // TODO: it is weird to pass mlir::triton::NVVM here since the conversion is
  // nvidia-specificontext
  m.def("add_to_llvmir", [](mlir::PassManager &pm, int32_t capability) {
    pm.addPass(mlir::triton::createConvertTritonGPUToLLVMPass(capability));
  });
  m.def("add_decompose_unsupported_conversions", [](mlir::PassManager &pm) {
    pm.addPass(NVIDIA::createDecomposeUnsupportedConversionsPass());
  });
}

void init_triton_nvidia_passes_ttnvgpuir(py::module &&m) {
  ADD_PASS_WRAPPER_1("add_plan_cta", mlir::createTritonNvidiaGPUPlanCTAPass,
                     mlir::triton::nvidia_gpu::ClusterInfo *);
  ADD_PASS_WRAPPER_0("add_fence_insertion",
                     mlir::createTritonNvidiaGPUFenceInsertionPass);
  ADD_PASS_WRAPPER_0("add_tma_lowering",
                     mlir::createTritonNvidiaGPUTMALoweringPass);
  ADD_PASS_WRAPPER_0("add_nvgpu_to_llvm",
                     mlir::triton::createConvertNVGPUToLLVMPass);
}

// Forward decls of cublas types
/* CUBLAS status type returns */
typedef enum {
  CUBLAS_STATUS_SUCCESS = 0,
  CUBLAS_STATUS_NOT_INITIALIZED = 1,
  CUBLAS_STATUS_ALLOC_FAILED = 3,
  CUBLAS_STATUS_INVALID_VALUE = 7,
  CUBLAS_STATUS_ARCH_MISMATCH = 8,
  CUBLAS_STATUS_MAPPING_ERROR = 11,
  CUBLAS_STATUS_EXECUTION_FAILED = 13,
  CUBLAS_STATUS_INTERNAL_ERROR = 14,
  CUBLAS_STATUS_NOT_SUPPORTED = 15,
  CUBLAS_STATUS_LICENSE_ERROR = 16
} cublasStatus_t;

struct cublasContext;
typedef struct cublasLtContext *cublasLtHandle_t;
struct cublasLtMatmulDescOpaque_t;
typedef cublasLtMatmulDescOpaque_t *cublasLtMatmulDesc_t;
struct cublasLtMatmulAlgo_t;

// Typedefs for cublas functions
typedef cublasStatus_t (*cublasLtCreate_t)(cublasLtHandle_t *);
typedef cublasStatus_t (*cublasLtDestroy_t)(cublasLtHandle_t);

cublasLtCreate_t _cublasLtCreate;
cublasLtDestroy_t _cublasLtDestroy;

void *cublas_handle = nullptr;

void load_cublas() {
  void *cublas_handle = dlopen("libcublas.so", RTLD_LAZY);
  if (!cublas_handle) {
    fprintf(stderr, "%s\n", dlerror());
    exit(1);
  }
  dlerror(); // Clear any existing error
  _cublasLtCreate = (cublasLtCreate_t)dlsym(cublas_handle, "cublasLtCreate");
  _cublasLtDestroy = (cublasLtDestroy_t)dlsym(cublas_handle, "cublasLtDestroy");
  const char *dlsym_error = dlerror();
  if (dlsym_error) {
    fprintf(stderr, "%s\n", dlsym_error);
    exit(1);
  }
}

void unload_cublas() { dlclose(cublas_handle); }

void dummy_cublas_call() {
  cublasLtHandle_t handle;
  _cublasLtCreate(&handle);
  _cublasLtDestroy(handle);
}

void init_triton_nvidia(py::module &&m) {
  auto passes = m.def_submodule("passes");
  init_triton_nvidia_passes_ttgpuir(passes.def_submodule("ttgpuir"));
  init_triton_nvidia_passes_ttnvgpuir(passes.def_submodule("ttnvgpuir"));

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
                    mlir::triton::nvgpu::NVGPUDialect>();
    mlir::registerNVVMDialectTranslation(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
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
  if (!cublas) {
    throw std::runtime_error("Failed to create cublas submodule");
  }
  cublas.def("load", &load_cublas);
  cublas.def("unload", &unload_cublas);
  cublas.def("dummy_call", &dummy_cublas_call);
}
