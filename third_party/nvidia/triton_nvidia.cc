#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "passes.h"
#include "triton/Conversion/NVGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/TargetSelect.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(mlir::triton::gpu::TMAMetadataTy);

void init_triton_nvidia_passes_ttgpuir(py::module &&m) {
  using namespace mlir::triton::gpu;
  ADD_PASS_WRAPPER_1("add_rewrite_tensor_pointer",
                     mlir::createTritonGPURewriteTensorPointerPass, int);
  // TODO: it is weird to pass mlir::triton::NVVM here since the conversion is
  // nvidia-specificontext
  m.def("add_to_llvmir", [](mlir::PassManager &pm, int32_t capability,
                            mlir::triton::gpu::TMAMetadataTy *tmaMetadata) {
    pm.addPass(createConvertTritonGPUToLLVMPass(capability, mlir::triton::NVVM,
                                                tmaMetadata));
  });
}

void init_triton_nvidia_passes_ttnvgpuir(py::module &&m) {
  ADD_PASS_WRAPPER_1("add_plan_cta", mlir::createTritonNvidiaGPUPlanCTAPass,
                     mlir::triton::nvidia_gpu::ClusterInfo *);
  ADD_PASS_WRAPPER_1("add_wsfeasibility_checking",
                     mlir::createTritonNvidiaGPUWSFeasibilityCheckingPass, int);
  ADD_PASS_WRAPPER_1("add_wsdecomposing",
                     mlir::createTritonNvidiaGPUWSDecomposingPass, int);
  ADD_PASS_WRAPPER_1("add_wsmutex", mlir::createTritonNvidiaGPUWSMutexPass,
                     int);
  ADD_PASS_WRAPPER_1("add_wsmaterialization",
                     mlir::createTritonNvidiaGPUWSMaterializationPass, int);
  ADD_PASS_WRAPPER_0("add_wsfixup_missing_attrs",
                     mlir::createTritonNvidiaGPUWSFixupMissingAttrs);
  ADD_PASS_WRAPPER_2("add_materialize_load_store",
                     mlir::createTritonNvidiaGPUMaterializeLoadStorePass, int,
                     int);
  ADD_PASS_WRAPPER_0("add_fence_insertion",
                     mlir::createTritonNvidiaGPUFenceInsertionPass);
  ADD_PASS_WRAPPER_0("add_nvgpu_to_llvm",
                     mlir::triton::createConvertNVGPUToLLVMPass);
  ADD_PASS_WRAPPER_3("add_wspipeline",
                     mlir::createTritonNvidiaGPUWSPipelinePass, int, int, int);

  m.def("is_ws_supported", [](mlir::ModuleOp &mod) -> bool {
    return mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect::getWSSupportedAttr(
        mod);
  });
}

void init_triton_nvidia(py::module &&m){
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

  // tma info
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
}
