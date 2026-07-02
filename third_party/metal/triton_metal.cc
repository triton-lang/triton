#include "TritonMetalToLLVM/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace py = nanobind;

namespace mlir::triton::metal {

constexpr int kApple7 = 7;   // M1
constexpr int kApple8 = 8;   // M2
constexpr int kApple9 = 9;   // M3
constexpr int kApple10 = 10; // M4

constexpr int kSharedAddressSpace = 3;
constexpr int kSimdWidth = 32;
constexpr int kMaxThreadgroupMemory = 32768;

} // namespace mlir::triton::metal

namespace {

void init_triton_metal_passes_ttgpuir(py::module_ &m) {
  m.def("add_allocate_shared_memory",
        [](mlir::PassManager &pm, int32_t gpuFamily) {
          pm.addPass(mlir::triton::metal::createAllocateSharedMemoryMetalPass(
              gpuFamily));
        });

  m.def("add_to_llvmir", [](mlir::PassManager &pm, int32_t gpuFamily) {
    pm.addPass(
        mlir::triton::metal::createConvertTritonMetalToLLVMPass(gpuFamily));
  });
}

} // namespace

void init_triton_metal(py::module_ &m) {
  m.doc() = "Triton Metal backend - Apple Silicon GPU support (Metal 4)";

  auto passes = m.def_submodule("passes");
  auto ttgpuir_m = passes.def_submodule("ttgpuir");
  init_triton_metal_passes_ttgpuir(ttgpuir_m);

  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::TritonDialect,
                    mlir::triton::gpu::TritonGPUDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  m.def("get_shared_address_space",
        []() { return mlir::triton::metal::kSharedAddressSpace; });
  m.def("get_simd_width", []() { return mlir::triton::metal::kSimdWidth; });
  m.def("get_max_threadgroup_memory",
        []() { return mlir::triton::metal::kMaxThreadgroupMemory; });
}
