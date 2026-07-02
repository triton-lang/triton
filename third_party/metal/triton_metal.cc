#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Analysis/Allocation.h"
#include "passes.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace py = nanobind;

namespace mlir::triton::metal {

// Metal GPU family constants
constexpr int kApple7 = 7;   // M1
constexpr int kApple8 = 8;   // M2
constexpr int kApple9 = 9;   // M3
constexpr int kApple10 = 10; // M4

// Metal shared memory address space (threadgroup)
constexpr int kSharedAddressSpace = 3;

// SIMD width for Apple Silicon GPUs
constexpr int kSimdWidth = 32;

// Max threadgroup memory in bytes (32 KB)
constexpr int kMaxThreadgroupMemory = 32768;

} // namespace mlir::triton::metal

namespace {

// Create a pass to allocate shared (threadgroup) memory for Metal.
// This reuses the existing ModuleAllocation infrastructure with Metal-specific
// parameters (no tensor memory, fixed 32 banks, 32-byte partition size).
void addAllocateSharedMemoryMetal(mlir::PassManager &pm, int32_t gpuFamily) {
  // Metal uses the same allocation infrastructure as NVIDIA/AMD but with:
  //   - No tensor memory (zeroed scratch)
  //   - 32 memory banks
  //   - Apple Silicon partition size: 32 bytes (aligned to SIMD width)
  pm.addPass(mlir::triton::createAllocateSharedMemoryPass(gpuFamily));
}

void init_triton_metal_passes_ttgpuir(py::module_ &m) {
  using namespace mlir::triton;

  // Allocate threadgroup (shared) memory for Metal kernels.
  // Maps TritonGPU shared memory operations to fixed offsets within
  // the Metal threadgroup memory allocation.
  m.def("add_allocate_shared_memory",
        [](mlir::PassManager &pm, int32_t gpuFamily) {
          addAllocateSharedMemoryMetal(pm, gpuFamily);
        });

  // Convert TritonGPU dialect to LLVM dialect for Metal targets.
  // Uses the shared TritonGPU-to-LLVM conversion infrastructure with
  // Metal-specific TargetInfo (SIMD shuffles, threadgroup barriers,
  // simdgroup_matrix ops instead of WMMA/MMA).
  m.def("add_to_llvmir",
        [](mlir::PassManager &pm, int32_t gpuFamily) {
          pm.addPass(mlir::triton::createConvertTritonGPUToLLVMPass(
              gpuFamily, /*ptxVersion=*/0,
              /*enableConcurrencySanitizer=*/false));
        });
}

} // namespace

void init_triton_metal(py::module_ &m) {
  m.doc() = "Triton Metal backend - Apple Silicon GPU support (Metal 4)";

  auto passes = m.def_submodule("passes");
  auto ttgpuir_m = passes.def_submodule("ttgpuir");
  init_triton_metal_passes_ttgpuir(ttgpuir_m);

  // Load required MLIR dialects for Metal compilation
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::TritonDialect,
                    mlir::triton::gpu::TritonGPUDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  // Metal device properties query
  m.def("get_shared_address_space",
        []() { return mlir::triton::metal::kSharedAddressSpace; });
  m.def("get_simd_width", []() { return mlir::triton::metal::kSimdWidth; });
  m.def("get_max_threadgroup_memory",
        []() { return mlir::triton::metal::kMaxThreadgroupMemory; });
}
