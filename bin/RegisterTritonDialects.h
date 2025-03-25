#pragma once

#ifdef __NVIDIA__
#include "third_party/nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#endif
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#ifdef __NVIDIA__
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#endif

// Below headers will allow registration to ROCm passes
#ifdef __AMD__
#include "TritonAMDGPUToLLVM/Passes.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "TritonAMDGPUTransforms/TritonGPUConversion.h"
#endif

#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#ifdef __NVIDIA__
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#endif

#ifdef __NVIDIA__
#include "nvidia/include/NVGPUToLLVM/Passes.h"
#include "nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h"
#endif
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Target/LLVMIR/Passes.h"

#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/InitAllPasses.h"

#include <dlfcn.h>
#include <iostream>
#include <string>

namespace mlir {
namespace test {
void registerTestAliasPass();
void registerTestAlignmentPass();
void registerTestAllocationPass();
void registerTestMembarPass();
} // namespace test
} // namespace mlir

inline void registerTritonDialects(mlir::DialectRegistry &registry) {
  mlir::registerAllPasses();
  mlir::registerTritonPasses();
  mlir::triton::gpu::registerTritonGPUPasses();
#ifdef __NVIDIA__
  mlir::registerTritonNvidiaGPUPasses();
  mlir::test::registerTestAliasPass();
  mlir::test::registerTestAlignmentPass();
  mlir::test::registerTestAllocationPass();
  mlir::test::registerTestMembarPass();
#endif
  mlir::triton::registerConvertTritonToTritonGPUPass();
  mlir::triton::registerAllocateSharedMemoryPass();
#ifdef __NVIDIA__
  mlir::triton::registerConvertTritonGPUToLLVMPass();
  mlir::triton::registerConvertNVGPUToLLVMPass();
  mlir::triton::registerDecomposeUnsupportedNVIDIAConversions();
#endif
  mlir::registerLLVMDIScope();

#ifdef __AMD__
  // TritonAMDGPUToLLVM passes
  mlir::triton::registerConvertTritonAMDGPUToLLVM();
  mlir::triton::registerConvertBuiltinFuncToLLVM();
  mlir::triton::registerDecomposeUnsupportedAMDConversions();

  // TritonAMDGPUTransforms passes
  mlir::registerTritonAMDGPUAccelerateMatmul();
  mlir::registerTritonAMDGPUOptimizeEpilogue();
  mlir::registerTritonAMDGPUReorderInstructions();
  mlir::registerTritonAMDGPUStreamPipeline();
#endif

  // TODO: register Triton & TritonGPU passes
  registry.insert<mlir::triton::TritonDialect, mlir::cf::ControlFlowDialect,
#ifdef __NVIDIA__
                  mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,
#endif
                  mlir::triton::gpu::TritonGPUDialect, mlir::math::MathDialect,
                  mlir::arith::ArithDialect, mlir::scf::SCFDialect,
                  mlir::gpu::GPUDialect, mlir::LLVM::LLVMDialect,
#ifdef __NVIDIA__
                  mlir::triton::nvgpu::NVGPUDialect,
#endif
#ifdef __AMD__
                  mlir::ROCDL::ROCDLDialect,
#endif
                  mlir::NVVM::NVVMDialect>();
}
