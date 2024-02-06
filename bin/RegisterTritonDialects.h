#pragma once
#include "triton/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "TritonAMDGPUToLLVM/Passes.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "TritonAMDGPUTransforms/TritonGPUConversion.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "nvidia/include/NVGPUToLLVM/Passes.h"
#include "nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Target/LLVMIR/Passes.h"

#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/InitAllPasses.h"
#include "triton/Tools/Sys/GetEnv.hpp"

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
  mlir::registerTritonGPUPasses();
  mlir::registerTritonNvidiaGPUPasses();
  mlir::test::registerTestAliasPass();
  mlir::test::registerTestAlignmentPass();
  mlir::test::registerTestAllocationPass();
  mlir::test::registerTestMembarPass();
  mlir::triton::registerConvertTritonToTritonGPUPass();
  mlir::triton::registerDecomposeUnsupportedConversionsPass();
  mlir::triton::registerAllocateSharedMemoryPass();
  mlir::triton::registerConvertTritonGPUToLLVMPass();
  mlir::triton::registerConvertNVGPUToLLVMPass();
  mlir::registerLLVMDIScope();

  // AMD GPU specific passes registration
  mlir::triton::registerConvertTritonAMDGPUToLLVM();

  // TODO: Uncomment when fixed undefined symbols(see below)
  // mlir::registerTritonAMDGPUPasses();

  // TODO: remove section below when line above works
  mlir::registerTritonAMDGPUAccelerateMatmul();
  // FIXME: createTritonAMDGPUCoalesce is not defined
  // mlir::registerTritonAMDGPUCoalesce();
  mlir::registerTritonAMDGPUDecomposeConversions();
  // FIXME: createTritonAMDGPUOptimizeDotOperands is not defined
  // mlir::registerTritonAMDGPUOptimizeDotOperands();
  mlir::registerTritonAMDGPUOptimizeEpilogue();
  // FIXME: createTritonAMDGPUPipeline is not defined
  // mlir::registerTritonAMDGPUPipeline();
  // FIXME: createTritonAMDGPUPrefetch is not defined
  // mlir::registerTritonAMDGPUPrefetch();
  mlir::registerTritonAMDGPURemoveLayoutConversions();
  mlir::registerTritonAMDGPUReorderInstructions();
  mlir::registerTritonAMDGPUStreamPipeline();

  // End of AMD GPU specific passes registration

  // TODO: register Triton & TritonGPU passes
  registry.insert<mlir::triton::TritonDialect, mlir::cf::ControlFlowDialect,
                  mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,
                  mlir::triton::gpu::TritonGPUDialect, mlir::math::MathDialect,
                  mlir::arith::ArithDialect, mlir::scf::SCFDialect,
                  mlir::gpu::GPUDialect, mlir::LLVM::LLVMDialect,
                  mlir::NVVM::NVVMDialect, mlir::triton::nvgpu::NVGPUDialect>();
}
