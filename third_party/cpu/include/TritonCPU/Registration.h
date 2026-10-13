#ifndef TRITON_THIRD_PARTY_CPU_INCLUDE_TRITONCPU_REGISTRATION_H_
#define TRITON_THIRD_PARTY_CPU_INCLUDE_TRITONCPU_REGISTRATION_H_

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"
#include "cpu/include/ScalarizePass/ScalarizeInterfaceImpl.h"
#include "cpu/include/TritonCPUToLLVM/Passes.h"
#include "cpu/include/TritonCPUTransforms/Passes.h"
#include "cpu/include/TritonToTritonCPU/Passes.h"
#include "mlir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/X86/X86Dialect.h"
#include "mlir/IR/DialectRegistry.h"

namespace mlir::triton::cpu {

inline void registerTritonCPU(DialectRegistry &registry) {
  registerTritonToTritonCPUPasses();
  registerTritonCPUTransformsPasses();
  registerTritonCPUToLLVMPasses();
  registerTritonOpScalarizeExternalModels(registry);
  registry.insert<TritonCPUDialect, memref::MemRefDialect,
                  vector::VectorDialect, x86::X86Dialect, tensor::TensorDialect,
                  arm_neon::ArmNeonDialect>();
}

} // namespace mlir::triton::cpu

#endif // TRITON_THIRD_PARTY_CPU_INCLUDE_TRITONCPU_REGISTRATION_H_
