#ifndef NVGPU_CONVERSION_PASSES_H
#define NVGPU_CONVERSION_PASSES_H

#include "NVGPUToLLVM/NVGPUToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "NVGPUToLLVM/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
