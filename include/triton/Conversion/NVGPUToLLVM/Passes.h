#ifndef NVGPU_CONVERSION_PASSES_H
#define NVGPU_CONVERSION_PASSES_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/NVGPUToLLVM/NVGPUToLLVMPass.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton/Conversion/NVGPUToLLVM/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
