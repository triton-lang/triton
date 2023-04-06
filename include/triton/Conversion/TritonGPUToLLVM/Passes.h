#ifndef TRITONGPU_CONVERSION_PASSES_H
#define TRITONGPU_CONVERSION_PASSES_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/TritonGPUToLLVMPass.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton/Conversion/TritonGPUToLLVM/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
