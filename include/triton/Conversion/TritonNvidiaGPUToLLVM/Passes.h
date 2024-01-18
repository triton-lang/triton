#ifndef TRITONNVIDIAGPU_CONVERSION_PASSES_H
#define TRITONNVIDIAGPU_CONVERSION_PASSES_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonNvidiaGPUToLLVM/TritonNvidiaGPUToLLVMPass.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton/Conversion/TritonNvidiaGPUToLLVM/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
