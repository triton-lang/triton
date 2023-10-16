#ifndef CONVERSION_TRITON_TO_TRITONGPUROCM_PASSES_H
#define CONVERSION_TRITON_TO_TRITONGPUROCM_PASSES_H

#include "triton/Conversion/TritonToTritonGPUROCM/TritonToTritonGPUPass.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton/Conversion/TritonToTritonGPUROCM/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
