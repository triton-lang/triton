#ifndef TRITON_CONVERSION_TO_CPU_PASSES_H
#define TRITON_CONVERSION_TO_CPU_PASSES_H

#include "triton/Conversion/TritonToTritonCPU/TritonToTritonCPUPass.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton/Conversion/TritonToTritonCPU/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
