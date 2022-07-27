#ifndef TRITON_CONVERSION_PASSES_H
#define TRITON_CONVERSION_PASSES_H

#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPU.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton/Conversion/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif