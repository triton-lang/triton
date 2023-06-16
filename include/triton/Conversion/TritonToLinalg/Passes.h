//===----------------------------------------------------------------------===//
//
// Copyright (c) Triton Project Contributors.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_TO_LINALG_CONVERSION_PASSES_H
#define TRITON_TO_LINALG_CONVERSION_PASSES_H

#include "triton/Conversion/TritonToLinalg/TritonToLinalg.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton/Conversion/TritonToLinalg/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
