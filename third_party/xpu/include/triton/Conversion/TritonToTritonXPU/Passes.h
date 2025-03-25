//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#ifndef TT2TTX_CONVERSION_PASSES_H
#define TT2TTX_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"
#include "triton/Conversion/TritonToTritonXPU/TritonToTritonXPUPass.h"

namespace mlir {

namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton/Conversion/TritonToTritonXPU/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TT2TTX_CONVERSION_PASSES_H
