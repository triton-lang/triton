//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#ifndef TRITON_DIALECT_TRITONXPU_TRANSFORMS_PASSES_H_
#define TRITON_DIALECT_TRITONXPU_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "triton/Analysis/UtilityXPU.h"          // helper
#include "triton/Dialect/TritonXPU/IR/Dialect.h" // dependentDialects
#include "llvm/ADT/TypeSwitch.h"                 // TypeSwitch
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h" // llvm_unreachable

namespace mlir {
namespace triton {
namespace xpu {

// Generate the pass class declarations.
#define GEN_PASS_DECL
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

} // namespace xpu
} // namespace triton
} // namespace mlir
#endif
