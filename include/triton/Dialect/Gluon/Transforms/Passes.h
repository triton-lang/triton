#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include <memory>

namespace mlir::triton::gluon {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "triton/Dialect/Gluon/Transforms/Passes.h.inc"

} // namespace mlir::triton::gluon
