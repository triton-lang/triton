#ifndef TRITON_CONVERSION_PASSDETAIL_H
#define TRITON_CONVERSION_PASSDETAIL_H

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {
namespace triton {

#define GEN_PASS_CLASSES
#include "triton/Conversion/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
