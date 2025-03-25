//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#ifndef TTX2LLVM_CONVERSION_TRITONXPUTOLLVM_PASSES_H
#define TTX2LLVM_CONVERSION_TRITONXPUTOLLVM_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

#define GEN_PASS_DECL
#include "triton/Conversion/TritonXPUToLLVM/Passes.h.inc"

namespace xpu {

// TODO[dyq]: can be used ?
// std::unique_ptr<OperationPass<ModuleOp>>
// createDecomposeUnsupportedConversionsPass(uint32_t xpu_arch);

} // namespace xpu

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonXPUToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonXPUToLLVMPass(uint32_t xpu_arch, uint32_t buffer_size);

#define GEN_PASS_REGISTRATION
#include "triton/Conversion/TritonXPUToLLVM/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TTX2LLVM_CONVERSION_TRITONXPUTOLLVM_PASSES_H
