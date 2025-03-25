//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#ifndef TRITON_CONVERSION_TT2TTX_TRITONTOTRITONXPUPASS_H
#define TRITON_CONVERSION_TT2TTX_TRITONTOTRITONXPUPASS_H

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

constexpr static char AttrXPUTargetName[] = "triton_xpu.target";

// Create the pass with buffer_size passed from cl::opt.
std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonToTritonXPUPass();

// Create the pass with buffer_size set explicitly.
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonToTritonXPUPass(uint32_t xpu_arch, uint32_t buffer_size,
                                   uint32_t core_num);

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TT2TTX_TRITONTOTRITONXPUPASS_H
