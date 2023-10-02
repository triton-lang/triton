#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_PASS_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_PASS_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Target/PTX/TmaMetadata.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

enum Target { NVVM, ROCDL, Default = NVVM };

#define GEN_PASS_DECL
#include "triton/Conversion/TritonGPUToLLVM/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonGPUToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>>
<<<<<<< HEAD
#ifdef USE_ROCM
createConvertTritonGPUToLLVMPass(int computeCapability = 80,
                                 bool isROCM = true);
#else
createConvertTritonGPUToLLVMPass(int computeCapability = 80,
                                 bool isROCM = false);
#endif
=======
createConvertTritonGPUToLLVMPass(const ConvertTritonGPUToLLVMOptions &options);

>>>>>>> 36fc54b6f28168d3644808bfe299f1ba06a36272
} // namespace triton

} // namespace mlir

#endif
