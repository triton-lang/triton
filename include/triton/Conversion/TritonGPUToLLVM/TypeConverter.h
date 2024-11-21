#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_TYPECONVERTER_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_TYPECONVERTER_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"

using namespace mlir;
using namespace mlir::triton;

class TritonGPUToLLVMTypeConverter : public LLVMTypeConverter {
public:
  using TypeConverter::convertType;

  TritonGPUToLLVMTypeConverter(MLIRContext *ctx, LowerToLLVMOptions &option,
                               const TargetInfoBase &targetInfo,
                               const DataLayoutAnalysis *analysis = nullptr);

  Type getElementTypeForStruct(triton::gpu::TensorOrMemDesc type);
  Type convertTritonPointerType(triton::PointerType type);
  Type convertTritonTensorType(RankedTensorType type,
                               const TargetInfoBase &targetInfo);
  Type convertMemDescType(triton::gpu::MemDescType type,
                          const TargetInfoBase &targetInfo);
  Type convertAsyncToken(triton::gpu::AsyncTokenType type);
};

#endif
