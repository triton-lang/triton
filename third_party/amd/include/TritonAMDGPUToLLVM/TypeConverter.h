#ifndef TRITON_CONVERSION_TRITONAMDGPU_TO_LLVM_TYPECONVERTER_H
#define TRITON_CONVERSION_TRITONAMDGPU_TO_LLVM_TYPECONVERTER_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"

using namespace mlir;
using namespace mlir::triton;

class TritonAMDGPUToLLVMTypeConverter : public TritonGPUToLLVMTypeConverter {
public:
  TritonAMDGPUToLLVMTypeConverter(MLIRContext *ctx,
                                  const LowerToLLVMOptions &options,
                                  const TargetInfoBase &targetInfo,
                                  const DataLayoutAnalysis *analysis = nullptr)
      : TritonGPUToLLVMTypeConverter(ctx, options, targetInfo, analysis) {
    addConversion([&](TensorDescType type) -> std::optional<Type> {
      return convertTensorDescType(type);
    });
  }

  Type convertTensorDescType(triton::TensorDescType type) {
    auto ctx = type.getContext();
    auto blockType = type.getBlockType();
    auto shape = blockType.getShape();

    // Determine the number of dwords based on tensor dimensions
    // 2D tensors: group0 (4) + group1 (8) = 12 dwords
    // 3D-5D tensors: group0 (4) + group1 (8) + group2 (4) + group3 (4) = 20
    // dwords
    int numDwords = (shape.size() > 2) ? (4 + 8 + 4 + 4) : (4 + 8);

    auto types = SmallVector<Type>(numDwords, IntegerType::get(ctx, 32));
    return LLVM::LLVMStructType::getLiteral(ctx, types);
  }
};

#endif
