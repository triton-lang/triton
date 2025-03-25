//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#ifndef TRITON_CONVERSION_TRITONXPU_TO_LLVM_TYPECONVERTER_H
#define TRITON_CONVERSION_TRITONXPU_TO_LLVM_TYPECONVERTER_H

// clang-format off
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
// clang-format on

using namespace mlir;
using namespace mlir::triton;

class TritonXPUToLLVMTypeConverter : public LLVMTypeConverter {
public:
  using TypeConverter::convertType;

  TritonXPUToLLVMTypeConverter(MLIRContext *ctx, LowerToLLVMOptions &option,
                               const DataLayoutAnalysis *analysis = nullptr);

  Type getElementTypeForStruct(TensorOrMemDesc type);
  Type convertTritonPointerType(triton::PointerType type);
  Type convertTritonTensorType(RankedTensorType type);
};

#endif // TRITON_CONVERSION_TRITONXPU_TO_LLVM_TYPECONVERTER_H
