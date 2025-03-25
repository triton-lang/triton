//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Defines utilities to use while converting to the TritonXPU dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DIALECT_TRITONXPU_TRANSFORMS_TRITONGPUCONVERSION_H_
#define TRITON_DIALECT_TRITONXPU_TRANSFORMS_TRITONGPUCONVERSION_H_

#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h" // TypeConverter

namespace mlir {

class TritonXPUTypeConverter : public TypeConverter {
public:
  TritonXPUTypeConverter(MLIRContext *context, uint32_t buffer_size,
                         uint32_t core_num);
  uint32_t getBufferSize() const { return buffer_size; }
  uint32_t getCoreNum() const { return core_num; }

private:
  MLIRContext *context;
  uint32_t buffer_size;
  uint32_t core_num;
};

class TritonXPUConversionTarget : public ConversionTarget {
public:
  explicit TritonXPUConversionTarget(MLIRContext &ctx,
                                     TritonXPUTypeConverter &typeConverter);
};

} // namespace mlir

#endif // TRITON_DIALECT_TRITONXPU_TRANSFORMS_TRITONGPUCONVERSION_H_
