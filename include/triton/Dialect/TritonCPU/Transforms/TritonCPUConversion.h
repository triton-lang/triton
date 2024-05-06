//===----------------------------------------------------------------------===//
//
// Defines utilities to use while converting to the TritonCPU dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DIALECT_TRITONCPU_TRANSFORMS_TRITONCPUCONVERSION_H_
#define TRITON_DIALECT_TRITONCPU_TRANSFORMS_TRITONCPUCONVERSION_H_

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

class TritonCPUTypeConverter : public TypeConverter {
public:
  TritonCPUTypeConverter(MLIRContext *context);

private:
  MLIRContext *context;
};

class TritonCPUConversionTarget : public ConversionTarget {

public:
  explicit TritonCPUConversionTarget(MLIRContext &ctx,
                                     TritonCPUTypeConverter &typeConverter);
};

} // namespace mlir

#endif // TRITON_DIALECT_TRITONCPU_TRANSFORMS_TRITONCPUCONVERSION_H_
