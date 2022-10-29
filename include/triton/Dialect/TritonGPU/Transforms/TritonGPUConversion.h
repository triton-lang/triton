//===----------------------------------------------------------------------===//
//
// Defines utilities to use while converting to the TritonGPU dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_TRITONGPUCONVERSION_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_TRITONGPUCONVERSION_H_

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

class TritonGPUTypeConverter : public TypeConverter {
public:
  TritonGPUTypeConverter(MLIRContext *context, int numWarps);

private:
  MLIRContext *context;
  int numWarps;
};

class TritonGPUConversionTarget : public ConversionTarget {

public:
  explicit TritonGPUConversionTarget(MLIRContext &ctx,
                                     TritonGPUTypeConverter &typeConverter);
};

} // namespace mlir

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_TRITONGPUCONVERSION_H_
