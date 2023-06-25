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
  TritonGPUTypeConverter(MLIRContext *context, int numWarps,
                         int threadsPerWarp);
  int getNumWarps() const { return numWarps; }
  int getThreadsPerWarp() const { return threadsPerWarp; }

private:
  MLIRContext *context;
  int numWarps;
  int threadsPerWarp;
};

class TritonGPUConversionTarget : public ConversionTarget {

public:
  explicit TritonGPUConversionTarget(MLIRContext &ctx,
                                     TritonGPUTypeConverter &typeConverter);
};

} // namespace mlir

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_TRITONGPUCONVERSION_H_
