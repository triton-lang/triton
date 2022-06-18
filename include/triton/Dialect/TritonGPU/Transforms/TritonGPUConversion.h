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
  TritonGPUTypeConverter(MLIRContext *context, int numThreads);
private:
  MLIRContext *context;
  int numThreads;
};

class TritonGPUConversionTarget : public ConversionTarget {
  TritonGPUTypeConverter &typeConverter;
public:
  explicit TritonGPUConversionTarget(MLIRContext &ctx, TritonGPUTypeConverter &typeConverter);

  /// update layouts & insert ConvertLayoutOp if necessary
  LogicalResult refineLayouts(ModuleOp mod, int numThreads);
};

} // namespace mlir

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_TRITONGPUCONVERSION_H_
