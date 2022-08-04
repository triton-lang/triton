#ifndef TRITON_CONVERSION_TRITONGPUTOLLVM_TRITONGPUTOLLVMPASS_H_
#define TRITON_CONVERSION_TRITONGPUTOLLVM_TRITONGPUTOLLVMPASS_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

class TritonLLVMConversionTarget : public ConversionTarget {
  mlir::LLVMTypeConverter &typeConverter;

public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx,
                                      mlir::LLVMTypeConverter &typeConverter);
};

namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonGPUToLLVMPass();

} // namespace triton

} // namespace mlir

#endif
