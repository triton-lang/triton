#ifndef TRITON_CONVERSION_TRITONGPUTOLLVM_TRITONGPUTOLLVMPASS_H_
#define TRITON_CONVERSION_TRITONGPUTOLLVM_TRITONGPUTOLLVMPASS_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx,
                                      mlir::LLVMTypeConverter &typeConverter);
};

class TritonLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMFunctionConversionTarget(
      MLIRContext &ctx, mlir::LLVMTypeConverter &typeConverter);
};

namespace triton {

// Names for identifying different NVVM annotations. It is used as attribute
// names in MLIR modules. Refer to
// https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#supported-properties for
// the full list.
struct NVVMMetadataField {
  static constexpr char MaxNTid[] = "nvvm.maxntid";
  static constexpr char Kernel[] = "nvvm.kernel";
};

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonGPUToLLVMPass();

} // namespace triton

} // namespace mlir

#endif
