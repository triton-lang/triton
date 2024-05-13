#ifndef TRITONCPU_CONVERSION_TRITONCPUTOLLVM_TYPECONVERTER_H
#define TRITONCPU_CONVERSION_TRITONCPUTOLLVM_TYPECONVERTER_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Dialect/TritonCPU/IR/Types.h"

using namespace mlir;
using namespace mlir::triton;

class TritonCPUToLLVMTypeConverter : public LLVMTypeConverter {
public:
  using TypeConverter::convertType;

  TritonCPUToLLVMTypeConverter(MLIRContext *ctx, LowerToLLVMOptions &option,
                               const DataLayoutAnalysis *analysis = nullptr);

  Type convertTritonPointerType(triton::PointerType type);
};

#endif
