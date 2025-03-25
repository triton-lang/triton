//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#include "triton/Dialect/TritonXPU/Transforms/TritonXPUConversion.h"

#include "mlir/IR/MLIRContext.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include <cstdint>

using namespace mlir;

//
// TypeConverter
//
TritonXPUTypeConverter::TritonXPUTypeConverter(MLIRContext *context,
                                               uint32_t buffer_size,
                                               uint32_t core_num)
    : context(context), buffer_size(buffer_size), core_num(core_num) {

  addConversion([](Type type) { return type; });

  addConversion([this](RankedTensorType tensorType) -> RankedTensorType {
    if (tensorType.getEncoding())
      return tensorType;

    ArrayRef<int64_t> shape = tensorType.getShape();
    triton::xpu::ClusterLayoutAttr encoding =
        triton::xpu::getDefaultClusterEncoding(
            this->context, shape, this->buffer_size, this->core_num);
    return RankedTensorType::get(shape, tensorType.getElementType(), encoding);
  });

  // TODO[dyq]: check addConversion for triton::PointerType

  //
  // Materializations
  //
  // This will be called when (newArgType != origArgType)
  // This should create newArg, and map(origArg, newArg)
  addArgumentMaterialization([&](OpBuilder &builder,
                                 RankedTensorType tensorType, ValueRange inputs,
                                 Location loc) -> std::optional<Value> {
    llvm_unreachable("Argument rematerialization should not happen in Triton "
                     "-> TritonXPU conversion");
    return std::nullopt;
  });

  // If the origValue still has live user(s), use this to
  // convert origValue to newValue
  addSourceMaterialization([&](OpBuilder &builder, RankedTensorType tensorType,
                               ValueRange inputs,
                               Location loc) -> std::optional<Value> {
    llvm_unreachable("Source rematerialization should not happen in Triton -> "
                     "TritonXPU Conversion");
    return std::nullopt;
  });

  // This will be called when (desiredType != newOperandType)
  // where, desiredType = typeConverter->convertType(origType)
  // NOTE: only for remapped values.
  addTargetMaterialization([&](OpBuilder &builder, RankedTensorType tensorType,
                               ValueRange inputs, Location loc) {
    auto cast =
        builder.create<triton::xpu::ConvertLayoutOp>(loc, tensorType, inputs);
    return std::optional<Value>(cast.getResult());
  });
}

//
// TritonXPUConversion
//
TritonXPUConversionTarget::TritonXPUConversionTarget(
    MLIRContext &context, TritonXPUTypeConverter &typeConverter)
    : ConversionTarget(context) {

  addLegalDialect<triton::xpu::TritonXPUDialect>();

  // Some ops from SCF are illegal
  // TODO[dyq]: addIllegalOp necessary?
  //   addIllegalOp<scf::ExecuteRegionOp, scf::ParallelOp, scf::ReduceOp,
  //                scf::ReduceReturnOp>();

  addDynamicallyLegalDialect<arith::ArithDialect, math::MathDialect,
                             triton::TritonDialect, cf::ControlFlowDialect,
                             scf::SCFDialect>([&](Operation *op) {
    bool hasLegalRegions = true;
    for (auto &region : op->getRegions()) {
      hasLegalRegions = hasLegalRegions && typeConverter.isLegal(&region);
    }
    if (hasLegalRegions && typeConverter.isLegal(op)) {
      return true;
    }
    return false;
  });

  // TODO[dyq]: XPUSDNN-CHECK check addDynamicallyLegalDialect for triton::DotOp
}
