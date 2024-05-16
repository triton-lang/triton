#include "triton/Dialect/TritonCPU/Transforms/TritonCPUConversion.h"

#include "mlir/IR/IRMapping.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"
#include <algorithm>
#include <numeric>

using namespace mlir;
using namespace mlir::triton::cpu;

//
// TypeConverter
//
TritonCPUTypeConverter::TritonCPUTypeConverter(MLIRContext *context)
    : context(context) {
  addConversion([](Type type) { return type; });

  // Add encoding for tensor
  addConversion([this](RankedTensorType tensorType) -> RankedTensorType {
    // TODO:
    return tensorType;
  });

  // Add encoding for tensor pointer
  addConversion([this](triton::PointerType ptrType) -> triton::PointerType {
    // Check whether tensor pointer `tt.ptr<tensor<>>`
    auto pointeeTensorType =
        dyn_cast<RankedTensorType>(ptrType.getPointeeType());
    if (pointeeTensorType == nullptr)
      return ptrType;

    // Add layout into the tensor
    auto convertedTensorType = convertType(pointeeTensorType);
    return triton::PointerType::get(convertedTensorType,
                                    ptrType.getAddressSpace());
  });

  //
  // Materializations
  //
  // This will be called when (newArgType != origArgType)
  // This will create newArg, and map(origArg, newArg)
  addArgumentMaterialization([&](OpBuilder &builder,
                                 RankedTensorType tensorType, ValueRange inputs,
                                 Location loc) -> std::optional<Value> {
    llvm_unreachable("Argument rematerialization should not happen in Triton "
                     "-> TritonCPU conversion");
    return std::nullopt;
  });

  // If the origValue still has live user(s), use this to
  // convert origValue to newValue
  addSourceMaterialization([&](OpBuilder &builder, RankedTensorType tensorType,
                               ValueRange inputs,
                               Location loc) -> std::optional<Value> {
    llvm_unreachable("Source rematerialization should not happen in Triton -> "
                     "TritonCPU Conversion");
    return std::nullopt;
  });

  // This will be called when (desiredType != newOperandType)
  // where, desiredType = typeConverter->convertType(origType)
  // NOTE: only for remapped values.
  addTargetMaterialization([&](OpBuilder &builder, RankedTensorType tensorType,
                               ValueRange inputs, Location loc) {
    llvm_unreachable("Source rematerialization should not happen in Triton -> "
                     "TritonCPU Conversion");
    return std::nullopt;
  });
}

//
// TritonCPUConversion
//
TritonCPUConversionTarget::TritonCPUConversionTarget(
    MLIRContext &context, TritonCPUTypeConverter &typeConverter)
    : ConversionTarget(context) {
  // TODO: we should also verify ops of TritonCPUDialect
  addLegalDialect<triton::cpu::TritonCPUDialect>();

  // Some ops from SCF are illegal
  addIllegalOp<scf::ExecuteRegionOp, scf::ParallelOp, scf::ReduceOp,
               scf::ReduceReturnOp>();

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

  // We have requirements for the data layouts
  addDynamicallyLegalOp<triton::DotOp>([](triton::DotOp dotOp) -> bool {
    Attribute aEncoding =
        cast<RankedTensorType>(dotOp.getA().getType()).getEncoding();
    Attribute bEncoding =
        cast<RankedTensorType>(dotOp.getB().getType()).getEncoding();
    // TODO:
    return false;
  });
}
