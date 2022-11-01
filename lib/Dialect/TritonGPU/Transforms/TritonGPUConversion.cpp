#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <algorithm>
#include <numeric>

using namespace mlir;
using namespace mlir::triton::gpu;

//
// TypeConverter
//
TritonGPUTypeConverter::TritonGPUTypeConverter(MLIRContext *context,
                                               int numWarps)
    : context(context), numWarps(numWarps) {
  // TODO: how does MLIR pick the right conversion?
  addConversion([](Type type) { return type; });
  addConversion([this](RankedTensorType tensorType) -> RankedTensorType {
    // types with encoding are already in the right format
    // TODO: check for layout encodings specifically
    if (tensorType.getEncoding())
      return tensorType;
    // pessimistic values for attributes:
    //   - 1 element per thread
    //   - order = arange(rank)
    ArrayRef<int64_t> shape = tensorType.getShape();
    int rank = shape.size();
    llvm::SmallVector<unsigned> order(rank);
    std::iota(order.begin(), order.end(), 0);
    llvm::SmallVector<unsigned> sizePerThread(rank, 1);
    Attribute encoding = triton::gpu::BlockedEncodingAttr::get(
        this->context, shape, sizePerThread, order, this->numWarps);
    return RankedTensorType::get(shape, tensorType.getElementType(), encoding);
  });

  //
  // materailizations
  //
  // This will be called when (newArgType != origArgType)
  // This will create newArg, and map(origArg, newArg)
  addArgumentMaterialization([&](OpBuilder &builder,
                                 RankedTensorType tensorType, ValueRange inputs,
                                 Location loc) {
    llvm_unreachable("Argument rematerialization not implemented");
    return llvm::None;
  });

  // If the origValue still has live user(s), use this to
  // convert origValue to newValue
  addSourceMaterialization([&](OpBuilder &builder, RankedTensorType tensorType,
                               ValueRange inputs, Location loc) {
    llvm_unreachable("Source rematerialization not implemented");
    return llvm::None;
  });

  // This will be called when (desiredType != newOperandType)
  // where, desiredType = typeConverter->convertType(origType)
  // NOTE: only for remapped values.
  addTargetMaterialization([&](OpBuilder &builder, RankedTensorType tensorType,
                               ValueRange inputs, Location loc) {
    auto cast =
        builder.create<triton::gpu::ConvertLayoutOp>(loc, tensorType, inputs);
    return Optional<Value>(cast.getResult());
    // return Optional<Value>(cast.getResult(0));
    // llvm_unreachable("Not implemented");
    // return llvm::None;
  });
}

//
// TritonGPUConversion
//
TritonGPUConversionTarget::TritonGPUConversionTarget(
    MLIRContext &context, TritonGPUTypeConverter &typeConverter)
    : ConversionTarget(context) {
  // TODO: we should also verify ops of TritonGPUDialect
  addLegalDialect<triton::gpu::TritonGPUDialect>();

  // Some ops from SCF are illegal
  addIllegalOp<scf::ExecuteRegionOp, scf::ParallelOp, scf::ReduceOp,
               scf::ReduceReturnOp>();

  addDynamicallyLegalDialect<arith::ArithmeticDialect, math::MathDialect,
                             triton::TritonDialect, StandardOpsDialect,
                             scf::SCFDialect>([&](Operation *op) {
    if (typeConverter.isLegal(op))
      return true;
    return false;
  });

  // We have requirements for the data layouts
  addDynamicallyLegalOp<triton::DotOp>([](triton::DotOp dotOp) -> bool {
    Attribute aEncoding =
        dotOp.a().getType().cast<RankedTensorType>().getEncoding();
    Attribute bEncoding =
        dotOp.b().getType().cast<RankedTensorType>().getEncoding();
    if (aEncoding && aEncoding.isa<triton::gpu::SharedEncodingAttr>() &&
        bEncoding && bEncoding.isa<triton::gpu::SharedEncodingAttr>())
      return true;
    return false;
  });
}
