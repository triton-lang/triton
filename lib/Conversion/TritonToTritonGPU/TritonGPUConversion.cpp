#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"

#include <algorithm>
#include <numeric>

#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
using namespace mlir::triton::gpu;

//
// TypeConverter
//
TritonGPUTypeConverter::TritonGPUTypeConverter(MLIRContext *context,
                                               int numWarps, int threadsPerWarp,
                                               int numCTAs,
                                               bool enableSourceRemat)
    : context(context), numWarps(numWarps), threadsPerWarp(threadsPerWarp),
      numCTAs(numCTAs) {
  addConversion([](Type type) { return type; });

  // Add encoding for tensor
  addConversion([this](RankedTensorType tensorType) -> RankedTensorType {
    // types with encoding are already in the right format
    // TODO: check for layout encodings more specifically
    if (tensorType.getEncoding())
      return tensorType;
    ArrayRef<int64_t> shape = tensorType.getShape();
    triton::gpu::BlockedEncodingAttr encoding =
        getDefaultBlockedEncoding(this->context, shape, this->numWarps,
                                  this->threadsPerWarp, this->numCTAs);
    return RankedTensorType::get(shape, tensorType.getElementType(), encoding);
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

  // If the origValue still has live user(s), use this to
  // convert origValue to newValue
  if (enableSourceRemat) {
    addSourceMaterialization([](OpBuilder &builder, RankedTensorType tensorType,
                                ValueRange inputs, Location loc) -> Value {
      return builder.create<UnrealizedConversionCastOp>(loc, tensorType, inputs)
          .getResult(0);
    });
  }

  // This will be called when (desiredType != newOperandType)
  // where, desiredType = typeConverter->convertType(origType)
  // NOTE: only for remapped values.
  addTargetMaterialization([](OpBuilder &builder, RankedTensorType tensorType,
                              ValueRange inputs, Location loc) {
    auto cast =
        builder.create<triton::gpu::ConvertLayoutOp>(loc, tensorType, inputs);
    return cast.getResult();
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

  addDynamicallyLegalDialect<arith::ArithDialect, math::MathDialect,
                             triton::TritonDialect, cf::ControlFlowDialect,
                             scf::SCFDialect, ub::UBDialect>(
      [&](Operation *op) { return isDynamicallyLegal(op, typeConverter); });

  // We have requirements for the data layouts
  addDynamicallyLegalOp<triton::DotOp>([](triton::DotOp dotOp) -> bool {
    Attribute aEncoding =
        cast<RankedTensorType>(dotOp.getA().getType()).getEncoding();
    Attribute bEncoding =
        cast<RankedTensorType>(dotOp.getB().getType()).getEncoding();
    if (aEncoding && isa<triton::gpu::DotOperandEncodingAttr>(aEncoding) &&
        bEncoding && isa<triton::gpu::DotOperandEncodingAttr>(bEncoding))
      return true;
    return false;
  });
}

bool TritonGPUConversionTarget::isDynamicallyLegal(
    Operation *op, const TypeConverter &typeConverter) {
  bool hasLegalRegions = true;
  for (auto &region : op->getRegions()) {
    hasLegalRegions = hasLegalRegions && typeConverter.isLegal(&region);
  }
  if (hasLegalRegions && typeConverter.isLegal(op)) {
    return true;
  }
  return false;
}

// This function returns the layout to use for gather/scatter indices. The
// `gather4` and `scatter4` TMA instructions require 4 consecutive indices.
// Thus, threads issuing these instructions must have all 4 index elements
// available.
static RankedTensorType getNewIndicesType(RankedTensorType type,
                                          unsigned numThreads,
                                          unsigned numWarps) {
  assert(type.getRank() == 1);
  auto enc = cast<DistributedEncodingTrait>(type.getEncoding());

  // Technically any layout where we have a pack of 4 neighbouring elements plus
  // broadcasted over the warp dimension is okay but for now we just pick a
  // layout.
  std::array<unsigned, 2> sizePerThread{1, 4};
  std::array<unsigned, 2> threadsPerWarp = {numThreads, 1};
  std::array<unsigned, 2> order = {1, 0};
  std::array<unsigned, 2> warpsPerCta = {1, numWarps};

  MLIRContext *ctx = type.getContext();
  auto ctaLayout = CTALayoutAttr::getDefault(ctx, /*rank=*/2);
  auto parentEncoding = BlockedEncodingAttr::get(
      ctx, sizePerThread, threadsPerWarp, warpsPerCta, order, ctaLayout);
  auto newEncoding = SliceEncodingAttr::get(ctx, /*dim=*/0, parentEncoding);
  if (enc == newEncoding)
    return {};

  return RankedTensorType::get(type.getShape(), type.getElementType(),
                               newEncoding);
}

// Function for converting any gather or scatter op that requires a specific
// index layout. This also handles converting result types if there are any.
static LogicalResult convertGatherScatterIndices(Operation *op,
                                                 OpOperand &indices,
                                                 ConversionPatternRewriter &b) {
  auto type = cast<RankedTensorType>(indices.get().getType());
  RankedTensorType newType =
      getNewIndicesType(type, lookupThreadsPerWarp(b), lookupNumWarps(op));
  if (!newType)
    return failure();
  Value index = b.create<ConvertLayoutOp>(op->getLoc(), newType, indices.get());
  indices.set(index);
  return success();
}

LogicalResult impl::convertGatherScatterOp(
    Operation *op, ValueRange operands, OpOperand &xOffsetsMutable,
    const TypeConverter &typeConverter, ConversionPatternRewriter &rewriter) {
  LogicalResult result = success();
  rewriter.modifyOpInPlace(op, [&] {
    for (auto [operand, value] : llvm::zip(op->getOpOperands(), operands))
      operand.set(value);
    for (OpResult result : op->getOpResults())
      result.setType(typeConverter.convertType(result.getType()));
    result = convertGatherScatterIndices(op, xOffsetsMutable, rewriter);
  });
  return result;
}
