#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TDMUtility.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {
// Validates that the tensor descriptor's strides and shared layout are
// compatible with TDM. Requirements:
//  - The shared order must be [rank-1, rank-2, ..., 0].
//  - All stride-1 dimensions must be consecutive trailing dims.
LogicalResult validateStridesAndSharedOrder(triton::MakeTensorDescOp op,
                                            Attribute sharedEnc,
                                            ArrayRef<int64_t> shape,
                                            ValueRange strides) {
  int rank = shape.size();
  auto sharedOrder = triton::gpu::getOrder(
      cast<triton::gpu::SharedEncodingTrait>(sharedEnc), shape);
  SmallVector<unsigned> expectedOrder(llvm::reverse(llvm::seq<unsigned>(rank)));
  if (sharedOrder != ArrayRef(expectedOrder)) {
    return op.emitError() << "requires shared order [rank-1, rank-2, ..., 0]";
  }

  auto isStride1 = [](Value v) {
    return getConstantIntValue(getAsOpFoldResult(v)).value_or(0) == 1;
  };
  auto reversedStrides = llvm::reverse(strides);
  auto firstNonStride1 = llvm::find_if_not(reversedStrides, isStride1);
  if (firstNonStride1 == reversedStrides.begin())
    return op.emitError() << "last dimension must have stride 1";
  if (llvm::any_of(llvm::make_range(firstNonStride1, reversedStrides.end()),
                   isStride1))
    return op.emitError() << "requires all stride 1 dimensions to be "
                             "consecutive starting from the last dimension";

  return success();
}

struct MakeTensorDescOpConversion
    : public ConvertOpToLLVMPattern<triton::MakeTensorDescOp> {
  using ConvertOpToLLVMPattern<
      triton::MakeTensorDescOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::MakeTensorDescOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto basePtr = adaptor.getBase();
    auto tensorShape = adaptor.getShape();
    auto tensorStride = adaptor.getStrides();
    auto result = op.getResult();

    auto tensorDescTy = result.getType();
    auto sharedEnc = tensorDescTy.getSharedLayout();
    if (!sharedEnc) {
      return rewriter.notifyMatchFailure(
          op, "Descriptor has no shared memory layout assigned.");
    }
    unsigned padInterval = 0;
    unsigned padAmount = 0;
    if (auto padEnc = getPaddedEncoding(sharedEnc)) {
      if (padEnc.getIntervals().size() != 1 || padEnc.getPaddings().size() != 1)
        return rewriter.notifyMatchFailure(
            op, "NYI: Multiple interval-padding pairs in TDM.");
      padInterval = padEnc.getIntervals()[0];
      padAmount = padEnc.getPaddings()[0];
    }

    Type elementType =
        getTypeConverter()->convertType(tensorDescTy.getElementType());
    SmallVector<int64_t> blockShape = to_vector(tensorDescTy.getShape());
    auto shapePerCTA = triton::gpu::getShapePerCTA(sharedEnc, blockShape);

    if (failed(validateStridesAndSharedOrder(op, sharedEnc, shapePerCTA,
                                             tensorStride))) {
      return failure();
    }
    auto sharedOrder = triton::gpu::getOrder(
        cast<triton::gpu::SharedEncodingTrait>(sharedEnc), shapePerCTA);
    // Lower the tensor descriptor to a base TDM descriptor.  The final hardware
    // descriptor is completed at each TDM op site because pred, LDS address,
    // barrier, and tile_dim* are op-local.
    // Returns 2 (2D) or 4 (3D-5D) vector groups; scalarize into 12 or 20
    // i32 scalars to match the flat MLIR struct type from
    // `convertTensorDescType` (matches the host-side TDMDescriptor ABI).
    SmallVector<Value> groups = LLVM::AMD::createTDMDescriptor(
        rewriter, loc, getTypeConverter(), elementType, blockShape.size(),
        padInterval, padAmount, tensorShape, tensorStride, basePtr);
    SmallVector<Value> scalars =
        mlir::LLVM::AMD::scalarizeTDMDescriptor(rewriter, loc, groups);

    auto desc = packLLElements(loc, getTypeConverter(), scalars, rewriter,
                               tensorDescTy);

    rewriter.replaceOp(op, desc);
    return success();
  }
};

struct UpdateTensorDescriptorOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::UpdateTensorDescriptorOp> {
  using ConvertOpToLLVMPattern<
      triton::amdgpu::UpdateTensorDescriptorOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::amdgpu::UpdateTensorDescriptorOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto tensorDescTy = op.getDesc().getType();
    Type elementType =
        getTypeConverter()->convertType(tensorDescTy.getElementType());
    SmallVector<int64_t> blockShape = to_vector(tensorDescTy.getShape());

    if (blockShape.size() != 2) {
      return rewriter.notifyMatchFailure(
          op, "UpdateTensorDescriptorOp lowering currently supports 2D only");
    }

    // Unpack the input descriptor into vector groups (group0: <4 x i32>,
    // group1: <8 x i32> for 2D).
    SmallVector<Value> groups =
        mlir::LLVM::AMD::unpackTDMDescriptor(rewriter, loc, adaptor.getDesc());
    assert(groups.size() == 2 && "2D descriptor expects 2 vector groups");
    Value group0 = groups[0];
    Value group1 = groups[1];

    SmallVector<Value> addOffsets = llvm::to_vector(adaptor.getAddOffsets());
    SmallVector<Value> setBounds = llvm::to_vector(adaptor.getSetBounds());

    Value destPtr;
    if (op.getDest()) {
      auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
          loc, adaptor.getDest(), elementType, rewriter);
      destPtr = smemObj.getBase();
    }
    Value barrierPtr;
    if (op.getBarrier()) {
      auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
          loc, adaptor.getBarrier(),
          getTypeConverter()->convertType(
              op.getBarrier().getType().getElementType()),
          rewriter);
      barrierPtr = smemObj.getBase();
    }
    Value pred = adaptor.getPred();

    mlir::LLVM::AMD::updateTensorDescriptor(
        rewriter, loc, elementType, blockShape, group0, group1, addOffsets,
        setBounds, destPtr, pred, barrierPtr);

    // Re-pack the mutated groups back into the flat MLIR struct that
    // matches convertTensorDescType / the host-side TDMDescriptor ABI.
    SmallVector<Value> mutated = {group0, group1};
    SmallVector<Value> scalars =
        mlir::LLVM::AMD::scalarizeTDMDescriptor(rewriter, loc, mutated);
    Value newDesc = packLLElements(loc, getTypeConverter(), scalars, rewriter,
                                   tensorDescTy);

    rewriter.replaceOp(op, newDesc);
    return success();
  }
};
} // namespace

void mlir::triton::AMD::populateTensorPtrOpsToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<MakeTensorDescOpConversion>(typeConverter, benefit);
  patterns.add<UpdateTensorDescriptorOpConversion>(typeConverter, benefit);
  return;
}
