#include "Conversion/ProtonGPUToLLVM/PatternProtonGPUOpToLLVM.h"
#include "Conversion/ProtonGPUToLLVM/GlobalScratchAllocOpToLLVM.h"
#include "Conversion/ProtonGPUToLLVM/TargetInfoBase.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::triton {
namespace proton::gpu {

namespace {

struct InitBufferIndexOpConversion
    : public ConvertOpToLLVMPattern<
          mlir::triton::proton::gpu::InitBufferIndexOp> {
  explicit InitBufferIndexOpConversion(
      LLVMTypeConverter &typeConverter,
      const proton::gpu::TargetInfoBase &targetInfo, PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<
            mlir::triton::proton::gpu::InitBufferIndexOp>(typeConverter,
                                                          benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(mlir::triton::proton::gpu::InitBufferIndexOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto ptrTy = ptr_ty(rewriter.getContext(), IndexPtrAddrSpace);
    auto indexPtr = rewriter.create<LLVM::AllocaOp>(
        loc, ptrTy, i32_ty, b.i32_val(1), /*alignment=*/0);
    b.store(b.i32_val(0), indexPtr);
    rewriter.replaceOp(op, indexPtr);
    return success();
  }

protected:
  const proton::gpu::TargetInfoBase &targetInfo;
};

struct ReadCounterOpConversion
    : public ConvertOpToLLVMPattern<mlir::triton::proton::gpu::ReadCounterOp> {
  explicit ReadCounterOpConversion(
      LLVMTypeConverter &typeConverter,
      const proton::gpu::TargetInfoBase &targetInfo, PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<mlir::triton::proton::gpu::ReadCounterOp>(
            typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(mlir::triton::proton::gpu::ReadCounterOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value clock = targetInfo.clock(rewriter, op.getLoc(), false);
    rewriter.replaceOp(op, clock);
    return success();
  }

protected:
  const proton::gpu::TargetInfoBase &targetInfo;
};

struct FinalizeOpConversion
    : public ConvertOpToLLVMPattern<mlir::triton::proton::gpu::FinalizeOp> {
  explicit FinalizeOpConversion(LLVMTypeConverter &typeConverter,
                                const proton::gpu::TargetInfoBase &targetInfo,
                                PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<mlir::triton::proton::gpu::FinalizeOp>(
            typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(mlir::triton::proton::gpu::FinalizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }

protected:
  const proton::gpu::TargetInfoBase &targetInfo;
};

struct StackAllocOpConversion
    : public ConvertOpToLLVMPattern<mlir::triton::proton::gpu::StackAllocOp> {
  explicit StackAllocOpConversion(LLVMTypeConverter &typeConverter,
                                  const proton::gpu::TargetInfoBase &targetInfo,
                                  PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<mlir::triton::proton::gpu::StackAllocOp>(
            typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(mlir::triton::proton::gpu::StackAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }

protected:
  const proton::gpu::TargetInfoBase &targetInfo;
};

struct SegmentBaseOpConversion
    : public ConvertOpToLLVMPattern<mlir::triton::proton::gpu::SegmentBaseOp> {
  explicit SegmentBaseOpConversion(
      LLVMTypeConverter &typeConverter,
      const proton::gpu::TargetInfoBase &targetInfo, PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<mlir::triton::proton::gpu::SegmentBaseOp>(
            typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(mlir::triton::proton::gpu::SegmentBaseOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto mod = op.getOperation()->getParentOfType<ModuleOp>();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    int numWarps = mlir::triton::gpu::lookupNumWarps(mod);
    auto granularity = op.getGranularity();
    auto warpIdsAttr = op.getWarpIdsAttr();

    llvm::ArrayRef<int> selectedWarpIds;
    llvm::SmallVector<int, 16> allWarpIds(numWarps, 0);
    for (int i = 0; i < numWarps; ++i)
      allWarpIds[i] = i;

    if (granularity == proton::gpu::Granularity::ALL) {
      selectedWarpIds = allWarpIds;
    } else if (granularity == proton::gpu::Granularity::SELECT) {
      assert(warpIdsAttr && "warp ids must be specified");
      selectedWarpIds = warpIdsAttr.asArrayRef();
    } else {
      mlir::emitError(loc, "granularity must be all or select");
      return failure();
    }

    Value curThreadId = getThreadId(rewriter, loc);
    Value threadsPerWarp =
        b.i32_val(triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod));
    Value curWarpId = b.udiv(curThreadId, threadsPerWarp);

    auto bufferTy =
        mlir::cast<triton::gpu::MemDescType>(op.getData().getType());
    const int bufferSizeInBytes =
        mlir::ShapedType::getNumElements(bufferTy.getShape()) *
        bufferTy.getElementType().getIntOrFloatBitWidth() / 8;
    const int segmentByteSize = bufferSizeInBytes / selectedWarpIds.size();
    int warpSegmentBase = 0;

    Value segmentBase = b.i32_val(-1);
    for (int warpId : selectedWarpIds) {
      segmentBase = b.select(b.icmp_eq(curWarpId, b.i32_val(warpId)),
                             b.i32_val(warpSegmentBase), segmentBase);
      warpSegmentBase += segmentByteSize;
    }

    rewriter.replaceOp(op, segmentBase);
    return success();
  }

protected:
  const proton::gpu::TargetInfoBase &targetInfo;
};

Type convertProtonMemDescType(triton::gpu::MemDescType type,
                              const TargetInfoBase &targetInfo) {
  auto ctx = type.getContext();
  // base ptr
  auto ptrType = LLVM::LLVMPointerType::get(
      ctx, targetInfo.getAddressSpace(type.getMemorySpace()));

  SmallVector<Type, 4> types;
  types.push_back(ptrType);
  auto rank = type.getRank();
  // offsets
  for (auto i = 0; i < rank; i++) {
    types.push_back(IntegerType::get(ctx, 32));
  }

  return LLVM::LLVMStructType::getLiteral(ctx, types);
}

} // namespace

void populateProtonGPUOpPatterns(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 const TargetInfoBase &targetInfo,
                                 PatternBenefit benefit) {
  populateGlobalScratchAllocOpToLLVMPattern(typeConverter, patterns, targetInfo,
                                            benefit);
  patterns.add<InitBufferIndexOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<ReadCounterOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<FinalizeOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<SegmentBaseOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<StackAllocOpConversion>(typeConverter, targetInfo, benefit);
}

void populateTypeConversions(LLVMTypeConverter &typeConverter,
                             const TargetInfoBase &targetInfo) {
  typeConverter.addConversion(
      [&](triton::gpu::MemDescType type) -> std::optional<Type> {
        return convertProtonMemDescType(type, targetInfo);
      });
}

} // namespace proton::gpu
} // namespace mlir::triton
