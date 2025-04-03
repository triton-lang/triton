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
struct CircularStoreOpConversion
    : public ConvertOpToLLVMPattern<
          mlir::triton::proton::gpu::CircularStoreOp> {
  explicit CircularStoreOpConversion(
      LLVMTypeConverter &typeConverter,
      const proton::gpu::TargetInfoBase &targetInfo, PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<
            mlir::triton::proton::gpu::CircularStoreOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(mlir::triton::proton::gpu::CircularStoreOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }

protected:
  const proton::gpu::TargetInfoBase &targetInfo;
};

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
    auto mod = op.getOperation()->getParentOfType<ModuleOp>();
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
    rewriter.replaceOp(op, {clock});
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
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    Location loc = op.getLoc();
    auto ctx = moduleOp.getContext();

    auto bufferTy =
        mlir::cast<triton::gpu::MemDescType>(op.getData().getType());
    auto rank = bufferTy.getRank();
    assert(rank > 0 && "Proton stack currently only supports 1-D shapes");

    const int bufferSizeInBytes =
        mlir::ShapedType::getNumElements(bufferTy.getShape()) *
        bufferTy.getElementType().getIntOrFloatBitWidth() / 8;

    auto bufferSizeVal = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getIntegerType(32),
        IntegerAttr::get(rewriter.getIntegerType(32), bufferSizeInBytes / 4));

    auto llvmPointerType = LLVM::LLVMPointerType::get(op->getContext());
    Type llvmInt32Type = IntegerType::get(op->getContext(), 32);
    Value arrayVal = rewriter.create<LLVM::AllocaOp>(
        loc, llvmPointerType, llvmInt32Type, bufferSizeVal);

    auto b = TritonLLVMOpBuilder(loc, rewriter);

    // TODO(crobeck): update if we ever support multi-rank stack alloc ops
    SmallVector<Type, 4> types = {ptr_ty(ctx)};
    SmallVector<Value, 4> elems = {arrayVal}; // i32 ptr - the start address

    auto structTy =
        LLVM::LLVMStructType::getLiteral(rewriter.getContext(), types);

    // return value
    Value llvmStruct = rewriter.create<LLVM::UndefOp>(loc, structTy);
    for (const auto &v : llvm::enumerate(elems)) {
      llvmStruct = b.insert_val(structTy, llvmStruct, v.value(), v.index());
    }
    rewriter.replaceOp(op, llvmStruct);
    return success();
  }

protected:
  const proton::gpu::TargetInfoBase &targetInfo;
};

} // namespace

void populateProtonGPUOpPatterns(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 const TargetInfoBase &targetInfo,
                                 PatternBenefit benefit) {
  populateGlobalScratchAllocOpToLLVMPattern(typeConverter, patterns, targetInfo,
                                            benefit);
  patterns.add<CircularStoreOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<InitBufferIndexOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<ReadCounterOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<FinalizeOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<StackAllocOpConversion>(typeConverter, targetInfo, benefit);
}

} // namespace proton::gpu
} // namespace mlir::triton
