#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace {

// TODO: unify with ConcurrencySanitizer.cpp
constexpr static int8_t WRITE_BIT = 1 << 0;
constexpr static int8_t READ_BIT = 1 << 1;

namespace tt = mlir::triton;
namespace ttg = tt::gpu;

BlockedEncodingAttr getBlockedEncoding(ModuleOp module, unsigned int size) {
  MLIRContext *ctx = module.getContext();
  unsigned int warps =
      mlir::cast<mlir::IntegerAttr>(module->getAttr("ttg.num-warps")).getInt();
  auto ctaLayout = CTALayoutAttr::getDefault(ctx, /*rank=*/1);
  return BlockedEncodingAttr::get(ctx,
                                  /*sizePerThread=*/{size},
                                  /*threadsPerWarp=*/{32},
                                  /*warpsPerCTA=*/{warps},
                                  /*order=*/{0}, ctaLayout);
}

Value createConstIntTensor(OpBuilder &builder, Location loc, ModuleOp module,
                           int val, Type elType, int64_t size) {
  assert(llvm::isPowerOf2_64(size) && "Expected power of 2");
  auto tensorType =
      RankedTensorType::get({size}, elType, getBlockedEncoding(module, size));
  auto denseAttr = DenseElementsAttr::get(
      tensorType, APInt(elType.getIntOrFloatBitWidth(), val));
  return builder.create<arith::ConstantOp>(loc, tensorType, denseAttr);
}

Value createConstIntTensor(OpBuilder &builder, Location loc, int val,
                           RankedTensorType tensorType) {
  auto denseAttr = DenseElementsAttr::get(
      tensorType,
      APInt(tensorType.getElementType().getIntOrFloatBitWidth(), val));
  return builder.create<arith::ConstantOp>(loc, tensorType, denseAttr);
}

struct SharedBufferPointersOpConversion
    : public ConvertOpToLLVMPattern<
          triton::gpu::ExperimentalSharedBufferPointersOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::ExperimentalSharedBufferPointersOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &b) const override {
    auto loc = op.getLoc();
    auto *ctx = b.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto values = adaptor.getOffsets();
    auto shMemBufs = createInitializedIntTensor(b, loc, module, values);
    auto base =
        getSharedMemoryBase(b, op->getParentOfType<FunctionOpInterface>());
    shMemBufs = b.create<arith::AddIOp>(
        loc, shMemBufs,
        b.create<triton::SplatOp>(loc, shMemBufs.getType(), base));
    b.replaceOp(op, shMemBufs);
    return success();
  }

  Value createInitializedIntTensor(OpBuilder &builder, Location loc,
                                   ModuleOp module,
                                   ArrayRef<int32_t> values) const {
    int64_t size = values.size();
    assert(llvm::isPowerOf2_64(size) && "Expected power of 2");
    auto tensorType = RankedTensorType::get({size}, builder.getIntegerType(64),
                                            getBlockedEncoding(module, size));
    SmallVector<APInt> apInts = llvm::to_vector(
        llvm::map_range(values, [](int32_t v) { return APInt(64, v); }));
    auto denseAttr = DenseElementsAttr::get(tensorType, apInts);
    return builder.create<arith::ConstantOp>(loc, tensorType, denseAttr);
  }

  Value getSharedMemoryBase(ConversionPatternRewriter &rewriter,
                            FunctionOpInterface func) const {
    Location loc = func.getLoc();
    Value base = LLVM::getStackPointer(rewriter, func);
    // Bitcast to i64
    auto i64Ty = rewriter.getIntegerType(64);
    TritonLLVMOpBuilder b(loc, rewriter);
    base = b.ptrtoint(i64Ty, base);
    return base;
  }
};

Value createCmpIntTensorScalar(OpBuilder &builder, Location loc, Value tensor,
                               Value scalar) {
  auto tensorTy = cast<RankedTensorType>(tensor.getType());
  auto scalarTy = scalar.getType();
  auto elemTy = tensorTy.getElementType();
  assert(scalarTy == elemTy &&
         "Expected scalar to be of the same type as the tensor elements");
  auto splat = builder.create<triton::SplatOp>(loc, tensorTy, scalar);
  auto cmp = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                           tensor, splat);
  return cmp;
}

Value createMemDescToI64(ConversionPatternRewriter &rewriter, Location loc,
                         const LLVMTypeConverter *typeConverter,
                         ttg::MemDescType memDescTy, Value sharedMemStruct) {
  Type srcElemTy = typeConverter->convertType(memDescTy.getElementType());
  auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, sharedMemStruct,
                                                       srcElemTy, rewriter);
  auto offsets = smemObj.getOffsets();
  auto strides = smemObj.getStrides(memDescTy, loc, rewriter);
  Value offset = dot(rewriter, loc, offsets, strides);
  TritonLLVMOpBuilder b(loc, rewriter);
  auto i64Ty = rewriter.getIntegerType(64);
  offset = b.zext(i64Ty, offset);
  return b.add(offset, b.ptrtoint(i64Ty, smemObj.getBase()));
}

struct CheckAsyncWriteWithMbarSharedOpConversion
    : public ConvertOpToLLVMPattern<
          triton::gpu::ExperimentalCheckAsyncWriteWithMbarSharedOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::ExperimentalCheckAsyncWriteWithMbarSharedOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &b) const override {

    Location loc = op.getLoc();
    b.setInsertionPoint(op);
    RankedTensorType barriersTy =
        cast<RankedTensorType>(op.getBarriers().getType());
    RankedTensorType statesTy =
        cast<RankedTensorType>(op.getStates().getType());
    Value zero_64b = createConstIntTensor(b, loc, 0, barriersTy);
    Value zero_8b = createConstIntTensor(b, loc, 0, statesTy);
    Value mbar = createMemDescToI64(b, loc, getTypeConverter(),
                                    op.getMbar().getType(), adaptor.getMbar());
    Value buffer =
        createMemDescToI64(b, loc, getTypeConverter(), op.getBuffer().getType(),
                           adaptor.getBuffer());
    Value mbarSplat = b.create<triton::SplatOp>(loc, barriersTy, mbar);
    Value buffers = op.getBuffers();
    Value states = op.getStates();
    Value barriers = op.getBarriers();

    // 1. Check if the buffer has outstanding accesses
    Value currBuf = createCmpIntTensorScalar(b, loc, buffers, buffer);
    Value rwSplat =
        createConstIntTensor(b, loc, WRITE_BIT | READ_BIT, statesTy);
    Value writeSplat = createConstIntTensor(b, loc, WRITE_BIT, statesTy);
    Value isRW = b.create<arith::AndIOp>(loc, states, rwSplat);
    Value isCurrBufRW = b.create<arith::SelectOp>(loc, currBuf, isRW, zero_8b);

    // Assert that the buffer is not being read or written
    auto isCurrBufRW_i1 = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                  isCurrBufRW, zero_8b);
    b.create<tt::AssertOp>(loc, isCurrBufRW_i1,
                           "TMA copy to buffer being read or written");

    // 2. Update the access state
    Value outStates = b.create<arith::SelectOp>(
        loc, currBuf, b.create<arith::OrIOp>(loc, states, writeSplat), states);
    Value outBarriers =
        b.create<arith::SelectOp>(loc, currBuf, mbarSplat, barriers);
    b.replaceOp(op, {outStates, outBarriers});
    return success();
  }
};

struct CheckWaitMbarOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::ExperimentalCheckWaitMbarOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(triton::gpu::ExperimentalCheckWaitMbarOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();
    b.setInsertionPoint(op);
    RankedTensorType barriersTy =
        cast<RankedTensorType>(op.getBarriers().getType());
    RankedTensorType stateTy = cast<RankedTensorType>(op.getStates().getType());
    Value zero_64b = createConstIntTensor(b, loc, 0, barriersTy);
    Value zero_8b = createConstIntTensor(b, loc, 0, stateTy);
    Value mbar = createMemDescToI64(b, loc, getTypeConverter(),
                                    op.getMbar().getType(), adaptor.getMbar());
    Value currBar = createCmpIntTensorScalar(b, loc, op.getBarriers(), mbar);

    Value outStates =
        b.create<arith::SelectOp>(loc, currBar, zero_8b, op.getStates());
    Value outBarriers =
        b.create<arith::SelectOp>(loc, currBar, zero_64b, op.getBarriers());
    b.replaceOp(op, {outStates, outBarriers});
    return success();
  }
};
} // namespace

void mlir::triton::populateInstrumentationToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<SharedBufferPointersOpConversion>(typeConverter);
  patterns.add<CheckAsyncWriteWithMbarSharedOpConversion>(typeConverter);
  patterns.add<CheckWaitMbarOpConversion>(typeConverter);
}
