#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Utility.h"

namespace {

namespace tt = mlir::triton;
namespace ttg = tt::gpu;
namespace tti = mlir::triton::instrument;

////////////////////////////////////////////
// Utility functions
////////////////////////////////////////////

Value createFullLike(OpBuilder &builder, Location loc, Value scalar,
                     RankedTensorType tensorTy) {
  auto scalarTy = scalar.getType();
  auto elemTy = tensorTy.getElementType();
  assert(scalarTy == elemTy &&
         "Expected scalar to be of the same type as the tensor elements");
  return builder.create<triton::SplatOp>(loc, tensorTy, scalar);
}

Value createCmpIntTensorScalar(OpBuilder &builder, Location loc, Value tensor,
                               Value scalar) {
  auto tensorTy = cast<RankedTensorType>(tensor.getType());
  auto splat = createFullLike(builder, loc, scalar, tensorTy);
  auto cmp = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                           tensor, splat);
  return cmp;
}

Value createMemDescToI64(RewriterBase &rewriter, Location loc,
                         const LLVMTypeConverter *typeConverter,
                         ttg::MemDescType memDescTy, Value sharedMemStruct) {
  Type srcElemTy = typeConverter->convertType(memDescTy.getElementType());
  int elemSize = srcElemTy.getIntOrFloatBitWidth() / 8;
  auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, sharedMemStruct,
                                                       srcElemTy, rewriter);
  auto offsets = smemObj.getOffsets();
  auto strides = smemObj.getStrides(memDescTy, loc, rewriter);
  Value offset = dot(rewriter, loc, offsets, strides);
  TritonLLVMOpBuilder b(loc, rewriter);
  offset = b.mul(offset, b.i32_val(elemSize));
  auto i64Ty = rewriter.getIntegerType(64);
  offset = b.zext(i64Ty, offset);
  return b.add(offset, b.ptrtoint(i64Ty, smemObj.getBase()));
}

Type getBarsElType(OpBuilder &b) { return b.getIntegerType(64); }

RankedTensorType getWriteBarsType(OpBuilder &b, RankedTensorType buffersType) {
  int size = buffersType.getShape()[0];
  assert(llvm::isPowerOf2_64(size) && "Expected power of 2");
  auto tensorType = RankedTensorType::get({size}, getBarsElType(b),
                                          buffersType.getEncoding());
  return tensorType;
}

RankedTensorType getReadBarsType(OpBuilder &b, RankedTensorType buffersType,
                                 RankedTensorType barriersType) {
  int size = buffersType.getShape()[0];
  assert(llvm::isPowerOf2_64(size) && "Expected power of 2");
  auto tensorType = RankedTensorType::get({size}, getBarsElType(b),
                                          buffersType.getEncoding());
  return tensorType;
}

Value convertAndBroadcast(OpBuilder &b, Location loc, Value tensor, int dim,
                          ArrayRef<int64_t> shape,
                          ttg::BlockedEncodingAttr encoding) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  auto resultType =
      RankedTensorType::get(shape, tensorType.getElementType(), encoding);
  auto slicedLayout =
      ttg::SliceEncodingAttr::get(b.getContext(), dim, encoding);
  tensor = b.create<ttg::ConvertLayoutOp>(
      loc, tensorType.cloneWithEncoding(slicedLayout), tensor);
  tensor = tti::expandOuterSlicedDim(b, loc, tensor);
  tensor = b.create<tt::BroadcastOp>(loc, resultType, tensor);
  return tensor;
}

std::tuple<Block *, Block *, Block *>
createIfBlock(ConversionPatternRewriter &b, Location loc, Value cnd) {
  // #prevBlock
  // if (condition) {
  //   #ifBlock
  // }
  // #thenBlock
  Block *prevBlock = b.getInsertionBlock();
  Block *ifBlock = b.splitBlock(prevBlock, b.getInsertionPoint());

  // Split a block after the call.
  Block *thenBlock = b.splitBlock(ifBlock, ifBlock->begin());
  b.setInsertionPointToEnd(ifBlock);
  b.create<LLVM::BrOp>(loc, thenBlock);
  b.setInsertionPointToEnd(prevBlock);
  b.create<LLVM::CondBrOp>(loc, cnd, ifBlock, thenBlock);
  b.setInsertionPointToStart(thenBlock);

  return {prevBlock, ifBlock, thenBlock};
}

////////////////////////////////////////////
// Patterns
////////////////////////////////////////////

struct AssertInThreadOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalAssertInThreadOp> {
  explicit AssertInThreadOpConversion(LLVMTypeConverter &typeConverter,
                                      const TargetInfoBase &targetInfo,
                                      PatternBenefit benefit)
      : ConvertOpToLLVMPattern<tti::ExperimentalAssertInThreadOp>(typeConverter,
                                                                  benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(tti::ExperimentalAssertInThreadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto tensorTy = cast<RankedTensorType>(op.getCondition().getType());
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    SmallVector<Value> condElems =
        unpackLLElements(loc, adaptor.getCondition(), rewriter);
    auto condTy = condElems[0].getType();
    bool check_any = adaptor.getCheckAny();

    // TODO: Check that all the values are available in the current thread

    Value condition = check_any ? b.int_val(condTy.getIntOrFloatBitWidth(), 0)
                                : b.int_val(condTy.getIntOrFloatBitWidth(), 1);

    assert(condTy.isSignedInteger() ||
           condTy.isSignlessInteger() &&
               "Unsupported type for assert_in_thread");
    Value zero = rewriter.create<LLVM::ConstantOp>(
        loc, condTy, rewriter.getZeroAttr(condTy));
    for (auto elem : condElems) {
      if (check_any) {
        condition = b.or_(condition, elem);
      } else {
        condition = b.and_(condition, elem);
      }
    }

    // Invert the condition - assert will be hit if the condition is true
    condition = b.xor_(condition, b.int_val(condTy.getIntOrFloatBitWidth(), 1));

    llAssert(op, condition, adaptor.getMessage(), rewriter);
    b.barrier();
    rewriter.eraseOp(op);
    return success();
  }

  void llAssert(Operation *op, Value condition, StringRef message,
                ConversionPatternRewriter &rewriter) const {

    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    StringRef file = "unknown";
    StringRef func = "unknown";
    int line = 0;
    int col = 0;

    while (auto callLoc = dyn_cast<CallSiteLoc>(loc))
      loc = callLoc.getCallee();

    if (auto fileLineColLoc = dyn_cast<FileLineColLoc>(loc)) {
      file = fileLineColLoc.getFilename();
      line = fileLineColLoc.getLine();
      col = fileLineColLoc.getColumn();
    }

    // Print the message only for the first thread
    Value threadId = getThreadId(*b.builder, loc);
    Value zero = b.int_val(threadId.getType().getIntOrFloatBitWidth(), 0);
    Value threadIdIsZero = b.icmp_eq(threadId, zero);
    condition = b.and_(condition, threadIdIsZero);

    auto [prevBlock, ifBlock, thenBlock] =
        createIfBlock(rewriter, loc, condition);

    rewriter.setInsertionPointToStart(ifBlock);
    targetInfo.assertFail(rewriter, loc, message, file, func, line);

    rewriter.setInsertionPointToStart(thenBlock);
  }

protected:
  const TargetInfoBase &targetInfo;
};

struct SharedBufferPointersOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalSharedBufferPointersOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(tti::ExperimentalSharedBufferPointersOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    auto loc = op.getLoc();
    auto *ctx = b.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto values = adaptor.getOffsets();
    auto encoding =
        cast<ttg::BlockedEncodingAttr>(op.getResult().getType().getEncoding());
    auto shMemBufs = createInitializedIntArrayTensor(b, loc, encoding, values);
    auto base =
        getSharedMemoryBase(b, op->getParentOfType<FunctionOpInterface>());
    shMemBufs = b.create<arith::AddIOp>(
        loc, shMemBufs,
        b.create<triton::SplatOp>(loc, shMemBufs.getType(), base));
    b.replaceOp(op, shMemBufs);
    return success();
  }

  Value createInitializedIntArrayTensor(OpBuilder &builder, Location loc,
                                        BlockedEncodingAttr encoding,
                                        ArrayRef<int32_t> values) const {
    int64_t size = values.size();
    assert(llvm::isPowerOf2_64(size) && "Expected power of 2");
    auto tensorType =
        RankedTensorType::get({size}, builder.getIntegerType(64), encoding);
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

struct CheckOutstandingWritesOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalCheckOutstandingWritesOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(tti::ExperimentalCheckOutstandingWritesOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();
    b.setInsertionPoint(op);
    if (op.getPred()) {
      auto [prevBlock, ifBlock, thenBlock] =
          createIfBlock(b, loc, op.getPred());
      b.setInsertionPointToStart(ifBlock);
    }
    TypedValue<RankedTensorType> buffers = op.getBuffers();
    RankedTensorType writeBarsType =
        cast<RankedTensorType>(op.getWriteBarsType());
    Value writeBars =
        tti::createLoadScratchMemory(b, loc, op.getWriteBars(), writeBarsType)
            ->getResult(0);
    Value buf = createMemDescToI64(b, loc, getTypeConverter(),
                                   op.getBuf().getType(), adaptor.getBuf());

    // Gluon pseudo-code:
    // curr_buf_bar = tl.where(bufs == buf, write_bars, 0)
    // tl.device_assert(curr_buf_bar == ttgl.zeros_like(curr_buf_bar), "Buffer
    // being accessed has outstanding writes")

    Value writeBarsZero = tti::createConstIntTensor(b, loc, 0, writeBarsType);
    Value buffersEqBuf = createCmpIntTensorScalar(b, loc, buffers, buf);
    Value currBufBar =
        b.create<arith::SelectOp>(loc, buffersEqBuf, writeBars, writeBarsZero);
    Value currBufBarEqZero = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, currBufBar, writeBarsZero);
    b.create<tti::ExperimentalAssertInThreadOp>(
        loc, currBufBarEqZero, "Buffer being accessed has outstanding writes",
        /*check_any=*/false);
    b.eraseOp(op);

    return success();
  }
};

struct CheckOutstandingReadsOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalCheckOutstandingReadsOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(tti::ExperimentalCheckOutstandingReadsOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPoint(op);
    if (op.getPred()) {
      auto [prevBlock, ifBlock, thenBlock] =
          createIfBlock(b, loc, op.getPred());
      b.setInsertionPointToStart(ifBlock);
    }
    TypedValue<RankedTensorType> buffers = op.getBuffers();
    RankedTensorType readBarsType =
        cast<RankedTensorType>(op.getReadBarsType());
    Value readBars =
        tti::createLoadScratchMemory(b, loc, op.getReadBars(), readBarsType)
            ->getResult(0);
    Value buf = createMemDescToI64(b, loc, getTypeConverter(),
                                   op.getBuf().getType(), adaptor.getBuf());

    // Gluon pseudo-code:
    // bufsEqBuf = bufs == buf
    // bufsEqBuf = ttgl.convert_layout(bufsEqBuf, ttgl.SliceLayout(1,
    // read_bars_layout))[:, None] curr_buf_bars = tl.where(bufsEqBuf,
    // read_bars, 0) tl.device_assert(curr_buf_bars ==
    // ttgl.zeros_like(curr_buf_bars), "Buffer being accessed has outstanding
    // reads")
    auto buffersEqBuf = createCmpIntTensorScalar(b, loc, buffers, buf);
    buffersEqBuf = convertAndBroadcast(
        b, loc, buffersEqBuf, 1, readBarsType.getShape(),
        cast<ttg::BlockedEncodingAttr>(readBarsType.getEncoding()));
    auto readBarsZero = tti::createConstIntTensor(b, loc, 0, readBarsType);
    auto currBufBar =
        b.create<arith::SelectOp>(loc, buffersEqBuf, readBars, readBarsZero);
    auto currBufBarEqZero = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, currBufBar, readBarsZero);
    b.create<tti::ExperimentalAssertInThreadOp>(
        loc, currBufBarEqZero, "Buffer being accessed has outstanding reads",
        /*check_any=*/false);
    b.eraseOp(op);
    return success();
  }
};

struct MarkAsWriteOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalMarkAsWriteOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(tti::ExperimentalMarkAsWriteOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPoint(op);
    if (op.getPred()) {
      auto [prevBlock, ifBlock, thenBlock] =
          createIfBlock(b, loc, op.getPred());
      b.setInsertionPointToStart(ifBlock);
    }
    TypedValue<RankedTensorType> buffers = op.getBuffers();
    RankedTensorType writeBarsType =
        cast<RankedTensorType>(op.getWriteBarsType());
    Value writeBars =
        tti::createLoadScratchMemory(b, loc, op.getWriteBars(), writeBarsType)
            ->getResult(0);
    Value buf = createMemDescToI64(b, loc, getTypeConverter(),
                                   op.getBuf().getType(), adaptor.getBuf());
    Value mbar = createMemDescToI64(b, loc, getTypeConverter(),
                                    op.getMbar().getType(), adaptor.getMbar());

    // Gluon pseudo-code:
    // write_bars = tl.where(bufs == buf, mbar, write_bars)

    auto buffersEqBuf = createCmpIntTensorScalar(b, loc, buffers, buf);
    writeBars = b.create<arith::SelectOp>(
        loc, buffersEqBuf, createFullLike(b, loc, mbar, writeBarsType),
        writeBars);
    tti::createStoreScratchMemory(b, loc, op.getWriteBars(), writeBars,
                                  writeBarsType);
    b.eraseOp(op);
    return success();
  }
};

struct MarkAsReadOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalMarkAsReadOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(tti::ExperimentalMarkAsReadOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPoint(op);
    if (op.getPred()) {
      auto [prevBlock, ifBlock, thenBlock] =
          createIfBlock(b, loc, op.getPred());
      b.setInsertionPointToStart(ifBlock);
    }
    TypedValue<RankedTensorType> buffers = op.getBuffers();
    TypedValue<RankedTensorType> barriers = op.getBarriers();
    RankedTensorType readBarsType =
        cast<RankedTensorType>(op.getReadBarsType());
    Value readBars =
        tti::createLoadScratchMemory(b, loc, op.getReadBars(), readBarsType)
            ->getResult(0);
    Value buf = createMemDescToI64(b, loc, getTypeConverter(),
                                   op.getBuf().getType(), adaptor.getBuf());
    Value mbar = createMemDescToI64(b, loc, getTypeConverter(),
                                    op.getMbar().getType(), adaptor.getMbar());

    // Gluon pseudo-code:
    // bufsEqBuf = bufs == buf
    // bufsEqBuf = ttgl.convert_layout(bufsEqBuf, ttgl.SliceLayout(1,
    // read_bars_layout))[:, None] barsEqMbar = bars == mbar barsEqMbar =
    // ttgl.convert_layout(barsEqMbar, ttgl.SliceLayout(0,
    // read_bars_layout))[None, :] bufAndBar = bufsEqBuf & barsEqMbar read_bars
    // = read_bars | bufAndBar

    auto buffersEqBuf = createCmpIntTensorScalar(b, loc, buffers, buf);
    buffersEqBuf = convertAndBroadcast(
        b, loc, buffersEqBuf, 1, readBarsType.getShape(),
        cast<ttg::BlockedEncodingAttr>(readBarsType.getEncoding()));
    auto barriersEqMbar = createCmpIntTensorScalar(b, loc, barriers, mbar);
    barriersEqMbar = convertAndBroadcast(
        b, loc, barriersEqMbar, 0, readBarsType.getShape(),
        cast<ttg::BlockedEncodingAttr>(readBarsType.getEncoding()));
    Value bufAndBar =
        b.create<arith::AndIOp>(loc, buffersEqBuf, barriersEqMbar);
    bufAndBar = b.create<arith::ExtUIOp>(loc, readBarsType, bufAndBar);
    readBars = b.create<arith::OrIOp>(loc, readBars, bufAndBar);

    tti::createStoreScratchMemory(b, loc, op.getReadBars(), readBars,
                                  readBarsType);
    b.eraseOp(op);
    return success();
  }
};

struct ClearWriteBarrierOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalClearWriteBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(tti::ExperimentalClearWriteBarrierOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPoint(op);
    if (op.getPred()) {
      auto [prevBlock, ifBlock, thenBlock] =
          createIfBlock(b, loc, op.getPred());
      b.setInsertionPointToStart(ifBlock);
    }
    RankedTensorType writeBarsType =
        cast<RankedTensorType>(op.getWriteBarsType());
    Value writeBars =
        tti::createLoadScratchMemory(b, loc, op.getWriteBars(), writeBarsType)
            ->getResult(0);
    Value mbar = createMemDescToI64(b, loc, getTypeConverter(),
                                    op.getMbar().getType(), adaptor.getMbar());

    // Gluon pseudo-code:
    // write_bars = tl.where(write_bars == mbar, 0, write_bars)

    auto writeBarsZero = tti::createConstIntTensor(b, loc, 0, writeBarsType);
    auto writeBarsEqMbar = createCmpIntTensorScalar(b, loc, writeBars, mbar);
    writeBars = b.create<arith::SelectOp>(loc, writeBarsEqMbar, writeBarsZero,
                                          writeBars);
    tti::createStoreScratchMemory(b, loc, op.getWriteBars(), writeBars,
                                  writeBarsType);
    b.eraseOp(op);
    return success();
  }
};

struct ClearReadBarrierOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalClearReadBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(tti::ExperimentalClearReadBarrierOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPoint(op);
    if (op.getPred()) {
      auto [prevBlock, ifBlock, thenBlock] =
          createIfBlock(b, loc, op.getPred());
      b.setInsertionPointToStart(ifBlock);
    }
    TypedValue<RankedTensorType> barriers = op.getBarriers();
    RankedTensorType readBarsType =
        cast<RankedTensorType>(op.getReadBarsType());
    Value readBars =
        tti::createLoadScratchMemory(b, loc, op.getReadBars(), readBarsType)
            ->getResult(0);
    Value mbar = createMemDescToI64(b, loc, getTypeConverter(),
                                    op.getMbar().getType(), adaptor.getMbar());

    // Gluon pseudo-code:
    // barsEqMbar = bars == mbar
    // barsEqMbar = ttgl.convert_layout(barsEqMbar, ttgl.SliceLayout(0,
    // read_bars_layout))[None, :] read_bars = tl.where(barsEqMbar, 0,
    // read_bars)

    auto readBarsZero = tti::createConstIntTensor(b, loc, 0, readBarsType);
    auto readBarsEqMbar = createCmpIntTensorScalar(b, loc, barriers, mbar);
    readBarsEqMbar = convertAndBroadcast(
        b, loc, readBarsEqMbar, 0, readBarsType.getShape(),
        cast<ttg::BlockedEncodingAttr>(readBarsType.getEncoding()));
    readBars =
        b.create<arith::SelectOp>(loc, readBarsEqMbar, readBarsZero, readBars);
    tti::createStoreScratchMemory(b, loc, op.getReadBars(), readBars,
                                  readBarsType);
    b.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::triton::populateInstrumentationToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<AssertInThreadOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<SharedBufferPointersOpConversion>(typeConverter);
  patterns.add<CheckOutstandingWritesOpConversion>(typeConverter);
  patterns.add<CheckOutstandingReadsOpConversion>(typeConverter);
  patterns.add<MarkAsWriteOpConversion>(typeConverter);
  patterns.add<MarkAsReadOpConversion>(typeConverter);
  patterns.add<ClearWriteBarrierOpConversion>(typeConverter);
  patterns.add<ClearReadBarrierOpConversion>(typeConverter);
}
