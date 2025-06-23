#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"

namespace {

// TODO: unify with ConcurrencySanitizer.cpp
constexpr static int8_t WRITE_BIT = 1 << 0;
constexpr static int8_t READ_BIT = 1 << 1;

namespace tt = mlir::triton;
namespace ttg = tt::gpu;
namespace tti = mlir::triton::instrument;

////////////////////////////////////////////
// Utility functions
////////////////////////////////////////////

Value createConstIntTensor(OpBuilder &builder, Location loc, int val,
                           RankedTensorType tensorType) {
  auto denseAttr = DenseElementsAttr::get(
      tensorType,
      APInt(tensorType.getElementType().getIntOrFloatBitWidth(), val));
  return builder.create<arith::ConstantOp>(loc, tensorType, denseAttr);
}

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

    // #block1
    // if (condition) {
    //   #block2
    //   __assertfail(message);
    // }
    // #block3
    Block *prevBlock = op->getBlock();

    Block *ifBlock = rewriter.splitBlock(prevBlock, op->getIterator());
    rewriter.setInsertionPointToStart(ifBlock);
    targetInfo.assertFail(rewriter, loc, message, file, func, line);

    // Split a block after the call.
    Block *thenBlock = rewriter.splitBlock(ifBlock, op->getIterator());
    rewriter.setInsertionPointToEnd(ifBlock);
    rewriter.create<LLVM::BrOp>(loc, thenBlock);
    rewriter.setInsertionPointToEnd(prevBlock);
    rewriter.create<LLVM::CondBrOp>(loc, condition, ifBlock, thenBlock);
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

struct CheckAsyncWriteWithMbarSharedOpConversion
    : public ConvertOpToLLVMPattern<
          tti::ExperimentalCheckAsyncWriteWithMbarSharedOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(tti::ExperimentalCheckAsyncWriteWithMbarSharedOp op,
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
    Value buffers = op.getBuffers();
    Value states = op.getStates();
    Value barriers = op.getBarriers();

    // 1. Check if the buffer has outstanding accesses
    Value currBuf = createCmpIntTensorScalar(b, loc, buffers, buffer);
    Value rwSplat =
        createConstIntTensor(b, loc, WRITE_BIT | READ_BIT, statesTy);
    Value isRW = b.create<arith::AndIOp>(loc, states, rwSplat);
    Value isCurrBufRW = b.create<arith::SelectOp>(loc, currBuf, isRW, zero_8b);

    // Assert that the buffer is not being read or written
    auto isCurrBufRW_i1 = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                  isCurrBufRW, zero_8b);
    b.create<tti::ExperimentalAssertInThreadOp>(
        loc, isCurrBufRW_i1, "TMA copy to buffer being read or written",
        /*check_any=*/false);

    // 2. Update the access state
    Value writeSplat = createConstIntTensor(b, loc, WRITE_BIT, statesTy);
    Value outStates = b.create<arith::SelectOp>(
        loc, currBuf, b.create<arith::OrIOp>(loc, states, writeSplat), states);
    Value mbarSplat = b.create<triton::SplatOp>(loc, barriersTy, mbar);
    Value outBarriers =
        b.create<arith::SelectOp>(loc, currBuf, mbarSplat, barriers);
    b.replaceOp(op, {outStates, outBarriers});
    return success();
  }
};

struct CheckWaitMbarOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalCheckWaitMbarOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(tti::ExperimentalCheckWaitMbarOp op,
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
  patterns.add<AssertInThreadOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<SharedBufferPointersOpConversion>(typeConverter);
  patterns.add<CheckAsyncWriteWithMbarSharedOpConversion>(typeConverter);
  patterns.add<CheckWaitMbarOpConversion>(typeConverter);
}
