#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "third_party/nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace {

namespace tt = mlir::triton;
namespace ttg = tt::gpu;
namespace tti = mlir::triton::instrument;
namespace ttng = mlir::triton::nvidia_gpu;

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

Value createCmpIntTensorScalar(
    OpBuilder &builder, Location loc, Value tensor, Value scalar,
    arith::CmpIPredicate predicate = arith::CmpIPredicate::eq) {
  auto tensorTy = cast<RankedTensorType>(tensor.getType());
  auto splat = createFullLike(builder, loc, scalar, tensorTy);
  auto cmp = builder.create<arith::CmpIOp>(loc, predicate, tensor, splat);
  return cmp;
}

Value createMemDescToI64(RewriterBase &rewriter, Location loc,
                         const LLVMTypeConverter *typeConverter,
                         ttg::MemDescType memDescTy, Value sharedMemStruct) {
  TritonLLVMOpBuilder b(loc, rewriter);
  if (isa<ttng::TensorMemoryEncodingAttr>(memDescTy.getEncoding())) {
    return b.ptrtoint(rewriter.getIntegerType(64), sharedMemStruct);
  }
  assert(isa<ttg::SharedEncodingTrait>(memDescTy.getEncoding()) &&
         "Unsupported memory encoding");
  Type srcElemTy = typeConverter->convertType(memDescTy.getElementType());
  auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, sharedMemStruct,
                                                       srcElemTy, rewriter);
  auto offset = smemObj.getShmemOffset(loc, rewriter, memDescTy);
  auto elemSize = srcElemTy.getIntOrFloatBitWidth() / 8;
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
                          RankedTensorType dstType) {
  auto shape = dstType.getShape();
  int j = 0;
  for (int i = 0; i < shape.size(); i++) {
    if (i != dim) {
      assert(shape[i] ==
                 cast<RankedTensorType>(tensor.getType()).getShape()[j] &&
             "Expected shape to be the same");
      j++;
    }
  }
  auto encoding = cast<ttg::BlockedEncodingAttr>(dstType.getEncoding());
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

Value createConvertLayout(OpBuilder &b, Location loc, Value tensor,
                          Attribute encoding) {
  RankedTensorType dstType =
      cast<RankedTensorType>(tensor.getType()).cloneWithEncoding(encoding);
  return b.create<ttg::ConvertLayoutOp>(loc, dstType, tensor);
}

Value createOneHot(OpBuilder &b, Location loc, int size, int index,
                   Attribute encoding) {
  int start = 0;
  int end = size;
  RankedTensorType type =
      RankedTensorType::get({size}, b.getI32Type(), encoding);
  Value arange = b.create<tt::MakeRangeOp>(loc, type, start, end);
  Value indexT = tti::createConstIntTensor(b, loc, index, type);
  return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, arange, indexT);
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

Value createMaxReduce(OpBuilder &b, Location loc, Value tensor, int axis) {
  OpBuilder::InsertionGuard guard(b);
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  auto reduceOp = b.create<tt::ReduceOp>(loc, std::vector<Value>{tensor}, axis);
  auto &region = reduceOp.getRegion();
  auto &block = region.emplaceBlock();
  block.addArguments({tensorType.getElementType(), tensorType.getElementType()},
                     {loc, loc});
  auto arg0 = block.getArgument(0);
  auto arg1 = block.getArgument(1);
  b.setInsertionPointToStart(&block);
  auto cmpOp =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, arg0, arg1);
  auto result = b.create<arith::SelectOp>(loc, cmpOp, arg0, arg1);
  auto returnOp = b.create<tt::ReduceReturnOp>(loc, std::vector<Value>{result});
  return reduceOp->getResult(0);
}

Value createOrReduce(OpBuilder &b, Location loc, Value tensor, int axis) {
  OpBuilder::InsertionGuard guard(b);
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  auto reduceOp = b.create<tt::ReduceOp>(loc, std::vector<Value>{tensor}, axis);
  auto &region = reduceOp.getRegion();
  auto &block = region.emplaceBlock();
  block.addArguments({tensorType.getElementType(), tensorType.getElementType()},
                     {loc, loc});
  auto arg0 = block.getArgument(0);
  auto arg1 = block.getArgument(1);
  b.setInsertionPointToStart(&block);
  auto result = b.create<arith::OrIOp>(loc, arg0, arg1);
  auto returnOp = b.create<tt::ReduceReturnOp>(loc, std::vector<Value>{result});
  return reduceOp->getResult(0);
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

    while (auto nameLoc = dyn_cast<NameLoc>(loc))
      loc = nameLoc.getChildLoc();

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

struct BufferPointersOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalBufferPointersOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(tti::ExperimentalBufferPointersOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto values = adaptor.getOffsets();
    auto encoding =
        cast<ttg::BlockedEncodingAttr>(op.getResult().getType().getEncoding());
    auto bufPointers =
        createInitializedIntArrayTensor(rewriter, loc, encoding, values);
    Value base = nullptr;
    if (op.getMemType() == tti::MemType::SHARED_MEM) {
      base = getSharedMemoryBase(rewriter,
                                 op->getParentOfType<FunctionOpInterface>());
    } else {
      assert(op.getMemType() == tti::MemType::TENSOR_MEM &&
             "Unsupported memory type");
      TritonLLVMOpBuilder b(loc, rewriter);
      base = rewriter.create<nvgpu::TensorMemoryBaseAddress>(loc);
      base = b.ptrtoint(i32_ty, base);
    }
    bufPointers = rewriter.create<arith::AddIOp>(
        loc, bufPointers,
        rewriter.create<triton::SplatOp>(loc, bufPointers.getType(), base));
    rewriter.replaceOp(op, bufPointers);
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

struct SetWriteVisibilityOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalSetWriteVisibilityOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(tti::ExperimentalSetWriteVisibilityOp op,
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
    RankedTensorType writeVisibilityType =
        cast<RankedTensorType>(op.getWriteVisibilityType());
    Value writeVisibility =
        tti::createLoadScratchMemory(b, loc, op.getWriteVisibility(),
                                     writeVisibilityType)
            ->getResult(0);
    Value buf = createMemDescToI64(b, loc, getTypeConverter(),
                                   op.getBuf().getType(), adaptor.getBuf());
    Value buffersEqBuf = createCmpIntTensorScalar(b, loc, buffers, buf);
    int thread = op.getThread();
    Value threadBit =
        tti::createConstIntTensor(b, loc, 1ULL << thread, writeVisibilityType);
    writeVisibility = b.create<arith::SelectOp>(loc, buffersEqBuf, threadBit,
                                                writeVisibility);

    // Print
    // b.create<tt::PrintOp>(loc, "wv: ", false, threadBit,
    // std::vector<int32_t>{0});

    tti::createStoreScratchMemory(b, loc, op.getWriteVisibility(),
                                  writeVisibility, writeVisibilityType);
    b.eraseOp(op);
    return success();
  }
};

struct SetReadVisibilityOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalSetReadVisibilityOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(tti::ExperimentalSetReadVisibilityOp op,
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
    RankedTensorType readVisibilityType =
        cast<RankedTensorType>(op.getReadVisibilityType());
    Value readVisibility =
        tti::createLoadScratchMemory(b, loc, op.getReadVisibility(),
                                     readVisibilityType)
            ->getResult(0);
    Value buf = createMemDescToI64(b, loc, getTypeConverter(),
                                   op.getBuf().getType(), adaptor.getBuf());
    Value buffersEqBuf = createCmpIntTensorScalar(b, loc, buffers, buf);
    buffersEqBuf =
        convertAndBroadcast(b, loc, buffersEqBuf, 1, readVisibilityType);
    Attribute threadOneHotEncoding = SliceEncodingAttr::get(
        b.getContext(), 0,
        cast<BlockedEncodingAttr>(readVisibilityType.getEncoding()));
    Value threadOneHot = createOneHot(b, loc, readVisibilityType.getShape()[1],
                                      op.getThread(), threadOneHotEncoding);
    threadOneHot =
        convertAndBroadcast(b, loc, threadOneHot, 0, readVisibilityType);
    Value threadBit = tti::createConstIntTensor(b, loc, 1ULL << op.getThread(),
                                                readVisibilityType);
    Value readVisibilityOrThreadBit =
        b.create<arith::OrIOp>(loc, readVisibility, threadBit);
    Value bufAndThread =
        b.create<arith::AndIOp>(loc, buffersEqBuf, threadOneHot);
    readVisibility = b.create<arith::SelectOp>(
        loc, bufAndThread, readVisibilityOrThreadBit, readVisibility);
    tti::createStoreScratchMemory(b, loc, op.getReadVisibility(),
                                  readVisibility, readVisibilityType);
    b.eraseOp(op);
    return success();
  }
};

struct ClearWriteTrackingOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalClearWriteTrackingOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(tti::ExperimentalClearWriteTrackingOp op,
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
    RankedTensorType writeTrackingType =
        cast<RankedTensorType>(op.getWriteTrackingType());
    Value writeTracking = tti::createLoadScratchMemory(
                              b, loc, op.getWriteTracking(), writeTrackingType)
                              ->getResult(0);
    Value buf = createMemDescToI64(b, loc, getTypeConverter(),
                                   op.getBuf().getType(), adaptor.getBuf());
    Value buffersEqBuf = createCmpIntTensorScalar(b, loc, buffers, buf);
    buffersEqBuf =
        convertAndBroadcast(b, loc, buffersEqBuf, 1, writeTrackingType);
    Value zero = tti::createConstIntTensor(b, loc, 0, writeTrackingType);
    writeTracking =
        b.create<arith::SelectOp>(loc, buffersEqBuf, zero, writeTracking);
    tti::createStoreScratchMemory(b, loc, op.getWriteTracking(), writeTracking,
                                  writeTrackingType);
    b.eraseOp(op);
    return success();
  }
};

struct ClearReadVisibilityOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalClearReadVisibilityOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(tti::ExperimentalClearReadVisibilityOp op,
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
    RankedTensorType readVisibilityType =
        cast<RankedTensorType>(op.getReadVisibilityType());
    Value readVisibility =
        tti::createLoadScratchMemory(b, loc, op.getReadVisibility(),
                                     readVisibilityType)
            ->getResult(0);
    Value buf = createMemDescToI64(b, loc, getTypeConverter(),
                                   op.getBuf().getType(), adaptor.getBuf());
    Value buffersEqBuf = createCmpIntTensorScalar(b, loc, buffers, buf);
    buffersEqBuf =
        convertAndBroadcast(b, loc, buffersEqBuf, 1, readVisibilityType);
    Value zero = tti::createConstIntTensor(b, loc, 0, readVisibilityType);
    readVisibility =
        b.create<arith::SelectOp>(loc, buffersEqBuf, zero, readVisibility);
    tti::createStoreScratchMemory(b, loc, op.getReadVisibility(),
                                  readVisibility, readVisibilityType);
    b.eraseOp(op);
    return success();
  }
};

struct ClearReadTrackingOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalClearReadTrackingOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(tti::ExperimentalClearReadTrackingOp op,
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
    RankedTensorType readTrackingType =
        cast<RankedTensorType>(op.getReadTrackingType());
    Value readTracking = tti::createLoadScratchMemory(
                             b, loc, op.getReadTracking(), readTrackingType)
                             ->getResult(0);
    Value buf = createMemDescToI64(b, loc, getTypeConverter(),
                                   op.getBuf().getType(), adaptor.getBuf());
    Value buffersEqBuf = createCmpIntTensorScalar(b, loc, buffers, buf);
    buffersEqBuf =
        convertAndBroadcast(b, loc, buffersEqBuf, 1, readTrackingType);
    Value zero = tti::createConstIntTensor(b, loc, 0, readTrackingType);
    readTracking =
        b.create<arith::SelectOp>(loc, buffersEqBuf, zero, readTracking);
    tti::createStoreScratchMemory(b, loc, op.getReadTracking(), readTracking,
                                  readTrackingType);
    b.eraseOp(op);
    return success();
  }
};

struct TrackVisibleWritesOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalTrackVisibleWritesOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(tti::ExperimentalTrackVisibleWritesOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();
    b.setInsertionPoint(op);
    if (op.getPred()) {
      auto [prevBlock, ifBlock, thenBlock] =
          createIfBlock(b, loc, op.getPred());
      b.setInsertionPointToStart(ifBlock);
    }
    TypedValue<RankedTensorType> barriers = op.getBarriers();
    RankedTensorType writeVisibilityType =
        cast<RankedTensorType>(op.getWriteVisibilityType());
    Value writeVisibility =
        tti::createLoadScratchMemory(b, loc, op.getWriteVisibility(),
                                     writeVisibilityType)
            ->getResult(0);
    RankedTensorType writeTrackingType =
        cast<RankedTensorType>(op.getWriteTrackingType());
    Value writeTracking = tti::createLoadScratchMemory(
                              b, loc, op.getWriteTracking(), writeTrackingType)
                              ->getResult(0);
    Value bar = createMemDescToI64(b, loc, getTypeConverter(),
                                   op.getMbar().getType(), adaptor.getMbar());
    Value barriersEqBar = createCmpIntTensorScalar(b, loc, barriers, bar);
    barriersEqBar =
        convertAndBroadcast(b, loc, barriersEqBar, 0, writeTrackingType);
    int thread = op.getThread();
    Value threadBit =
        tti::createConstIntTensor(b, loc, 1ULL << thread, writeVisibilityType);
    Value visibleWrites =
        b.create<arith::AndIOp>(loc, writeVisibility, threadBit);
    visibleWrites = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                            visibleWrites, threadBit);
    visibleWrites =
        convertAndBroadcast(b, loc, visibleWrites, 1, writeTrackingType);
    Value barAndVisible =
        b.create<arith::AndIOp>(loc, barriersEqBar, visibleWrites);
    Value writeTrackingOne =
        tti::createConstIntTensor(b, loc, 1, writeTrackingType);
    writeTracking = b.create<arith::SelectOp>(loc, barAndVisible,
                                              writeTrackingOne, writeTracking);
    tti::createStoreScratchMemory(b, loc, op.getWriteTracking(), writeTracking,
                                  writeTrackingType);
    b.eraseOp(op);

    return success();
  }
};

struct TrackVisibleReadsOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalTrackVisibleReadsOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(tti::ExperimentalTrackVisibleReadsOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();
    b.setInsertionPoint(op);
    if (op.getPred()) {
      auto [prevBlock, ifBlock, thenBlock] =
          createIfBlock(b, loc, op.getPred());
      b.setInsertionPointToStart(ifBlock);
    }
    TypedValue<RankedTensorType> barriers = op.getBarriers();
    RankedTensorType readVisibilityType =
        cast<RankedTensorType>(op.getReadVisibilityType());
    Value readVisibility =
        tti::createLoadScratchMemory(b, loc, op.getReadVisibility(),
                                     readVisibilityType)
            ->getResult(0);
    RankedTensorType readTrackingType =
        cast<RankedTensorType>(op.getReadTrackingType());
    Value readTracking = tti::createLoadScratchMemory(
                             b, loc, op.getReadTracking(), readTrackingType)
                             ->getResult(0);
    Value bar = createMemDescToI64(b, loc, getTypeConverter(),
                                   op.getMbar().getType(), adaptor.getMbar());
    Value barriersEqBar = createCmpIntTensorScalar(b, loc, barriers, bar);
    barriersEqBar =
        convertAndBroadcast(b, loc, barriersEqBar, 0, readTrackingType);
    // TODO: getSingleDimSliceEncoding
    Attribute threadOneHotEncoding = SliceEncodingAttr::get(
        b.getContext(), 0,
        cast<BlockedEncodingAttr>(readVisibilityType.getEncoding()));
    Value threadOneHot = createOneHot(b, loc, readVisibilityType.getShape()[1],
                                      op.getThread(), threadOneHotEncoding);
    threadOneHot =
        convertAndBroadcast(b, loc, threadOneHot, 0, readVisibilityType);
    Value readVisibilityZero =
        tti::createConstIntTensor(b, loc, 0, readVisibilityType);
    Value visibleReads = b.create<arith::SelectOp>(
        loc, threadOneHot, readVisibility, readVisibilityZero);
    visibleReads = createOrReduce(b, loc, visibleReads, 1);
    visibleReads =
        convertAndBroadcast(b, loc, visibleReads, 1, readTrackingType);
    Value readTrackingOrVisible =
        b.create<arith::OrIOp>(loc, readTracking, visibleReads);
    readTracking = b.create<arith::SelectOp>(
        loc, barriersEqBar, readTrackingOrVisible, readTracking);

    tti::createStoreScratchMemory(b, loc, op.getReadTracking(), readTracking,
                                  readTrackingType);
    b.eraseOp(op);
    return success();
  }
};

struct TransferVisibleWritesOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalTransferVisibleWritesOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(tti::ExperimentalTransferVisibleWritesOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();
    b.setInsertionPoint(op);
    if (op.getPred()) {
      auto [prevBlock, ifBlock, thenBlock] =
          createIfBlock(b, loc, op.getPred());
      b.setInsertionPointToStart(ifBlock);
    }
    TypedValue<RankedTensorType> barriers = op.getBarriers();
    RankedTensorType writeVisibilityType =
        cast<RankedTensorType>(op.getWriteVisibilityType());
    Value writeVisibility =
        tti::createLoadScratchMemory(b, loc, op.getWriteVisibility(),
                                     writeVisibilityType)
            ->getResult(0);
    RankedTensorType writeTrackingType =
        cast<RankedTensorType>(op.getWriteTrackingType());
    Value writeTracking = tti::createLoadScratchMemory(
                              b, loc, op.getWriteTracking(), writeTrackingType)
                              ->getResult(0);
    Value bar = createMemDescToI64(b, loc, getTypeConverter(),
                                   op.getMbar().getType(), adaptor.getMbar());
    Value thread =
        tti::createConstIntTensor(b, loc, op.getThread(), writeVisibilityType);
    Value barriersEqBar = createCmpIntTensorScalar(b, loc, barriers, bar);
    barriersEqBar =
        convertAndBroadcast(b, loc, barriersEqBar, 0, writeTrackingType);
    Value writeTrackingZero =
        tti::createConstIntTensor(b, loc, 0, writeTrackingType);
    Value trackingBuffers = b.create<arith::SelectOp>(
        loc, barriersEqBar, writeTracking, writeTrackingZero);
    trackingBuffers = createOrReduce(b, loc, trackingBuffers, 1);
    trackingBuffers = createConvertLayout(b, loc, trackingBuffers,
                                          writeVisibilityType.getEncoding());
    trackingBuffers =
        b.create<arith::ExtUIOp>(loc, writeVisibilityType, trackingBuffers);
    Value trackingThreadBit =
        b.create<arith::ShLIOp>(loc, trackingBuffers, thread);
    writeVisibility =
        b.create<arith::OrIOp>(loc, writeVisibility, trackingThreadBit);
    tti::createStoreScratchMemory(b, loc, op.getWriteVisibility(),
                                  writeVisibility, writeVisibilityType);
    b.eraseOp(op);
    return success();
  }
};

struct TransferVisibleReadsOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalTransferVisibleReadsOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(tti::ExperimentalTransferVisibleReadsOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();
    b.setInsertionPoint(op);
    if (op.getPred()) {
      auto [prevBlock, ifBlock, thenBlock] =
          createIfBlock(b, loc, op.getPred());
      b.setInsertionPointToStart(ifBlock);
    }
    TypedValue<RankedTensorType> barriers = op.getBarriers();
    RankedTensorType readVisibilityType =
        cast<RankedTensorType>(op.getReadVisibilityType());
    Value readVisibility =
        tti::createLoadScratchMemory(b, loc, op.getReadVisibility(),
                                     readVisibilityType)
            ->getResult(0);
    RankedTensorType readTrackingType =
        cast<RankedTensorType>(op.getReadTrackingType());
    Value readTracking = tti::createLoadScratchMemory(
                             b, loc, op.getReadTracking(), readTrackingType)
                             ->getResult(0);
    Value bar = createMemDescToI64(b, loc, getTypeConverter(),
                                   op.getMbar().getType(), adaptor.getMbar());
    Value barriersEqBar = createCmpIntTensorScalar(b, loc, barriers, bar);
    barriersEqBar =
        convertAndBroadcast(b, loc, barriersEqBar, 0, readTrackingType);
    Value readTrackingZero =
        tti::createConstIntTensor(b, loc, 0, readTrackingType);
    Value trackingBar = b.create<arith::SelectOp>(
        loc, barriersEqBar, readTracking, readTrackingZero);
    trackingBar = createOrReduce(b, loc, trackingBar, 1);
    trackingBar =
        convertAndBroadcast(b, loc, trackingBar, 1, readVisibilityType);
    Value readVisibilityOrTracking =
        b.create<arith::OrIOp>(loc, readVisibility, trackingBar);
    Attribute threadOneHotEncoding = SliceEncodingAttr::get(
        b.getContext(), 0,
        cast<BlockedEncodingAttr>(readVisibilityType.getEncoding()));
    Value threadOneHot = createOneHot(b, loc, readVisibilityType.getShape()[1],
                                      op.getThread(), threadOneHotEncoding);
    threadOneHot =
        convertAndBroadcast(b, loc, threadOneHot, 0, readVisibilityType);
    readVisibility = b.create<arith::SelectOp>(
        loc, threadOneHot, readVisibilityOrTracking, readVisibility);
    tti::createStoreScratchMemory(b, loc, op.getReadVisibility(),
                                  readVisibility, readVisibilityType);
    b.eraseOp(op);
    return success();
  }
};

struct VerifyWriteVisibilityOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalVerifyWriteVisibilityOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(tti::ExperimentalVerifyWriteVisibilityOp op,
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
    RankedTensorType writeVisibilityType =
        cast<RankedTensorType>(op.getWriteVisibilityType());
    Value writeVisibility =
        tti::createLoadScratchMemory(b, loc, op.getWriteVisibility(),
                                     writeVisibilityType)
            ->getResult(0);

    // Print
    // b.create<tt::PrintOp>(loc, "wv: ", false, writeVisibility,
    // std::vector<int32_t>{0});

    Value buf = createMemDescToI64(b, loc, getTypeConverter(),
                                   op.getBuf().getType(), adaptor.getBuf());
    Value buffersEqBuf = createCmpIntTensorScalar(b, loc, buffers, buf);
    Value writeVisibilityZero =
        tti::createConstIntTensor(b, loc, 0, writeVisibilityType);
    Value bufVisibility = b.create<arith::SelectOp>(
        loc, buffersEqBuf, writeVisibility, writeVisibilityZero);
    Value noOneIsWriting = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, bufVisibility, writeVisibilityZero);
    Value thread =
        tti::createConstIntTensor(b, loc, op.getThread(), writeVisibilityType);
    buffersEqBuf =
        b.create<arith::ExtUIOp>(loc, writeVisibilityType, buffersEqBuf);
    Value bufferThreadBit = b.create<arith::ShLIOp>(loc, buffersEqBuf, thread);
    Value bufferHasVisibility =
        b.create<arith::AndIOp>(loc, bufVisibility, bufferThreadBit);
    bufferHasVisibility = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, bufferHasVisibility, bufferThreadBit);
    Value writeVisible =
        b.create<arith::OrIOp>(loc, noOneIsWriting, bufferHasVisibility);
    b.create<tti::ExperimentalAssertInThreadOp>(
        loc, writeVisible, "Buffer being accessed has outstanding writes",
        /*check_any=*/false);
    b.eraseOp(op);
    return success();
  }
};

struct VerifyReadVisibilityOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalVerifyReadVisibilityOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(tti::ExperimentalVerifyReadVisibilityOp op,
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
    RankedTensorType readVisibilityType =
        cast<RankedTensorType>(op.getReadVisibilityType());
    Value readVisibility =
        tti::createLoadScratchMemory(b, loc, op.getReadVisibility(),
                                     readVisibilityType)
            ->getResult(0);

    Value buf = createMemDescToI64(b, loc, getTypeConverter(),
                                   op.getBuf().getType(), adaptor.getBuf());
    Value buffersEqBuf = createCmpIntTensorScalar(b, loc, buffers, buf);
    buffersEqBuf =
        convertAndBroadcast(b, loc, buffersEqBuf, 1, readVisibilityType);
    Value readVisibilityZero =
        tti::createConstIntTensor(b, loc, 0, readVisibilityType);
    Value bufVisibility = b.create<arith::SelectOp>(
        loc, buffersEqBuf, readVisibility, readVisibilityZero);
    Value totalVisibility = createOrReduce(b, loc, bufVisibility, 1);
    Attribute threadOneHotEncoding = SliceEncodingAttr::get(
        b.getContext(), 0,
        cast<BlockedEncodingAttr>(readVisibilityType.getEncoding()));
    Value threadOneHot = createOneHot(b, loc, readVisibilityType.getShape()[1],
                                      op.getThread(), threadOneHotEncoding);
    threadOneHot =
        convertAndBroadcast(b, loc, threadOneHot, 0, readVisibilityType);
    Value threadBit = tti::createConstIntTensor(b, loc, 1ULL << op.getThread(),
                                                readVisibilityType);
    Value readVisibilityOrThreadBit =
        b.create<arith::OrIOp>(loc, readVisibility, threadBit);
    Value bufThreadVisibility = b.create<arith::SelectOp>(
        loc, threadOneHot, bufVisibility, readVisibilityZero);
    bufThreadVisibility = createOrReduce(b, loc, bufThreadVisibility, 1);
    // Thread must have visivility that is a superset of read visibility of all
    // other threads
    Value threadAndTotalVisibility =
        b.create<arith::AndIOp>(loc, bufThreadVisibility, totalVisibility);
    Value hasVisibility =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                threadAndTotalVisibility, totalVisibility);
    b.create<tti::ExperimentalAssertInThreadOp>(
        loc, hasVisibility, "Buffer being accessed has outstanding reads",
        /*check_any=*/false);
    b.eraseOp(op);
    return success();
  }
};

struct SetWriteStateOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalSetWriteStateOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(tti::ExperimentalSetWriteStateOp op,
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
    RankedTensorType writeStateType =
        cast<RankedTensorType>(op.getWriteStateType());
    Value writeState =
        tti::createLoadScratchMemory(b, loc, op.getWriteState(), writeStateType)
            ->getResult(0);
    int notHwPipelined = op.getHwPipelined() ? 0 : 1;
    Value buf = createMemDescToI64(b, loc, getTypeConverter(),
                                   op.getBuf().getType(), adaptor.getBuf());

    // Gluon pseudo-code:
    // val = 1 | (not_hw_pipelined << 1)
    // write_state = tl.where(bufs == buf, 1, write_state)

    int val = 1 | (notHwPipelined << 1);
    auto buffersEqBuf = createCmpIntTensorScalar(b, loc, buffers, buf);
    writeState = b.create<arith::SelectOp>(
        loc, buffersEqBuf,
        tti::createConstIntTensor(b, loc, val, writeStateType), writeState);
    tti::createStoreScratchMemory(b, loc, op.getWriteState(), writeState,
                                  writeStateType);
    b.eraseOp(op);
    return success();
  }
};

struct CommitWriteWithBarrierOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalCommitWriteWithBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(tti::ExperimentalCommitWriteWithBarrierOp op,
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
    RankedTensorType writeBarsType =
        cast<RankedTensorType>(op.getWriteBarsType());
    Value writeBars =
        tti::createLoadScratchMemory(b, loc, op.getWriteBars(), writeBarsType)
            ->getResult(0);
    RankedTensorType writeStateType =
        cast<RankedTensorType>(op.getWriteStateType());
    Value writeState =
        tti::createLoadScratchMemory(b, loc, op.getWriteState(), writeStateType)
            ->getResult(0);
    Value mbar = createMemDescToI64(b, loc, getTypeConverter(),
                                    op.getMbar().getType(), adaptor.getMbar());

    // clang-format off
    // Gluon pseudo-code:
    // write_state = ttgl.convert_layout(write_state, ttgl.SliceLayout(1, write_bars_layout))[:, None]
    // barsEqMbar = barriers == mbar
    // barsEqMbar = ttgl.convert_layout(barsEqMbar, ttgl.SliceLayout(0, write_bars_layout))[None, :]
    // stateAndBar = write_state & barsEqMbar
    // write_bars = write_bars | stateAndBar
    // clang-format on

    writeState = convertAndBroadcast(b, loc, writeState, 1, writeBarsType);
    auto barriersEqMbar = createCmpIntTensorScalar(b, loc, barriers, mbar);
    barriersEqMbar =
        convertAndBroadcast(b, loc, barriersEqMbar, 0, writeBarsType);
    barriersEqMbar =
        b.create<arith::ExtUIOp>(loc, writeBarsType, barriersEqMbar);
    Value stateAndBar =
        b.create<arith::AndIOp>(loc, writeState, barriersEqMbar);
    writeBars = b.create<arith::OrIOp>(loc, writeBars, stateAndBar);
    tti::createStoreScratchMemory(b, loc, op.getWriteBars(), writeBars,
                                  writeBarsType);
    b.eraseOp(op);
    return success();
  }
};

struct SetReadBarrierOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalSetReadBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(tti::ExperimentalSetReadBarrierOp op,
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

    // clang-format off
    // Gluon pseudo-code:
    // bufsEqBuf = bufs == buf
    // bufsEqBuf = ttgl.convert_layout(bufsEqBuf, ttgl.SliceLayout(1, read_bars_layout))[:, None]
    // barsEqMbar = bars == mbar
    // barsEqMbar = ttgl.convert_layout(barsEqMbar, ttgl.SliceLayout(0, read_bars_layout))[None, :]
    // bufAndBar = bufsEqBuf & barsEqMbar
    // read_bars = read_bars | bufAndBar
    // clang-format on

    auto buffersEqBuf = createCmpIntTensorScalar(b, loc, buffers, buf);
    buffersEqBuf = convertAndBroadcast(b, loc, buffersEqBuf, 1, readBarsType);
    auto barriersEqMbar = createCmpIntTensorScalar(b, loc, barriers, mbar);
    barriersEqMbar =
        convertAndBroadcast(b, loc, barriersEqMbar, 0, readBarsType);
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
    TypedValue<RankedTensorType> barriers = op.getBarriers();
    RankedTensorType writeBarsType =
        cast<RankedTensorType>(op.getWriteBarsType());
    Value writeBars =
        tti::createLoadScratchMemory(b, loc, op.getWriteBars(), writeBarsType)
            ->getResult(0);
    RankedTensorType writeStateType =
        cast<RankedTensorType>(op.getWriteStateType());
    Value writeState =
        tti::createLoadScratchMemory(b, loc, op.getWriteState(), writeStateType)
            ->getResult(0);
    Value mbar = createMemDescToI64(b, loc, getTypeConverter(),
                                    op.getMbar().getType(), adaptor.getMbar());

    // clang-format off
    // Gluon pseudo-code:
    // barsEqMbar = barriers == mbar
    // barsEqMbar = ttgl.convert_layout(barsEqMbar, ttgl.SliceLayout(0, write_bars_layout))[None, :]
    // writeBarsForMbar = write_bars & barsEqMbar
    // writeBarsForMbar = ttgl.reduce(writeBarsForMbar, axis=1, combine_fn=max)
    // writeBarsForMbar is now a tensor of shape [num_buffers, 1] containing the
    // non-zero entries where the buffer was being tracked by the barrier.
    // write_state = tl.where(writeBarsForMbar != 0, 0, write_state)
    // write_bars = tl.where(barsEqMbar, 0, write_bars)
    // clang-format on

    auto barriersEqMbar = createCmpIntTensorScalar(b, loc, barriers, mbar);
    barriersEqMbar =
        convertAndBroadcast(b, loc, barriersEqMbar, 0, writeBarsType);
    Value barriersEqMbarI8 =
        b.create<arith::ExtUIOp>(loc, writeBarsType, barriersEqMbar);
    Value writeBarsForMbar =
        b.create<arith::AndIOp>(loc, writeBars, barriersEqMbarI8);
    writeBarsForMbar = createMaxReduce(b, loc, writeBarsForMbar, 1);
    Value writeStateZero = tti::createConstIntTensor(b, loc, 0, writeStateType);
    Value writeBarsForMbarNonZero = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne, writeBarsForMbar, writeStateZero);
    writeState = b.create<arith::SelectOp>(loc, writeBarsForMbarNonZero,
                                           writeStateZero, writeState);
    tti::createStoreScratchMemory(b, loc, op.getWriteState(), writeState,
                                  writeStateType);
    Value writeBarsZero = tti::createConstIntTensor(b, loc, 0, writeBarsType);
    writeBars = b.create<arith::SelectOp>(loc, barriersEqMbar, writeBarsZero,
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

    // clang-format off
    // Gluon pseudo-code:
    // barsEqMbar = bars == mbar
    // barsEqMbar = ttgl.convert_layout(barsEqMbar, ttgl.SliceLayout(0, read_bars_layout))[None, :]
    // read_bars = tl.where(barsEqMbar, 0, read_bars)
    // clang format on

    auto readBarsZero = tti::createConstIntTensor(b, loc, 0, readBarsType);
    auto readBarsEqMbar = createCmpIntTensorScalar(b, loc, barriers, mbar);
    readBarsEqMbar = convertAndBroadcast(
        b, loc, readBarsEqMbar, 0, readBarsType);
    readBars =
        b.create<arith::SelectOp>(loc, readBarsEqMbar, readBarsZero, readBars);
    tti::createStoreScratchMemory(b, loc, op.getReadBars(), readBars,
                                  readBarsType);
    b.eraseOp(op);
    return success();
  }
};

struct CheckBarrierWritesClearedOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalCheckBarrierWritesClearedOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(tti::ExperimentalCheckBarrierWritesClearedOp op,
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
    RankedTensorType writeBarsType =
        cast<RankedTensorType>(op.getWriteBarsType());
    Value writeBars =
        tti::createLoadScratchMemory(b, loc, op.getWriteBars(), writeBarsType)
            ->getResult(0);
    Value mbar = createMemDescToI64(b, loc, getTypeConverter(),
                                    op.getMbar().getType(), adaptor.getMbar());

    // clang-format off
    // Gluon pseudo-code:
    // barsEqMbar = bars == mbar
    // barsEqMbar = ttgl.convert_layout(barsEqMbar, ttgl.SliceLayout(0, write_bars_layout))[None, :]
    // currWriteBars = write_bars & barsEqMbar
    // tl.device_assert(currWriteBars == 0, "Barrier is being reused while still tracking writes")
    // clang-format on

    auto writeBarsZero = tti::createConstIntTensor(b, loc, 0, writeBarsType);
    auto barsEqMbar = createCmpIntTensorScalar(b, loc, barriers, mbar);
    barsEqMbar = convertAndBroadcast(b, loc, barsEqMbar, 0, writeBarsType);
    barsEqMbar = b.create<arith::ExtUIOp>(loc, writeBarsType, barsEqMbar);
    Value currWriteBars = b.create<arith::AndIOp>(loc, writeBars, barsEqMbar);
    Value currWriteBarsEqZero = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, currWriteBars, writeBarsZero);
    b.create<tti::ExperimentalAssertInThreadOp>(
        loc, currWriteBarsEqZero,
        "Barrier is being reused while still tracking writes", false);
    b.eraseOp(op);
    return success();
  }
};

struct StageAccessForCommitOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalStageAccessForCommitOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(tti::ExperimentalStageAccessForCommitOp op,
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
    RankedTensorType writeCommitsType =
        cast<RankedTensorType>(op.getOutstandingCommitsType());
    Value writeCommits =
        tti::createLoadScratchMemory(b, loc, op.getOutstandingCommits(),
                                     writeCommitsType)
            ->getResult(0);
    Value buf = createMemDescToI64(b, loc, getTypeConverter(),
                                   op.getBuf().getType(), adaptor.getBuf());

    // Gluon pseudo-code:
    // write_commits = tl.where(bufs == buf, -1, write_commits)

    auto buffersEqBuf = createCmpIntTensorScalar(b, loc, buffers, buf);
    auto writeCommitsMinusOne =
        tti::createConstIntTensor(b, loc, -1, writeCommitsType);
    writeCommits = b.create<arith::SelectOp>(
        loc, buffersEqBuf, writeCommitsMinusOne, writeCommits);
    tti::createStoreScratchMemory(b, loc, op.getOutstandingCommits(),
                                  writeCommits, writeCommitsType);
    b.eraseOp(op);
    return success();
  }
};

struct CommitAccessesOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalCommitAccessesOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(tti::ExperimentalCommitAccessesOp op,
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
    RankedTensorType writeCommitsType =
        cast<RankedTensorType>(op.getOutstandingCommitsType());
    Value writeCommits =
        tti::createLoadScratchMemory(b, loc, op.getOutstandingCommits(),
                                     writeCommitsType)
            ->getResult(0);

    // clang-format off
    // Gluon pseudo-code:
    // write_commits = tl.where(write_commits > 0, write_commits + 1, write_commits)
    // write_commits = tl.where(write_commits == -1, 1, write_commits)
    // clang-format on

    Type elementType = writeCommitsType.getElementType();
    Value minusOne = b.create<arith::ConstantOp>(
        loc, elementType, b.getIntegerAttr(elementType, -1));
    Value zero = b.create<arith::ConstantOp>(loc, elementType,
                                             b.getIntegerAttr(elementType, 0));
    Value writeCommitsOne =
        tti::createConstIntTensor(b, loc, 1, writeCommitsType);

    auto writeCommitsGtZero = createCmpIntTensorScalar(
        b, loc, writeCommits, zero, arith::CmpIPredicate::sgt);
    auto writeCommitsPlusOne =
        b.create<arith::AddIOp>(loc, writeCommits, writeCommitsOne);
    writeCommits = b.create<arith::SelectOp>(loc, writeCommitsGtZero,
                                             writeCommitsPlusOne, writeCommits);

    auto writeCommitsEqMinusOne =
        createCmpIntTensorScalar(b, loc, writeCommits, minusOne);
    writeCommits = b.create<arith::SelectOp>(loc, writeCommitsEqMinusOne,
                                             writeCommitsOne, writeCommits);
    tti::createStoreScratchMemory(b, loc, op.getOutstandingCommits(),
                                  writeCommits, writeCommitsType);
    b.eraseOp(op);
    return success();
  }
};

struct ClearOutstandingCommitsOpConversion
    : public ConvertOpToLLVMPattern<
          tti::ExperimentalClearOutstandingCommitsOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(tti::ExperimentalClearOutstandingCommitsOp op,
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
    RankedTensorType outstandingCommitsType =
        cast<RankedTensorType>(op.getOutstandingCommitsType());
    Value outstandingCommits =
        tti::createLoadScratchMemory(b, loc, op.getOutstandingCommits(),
                                     outstandingCommitsType)
            ->getResult(0);

    // clang-format off
    // Gluon pseudo-code:
    // outstanding_commits = tl.where(outstanding_commits > outstanding_num, 0, outstanding_commits)
    // clang-format on

    Type elementType = outstandingCommitsType.getElementType();
    Value outstandingNum = b.create<arith::ConstantOp>(
        loc, elementType,
        b.getIntegerAttr(elementType, op.getOutstandingNum()));
    Value outstandingCommitsZero =
        tti::createConstIntTensor(b, loc, 0, outstandingCommitsType);
    auto outstandingCommitsGtOutstandingNum = createCmpIntTensorScalar(
        b, loc, outstandingCommits, outstandingNum, arith::CmpIPredicate::sgt);
    outstandingCommits =
        b.create<arith::SelectOp>(loc, outstandingCommitsGtOutstandingNum,
                                  outstandingCommitsZero, outstandingCommits);
    tti::createStoreScratchMemory(b, loc, op.getOutstandingCommits(),
                                  outstandingCommits, outstandingCommitsType);
    b.eraseOp(op);
    return success();
  }
};

struct CheckOutstandingCommitsOpConversion
    : public ConvertOpToLLVMPattern<
          tti::ExperimentalCheckOutstandingCommitsOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(tti::ExperimentalCheckOutstandingCommitsOp op,
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
    RankedTensorType outstandingCommitsType =
        cast<RankedTensorType>(op.getOutstandingCommitsType());
    Value outstandingCommits =
        tti::createLoadScratchMemory(b, loc, op.getOutstandingCommits(),
                                     outstandingCommitsType)
            ->getResult(0);
    Value buf = createMemDescToI64(b, loc, getTypeConverter(),
                                   op.getBuf().getType(), adaptor.getBuf());
    StringRef pendingAccessType = op.getPendingAccessType();

    // clang-format off
    // Gluon pseudo-code:
    // curr_commits = tl.where(buf == buffers, outstanding_commits, 0)
    // tl.device_assert(curr_commits == 0, "Accessing buffer with pending access. Pending access type: ")
    // clang-format on

    Type elementType = outstandingCommitsType.getElementType();
    auto buffersEqBuf = createCmpIntTensorScalar(b, loc, buffers, buf);
    auto zero = b.create<arith::ConstantOp>(loc, elementType,
                                            b.getIntegerAttr(elementType, 0));
    auto outstandingCommitsZero =
        tti::createConstIntTensor(b, loc, 0, outstandingCommitsType);
    auto currCommits = b.create<arith::SelectOp>(
        loc, buffersEqBuf, outstandingCommits, outstandingCommitsZero);
    auto currCommitsEqZero =
        createCmpIntTensorScalar(b, loc, currCommits, zero);
    std::string message =
        "Accessing buffer with pending access. Pending access type: " +
        pendingAccessType.str();
    b.create<tti::ExperimentalAssertInThreadOp>(
        loc, currCommitsEqZero, b.getStringAttr(message), false);
    b.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::triton::populateInstrumentationToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<AssertInThreadOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<BufferPointersOpConversion>(typeConverter);
  patterns.add<SetWriteVisibilityOpConversion>(typeConverter);
  patterns.add<SetReadVisibilityOpConversion>(typeConverter);
  patterns.add<ClearWriteTrackingOpConversion>(typeConverter);
  patterns.add<ClearReadVisibilityOpConversion>(typeConverter);
  patterns.add<ClearReadTrackingOpConversion>(typeConverter);
  patterns.add<TrackVisibleWritesOpConversion>(typeConverter);
  patterns.add<TrackVisibleReadsOpConversion>(typeConverter);
  patterns.add<TransferVisibleWritesOpConversion>(typeConverter);
  patterns.add<TransferVisibleReadsOpConversion>(typeConverter);
  patterns.add<VerifyWriteVisibilityOpConversion>(typeConverter);
  patterns.add<VerifyReadVisibilityOpConversion>(typeConverter);
  patterns.add<SetWriteStateOpConversion>(typeConverter);
  patterns.add<CommitWriteWithBarrierOpConversion>(typeConverter);
  patterns.add<SetReadBarrierOpConversion>(typeConverter);
  patterns.add<ClearWriteBarrierOpConversion>(typeConverter);
  patterns.add<ClearReadBarrierOpConversion>(typeConverter);
  patterns.add<CheckBarrierWritesClearedOpConversion>(typeConverter);
  patterns.add<StageAccessForCommitOpConversion>(typeConverter);
  patterns.add<CommitAccessesOpConversion>(typeConverter);
  patterns.add<ClearOutstandingCommitsOpConversion>(typeConverter);
  patterns.add<CheckOutstandingCommitsOpConversion>(typeConverter);
}
