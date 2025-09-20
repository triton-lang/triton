#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "third_party/nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "third_party/nvidia/include/TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Utility.h"
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

Value createColumnMask(OpBuilder &b, Location loc, int column,
                       RankedTensorType tensorType) {
  Attribute encoding = tti::getSingleDimSliceEncoding(
      cast<ttg::BlockedEncodingAttr>(tensorType.getEncoding()), 1);
  Value columnMask =
      createOneHot(b, loc, tensorType.getShape()[1], column, encoding);
  return convertAndBroadcast(b, loc, columnMask, 0, tensorType);
}

Value createMultiColumnMask(OpBuilder &b, Location loc, uint64_t columnMask,
                            RankedTensorType tensorType) {
  RankedTensorType i1TensorType =
      cast<RankedTensorType>(tensorType.cloneWith(std::nullopt, b.getI1Type()));
  Value columnMaskVal = tti::createConstIntTensor(b, loc, 0, i1TensorType);
  for (int i = 0; i < 64; i++) {
    if (columnMask & (1ULL << i)) {
      columnMaskVal = b.create<arith::OrIOp>(
          loc, columnMaskVal, createColumnMask(b, loc, i, tensorType));
    }
  }
  return columnMaskVal;
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

struct LockAcquireOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalLockAcquireOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(tti::ExperimentalLockAcquireOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();
    b.setInsertionPoint(op);
    Value lock = op.getLock();

    Type elType = cast<PointerType>(lock.getType()).getPointeeType();
    assert(elType == b.getI32Type() && "Expected i32 lock element type");

    // Build: do { old = atom.global.acquire.cas.b32 [lock], 0, 1; } while (old
    // != 0);
    Block *prevBlock2 = b.getInsertionBlock();
    Block *whileBlock = b.splitBlock(prevBlock2, b.getInsertionPoint());
    Block *endBlock = b.splitBlock(whileBlock, whileBlock->begin());
    b.setInsertionPointToEnd(prevBlock2);
    Value elect = mlir::LLVM::NVIDIA::createElectPredicateWarp0(loc, b);
    if (op.getPred()) {
      elect = b.create<arith::AndIOp>(loc, elect, op.getPred());
    }
    b.create<LLVM::CondBrOp>(loc, elect, whileBlock, endBlock);

    b.setInsertionPointToEnd(whileBlock);

    auto i32 = b.getI32Type();
    Value zero =
        b.create<arith::ConstantOp>(loc, i32, b.getIntegerAttr(i32, 0));
    Value one = b.create<arith::ConstantOp>(loc, i32, b.getIntegerAttr(i32, 1));

    // Inline PTX CAS: old = atom.global.acquire.gpu.cas.b32 [lock], 0, 1
    // Use converted lock pointer from adaptor for addressing
    PTXBuilder ptx;
    auto *dstOpr = ptx.newOperand("=r", /*init=*/true);
    auto *ptrOpr = ptx.newAddrOperand(adaptor.getLock(), "l");
    auto *cmpOpr = ptx.newOperand(zero, "r");
    auto *valOpr = ptx.newOperand(one, "r");
    auto &atom = *ptx.create<PTXInstr>("atom");
    atom.global().o("acquire").o("gpu").o("cas").o("b32");
    atom(dstOpr, ptrOpr, cmpOpr, valOpr);
    Value old = ptx.launch(b, loc, i32);

    // while (old != 0) loop
    Value cond =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, old, zero);
    b.create<LLVM::CondBrOp>(loc, cond, whileBlock, endBlock);

    b.setInsertionPointToStart(endBlock);
    b.create<mlir::gpu::BarrierOp>(loc);
    b.eraseOp(op);
    return success();
  }
};

struct LockReleaseOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalLockReleaseOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(tti::ExperimentalLockReleaseOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();
    b.setInsertionPoint(op);
    Value lock = op.getLock();
    if (op.getPred()) {
      auto [prevBlock, ifBlock, thenBlock] =
          createIfBlock(b, loc, op.getPred());
      b.setInsertionPointToStart(ifBlock);
    }

    Type elType = cast<PointerType>(lock.getType()).getPointeeType();
    assert(elType == b.getI32Type() && "Expected i32 lock element type");

    b.create<mlir::gpu::BarrierOp>(loc);
    Value zero =
        b.create<arith::ConstantOp>(loc, elType, b.getIntegerAttr(elType, 0));
    b.create<triton::AtomicRMWOp>(loc, elType, RMWOp::XCHG, lock, zero, nullptr,
                                  MemSemantic::ACQUIRE_RELEASE,
                                  MemSyncScope::GPU);
    b.eraseOp(op);
    return success();
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
    Value threadBit = tti::createConstIntTensor(b, loc, op.getThreadMask(),
                                                writeVisibilityType);
    writeVisibility = b.create<arith::SelectOp>(loc, buffersEqBuf, threadBit,
                                                writeVisibility);

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
    Value threadColumnMask =
        createMultiColumnMask(b, loc, op.getThreadMask(), readVisibilityType);
    Value threadBit = tti::createConstIntTensor(b, loc, op.getThreadMask(),
                                                readVisibilityType);
    Value readVisibilityOrThreadBit =
        b.create<arith::OrIOp>(loc, readVisibility, threadBit);
    Value bufAndThread =
        b.create<arith::AndIOp>(loc, buffersEqBuf, threadColumnMask);
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
    Value threadColumnMask =
        createColumnMask(b, loc, op.getThread(), readVisibilityType);
    Value readVisibilityZero =
        tti::createConstIntTensor(b, loc, 0, readVisibilityType);
    Value visibleReads = b.create<arith::SelectOp>(
        loc, threadColumnMask, readVisibility, readVisibilityZero);
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
    RankedTensorType trackingBuffersType =
        cast<RankedTensorType>(trackingBuffers.getType());
    Value trackingBuffersOne =
        tti::createConstIntTensor(b, loc, 1, trackingBuffersType);
    trackingBuffers = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, trackingBuffers, trackingBuffersOne);
    Value threadMask = tti::createConstIntTensor(b, loc, op.getThreadMask(),
                                                 writeVisibilityType);
    Value writeVisibilityZero =
        tti::createConstIntTensor(b, loc, 0, writeVisibilityType);
    Value trackingThreadBit = b.create<arith::SelectOp>(
        loc, trackingBuffers, threadMask, writeVisibilityZero);
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
    Value threadColumnMask =
        createMultiColumnMask(b, loc, op.getThreadMask(), readVisibilityType);
    readVisibility = b.create<arith::SelectOp>(
        loc, threadColumnMask, readVisibilityOrTracking, readVisibility);
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
    std::string message = "Buffer being accessed has outstanding writes.";
    if (!op.getOperandName().str().empty()) {
      message += " Operand: " + op.getOperandName().str();
    }
    b.create<tti::ExperimentalAssertInThreadOp>(loc, writeVisible,
                                                b.getStringAttr(message),
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
    Value threadColumnMask =
        createColumnMask(b, loc, op.getThread(), readVisibilityType);
    Value threadBit = tti::createConstIntTensor(b, loc, 1ULL << op.getThread(),
                                                readVisibilityType);
    Value readVisibilityOrThreadBit =
        b.create<arith::OrIOp>(loc, readVisibility, threadBit);
    Value bufThreadVisibility = b.create<arith::SelectOp>(
        loc, threadColumnMask, bufVisibility, readVisibilityZero);
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
    buffersEqBuf =
        convertAndBroadcast(b, loc, buffersEqBuf, 1, writeCommitsType);
    Value threadColumnMask =
        createColumnMask(b, loc, op.getThread(), writeCommitsType);
    Value bufAndThread =
        b.create<arith::AndIOp>(loc, buffersEqBuf, threadColumnMask);
    auto writeCommitsMinusOne =
        tti::createConstIntTensor(b, loc, -1, writeCommitsType);
    writeCommits = b.create<arith::SelectOp>(
        loc, bufAndThread, writeCommitsMinusOne, writeCommits);
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

    Value threadColumnMask =
        createColumnMask(b, loc, op.getThread(), writeCommitsType);
    auto writeCommitsGtZero = createCmpIntTensorScalar(
        b, loc, writeCommits, zero, arith::CmpIPredicate::sgt);
    writeCommitsGtZero =
        b.create<arith::AndIOp>(loc, writeCommitsGtZero, threadColumnMask);
    auto writeCommitsPlusOne =
        b.create<arith::AddIOp>(loc, writeCommits, writeCommitsOne);
    writeCommits = b.create<arith::SelectOp>(loc, writeCommitsGtZero,
                                             writeCommitsPlusOne, writeCommits);

    auto writeCommitsEqMinusOne =
        createCmpIntTensorScalar(b, loc, writeCommits, minusOne);
    writeCommitsEqMinusOne =
        b.create<arith::AndIOp>(loc, writeCommitsEqMinusOne, threadColumnMask);
    writeCommits = b.create<arith::SelectOp>(loc, writeCommitsEqMinusOne,
                                             writeCommitsOne, writeCommits);
    tti::createStoreScratchMemory(b, loc, op.getOutstandingCommits(),
                                  writeCommits, writeCommitsType);
    b.eraseOp(op);
    return success();
  }
};

struct ClearOutstandingCommitsTransferWritesOpConversion
    : public ConvertOpToLLVMPattern<
          tti::ExperimentalClearOutstandingCommitsTransferWritesOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(tti::ExperimentalClearOutstandingCommitsTransferWritesOp op,
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
    RankedTensorType writeVisibilityType =
        cast<RankedTensorType>(op.getWriteVisibilityType());
    Value writeVisibility =
        tti::createLoadScratchMemory(b, loc, op.getWriteVisibility(),
                                     writeVisibilityType)
            ->getResult(0);

    Type elementType = outstandingCommitsType.getElementType();
    Value outstandingNum = b.create<arith::ConstantOp>(
        loc, elementType,
        b.getIntegerAttr(elementType, op.getOutstandingNum()));
    Value outstandingCommitsZero =
        tti::createConstIntTensor(b, loc, 0, outstandingCommitsType);
    Value threadColumnMask =
        createColumnMask(b, loc, op.getThread(), outstandingCommitsType);

    auto outstandingCommitsGtOutstandingNum = createCmpIntTensorScalar(
        b, loc, outstandingCommits, outstandingNum, arith::CmpIPredicate::sgt);
    outstandingCommitsGtOutstandingNum = b.create<arith::AndIOp>(
        loc, outstandingCommitsGtOutstandingNum, threadColumnMask);

    // Update write visibility rows: reduce per-thread mask to row mask,
    // and set the current thread bit only for rows where mask is true.
    Value rowMask =
        createOrReduce(b, loc, outstandingCommitsGtOutstandingNum, 1);
    // writeVisibilityType can be rank-1 (e.g., tensor<2xi64>), so do NOT
    // broadcast along dim=1. The select condition should match the row shape.
    Value transferThreadsBitsValue = tti::createConstIntTensor(
        b, loc, op.getTransferThreadMask(), writeVisibilityType);
    Value writeVisibilityOrThreadBit =
        b.create<arith::OrIOp>(loc, writeVisibility, transferThreadsBitsValue);
    writeVisibility = b.create<arith::SelectOp>(
        loc, rowMask, writeVisibilityOrThreadBit, writeVisibility);

    // Print
    // b.create<tt::PrintOp>(loc, "wv: ", false, writeVisibility,
    //                       std::vector<int32_t>{0});

    tti::createStoreScratchMemory(b, loc, op.getWriteVisibility(),
                                  writeVisibility, writeVisibilityType);

    // Clear outstanding commits entries
    outstandingCommits =
        b.create<arith::SelectOp>(loc, outstandingCommitsGtOutstandingNum,
                                  outstandingCommitsZero, outstandingCommits);
    tti::createStoreScratchMemory(b, loc, op.getOutstandingCommits(),
                                  outstandingCommits, outstandingCommitsType);
    b.eraseOp(op);
    return success();
  }
};

struct ClearOutstandingCommitsTransferReadsOpConversion
    : public ConvertOpToLLVMPattern<
          tti::ExperimentalClearOutstandingCommitsTransferReadsOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(tti::ExperimentalClearOutstandingCommitsTransferReadsOp op,
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
    RankedTensorType readVisibilityType =
        cast<RankedTensorType>(op.getReadVisibilityType());
    Value readVisibility =
        tti::createLoadScratchMemory(b, loc, op.getReadVisibility(),
                                     readVisibilityType)
            ->getResult(0);

    Type elementType = outstandingCommitsType.getElementType();
    Value outstandingNum = b.create<arith::ConstantOp>(
        loc, elementType,
        b.getIntegerAttr(elementType, op.getOutstandingNum()));
    Value outstandingCommitsZero =
        tti::createConstIntTensor(b, loc, 0, outstandingCommitsType);
    Value threadColumnMask =
        createColumnMask(b, loc, op.getThread(), outstandingCommitsType);
    auto outstandingCommitsGtOutstandingNum = createCmpIntTensorScalar(
        b, loc, outstandingCommits, outstandingNum, arith::CmpIPredicate::sgt);
    outstandingCommitsGtOutstandingNum = b.create<arith::AndIOp>(
        loc, outstandingCommitsGtOutstandingNum, threadColumnMask);

    // Update read visibility: set current thread bit for rows with mask
    Value rowMask =
        createOrReduce(b, loc, outstandingCommitsGtOutstandingNum, 1);
    rowMask = convertAndBroadcast(b, loc, rowMask, 1, readVisibilityType);
    Value transferThreadsBitsValue = tti::createConstIntTensor(
        b, loc, op.getTransferThreadMask(), readVisibilityType);
    Value readVisibilityOrThreadBit =
        b.create<arith::OrIOp>(loc, readVisibility, transferThreadsBitsValue);
    readVisibility = b.create<arith::SelectOp>(
        loc, rowMask, readVisibilityOrThreadBit, readVisibility);
    tti::createStoreScratchMemory(b, loc, op.getReadVisibility(),
                                  readVisibility, readVisibilityType);

    // Clear outstanding commits entries
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

    Type elementType = outstandingCommitsType.getElementType();
    // Select the buffer row across all threads and check it equals zero
    // (including -1 case).
    Value buffersEqBuf = createCmpIntTensorScalar(b, loc, buffers, buf);
    buffersEqBuf =
        convertAndBroadcast(b, loc, buffersEqBuf, 1, outstandingCommitsType);
    auto zeroTensor =
        tti::createConstIntTensor(b, loc, 0, outstandingCommitsType);
    auto selectedRows = b.create<arith::SelectOp>(
        loc, buffersEqBuf, outstandingCommits, zeroTensor);
    auto selectedEqZero = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                  selectedRows, zeroTensor);
    std::string message =
        "Accessing buffer with pending access. Pending access type: " +
        pendingAccessType.str();
    b.create<tti::ExperimentalAssertInThreadOp>(
        loc, selectedEqZero, b.getStringAttr(message), false);
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
  patterns.add<LockAcquireOpConversion>(typeConverter);
  patterns.add<LockReleaseOpConversion>(typeConverter);
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
  patterns.add<StageAccessForCommitOpConversion>(typeConverter);
  patterns.add<CommitAccessesOpConversion>(typeConverter);
  patterns.add<ClearOutstandingCommitsTransferWritesOpConversion>(
      typeConverter);
  patterns.add<ClearOutstandingCommitsTransferReadsOpConversion>(typeConverter);
  patterns.add<CheckOutstandingCommitsOpConversion>(typeConverter);
}
