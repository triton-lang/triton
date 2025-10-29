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
#include <limits>

namespace {

namespace tt = mlir::triton;
namespace ttg = tt::gpu;
namespace tti = mlir::triton::instrument;
namespace ttng = mlir::triton::nvidia_gpu;

////////////////////////////////////////////
// Utility functions
////////////////////////////////////////////

namespace BarrierBits {
constexpr unsigned phaseBit = 0;
constexpr unsigned initCountLsb = 1;
constexpr unsigned currentCountLsb = 9;
constexpr unsigned countBitWidth = 8;
constexpr unsigned countMask = (1u << countBitWidth) - 1;
} // namespace BarrierBits

namespace WaitingBits {
constexpr unsigned bitsPerThread = 2;
constexpr unsigned flagBit = 0;
constexpr unsigned phaseBit = 1;

constexpr uint32_t makeInterleavedMask(unsigned bit) {
  uint32_t mask = 0;
  for (unsigned i = 0; i < tti::NUM_THREADS; ++i)
    mask |= 1u << (bitsPerThread * i + bit);
  return mask;
}

constexpr uint32_t flagMask = makeInterleavedMask(flagBit);
constexpr uint32_t phaseMask = makeInterleavedMask(phaseBit);
} // namespace WaitingBits

static uint64_t expandActiveMask(uint64_t activeMask) {
  uint64_t expanded = 0;
  for (unsigned i = 0; i < tti::NUM_THREADS; ++i) {
    if (activeMask & (1ull << i))
      expanded |=
          1ull << (WaitingBits::bitsPerThread * i + WaitingBits::flagBit);
  }
  return expanded;
}

Value createFullLike(OpBuilder &builder, Location loc, Value scalar,
                     RankedTensorType tensorTy) {
  auto scalarTy = scalar.getType();
  auto elemTy = tensorTy.getElementType();
  assert(scalarTy == elemTy &&
         "Expected scalar to be of the same type as the tensor elements");
  return triton::SplatOp::create(builder, loc, tensorTy, scalar);
}

Value createCmpIntTensorScalar(
    OpBuilder &builder, Location loc, Value tensor, Value scalar,
    arith::CmpIPredicate predicate = arith::CmpIPredicate::eq) {
  auto tensorTy = cast<RankedTensorType>(tensor.getType());
  auto splat = createFullLike(builder, loc, scalar, tensorTy);
  auto cmp = arith::CmpIOp::create(builder, loc, predicate, tensor, splat);
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
  tensor = ttg::ConvertLayoutOp::create(
      b, loc, tensorType.cloneWithEncoding(slicedLayout), tensor);
  tensor = tti::expandOuterSlicedDim(b, loc, tensor);
  tensor = tt::BroadcastOp::create(b, loc, resultType, tensor);
  return tensor;
}

Value createConvertLayout(OpBuilder &b, Location loc, Value tensor,
                          Attribute encoding) {
  RankedTensorType dstType =
      cast<RankedTensorType>(tensor.getType()).cloneWithEncoding(encoding);
  return ttg::ConvertLayoutOp::create(b, loc, dstType, tensor);
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
  LLVM::BrOp::create(b, loc, thenBlock);
  b.setInsertionPointToEnd(prevBlock);
  LLVM::CondBrOp::create(b, loc, cnd, ifBlock, thenBlock);
  b.setInsertionPointToStart(thenBlock);

  return {prevBlock, ifBlock, thenBlock};
}

Value createOneHot(OpBuilder &b, Location loc, int size, int index,
                   Attribute encoding) {
  int start = 0;
  int end = size;
  RankedTensorType type =
      RankedTensorType::get({size}, b.getI32Type(), encoding);
  Value arange = tt::MakeRangeOp::create(b, loc, type, start, end);
  Value indexT = tti::createConstIntTensor(b, loc, index, type);
  return arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, arange,
                               indexT);
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
      columnMaskVal = arith::OrIOp::create(
          b, loc, columnMaskVal, createColumnMask(b, loc, i, tensorType));
    }
  }
  return columnMaskVal;
}

Value createOrReduce(OpBuilder &b, Location loc, Value tensor, int axis) {
  OpBuilder::InsertionGuard guard(b);
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  auto reduceOp =
      tt::ReduceOp::create(b, loc, std::vector<Value>{tensor}, axis);
  auto &region = reduceOp.getRegion();
  auto &block = region.emplaceBlock();
  block.addArguments({tensorType.getElementType(), tensorType.getElementType()},
                     {loc, loc});
  auto arg0 = block.getArgument(0);
  auto arg1 = block.getArgument(1);
  b.setInsertionPointToStart(&block);
  auto result = arith::OrIOp::create(b, loc, arg0, arg1);
  auto returnOp =
      tt::ReduceReturnOp::create(b, loc, std::vector<Value>{result});
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
    Value zero = LLVM::ConstantOp::create(rewriter, loc, condTy,
                                          rewriter.getZeroAttr(condTy));
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
      base = nvgpu::TensorMemoryBaseAddress::create(rewriter, loc);
      base = b.ptrtoint(i32_ty, base);
    }
    bufPointers = arith::AddIOp::create(
        rewriter, loc, bufPointers,
        triton::SplatOp::create(rewriter, loc, bufPointers.getType(), base));
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
    return arith::ConstantOp::create(builder, loc, tensorType, denseAttr);
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
      elect = arith::AndIOp::create(b, loc, elect, op.getPred());
    }
    LLVM::CondBrOp::create(b, loc, elect, whileBlock, endBlock);

    b.setInsertionPointToEnd(whileBlock);

    auto i32 = b.getI32Type();
    Value zero =
        arith::ConstantOp::create(b, loc, i32, b.getIntegerAttr(i32, 0));
    Value one =
        arith::ConstantOp::create(b, loc, i32, b.getIntegerAttr(i32, 1));

    // Inline PTX CAS: old = atom.global.acquire.gpu.cas.b32 [lock], 0, 1
    // Use converted lock pointer from adaptor for addressing
    PTXBuilder ptx;
    auto *dstOpr = ptx.newOperand("=r", /*init=*/true);
    auto *ptrOpr = ptx.newAddrOperand(adaptor.getLock(), "l");
    auto *cmpOpr = ptx.newOperand(zero, "r");
    auto *valOpr = ptx.newOperand(one, "r");
    auto &atom = *ptx.create("atom");
    atom.global().o("acquire").o("gpu").o("cas").o("b32");
    atom(dstOpr, ptrOpr, cmpOpr, valOpr);
    Value old = ptx.launch(b, loc, i32);

    // while (old != 0) loop
    Value cond =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne, old, zero);
    LLVM::CondBrOp::create(b, loc, cond, whileBlock, endBlock);

    b.setInsertionPointToStart(endBlock);
    mlir::gpu::BarrierOp::create(b, loc);
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

    mlir::gpu::BarrierOp::create(b, loc);
    Value zero =
        arith::ConstantOp::create(b, loc, elType, b.getIntegerAttr(elType, 0));
    triton::AtomicRMWOp::create(b, loc, elType, RMWOp::XCHG, lock, zero,
                                nullptr, MemSemantic::ACQUIRE_RELEASE,
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
    writeVisibility = arith::SelectOp::create(b, loc, buffersEqBuf, threadBit,
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
        arith::OrIOp::create(b, loc, readVisibility, threadBit);
    Value bufAndThread =
        arith::AndIOp::create(b, loc, buffersEqBuf, threadColumnMask);
    readVisibility = arith::SelectOp::create(
        b, loc, bufAndThread, readVisibilityOrThreadBit, readVisibility);
    tti::createStoreScratchMemory(b, loc, op.getReadVisibility(),
                                  readVisibility, readVisibilityType);
    b.eraseOp(op);
    return success();
  }
};

struct CopyWriteVisibilityOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalCopyWriteVisibilityOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(tti::ExperimentalCopyWriteVisibilityOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();
    b.setInsertionPoint(op);
    if (op.getPred()) {
      auto [prevBlock, ifBlock, thenBlock] =
          createIfBlock(b, loc, op.getPred());
      b.setInsertionPointToStart(ifBlock);
    }

    RankedTensorType writeVisibilityType =
        cast<RankedTensorType>(op.getWriteVisibilityType());
    Value writeVisibility =
        tti::createLoadScratchMemory(b, loc, op.getWriteVisibility(),
                                     writeVisibilityType)
            ->getResult(0);

    constexpr uint64_t fullMask =
        tti::THREADS_BITMASK_SIZE >= 64
            ? std::numeric_limits<uint64_t>::max()
            : ((1ull << tti::THREADS_BITMASK_SIZE) - 1);
    uint64_t destMaskVal = static_cast<uint64_t>(op.getDestMask());
    uint64_t clearMaskVal = fullMask & ~destMaskVal;

    Value destMaskTensor = tti::createConstIntTensor(
        b, loc, static_cast<int64_t>(destMaskVal), writeVisibilityType);
    Value clearMaskTensor = tti::createConstIntTensor(
        b, loc, static_cast<int64_t>(clearMaskVal), writeVisibilityType);
    Value cleared =
        arith::AndIOp::create(b, loc, writeVisibility, clearMaskTensor);

    uint64_t sourceMaskVal = 1ull << op.getSourceThread();
    Value sourceMaskTensor = tti::createConstIntTensor(
        b, loc, static_cast<int64_t>(sourceMaskVal), writeVisibilityType);
    Value zeroTensor =
        tti::createConstIntTensor(b, loc, 0, writeVisibilityType);
    Value sourceBits =
        arith::AndIOp::create(b, loc, writeVisibility, sourceMaskTensor);
    Value sourceIsSet = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne,
                                              sourceBits, zeroTensor);
    Value replicated = arith::SelectOp::create(b, loc, sourceIsSet,
                                               destMaskTensor, zeroTensor);

    Value updated = arith::OrIOp::create(b, loc, cleared, replicated);
    tti::createStoreScratchMemory(b, loc, op.getWriteVisibility(), updated,
                                  writeVisibilityType);
    b.eraseOp(op);
    return success();
  }
};

struct CopyReadVisibilityOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalCopyReadVisibilityOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(tti::ExperimentalCopyReadVisibilityOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();
    b.setInsertionPoint(op);
    if (op.getPred()) {
      auto [prevBlock, ifBlock, thenBlock] =
          createIfBlock(b, loc, op.getPred());
      b.setInsertionPointToStart(ifBlock);
    }

    RankedTensorType readVisibilityType =
        cast<RankedTensorType>(op.getReadVisibilityType());
    Value readVisibility =
        tti::createLoadScratchMemory(b, loc, op.getReadVisibility(),
                                     readVisibilityType)
            ->getResult(0);

    Value zeroTensor = tti::createConstIntTensor(b, loc, 0, readVisibilityType);
    Value destMaskTensor =
        createMultiColumnMask(b, loc, op.getDestMask(), readVisibilityType);

    Value cleared = arith::SelectOp::create(b, loc, destMaskTensor, zeroTensor,
                                            readVisibility);

    Value sourceColumnMask =
        createColumnMask(b, loc, op.getSourceThread(), readVisibilityType);
    Value sourceColumn = arith::SelectOp::create(b, loc, sourceColumnMask,
                                                 readVisibility, zeroTensor);
    Value sourceVector = createOrReduce(b, loc, sourceColumn, /*axis=*/1);
    Value broadcastRow =
        convertAndBroadcast(b, loc, sourceVector, 1, readVisibilityType);
    Value replicated = arith::SelectOp::create(b, loc, destMaskTensor,
                                               broadcastRow, zeroTensor);

    Value updated = arith::OrIOp::create(b, loc, cleared, replicated);
    tti::createStoreScratchMemory(b, loc, op.getReadVisibility(), updated,
                                  readVisibilityType);
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
        arith::SelectOp::create(b, loc, buffersEqBuf, zero, writeTracking);
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
        arith::SelectOp::create(b, loc, buffersEqBuf, zero, readVisibility);
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
        arith::SelectOp::create(b, loc, buffersEqBuf, zero, readTracking);
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
        arith::AndIOp::create(b, loc, writeVisibility, threadBit);
    visibleWrites = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                                          visibleWrites, threadBit);
    visibleWrites =
        convertAndBroadcast(b, loc, visibleWrites, 1, writeTrackingType);
    Value barAndVisible =
        arith::AndIOp::create(b, loc, barriersEqBar, visibleWrites);
    Value writeTrackingOne =
        tti::createConstIntTensor(b, loc, 1, writeTrackingType);
    writeTracking = arith::SelectOp::create(b, loc, barAndVisible,
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
    Value visibleReads = arith::SelectOp::create(
        b, loc, threadColumnMask, readVisibility, readVisibilityZero);
    visibleReads = createOrReduce(b, loc, visibleReads, 1);
    visibleReads =
        convertAndBroadcast(b, loc, visibleReads, 1, readTrackingType);
    Value readTrackingOrVisible =
        arith::OrIOp::create(b, loc, readTracking, visibleReads);
    readTracking = arith::SelectOp::create(b, loc, barriersEqBar,
                                           readTrackingOrVisible, readTracking);

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
    Value trackingBuffers = arith::SelectOp::create(
        b, loc, barriersEqBar, writeTracking, writeTrackingZero);
    trackingBuffers = createOrReduce(b, loc, trackingBuffers, 1);
    trackingBuffers = createConvertLayout(b, loc, trackingBuffers,
                                          writeVisibilityType.getEncoding());
    RankedTensorType trackingBuffersType =
        cast<RankedTensorType>(trackingBuffers.getType());
    Value trackingBuffersOne =
        tti::createConstIntTensor(b, loc, 1, trackingBuffersType);
    trackingBuffers = arith::CmpIOp::create(
        b, loc, arith::CmpIPredicate::eq, trackingBuffers, trackingBuffersOne);
    Value threadMask = tti::createConstIntTensor(b, loc, op.getThreadMask(),
                                                 writeVisibilityType);
    Value writeVisibilityZero =
        tti::createConstIntTensor(b, loc, 0, writeVisibilityType);
    Value trackingThreadBit = arith::SelectOp::create(
        b, loc, trackingBuffers, threadMask, writeVisibilityZero);
    writeVisibility =
        arith::OrIOp::create(b, loc, writeVisibility, trackingThreadBit);
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
    Value trackingBar = arith::SelectOp::create(b, loc, barriersEqBar,
                                                readTracking, readTrackingZero);
    trackingBar = createOrReduce(b, loc, trackingBar, 1);
    trackingBar =
        convertAndBroadcast(b, loc, trackingBar, 1, readVisibilityType);
    Value readVisibilityOrTracking =
        arith::OrIOp::create(b, loc, readVisibility, trackingBar);
    Value threadColumnMask =
        createMultiColumnMask(b, loc, op.getThreadMask(), readVisibilityType);
    readVisibility = arith::SelectOp::create(
        b, loc, threadColumnMask, readVisibilityOrTracking, readVisibility);
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
    Value bufVisibility = arith::SelectOp::create(
        b, loc, buffersEqBuf, writeVisibility, writeVisibilityZero);
    Value noOneIsWriting = arith::CmpIOp::create(
        b, loc, arith::CmpIPredicate::eq, bufVisibility, writeVisibilityZero);
    Value thread =
        tti::createConstIntTensor(b, loc, op.getThread(), writeVisibilityType);
    buffersEqBuf =
        arith::ExtUIOp::create(b, loc, writeVisibilityType, buffersEqBuf);
    Value bufferThreadBit = arith::ShLIOp::create(b, loc, buffersEqBuf, thread);
    Value bufferHasVisibility =
        arith::AndIOp::create(b, loc, bufVisibility, bufferThreadBit);
    bufferHasVisibility = arith::CmpIOp::create(
        b, loc, arith::CmpIPredicate::eq, bufferHasVisibility, bufferThreadBit);
    Value writeVisible =
        arith::OrIOp::create(b, loc, noOneIsWriting, bufferHasVisibility);

    std::string message = "Buffer being accessed has outstanding writes.";
    if (!op.getOperandName().str().empty()) {
      message += " Operand: " + op.getOperandName().str();
    }
    tti::ExperimentalAssertInThreadOp::create(b, loc, writeVisible,
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
    Value bufVisibility = arith::SelectOp::create(
        b, loc, buffersEqBuf, readVisibility, readVisibilityZero);
    Value totalVisibility = createOrReduce(b, loc, bufVisibility, 1);
    Value threadColumnMask =
        createColumnMask(b, loc, op.getThread(), readVisibilityType);
    Value threadBit = tti::createConstIntTensor(b, loc, 1ULL << op.getThread(),
                                                readVisibilityType);
    Value readVisibilityOrThreadBit =
        arith::OrIOp::create(b, loc, readVisibility, threadBit);
    Value bufThreadVisibility = arith::SelectOp::create(
        b, loc, threadColumnMask, bufVisibility, readVisibilityZero);
    bufThreadVisibility = createOrReduce(b, loc, bufThreadVisibility, 1);
    // Thread must have visivility that is a superset of read visibility of all
    // other threads
    Value threadAndTotalVisibility =
        arith::AndIOp::create(b, loc, bufThreadVisibility, totalVisibility);
    Value hasVisibility =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                              threadAndTotalVisibility, totalVisibility);
    tti::ExperimentalAssertInThreadOp::create(
        b, loc, hasVisibility, "Buffer being accessed has outstanding reads",
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
        arith::AndIOp::create(b, loc, buffersEqBuf, threadColumnMask);
    auto writeCommitsMinusOne =
        tti::createConstIntTensor(b, loc, -1, writeCommitsType, true);
    writeCommits = arith::SelectOp::create(b, loc, bufAndThread,
                                           writeCommitsMinusOne, writeCommits);
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
    Value minusOne = arith::ConstantOp::create(
        b, loc, elementType, b.getIntegerAttr(elementType, -1));
    Value zero = arith::ConstantOp::create(b, loc, elementType,
                                           b.getIntegerAttr(elementType, 0));
    Value writeCommitsOne =
        tti::createConstIntTensor(b, loc, 1, writeCommitsType);

    Value threadColumnMask =
        createColumnMask(b, loc, op.getThread(), writeCommitsType);
    auto writeCommitsGtZero = createCmpIntTensorScalar(
        b, loc, writeCommits, zero, arith::CmpIPredicate::sgt);
    writeCommitsGtZero =
        arith::AndIOp::create(b, loc, writeCommitsGtZero, threadColumnMask);
    auto writeCommitsPlusOne =
        arith::AddIOp::create(b, loc, writeCommits, writeCommitsOne);
    writeCommits = arith::SelectOp::create(b, loc, writeCommitsGtZero,
                                           writeCommitsPlusOne, writeCommits);

    auto writeCommitsEqMinusOne =
        createCmpIntTensorScalar(b, loc, writeCommits, minusOne);
    writeCommitsEqMinusOne =
        arith::AndIOp::create(b, loc, writeCommitsEqMinusOne, threadColumnMask);
    writeCommits = arith::SelectOp::create(b, loc, writeCommitsEqMinusOne,
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
    Value outstandingNum = arith::ConstantOp::create(
        b, loc, elementType,
        b.getIntegerAttr(elementType, op.getOutstandingNum()));
    Value outstandingCommitsZero =
        tti::createConstIntTensor(b, loc, 0, outstandingCommitsType);
    Value threadColumnMask =
        createColumnMask(b, loc, op.getThread(), outstandingCommitsType);

    auto outstandingCommitsGtOutstandingNum = createCmpIntTensorScalar(
        b, loc, outstandingCommits, outstandingNum, arith::CmpIPredicate::sgt);
    outstandingCommitsGtOutstandingNum = arith::AndIOp::create(
        b, loc, outstandingCommitsGtOutstandingNum, threadColumnMask);

    // Update write visibility rows: reduce per-thread mask to row mask,
    // and set the current thread bit only for rows where mask is true.
    Value rowMask =
        createOrReduce(b, loc, outstandingCommitsGtOutstandingNum, 1);
    // writeVisibilityType can be rank-1 (e.g., tensor<2xi64>), so do NOT
    // broadcast along dim=1. The select condition should match the row shape.
    Value transferThreadsBitsValue = tti::createConstIntTensor(
        b, loc, op.getTransferThreadMask(), writeVisibilityType);
    Value writeVisibilityOrThreadBit =
        arith::OrIOp::create(b, loc, writeVisibility, transferThreadsBitsValue);
    writeVisibility = arith::SelectOp::create(
        b, loc, rowMask, writeVisibilityOrThreadBit, writeVisibility);

    // Print
    // tt::PrintOp::create(b, loc, "wv: ", false, writeVisibility,
    //                       std::vector<int32_t>{0});

    tti::createStoreScratchMemory(b, loc, op.getWriteVisibility(),
                                  writeVisibility, writeVisibilityType);

    // Clear outstanding commits entries
    outstandingCommits =
        arith::SelectOp::create(b, loc, outstandingCommitsGtOutstandingNum,
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
    Value outstandingNum = arith::ConstantOp::create(
        b, loc, elementType,
        b.getIntegerAttr(elementType, op.getOutstandingNum()));
    Value outstandingCommitsZero =
        tti::createConstIntTensor(b, loc, 0, outstandingCommitsType);
    Value threadColumnMask =
        createColumnMask(b, loc, op.getThread(), outstandingCommitsType);
    auto outstandingCommitsGtOutstandingNum = createCmpIntTensorScalar(
        b, loc, outstandingCommits, outstandingNum, arith::CmpIPredicate::sgt);
    outstandingCommitsGtOutstandingNum = arith::AndIOp::create(
        b, loc, outstandingCommitsGtOutstandingNum, threadColumnMask);

    // Update read visibility: set current thread bit for rows with mask
    Value rowMask =
        createOrReduce(b, loc, outstandingCommitsGtOutstandingNum, 1);
    rowMask = convertAndBroadcast(b, loc, rowMask, 1, readVisibilityType);
    Value transferThreadsBitsValue = tti::createConstIntTensor(
        b, loc, op.getTransferThreadMask(), readVisibilityType);
    Value readVisibilityOrThreadBit =
        arith::OrIOp::create(b, loc, readVisibility, transferThreadsBitsValue);
    readVisibility = arith::SelectOp::create(
        b, loc, rowMask, readVisibilityOrThreadBit, readVisibility);
    tti::createStoreScratchMemory(b, loc, op.getReadVisibility(),
                                  readVisibility, readVisibilityType);

    // Clear outstanding commits entries
    outstandingCommits =
        arith::SelectOp::create(b, loc, outstandingCommitsGtOutstandingNum,
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
    auto selectedRows = arith::SelectOp::create(b, loc, buffersEqBuf,
                                                outstandingCommits, zeroTensor);
    auto selectedEqZero = arith::CmpIOp::create(
        b, loc, arith::CmpIPredicate::eq, selectedRows, zeroTensor);
    std::string message =
        "Accessing buffer with pending access. Pending access type: " +
        pendingAccessType.str();
    tti::ExperimentalAssertInThreadOp::create(b, loc, selectedEqZero,
                                              b.getStringAttr(message), false);
    b.eraseOp(op);
    return success();
  }
};

struct InitBarrierStateOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalInitBarrierStateOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(tti::ExperimentalInitBarrierStateOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPoint(op);

    auto statesType = cast<RankedTensorType>(op.getStatesType());
    Value states =
        tti::createLoadScratchMemory(b, loc, op.getStates(), statesType)
            ->getResult(0);
    TypedValue<RankedTensorType> barriers = op.getBarriers();
    Value bar = createMemDescToI64(b, loc, getTypeConverter(),
                                   op.getMbar().getType(), adaptor.getMbar());
    Value mask = createCmpIntTensorScalar(b, loc, barriers, bar);

    uint64_t count = op.getCount() & BarrierBits::countMask;
    uint64_t init = (count << BarrierBits::initCountLsb) |
                    (count << BarrierBits::currentCountLsb);
    Value newState = tti::createConstIntTensor(b, loc, init, statesType);
    Value updated = arith::SelectOp::create(b, loc, mask, newState, states);
    tti::createStoreScratchMemory(b, loc, op.getStates(), updated, statesType);
    b.eraseOp(op);
    return success();
  }
};

struct VerifyBarrierArriveOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalVerifyBarrierArriveOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(tti::ExperimentalVerifyBarrierArriveOp op,
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

    auto statesType = cast<RankedTensorType>(op.getStatesType());
    Value states =
        tti::createLoadScratchMemory(b, loc, op.getStates(), statesType)
            ->getResult(0);
    TypedValue<RankedTensorType> barriers = op.getBarriers();
    Value bar = createMemDescToI64(b, loc, getTypeConverter(),
                                   op.getMbar().getType(), adaptor.getMbar());
    Value mask = createCmpIntTensorScalar(b, loc, barriers, bar);

    Value zero32 = tti::createConstIntTensor(b, loc, 0, statesType);
    Value maskFF =
        tti::createConstIntTensor(b, loc, BarrierBits::countMask, statesType);
    Value shiftNine = tti::createConstIntTensor(
        b, loc, BarrierBits::currentCountLsb, statesType);

    Value currentCount = arith::ShRUIOp::create(b, loc, states, shiftNine);
    currentCount = arith::AndIOp::create(b, loc, currentCount, maskFF);

    Value arriveCount = tti::createConstIntTensor(
        b, loc, op.getCount() & BarrierBits::countMask, statesType);
    Value newCurrent = arith::SubIOp::create(b, loc, currentCount, arriveCount);
    Value newCurrentMasked =
        arith::SelectOp::create(b, loc, mask, newCurrent, zero32);
    Value nonNegative = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge,
                                              newCurrentMasked, zero32);
    tti::ExperimentalAssertInThreadOp::create(
        b, loc, nonNegative,
        b.getStringAttr(
            "Barrier arrive underflow: current count would become negative"),
        false);

    b.eraseOp(op);
    return success();
  }
};

struct UpdateBarrierStateOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalUpdateBarrierStateOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(tti::ExperimentalUpdateBarrierStateOp op,
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

    auto statesType = cast<RankedTensorType>(op.getStatesType());
    Value states =
        tti::createLoadScratchMemory(b, loc, op.getStates(), statesType)
            ->getResult(0);
    TypedValue<RankedTensorType> barriers = op.getBarriers();
    Value bar = createMemDescToI64(b, loc, getTypeConverter(),
                                   op.getMbar().getType(), adaptor.getMbar());
    Value mask = createCmpIntTensorScalar(b, loc, barriers, bar);
    Value maskI32 = arith::ExtUIOp::create(b, loc, statesType, mask);

    Value zero32 = tti::createConstIntTensor(b, loc, 0, statesType);
    Value one32 = tti::createConstIntTensor(b, loc, 1, statesType);
    Value maskFF =
        tti::createConstIntTensor(b, loc, BarrierBits::countMask, statesType);
    Value shiftOne = tti::createConstIntTensor(
        b, loc, BarrierBits::initCountLsb, statesType);
    Value shiftNine = tti::createConstIntTensor(
        b, loc, BarrierBits::currentCountLsb, statesType);

    Value phase = arith::AndIOp::create(b, loc, states, one32);
    Value initCount = arith::ShRUIOp::create(b, loc, states, shiftOne);
    initCount = arith::AndIOp::create(b, loc, initCount, maskFF);
    Value currentCount = arith::ShRUIOp::create(b, loc, states, shiftNine);
    currentCount = arith::AndIOp::create(b, loc, currentCount, maskFF);

    Value arriveCount = tti::createConstIntTensor(
        b, loc, op.getCount() & BarrierBits::countMask, statesType);
    Value newCurrent = arith::SubIOp::create(b, loc, currentCount, arriveCount);
    Value newCurrentMasked =
        arith::SelectOp::create(b, loc, mask, newCurrent, currentCount);

    Value zeroCond = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                                           newCurrentMasked, zero32);
    zeroCond = arith::AndIOp::create(b, loc, zeroCond, mask);
    Value zeroCondI32 = arith::ExtUIOp::create(b, loc, statesType, zeroCond);
    Value newPhase = arith::XOrIOp::create(b, loc, phase, zeroCondI32);
    Value newCurrentValue =
        arith::SelectOp::create(b, loc, zeroCond, initCount, newCurrentMasked);

    Value initField = arith::ShLIOp::create(b, loc, initCount, shiftOne);
    Value currentField =
        arith::ShLIOp::create(b, loc, newCurrentValue, shiftNine);
    Value newState = arith::OrIOp::create(b, loc, newPhase, initField);
    newState = arith::OrIOp::create(b, loc, newState, currentField);
    Value updated = arith::SelectOp::create(b, loc, mask, newState, states);
    tti::createStoreScratchMemory(b, loc, op.getStates(), updated, statesType);
    b.eraseOp(op);
    return success();
  }
};

struct SetWaitingOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalSetWaitingOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(tti::ExperimentalSetWaitingOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();
    b.setInsertionPoint(op);
    if (op.getPred()) {
      auto [prevBlock, ifBlock, thenBlock] =
          createIfBlock(b, loc, op.getPred());
      b.setInsertionPointToStart(ifBlock);
    }

    auto waitingType = cast<RankedTensorType>(op.getWaitingType());
    Value waiting =
        tti::createLoadScratchMemory(b, loc, op.getWaiting(), waitingType)
            ->getResult(0);
    TypedValue<RankedTensorType> barriers = op.getBarriers();
    Value bar = createMemDescToI64(b, loc, getTypeConverter(),
                                   op.getMbar().getType(), adaptor.getMbar());
    Value barriersEqBar = createCmpIntTensorScalar(b, loc, barriers, bar);

    unsigned baseThread = op.getBaseThread();
    unsigned flagShift =
        WaitingBits::bitsPerThread * baseThread + WaitingBits::flagBit;
    unsigned phaseShift =
        WaitingBits::bitsPerThread * baseThread + WaitingBits::phaseBit;
    uint32_t flagMask = 1u << flagShift;
    uint32_t phaseMask = 1u << phaseShift;
    uint32_t clearMask = ~(flagMask | phaseMask);

    Value clearMaskTensor =
        tti::createConstIntTensor(b, loc, clearMask, waitingType);
    Value clearedWaiting =
        arith::AndIOp::create(b, loc, waiting, clearMaskTensor);

    Value flagMaskTensor =
        tti::createConstIntTensor(b, loc, flagMask, waitingType);
    Value withFlag =
        arith::OrIOp::create(b, loc, clearedWaiting, flagMaskTensor);

    Value phaseScalar = arith::AndIOp::create(
        b, loc, op.getPhase(), arith::ConstantIntOp::create(b, loc, 1, 32));
    Value phaseTensor =
        triton::SplatOp::create(b, loc, waitingType, phaseScalar);
    Value shiftTensor =
        tti::createConstIntTensor(b, loc, phaseShift, waitingType);
    Value phaseBits = arith::ShLIOp::create(b, loc, phaseTensor, shiftTensor);
    Value pendingWaiting = arith::OrIOp::create(b, loc, withFlag, phaseBits);

    Value newWaiting =
        arith::SelectOp::create(b, loc, barriersEqBar, pendingWaiting, waiting);
    tti::createStoreScratchMemory(b, loc, op.getWaiting(), newWaiting,
                                  waitingType);
    b.eraseOp(op);
    return success();
  }
};

struct CheckAllActiveWaitingOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalCheckAllActiveWaitingOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(tti::ExperimentalCheckAllActiveWaitingOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();
    b.setInsertionPoint(op);
    if (op.getPred()) {
      auto [prevBlock, ifBlock, thenBlock] =
          createIfBlock(b, loc, op.getPred());
      b.setInsertionPointToStart(ifBlock);
    }

    auto waitingType = cast<RankedTensorType>(op.getWaitingType());
    auto barrierStatesType = cast<RankedTensorType>(op.getBarrierStatesType());
    Value waiting =
        tti::createLoadScratchMemory(b, loc, op.getWaiting(), waitingType)
            ->getResult(0);
    Value barrierStates = tti::createLoadScratchMemory(
                              b, loc, op.getBarrierStates(), barrierStatesType)
                              ->getResult(0);

    Value flagMaskTensor =
        tti::createConstIntTensor(b, loc, WaitingBits::flagMask, waitingType);
    Value phaseMaskTensor =
        tti::createConstIntTensor(b, loc, WaitingBits::phaseMask, waitingType);

    Value flags = arith::AndIOp::create(b, loc, waiting, flagMaskTensor);
    Value phases = arith::AndIOp::create(b, loc, waiting, phaseMaskTensor);
    Value shiftOneTensor = tti::createConstIntTensor(b, loc, 1, waitingType);
    Value phasesAligned =
        arith::ShRUIOp::create(b, loc, phases, shiftOneTensor);

    Value phasesComplement =
        arith::XOrIOp::create(b, loc, phasesAligned, flagMaskTensor);
    Value waitingPhase0 =
        arith::AndIOp::create(b, loc, flags, phasesComplement);
    Value waitingPhase1 = arith::AndIOp::create(b, loc, flags, phasesAligned);

    Value oneState = tti::createConstIntTensor(b, loc, 1, barrierStatesType);
    Value barrierPhase = arith::AndIOp::create(b, loc, barrierStates, oneState);
    Value phaseIsOne = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                                             barrierPhase, oneState);

    Value effectiveWaiting = arith::SelectOp::create(
        b, loc, phaseIsOne, waitingPhase1, waitingPhase0);
    Value waitingOr = createOrReduce(b, loc, effectiveWaiting, /*axis=*/0);

    uint64_t expandedMaskValue = expandActiveMask(op.getActiveMask());
    Value expandedMask = arith::ConstantIntOp::create(
        b, loc, expandedMaskValue, waitingOr.getType().getIntOrFloatBitWidth());
    Value waitingMasked =
        arith::AndIOp::create(b, loc, waitingOr, expandedMask);
    Value eq = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                                     waitingMasked, expandedMask);
    Value vTrue = arith::ConstantOp::create(b, loc, eq.getType(),
                                            b.getIntegerAttr(eq.getType(), 1));
    Value ok = arith::XOrIOp::create(b, loc, eq, vTrue);
    tti::ExperimentalAssertInThreadOp::create(
        b, loc, ok,
        b.getStringAttr(
            "Deadlock detected: all active threads are waiting on mbarriers"),
        false);
    b.eraseOp(op);
    return success();
  }
};

struct ClearWaitingOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalClearWaitingOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(tti::ExperimentalClearWaitingOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();
    b.setInsertionPoint(op);
    if (op.getPred()) {
      auto [prevBlock, ifBlock, thenBlock] =
          createIfBlock(b, loc, op.getPred());
      b.setInsertionPointToStart(ifBlock);
    }

    auto waitingType = cast<RankedTensorType>(op.getWaitingType());
    Value waiting =
        tti::createLoadScratchMemory(b, loc, op.getWaiting(), waitingType)
            ->getResult(0);
    TypedValue<RankedTensorType> barriers = op.getBarriers();
    Value bar = createMemDescToI64(b, loc, getTypeConverter(),
                                   op.getMbar().getType(), adaptor.getMbar());
    Value barriersEqBar = createCmpIntTensorScalar(b, loc, barriers, bar);

    unsigned baseThread = op.getBaseThread();
    unsigned flagShift =
        WaitingBits::bitsPerThread * baseThread + WaitingBits::flagBit;
    unsigned phaseShift =
        WaitingBits::bitsPerThread * baseThread + WaitingBits::phaseBit;
    uint32_t clearMask = ~((1u << flagShift) | (1u << phaseShift));

    Value clearMaskTensor =
        tti::createConstIntTensor(b, loc, clearMask, waitingType);
    Value cleared = arith::AndIOp::create(b, loc, waiting, clearMaskTensor);
    Value newWaiting =
        arith::SelectOp::create(b, loc, barriersEqBar, cleared, waiting);
    tti::createStoreScratchMemory(b, loc, op.getWaiting(), newWaiting,
                                  waitingType);
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
  patterns.add<CopyWriteVisibilityOpConversion>(typeConverter);
  patterns.add<CopyReadVisibilityOpConversion>(typeConverter);
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
  patterns.add<InitBarrierStateOpConversion>(typeConverter);
  patterns.add<VerifyBarrierArriveOpConversion>(typeConverter);
  patterns.add<UpdateBarrierStateOpConversion>(typeConverter);
  // Deadlock detection
  patterns.add<SetWaitingOpConversion>(typeConverter);
  patterns.add<CheckAllActiveWaitingOpConversion>(typeConverter);
  patterns.add<ClearWaitingOpConversion>(typeConverter);
}
