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

Value createMemDescToI32(RewriterBase &rewriter, Location loc,
                         const LLVMTypeConverter *typeConverter,
                         ttg::MemDescType memDescTy, Value sharedMemStruct) {
  TritonLLVMOpBuilder b(loc, rewriter);
  auto i32Ty = rewriter.getIntegerType(32);
  if (isa<ttng::TensorMemorySpaceAttr>(memDescTy.getMemorySpace())) {
    return b.ptrtoint(i32Ty, sharedMemStruct);
  }
  assert(isa<ttg::SharedEncodingTrait>(memDescTy.getEncoding()) &&
         "Unsupported memory encoding");
  Type srcElemTy = typeConverter->convertType(memDescTy.getElementType());
  auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, sharedMemStruct,
                                                       srcElemTy, rewriter);
  auto offset = smemObj.getShmemOffset(loc, rewriter, memDescTy);
  auto elemSize = srcElemTy.getIntOrFloatBitWidth() / 8;
  offset = b.mul(offset, b.i32_val(elemSize));
  return b.add(offset, b.ptrtoint(i32Ty, smemObj.getBase()));
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
    if (isa<RankedTensorType>(op.getCondition().getType())) {
      // Add a barrier to avoid a race condition in case an assert is followed
      // by an op that may trap if the assert condition is true. Since the
      // tensor in those two operations may have different layout we need to
      // make sure all the threads are done executing the assert before going to
      // the next op.
      b.barrier();
    }
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

struct BufferDescriptorsOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalBufferDescriptorsOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(tti::ExperimentalBufferDescriptorsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto encoding =
        cast<ttg::BlockedEncodingAttr>(op.getResult().getType().getEncoding());
    auto offsets = adaptor.getOffsets();
    auto lengths = adaptor.getLengths();
    assert(offsets.size() == lengths.size() && "Mismatched descriptor arrays");

    auto tensorType = cast<RankedTensorType>(op.getResult().getType());

    SmallVector<uint64_t> offsetVals;
    offsetVals.reserve(offsets.size());
    for (int32_t offset : offsets)
      offsetVals.push_back(static_cast<uint32_t>(offset));
    Value pointerTensor =
        createInitializedIntArrayTensor(rewriter, loc, encoding, offsetVals);

    TritonLLVMOpBuilder b(loc, rewriter);
    auto i64Ty = rewriter.getIntegerType(64);
    Value baseTensor = nullptr;
    if (op.getMemType() == tti::MemType::SHARED_MEM) {
      auto func = op->getParentOfType<FunctionOpInterface>();
      Value base = getSharedMemoryBase(rewriter, func);
      baseTensor = triton::SplatOp::create(rewriter, loc, tensorType, base);
    } else {
      assert(op.getMemType() == tti::MemType::TENSOR_MEM &&
             "Unsupported memory type");
      Value basePtr = nvgpu::TensorMemoryBaseAddress::create(rewriter, loc);
      Value base = b.ptrtoint(i64Ty, basePtr);
      baseTensor = triton::SplatOp::create(rewriter, loc, tensorType, base);
    }

    pointerTensor = arith::AddIOp::create(
        rewriter, loc, pointerTensor.getType(), pointerTensor, baseTensor);

    SmallVector<uint64_t> maskVals(offsets.size(), 0xffffffffu);
    Value maskTensor =
        createInitializedIntArrayTensor(rewriter, loc, encoding, maskVals);
    Value trimmedPointers = arith::AndIOp::create(
        rewriter, loc, pointerTensor.getType(), pointerTensor, maskTensor);

    SmallVector<uint64_t> lengthVals;
    lengthVals.reserve(lengths.size());
    for (int32_t length : lengths)
      lengthVals.push_back(static_cast<uint64_t>(static_cast<uint32_t>(length))
                           << 32);
    Value lengthTensor =
        createInitializedIntArrayTensor(rewriter, loc, encoding, lengthVals);

    auto bufDescriptors =
        arith::OrIOp::create(rewriter, loc, trimmedPointers.getType(),
                             trimmedPointers, lengthTensor);
    rewriter.replaceOp(op, bufDescriptors);
    return success();
  }

  Value createInitializedIntArrayTensor(OpBuilder &builder, Location loc,
                                        BlockedEncodingAttr encoding,
                                        ArrayRef<uint64_t> values) const {
    int64_t size = values.size();
    assert(llvm::isPowerOf2_64(size) && "Expected power of 2");
    auto tensorType =
        RankedTensorType::get({size}, builder.getIntegerType(64), encoding);
    SmallVector<APInt> apInts = llvm::to_vector(
        llvm::map_range(values, [](uint64_t v) { return APInt(64, v); }));
    auto denseAttr = DenseElementsAttr::get(tensorType, apInts);
    return arith::ConstantOp::create(builder, loc, tensorType, denseAttr);
  }

  Value getSharedMemoryBase(ConversionPatternRewriter &rewriter,
                            FunctionOpInterface func) const {
    Location loc = func.getLoc();
    Value basePtr = LLVM::getStackPointer(rewriter, func);
    auto i64Ty = rewriter.getIntegerType(64);
    TritonLLVMOpBuilder b(loc, rewriter);
    return b.ptrtoint(i64Ty, basePtr);
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

struct MemDescToI32OpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalMemDescToI32Op> {
public:
  using ConvertOpToLLVMPattern<
      tti::ExperimentalMemDescToI32Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(tti::ExperimentalMemDescToI32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value converted =
        createMemDescToI32(rewriter, loc, getTypeConverter(),
                           op.getMemdesc().getType(), adaptor.getMemdesc());
    rewriter.replaceOp(op, converted);
    return success();
  }
};

} // namespace

void mlir::triton::populateInstrumentationToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<AssertInThreadOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<BufferDescriptorsOpConversion>(typeConverter);
  patterns.add<LockAcquireOpConversion>(typeConverter);
  patterns.add<LockReleaseOpConversion>(typeConverter);
  patterns.add<MemDescToI32OpConversion>(typeConverter);
}
