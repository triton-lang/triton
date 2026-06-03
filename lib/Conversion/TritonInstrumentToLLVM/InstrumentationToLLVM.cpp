#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "third_party/nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "third_party/nvidia/include/TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include <limits>

namespace {

namespace tt = mlir::triton;
namespace ttg = tt::gpu;
namespace tti = mlir::triton::instrument;
namespace ttng = mlir::triton::nvidia_gpu;

// The first 24 bits of the shared memory object are CTA-invariant
// The next 4 bits are the CTA index
constexpr uint32_t kSharedMemoryObjectMask = (1u << 24) - 1;

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
  return b.and_(b.add(offset, b.ptrtoint(i32Ty, smemObj.getBase())),
                b.i32_val(kSharedMemoryObjectMask));
}

////////////////////////////////////////////
// Patterns
////////////////////////////////////////////

struct AssertUniformOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalAssertUniformOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(tti::ExperimentalAssertUniformOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    TritonLLVMIRRewriter b(op.getLoc(), rewriter);
    Value tid = getThreadId(b, op.getLoc());
    Value threadIdIsZero = b.icmp_eq(tid, b.i32_val(0));

    auto [prevBlock, ifBlock, thenBlock] =
        createIfBlock(rewriter, op.getLoc(), threadIdIsZero);
    rewriter.setInsertionPointToStart(ifBlock);
    AssertOp::create(rewriter, op.getLoc(), adaptor.getCondition(),
                     adaptor.getMessage());
    rewriter.eraseOp(op);
    rewriter.setInsertionPointToStart(thenBlock);
    return success();
  }
};

struct BufferDescriptorsOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalBufferDescriptorsOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(tti::ExperimentalBufferDescriptorsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto offsets = adaptor.getOffsets();
    auto lengths = adaptor.getLengths();
    assert(offsets.size() == lengths.size() && "Mismatched descriptor arrays");

    auto tensorType = cast<RankedTensorType>(op.getResult().getType());
    auto encoding =
        cast<ttg::DistributedEncodingTrait>(tensorType.getEncoding());
    assert(tensorType.getRank() == 1 &&
           "descriptor tables must have shape [descriptor]");
    assert(static_cast<int64_t>(offsets.size()) ==
               tensorType.getShape().back() &&
           "Descriptor data must match the descriptor dimension");

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

    SmallVector<uint64_t> maskVals(offsets.size(),
                                   op.getMemType() == tti::MemType::SHARED_MEM
                                       ? kSharedMemoryObjectMask
                                       : 0xffffffffu);
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

    Value bufDescriptors =
        arith::OrIOp::create(rewriter, loc, trimmedPointers.getType(),
                             trimmedPointers, lengthTensor);
    rewriter.replaceOp(op, bufDescriptors);
    return success();
  }

  Value createInitializedIntArrayTensor(OpBuilder &builder, Location loc,
                                        ttg::DistributedEncodingTrait encoding,
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
  explicit LockAcquireOpConversion(LLVMTypeConverter &typeConverter,
                                   const TargetInfoBase &targetInfo)
      : ConvertOpToLLVMPattern<tti::ExperimentalLockAcquireOp>(typeConverter),
        targetInfo(targetInfo) {}

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

    Value elect;
    if (targetInfo.isCuda()) {
      elect = mlir::LLVM::NVIDIA::createElectPredicateWarp0(loc, b);
    } else {
      TritonLLVMOpBuilder tb(loc, b);
      auto [laneId, warpId] = getLaneAndWarpId(b, loc);
      Value lane0 = tb.icmp_eq(laneId, tb.i32_val(0));
      Value warp0 = tb.icmp_eq(warpId, tb.i32_val(0));
      elect = tb.and_(lane0, warp0);
    }
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

    if (targetInfo.isCuda()) {
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
    } else {
      Value oldVal = LLVM::AtomicRMWOp::create(
          b, loc, LLVM::AtomicBinOp::xchg, adaptor.getLock(), one,
          LLVM::AtomicOrdering::acquire,
          StringAttr::get(b.getContext(), "agent"));
      Value acquired =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, oldVal, zero);
      LLVM::CondBrOp::create(b, loc, acquired, endBlock, whileBlock);
    }

    b.setInsertionPointToStart(endBlock);
    triton::gpu::BarrierOp::create(b, loc,
                                   triton::gpu::AddrSpace::GlobalRead |
                                       triton::gpu::AddrSpace::GlobalWrite);
    b.eraseOp(op);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

struct LockReleaseOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalLockReleaseOp> {
  explicit LockReleaseOpConversion(LLVMTypeConverter &typeConverter,
                                   const TargetInfoBase &targetInfo)
      : ConvertOpToLLVMPattern<tti::ExperimentalLockReleaseOp>(typeConverter),
        targetInfo(targetInfo) {}

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

    triton::gpu::BarrierOp::create(b, loc,
                                   triton::gpu::AddrSpace::GlobalRead |
                                       triton::gpu::AddrSpace::GlobalWrite);

    auto i32 = b.getI32Type();
    Value zero =
        arith::ConstantOp::create(b, loc, i32, b.getIntegerAttr(i32, 0));

    if (targetInfo.isCuda()) {
      Value elect = mlir::LLVM::NVIDIA::createElectPredicateWarp0(loc, b);

      PTXBuilder ptx;
      auto *dstOpr = ptx.newOperand("=r", /*init=*/true);
      auto *ptrOpr = ptx.newAddrOperand(adaptor.getLock(), "l");
      auto *valOpr = ptx.newOperand(zero, "r");
      auto &atom = *ptx.create("atom");
      atom.global().o("acq_rel").o("gpu").o("exch").o("b32");
      atom(dstOpr, ptrOpr, valOpr).predicate(elect);
      ptx.launch(b, loc, i32);
    } else {
      LLVM::AtomicRMWOp::create(b, loc, LLVM::AtomicBinOp::xchg,
                                adaptor.getLock(), zero,
                                LLVM::AtomicOrdering::release,
                                StringAttr::get(b.getContext(), "agent"));
    }

    b.eraseOp(op);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
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

struct ClusterCTAIdOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalClusterCTAIdOp> {
  ClusterCTAIdOpConversion(const LLVMTypeConverter &converter,
                           const TargetInfoBase &targetInfo,
                           PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<tti::ExperimentalClusterCTAIdOp>(converter,
                                                                benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(tti::ExperimentalClusterCTAIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value blockId = targetInfo.getClusterCTAId(rewriter, loc);
    rewriter.replaceOp(op, blockId);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::populateInstrumentationToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo) {
  patterns.add<AssertUniformOpConversion>(typeConverter);
  patterns.add<BufferDescriptorsOpConversion>(typeConverter);
  patterns.add<LockAcquireOpConversion>(typeConverter, targetInfo);
  patterns.add<LockReleaseOpConversion>(typeConverter, targetInfo);
  patterns.add<MemDescToI32OpConversion>(typeConverter);
  patterns.add<ClusterCTAIdOpConversion>(typeConverter, targetInfo);
}
