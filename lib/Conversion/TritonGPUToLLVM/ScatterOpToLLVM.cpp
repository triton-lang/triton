#include "ReduceScanCommon.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {
class ScatterOpConversion : public ConvertOpToLLVMPattern<ScatterOp> {
public:
  ScatterOpConversion(LLVMTypeConverter &typeConverter,
                      const TargetInfoBase &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(ScatterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  // Codegen the scatter via shared memory scratch space.
  void emitScatterInShared(ScatterOp op, OpAdaptor adaptor,
                           ConversionPatternRewriter &rewriter) const;
  // Codegen a warp-local scatter by shuffling source values and selecting
  // destination updates entirely within the warp.
  void emitWarpLocalScatter(ScatterOp op, OpAdaptor adaptor,
                            ConversionPatternRewriter &rewriter) const;

  const TargetInfoBase &targetInfo;
};

static Value convertIndexToI32(Location loc, Value index,
                               ConversionPatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  unsigned idxWidth = index.getType().getIntOrFloatBitWidth();
  if (idxWidth > 32) {
    index = b.trunc(i32_ty, index);
  } else if (idxWidth < 32) {
    index = b.zext(i32_ty, index);
  }
  return index;
}

static Value applyKnownScatterCombine(Location loc,
                                      ConversionPatternRewriter &rewriter,
                                      StringRef reduceKind, Value lhs,
                                      Value rhs) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  if (reduceKind == "add")
    return b.add(lhs, rhs);
  if (reduceKind == "fadd")
    return b.fadd(lhs, rhs);
  if (reduceKind == "max")
    return b.smax(lhs, rhs);
  if (reduceKind == "umax")
    return b.umax(lhs, rhs);
  if (reduceKind == "fmax")
    return b.fmax(lhs, rhs);
  if (reduceKind == "min")
    return b.smin(lhs, rhs);
  if (reduceKind == "umin")
    return b.umin(lhs, rhs);
  if (reduceKind == "fmin")
    return b.fmin(lhs, rhs);
  if (reduceKind == "and")
    return b.and_(lhs, rhs);
  if (reduceKind == "or")
    return b.or_(lhs, rhs);
  if (reduceKind == "xor")
    return b.xor_(lhs, rhs);
  llvm_unreachable("unsupported scatter reduce kind");
}

static Value applyScatterCombine(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 ScatterOp op, Value lhs, Value rhs,
                                 Value pred = Value()) {
  Value combined;
  if (auto reduceKindAttr = op.getReduceKindAttr()) {
    combined = applyKnownScatterCombine(loc, rewriter,
                                        reduceKindAttr.getValue(), lhs, rhs);
  } else {
    combined =
        applyCombineOp(loc, rewriter, op.getCombineOp(), {lhs}, {rhs})[0];
  }
  if (pred) {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    return b.select(pred, combined, lhs);
  }
  return combined;
}

static std::optional<LLVM::AtomicBinOp>
getScatterFastAtomicBinOp(ScatterOp op) {
  auto reduceKindAttr = op.getReduceKindAttr();
  if (!reduceKindAttr)
    return std::nullopt;

  Type elemTy = op.getDst().getType().getElementType();
  unsigned bitWidth = elemTy.getIntOrFloatBitWidth();
  bool supports32Or64 = bitWidth == 32 || bitWidth == 64;
  StringRef reduceKind = reduceKindAttr.getValue();

  if (reduceKind == "add" && isa<IntegerType>(elemTy) && supports32Or64)
    return LLVM::AtomicBinOp::add;
  if (reduceKind == "fadd" && isa<FloatType>(elemTy) && supports32Or64)
    return LLVM::AtomicBinOp::fadd;
  if (reduceKind == "max" && isa<IntegerType>(elemTy) && supports32Or64)
    return LLVM::AtomicBinOp::max;
  if (reduceKind == "min" && isa<IntegerType>(elemTy) && supports32Or64)
    return LLVM::AtomicBinOp::min;
  if (reduceKind == "umax" && isa<IntegerType>(elemTy) && supports32Or64)
    return LLVM::AtomicBinOp::umax;
  if (reduceKind == "umin" && isa<IntegerType>(elemTy) && supports32Or64)
    return LLVM::AtomicBinOp::umin;
  if (reduceKind == "and" && isa<IntegerType>(elemTy) && supports32Or64)
    return LLVM::AtomicBinOp::_and;
  if (reduceKind == "or" && isa<IntegerType>(elemTy) && supports32Or64)
    return LLVM::AtomicBinOp::_or;
  if (reduceKind == "xor" && isa<IntegerType>(elemTy) && supports32Or64)
    return LLVM::AtomicBinOp::_xor;
  return std::nullopt;
}

static Value countTrailingZeros(Location loc,
                                ConversionPatternRewriter &rewriter,
                                Value value) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type valueTy = value.getType();
  Value isZeroUndef = b.false_val();
  return LLVM::createLLVMIntrinsicCallOp(rewriter, loc, "llvm.cttz", {valueTy},
                                         {value, isZeroUndef})
      .getResult(0);
}

LogicalResult ScatterOpConversion::matchAndRewrite(
    ScatterOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  ScatterLoweringHelper helper(op);
  if (helper.isWarpLocal()) {
    emitWarpLocalScatter(op, adaptor, rewriter);
  } else {
    emitScatterInShared(op, adaptor, rewriter);
  }
  return success();
}

void ScatterOpConversion::emitScatterInShared(
    ScatterOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  RankedTensorType dstType = op.getDst().getType();
  RankedTensorType srcType = op.getSrc().getType();
  bool hasCombine = !op.getCombineOp().empty() || op.getReduceKindAttr();
  bool includeSelf = op.getIncludeSelf();

  SmallVector<unsigned> dstShapePerCTA =
      convertType<unsigned>(triton::gpu::getShapePerCTA(dstType));
  SmallVector<unsigned> srcShapePerCTA =
      convertType<unsigned>(triton::gpu::getShapePerCTA(srcType));

  SmallVector<Value> dstValues =
      unpackLLElements(loc, adaptor.getDst(), rewriter);
  SmallVector<SmallVector<Value>> dstIndices =
      emitIndices(loc, rewriter, targetInfo, dstType.getEncoding(), dstType,
                  /*withCTAOffset=*/true);

  Value smemBase = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op);
  Type elemType = getTypeConverter()->convertType(dstType.getElementType());
  Value truePred = b.true_val();

  if (hasCombine) {
    // Shared reduction path overview:
    // 1) Initialize shared value/flag arrays for the destination tile.
    // 2) Locally fold duplicates within each thread.
    // 3) For each source element index, form same-dst groups within a warp.
    // 4) Only one lane (group leader) publishes the aggregated value.
    //    - known op + include_self: direct atomicrmw fast path
    //    - otherwise: CAS-guarded shared update state machine
    // 5) Barrier and read final dst values from shared memory.
    size_t dstElems = product(dstShapePerCTA);
    unsigned elemBytes = ceil<unsigned>(dstType.getElementTypeBitWidth(), 8);
    auto alignUp = [](size_t value, size_t align) {
      return (value + align - 1) / align * align;
    };
    auto fastAtomicBinOp =
        includeSelf ? getScatterFastAtomicBinOp(op) : std::nullopt;
    bool useFastAtomic = fastAtomicBinOp.has_value();
    size_t valuesBytes = dstElems * elemBytes;
    Value valuesBase = smemBase;
    Value flagsBase;
    if (!useFastAtomic) {
      size_t flagsOffsetBytes = alignUp(valuesBytes, sizeof(int32_t));
      flagsBase = b.gep(smemBase.getType(), i8_ty, smemBase,
                        b.i32_val(flagsOffsetBytes));
    }

    // Initialize shared memory values and flags for the destination tile.
    // Flag values: 0 = empty, 1 = locked, 2 = full.
    for (auto [value, indices] : llvm::zip(dstValues, dstIndices)) {
      Value offset = LLVM::linearize(rewriter, loc, indices, dstShapePerCTA);
      Value valuePtr =
          b.gep(valuesBase.getType(), elemType, valuesBase, offset);
      if (includeSelf || useFastAtomic) {
        targetInfo.storeShared(rewriter, loc, valuePtr, value, truePred);
      }
      if (!useFastAtomic) {
        Value flagPtr = b.gep(flagsBase.getType(), i32_ty, flagsBase, offset);
        if (includeSelf) {
          targetInfo.storeShared(rewriter, loc, flagPtr, b.i32_val(2),
                                 truePred);
        } else {
          targetInfo.storeShared(rewriter, loc, flagPtr, b.i32_val(0),
                                 truePred);
        }
      }
    }

    // Ensure all lanes see a consistent shared-memory initialization before
    // any lane starts publishing source updates.
    b.barrier(triton::gpu::AddrSpace::Local);

    SmallVector<Value> srcValues =
        unpackLLElements(loc, adaptor.getSrc(), rewriter);
    SmallVector<Value> idxValues =
        unpackLLElements(loc, adaptor.getIndices(), rewriter);
    SmallVector<SmallVector<Value>> srcIndices =
        emitIndices(loc, rewriter, targetInfo, srcType.getEncoding(), srcType,
                    /*withCTAOffset=*/true);

    assert(srcValues.size() == idxValues.size());
    assert(srcValues.size() == srcIndices.size());

    unsigned axis = op.getAxis();
    auto successOrdering = LLVM::AtomicOrdering::monotonic;
    auto failureOrdering = LLVM::AtomicOrdering::monotonic;
    StringRef syncScope = "block";

    // Precompute offsets and validity.
    SmallVector<Value> srcOffsets;
    SmallVector<Value> srcValid;
    srcOffsets.reserve(srcValues.size());
    srcValid.reserve(srcValues.size());

    for (auto [idxValue, indices] : llvm::zip(idxValues, srcIndices)) {
      Value dstAxis = convertIndexToI32(loc, idxValue, rewriter);
      Value dstAxisInBounds =
          b.icmp_ult(dstAxis, b.i32_val(dstType.getShape()[axis]));
      indices[axis] = dstAxis;
      Value dstOffset = LLVM::linearize(rewriter, loc, indices, dstShapePerCTA);
      srcOffsets.push_back(dstOffset);
      srcValid.push_back(dstAxisInBounds);
    }

    // Per-thread local reduction: combine duplicate destination offsets among
    // elements owned by this thread. This reduces later inter-thread traffic.
    for (size_t i = 0; i < srcValues.size(); ++i) {
      for (size_t j = 0; j < i; ++j) {
        Value bothValid = b.and_(srcValid[i], srcValid[j]);
        Value sameOffset = b.icmp_eq(srcOffsets[i], srcOffsets[j]);
        Value matches = b.and_(bothValid, sameOffset);
        // Combine unconditionally, then select based on matches to avoid
        // creating control flow in the local-reduction loop.
        Value combined =
            applyScatterCombine(loc, rewriter, op, srcValues[j], srcValues[i]);
        srcValues[j] = b.select(matches, combined, srcValues[j]);
        srcValid[i] = b.select(matches, b.false_val(), srcValid[i]);
      }
    }

    // Pack per-thread candidates into LLVM vectors to keep the hot path in SSA
    // values and avoid local stack traffic.
    Block *entryBlock = rewriter.getInsertionBlock();
    Value srcValuesVec = packLLVector(loc, srcValues, rewriter);
    Value srcOffsetsVec = packLLVector(loc, srcOffsets, rewriter);
    Value srcValidVec = packLLVector(loc, srcValid, rewriter);

    // Per-element update loop after local reduction.
    // We use warp aggregation (ballot + shuffleIdx) so one lane per equal
    // destination offset performs the shared update.
    Block *postBlock =
        rewriter.splitBlock(entryBlock, rewriter.getInsertionPoint());
    auto *parent = entryBlock->getParent();
    rewriter.setInsertionPointToEnd(entryBlock);

    auto insertIt = Region::iterator(postBlock);
    auto sharedPtrTy = smemBase.getType();
    int threadsPerWarp = triton::gpu::lookupThreadsPerWarp(rewriter);
    Type maskTy = int_ty(threadsPerWarp);
    Value zeroMask = b.int_val(threadsPerWarp, 0);
    Value oneMask = b.int_val(threadsPerWarp, 1);
    Value allOnesMask = b.int_val(threadsPerWarp, -1);
    Value laneId = getLaneAndWarpId(rewriter, loc).first;
    Value laneShift = laneId;
    if (threadsPerWarp > 32)
      laneShift = b.zext(maskTy, laneId);
    Value laneBit = b.shl(oneMask, laneShift);

    Block *loopHeader = rewriter.createBlock(parent, insertIt, {i32_ty}, {loc});
    Block *loopBody = rewriter.createBlock(parent, insertIt, {i32_ty}, {loc});
    Block *reduceHeader = rewriter.createBlock(
        parent, insertIt, {i32_ty, i32_ty, i1_ty, elemType, elemType, maskTy},
        {loc, loc, loc, loc, loc, loc});
    Block *reduceStep = rewriter.createBlock(
        parent, insertIt, {i32_ty, i32_ty, i1_ty, elemType, elemType, maskTy},
        {loc, loc, loc, loc, loc, loc});
    Block *afterReduce = rewriter.createBlock(parent, insertIt,
                                              {i32_ty, i32_ty, i1_ty, elemType},
                                              {loc, loc, loc, loc});
    Block *loopLatch = rewriter.createBlock(parent, insertIt, {i32_ty}, {loc});
    Block *atomicBlock = nullptr;
    Block *retryHeader = nullptr;
    Block *retryAttempt = nullptr;
    Block *updateBlock = nullptr;
    Block *tryCombineBlock = nullptr;
    Block *initBlock = nullptr;
    Block *combineBlock = nullptr;
    Block *retryTail = nullptr;
    if (useFastAtomic) {
      atomicBlock = rewriter.createBlock(
          parent, insertIt, {i32_ty, sharedPtrTy, elemType}, {loc, loc, loc});
    } else {
      retryHeader = rewriter.createBlock(
          parent, insertIt, {i32_ty, sharedPtrTy, sharedPtrTy, elemType, i1_ty},
          {loc, loc, loc, loc, loc});
      retryAttempt = rewriter.createBlock(
          parent, insertIt, {i32_ty, sharedPtrTy, sharedPtrTy, elemType, i1_ty},
          {loc, loc, loc, loc, loc});
      updateBlock = rewriter.createBlock(
          parent, insertIt, {i32_ty, sharedPtrTy, sharedPtrTy, elemType},
          {loc, loc, loc, loc});
      tryCombineBlock = rewriter.createBlock(
          parent, insertIt, {i32_ty, sharedPtrTy, sharedPtrTy, elemType},
          {loc, loc, loc, loc});
      initBlock = rewriter.createBlock(
          parent, insertIt, {i32_ty, sharedPtrTy, sharedPtrTy, elemType},
          {loc, loc, loc, loc});
      combineBlock = rewriter.createBlock(
          parent, insertIt, {i32_ty, sharedPtrTy, sharedPtrTy, elemType},
          {loc, loc, loc, loc});
      retryTail = rewriter.createBlock(
          parent, insertIt,
          {i32_ty, sharedPtrTy, sharedPtrTy, elemType, i1_ty, i1_ty},
          {loc, loc, loc, loc, loc, loc});
    }

    rewriter.setInsertionPointToEnd(entryBlock);
    LLVM::BrOp::create(rewriter, loc, b.i32_val(0), loopHeader);

    // loopHeader: check bounds
    rewriter.setInsertionPointToStart(loopHeader);
    Value i = loopHeader->getArgument(0);
    Value inRange = b.icmp_slt(i, b.i32_val(srcValues.size()));
    LLVM::CondBrOp::create(rewriter, loc, inRange, loopBody, ValueRange{i},
                           postBlock, ValueRange{});

    // loopBody: for logical element index `i`, each lane loads one candidate.
    // We then discover equal destination offsets within the warp and reduce
    // them to a single leader-lane contribution.
    rewriter.setInsertionPointToStart(loopBody);
    Value bodyIdx = loopBody->getArgument(0);
    Value srcValue = b.extract_element(srcValuesVec, bodyIdx);
    Value dstOffset = b.extract_element(srcOffsetsVec, bodyIdx);
    Value isValid = b.extract_element(srcValidVec, bodyIdx);

    // Active mask for lanes whose element `i` is in-bounds.
    Value activeMask =
        targetInfo.ballot(rewriter, loc, maskTy, cast<Value>(isValid));

    // Compute the equivalent of match_any(dstOffset) over active lanes.
    // Targets may use native instructions; otherwise this falls back to a
    // ballot-based implementation.
    Value groupMask =
        targetInfo.matchAny(rewriter, loc, maskTy, dstOffset, activeMask);

    // Leader lane = lowest set bit in groupMask. Only leader publishes the
    // final reduced value for this (warp, i, dstOffset) group.
    Value groupNeg = b.sub(zeroMask, groupMask);
    Value leaderBit = b.and_(groupMask, groupNeg);
    Value isLeader = b.icmp_eq(laneBit, leaderBit);
    Value pending = b.and_(isValid, isLeader);

    // Iterate only non-leader bits in groupMask using cttz(mask):
    // avoids a fixed-width 0..warp_size-1 scan and skips inactive lanes.
    Value remainingMask = b.and_(groupMask, b.xor_(leaderBit, allOnesMask));
    LLVM::BrOp::create(rewriter, loc,
                       ValueRange{bodyIdx, dstOffset, pending, srcValue,
                                  srcValue, remainingMask},
                       reduceHeader);

    // reduceHeader: consume group lanes until remainingMask == 0.
    rewriter.setInsertionPointToStart(reduceHeader);
    Value rhIdx = reduceHeader->getArgument(0);
    Value rhDstOffset = reduceHeader->getArgument(1);
    Value rhPending = reduceHeader->getArgument(2);
    Value rhSrcValue = reduceHeader->getArgument(3);
    Value rhReducedValue = reduceHeader->getArgument(4);
    Value rhRemainingMask = reduceHeader->getArgument(5);
    Value hasRemaining = b.icmp_ne(rhRemainingMask, zeroMask);
    LLVM::CondBrOp::create(
        rewriter, loc, hasRemaining, reduceStep,
        ValueRange{rhIdx, rhDstOffset, rhPending, rhSrcValue, rhReducedValue,
                   rhRemainingMask},
        afterReduce, ValueRange{rhIdx, rhDstOffset, rhPending, rhReducedValue});

    // reduceStep: pick next lane via cttz, shuffle its source value, fold it
    // into the running accumulator, and clear that lane bit.
    rewriter.setInsertionPointToStart(reduceStep);
    Value rsIdx = reduceStep->getArgument(0);
    Value rsDstOffset = reduceStep->getArgument(1);
    Value rsPending = reduceStep->getArgument(2);
    Value rsSrcValue = reduceStep->getArgument(3);
    Value rsReducedValue = reduceStep->getArgument(4);
    Value rsRemainingMask = reduceStep->getArgument(5);
    Value lane = countTrailingZeros(loc, rewriter, rsRemainingMask);
    Value laneI32 = lane;
    unsigned laneWidth = lane.getType().getIntOrFloatBitWidth();
    if (laneWidth > 32)
      laneI32 = b.trunc(i32_ty, lane);
    else if (laneWidth < 32)
      laneI32 = b.zext(i32_ty, lane);
    Value shuffled = targetInfo.shuffleIdx(rewriter, loc, rsSrcValue, laneI32);
    Value nextReduced =
        applyScatterCombine(loc, rewriter, op, rsReducedValue, shuffled);
    Value laneShiftDyn = laneI32;
    if (threadsPerWarp > 32)
      laneShiftDyn = b.zext(maskTy, laneI32);
    Value laneMask = b.shl(oneMask, laneShiftDyn);
    Value nextRemaining =
        b.and_(rsRemainingMask, b.xor_(laneMask, allOnesMask));
    LLVM::BrOp::create(rewriter, loc,
                       ValueRange{rsIdx, rsDstOffset, rsPending, rsSrcValue,
                                  nextReduced, nextRemaining},
                       reduceHeader);

    rewriter.setInsertionPointToStart(afterReduce);
    Value arIdx = afterReduce->getArgument(0);
    Value arDstOffset = afterReduce->getArgument(1);
    Value arPending = afterReduce->getArgument(2);
    Value arReducedValue = afterReduce->getArgument(3);
    Value valuePtr =
        b.gep(valuesBase.getType(), elemType, valuesBase, arDstOffset);
    if (useFastAtomic) {
      // Fast path for known associative/commutative ops when include_self=true:
      // dst tile was pre-initialized from dst, so each leader can atomicrmw
      // directly without a flag/CAS protocol.
      LLVM::CondBrOp::create(rewriter, loc, arPending, atomicBlock,
                             ValueRange{arIdx, valuePtr, arReducedValue},
                             loopLatch, ValueRange{arIdx});

      rewriter.setInsertionPointToStart(atomicBlock);
      Value atIdx = atomicBlock->getArgument(0);
      Value atValuePtr = atomicBlock->getArgument(1);
      Value atReducedValue = atomicBlock->getArgument(2);
      LLVM::AtomicRMWOp::create(rewriter, loc, *fastAtomicBinOp, atValuePtr,
                                atReducedValue,
                                LLVM::AtomicOrdering::monotonic);
      LLVM::BrOp::create(rewriter, loc, atIdx, loopLatch);
    } else {
      // Generic path (callable combine_fn or include_self=false):
      // CAS protocol on per-offset flags to serialize load+combine+store.
      Value flagPtr =
          b.gep(flagsBase.getType(), i32_ty, flagsBase, arDstOffset);
      LLVM::BrOp::create(
          rewriter, loc,
          ValueRange{arIdx, flagPtr, valuePtr, arReducedValue, arPending},
          retryHeader);

      // retryHeader: iterate while any leader lane is still pending.
      rewriter.setInsertionPointToStart(retryHeader);
      Value rhIdx = retryHeader->getArgument(0);
      Value rhFlagPtr = retryHeader->getArgument(1);
      Value rhValuePtr = retryHeader->getArgument(2);
      Value rhReducedValue = retryHeader->getArgument(3);
      Value rhPending = retryHeader->getArgument(4);
      Value retryMask =
          targetInfo.ballot(rewriter, loc, maskTy, cast<Value>(rhPending));
      Value anyPending = b.icmp_ne(retryMask, zeroMask);
      LLVM::CondBrOp::create(
          rewriter, loc, anyPending, retryAttempt,
          ValueRange{rhIdx, rhFlagPtr, rhValuePtr, rhReducedValue, rhPending},
          loopLatch, ValueRange{rhIdx});

      // retryAttempt: only pending leader lanes attempt state transitions.
      rewriter.setInsertionPointToStart(retryAttempt);
      Value raIdx = retryAttempt->getArgument(0);
      Value raFlagPtr = retryAttempt->getArgument(1);
      Value raValuePtr = retryAttempt->getArgument(2);
      Value raReducedValue = retryAttempt->getArgument(3);
      Value raPending = retryAttempt->getArgument(4);
      LLVM::CondBrOp::create(
          rewriter, loc, raPending, updateBlock,
          ValueRange{raIdx, raFlagPtr, raValuePtr, raReducedValue}, retryTail,
          ValueRange{raIdx, raFlagPtr, raValuePtr, raReducedValue, raPending,
                     b.false_val()});

      // updateBlock:
      // - include_self=false: try empty->locked (0->1) for first writer.
      // - include_self=true: slot is already full, skip directly to
      // full->locked.
      rewriter.setInsertionPointToStart(updateBlock);
      Value updIdx = updateBlock->getArgument(0);
      Value updFlagPtr = updateBlock->getArgument(1);
      Value updValuePtr = updateBlock->getArgument(2);
      Value updReducedValue = updateBlock->getArgument(3);
      if (includeSelf) {
        // Slot is always full when include_self=true, so skip 0->1 CAS.
        Value casCombine = LLVM::AtomicCmpXchgOp::create(
            rewriter, loc, updFlagPtr, b.i32_val(2), b.i32_val(1),
            successOrdering, failureOrdering, syncScope);
        Value combineSuccess = b.extract_val(i1_ty, casCombine, 1);
        LLVM::CondBrOp::create(
            rewriter, loc, combineSuccess, combineBlock,
            ValueRange{updIdx, updFlagPtr, updValuePtr, updReducedValue},
            retryTail,
            ValueRange{updIdx, updFlagPtr, updValuePtr, updReducedValue,
                       b.true_val(), b.false_val()});
      } else {
        Value casInit = LLVM::AtomicCmpXchgOp::create(
            rewriter, loc, updFlagPtr, b.i32_val(0), b.i32_val(1),
            successOrdering, failureOrdering, syncScope);
        Value initSuccess = b.extract_val(i1_ty, casInit, 1);
        LLVM::CondBrOp::create(
            rewriter, loc, initSuccess, initBlock,
            ValueRange{updIdx, updFlagPtr, updValuePtr, updReducedValue},
            tryCombineBlock,
            ValueRange{updIdx, updFlagPtr, updValuePtr, updReducedValue});
      }

      // tryCombineBlock: try full->locked (2->1) to perform read-combine-write.
      rewriter.setInsertionPointToStart(tryCombineBlock);
      Value tryIdx = tryCombineBlock->getArgument(0);
      Value tryFlagPtr = tryCombineBlock->getArgument(1);
      Value tryValuePtr = tryCombineBlock->getArgument(2);
      Value tryReducedValue = tryCombineBlock->getArgument(3);
      Value casCombine = LLVM::AtomicCmpXchgOp::create(
          rewriter, loc, tryFlagPtr, b.i32_val(2), b.i32_val(1),
          successOrdering, failureOrdering, syncScope);
      Value combineSuccess = b.extract_val(i1_ty, casCombine, 1);
      LLVM::CondBrOp::create(
          rewriter, loc, combineSuccess, combineBlock,
          ValueRange{tryIdx, tryFlagPtr, tryValuePtr, tryReducedValue},
          retryTail,
          ValueRange{tryIdx, tryFlagPtr, tryValuePtr, tryReducedValue,
                     b.true_val(), b.false_val()});

      // initBlock: publish first value for this offset, then mark full (2).
      rewriter.setInsertionPointToStart(initBlock);
      Value initIdx = initBlock->getArgument(0);
      Value initFlagPtr = initBlock->getArgument(1);
      Value initValuePtr = initBlock->getArgument(2);
      Value initReducedValue = initBlock->getArgument(3);
      targetInfo.storeShared(rewriter, loc, initValuePtr, initReducedValue,
                             truePred);
      LLVM::AtomicRMWOp::create(rewriter, loc, LLVM::AtomicBinOp::xchg,
                                initFlagPtr, b.i32_val(2),
                                LLVM::AtomicOrdering::monotonic);
      LLVM::BrOp::create(rewriter, loc,
                         ValueRange{initIdx, initFlagPtr, initValuePtr,
                                    initReducedValue, b.true_val(),
                                    b.true_val()},
                         retryTail);

      // combineBlock: read current value, combine with lane aggregate, store,
      // then mark full (2) to release the slot.
      rewriter.setInsertionPointToStart(combineBlock);
      Value combIdx = combineBlock->getArgument(0);
      Value combFlagPtr = combineBlock->getArgument(1);
      Value combValuePtr = combineBlock->getArgument(2);
      Value combReducedValue = combineBlock->getArgument(3);
      Value acc = targetInfo.loadShared(rewriter, loc, combValuePtr, elemType,
                                        truePred);
      Value combined =
          applyScatterCombine(loc, rewriter, op, acc, combReducedValue);
      targetInfo.storeShared(rewriter, loc, combValuePtr, combined, truePred);
      LLVM::AtomicRMWOp::create(rewriter, loc, LLVM::AtomicBinOp::xchg,
                                combFlagPtr, b.i32_val(2),
                                LLVM::AtomicOrdering::monotonic);
      LLVM::BrOp::create(rewriter, loc,
                         ValueRange{combIdx, combFlagPtr, combValuePtr,
                                    combReducedValue, b.true_val(),
                                    b.true_val()},
                         retryTail);

      // retryTail: clear pending on success; otherwise retry.
      rewriter.setInsertionPointToStart(retryTail);
      Value rtIdx = retryTail->getArgument(0);
      Value rtFlagPtr = retryTail->getArgument(1);
      Value rtValuePtr = retryTail->getArgument(2);
      Value rtReducedValue = retryTail->getArgument(3);
      Value rtPending = retryTail->getArgument(4);
      Value rtSuccess = retryTail->getArgument(5);
      Value newPending = b.and_(rtPending, b.xor_(rtSuccess, b.true_val()));
      LLVM::BrOp::create(
          rewriter, loc,
          ValueRange{rtIdx, rtFlagPtr, rtValuePtr, rtReducedValue, newPending},
          retryHeader);
    }

    // loopLatch: increment index and continue.
    rewriter.setInsertionPointToStart(loopLatch);
    Value latchIdx = loopLatch->getArgument(0);
    Value nextIdx = b.add(latchIdx, b.i32_val(1));
    LLVM::BrOp::create(rewriter, loc, nextIdx, loopHeader);

    rewriter.setInsertionPointToStart(postBlock);

    // Ensure all warp updates are visible before final readback.
    b.barrier(triton::gpu::AddrSpace::Local);

    SmallVector<Value> results(dstIndices.size());
    for (auto [i, indices] : llvm::enumerate(dstIndices)) {
      Value offset = LLVM::linearize(rewriter, loc, indices, dstShapePerCTA);
      Value valuePtr =
          b.gep(valuesBase.getType(), elemType, valuesBase, offset);
      if (includeSelf || useFastAtomic) {
        results[i] =
            targetInfo.loadShared(rewriter, loc, valuePtr, elemType, truePred);
      } else {
        Value flagPtr = b.gep(flagsBase.getType(), i32_ty, flagsBase, offset);
        Value flag =
            targetInfo.loadShared(rewriter, loc, flagPtr, i32_ty, truePred);
        Value hasValue = b.icmp_eq(flag, b.i32_val(2));
        Value val =
            targetInfo.loadShared(rewriter, loc, valuePtr, elemType, truePred);
        results[i] = b.select(hasValue, val, dstValues[i]);
      }
    }

    Value packed = packLLElements(loc, getTypeConverter(), results, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, packed);
    return;
  }

  assert(dstValues.size() == dstIndices.size());
  for (auto [value, indices] : llvm::zip(dstValues, dstIndices)) {
    Value offset = LLVM::linearize(rewriter, loc, indices, dstShapePerCTA);
    Value ptr = b.gep(smemBase.getType(), elemType, smemBase, offset);
    targetInfo.storeShared(rewriter, loc, ptr, value, truePred);
  }

  // Ensure all destination initialization stores are visible before scattering.
  b.barrier(triton::gpu::AddrSpace::Local);

  SmallVector<Value> srcValues =
      unpackLLElements(loc, adaptor.getSrc(), rewriter);
  SmallVector<Value> idxValues =
      unpackLLElements(loc, adaptor.getIndices(), rewriter);
  SmallVector<SmallVector<Value>> srcIndices =
      emitIndices(loc, rewriter, targetInfo, srcType.getEncoding(), srcType,
                  /*withCTAOffset=*/true);

  assert(srcValues.size() == idxValues.size());
  assert(srcValues.size() == srcIndices.size());

  unsigned axis = op.getAxis();
  for (auto [srcValue, idxValue, indices] :
       llvm::zip(srcValues, idxValues, srcIndices)) {
    indices[axis] = convertIndexToI32(loc, idxValue, rewriter);
    Value offset = LLVM::linearize(rewriter, loc, indices, dstShapePerCTA);
    Value ptr = b.gep(smemBase.getType(), elemType, smemBase, offset);
    targetInfo.storeShared(rewriter, loc, ptr, srcValue, truePred);
  }

  // Ensure all scatter writes are visible before reading final outputs.
  b.barrier(triton::gpu::AddrSpace::Local);

  SmallVector<Value> results(dstIndices.size());
  for (auto [i, indices] : llvm::enumerate(dstIndices)) {
    Value offset = LLVM::linearize(rewriter, loc, indices, dstShapePerCTA);
    Value ptr = b.gep(smemBase.getType(), elemType, smemBase, offset);
    results[i] = targetInfo.loadShared(rewriter, loc, ptr, elemType, truePred);
  }

  Value packed =
      packLLElements(loc, getTypeConverter(), results, rewriter, op.getType());
  rewriter.replaceOp(op, packed);
}

void ScatterOpConversion::emitWarpLocalScatter(
    ScatterOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  MLIRContext *ctx = op.getContext();
  Location loc = op.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  RankedTensorType dstType = op.getDst().getType();
  RankedTensorType srcType = op.getSrc().getType();

  StringAttr kBlock = str_attr("block");
  StringAttr kWarp = str_attr("warp");
  StringAttr kLane = str_attr("lane");
  StringAttr kRegister = str_attr("register");
  StringAttr kScatterDim = rewriter.getStringAttr("dim" + Twine(op.getAxis()));

  SmallVector<StringAttr> allDims, otherDims;
  for (unsigned dim = 0, rank = dstType.getRank(); dim < rank; ++dim) {
    allDims.push_back(str_attr("dim" + Twine(dim)));
    if (dim != op.getAxis())
      otherDims.push_back(allDims.back());
  }

  LinearLayout dstLayout = toLinearLayout(dstType);
  LinearLayout srcLayout = toLinearLayout(srcType);
  LinearLayout invDstLayout = dstLayout.pseudoinvert();
  bool hasCombine = !op.getCombineOp().empty() || op.getReduceKindAttr();
  bool includeSelf = op.getIncludeSelf();
  // We split the destination inverse layout into:
  // - "other" dims: all logical dims except the scatter axis.
  // - "axis" dim: the logical scatter axis only.
  // This lets us precompute the physical components that depend only on the
  // other dims (static per source element), and defer the axis-dependent part
  // to runtime (because indices are dynamic).
  LinearLayout invDstOtherLayout =
      invDstLayout.sublayout(otherDims, {kLane, kRegister});
  LinearLayout invDstAxisLayout =
      invDstLayout.sublayout({kScatterDim}, {kLane, kRegister});
  SmallVector<Value> dstValues =
      unpackLLElements(loc, adaptor.getDst(), rewriter);
  SmallVector<Value> srcValues =
      unpackLLElements(loc, adaptor.getSrc(), rewriter);
  SmallVector<Value> idxValues =
      unpackLLElements(loc, adaptor.getIndices(), rewriter);

  unsigned srcRegsPerThread = srcLayout.getInDimSize(kRegister);
  assert(srcRegsPerThread == srcValues.size());
  assert(srcRegsPerThread == idxValues.size());
  unsigned dstRegsPerThread = dstLayout.getInDimSize(kRegister);
  assert(dstRegsPerThread == dstValues.size());
  unsigned numLanes = srcLayout.getInDimSize(kLane);

  Value laneId = getLaneAndWarpId(rewriter, loc).first;
  auto getOutPos = [&](const LinearLayout &layout, StringAttr dim) {
    auto it = llvm::find(layout.getOutDimNames(), dim);
    assert(it != layout.getOutDimNames().end() && "missing output dim");
    return static_cast<int>(std::distance(layout.getOutDimNames().begin(), it));
  };
  int dstOtherLaneOutPos = getOutPos(invDstOtherLayout, kLane);
  int dstOtherRegOutPos = getOutPos(invDstOtherLayout, kRegister);
  int dstAxisLaneOutPos = getOutPos(invDstAxisLayout, kLane);
  int dstAxisRegOutPos = getOutPos(invDstAxisLayout, kRegister);
  // Precompute the set of physical (lane,reg) "parts" contributed by the axis
  // dimension alone. Later we can XOR these with the "other-dims" base to get
  // the final physical destination.
  SmallVector<int32_t> dstAxisLaneParts;
  SmallVector<int32_t> dstAxisRegParts;
  llvm::SmallDenseSet<int32_t> dstAxisLanePartSet;
  llvm::SmallDenseSet<int32_t> dstAxisRegPartSet;
  for (int32_t axisVal = 0; axisVal < dstType.getShape()[op.getAxis()];
       ++axisVal) {
    SmallVector<std::pair<StringAttr, int32_t>> axisPhysCoordsConst =
        invDstAxisLayout.apply({{kScatterDim, axisVal}});
    int32_t lanePart = axisPhysCoordsConst[dstAxisLaneOutPos].second;
    int32_t regPart = axisPhysCoordsConst[dstAxisRegOutPos].second;
    if (dstAxisLanePartSet.insert(lanePart).second)
      dstAxisLaneParts.push_back(lanePart);
    if (dstAxisRegPartSet.insert(regPart).second)
      dstAxisRegParts.push_back(regPart);
  }

  struct SrcCandidateInfo {
    // The precomputed destination physical base contributed by non-axis dims.
    int32_t dstLaneBase;
    int32_t dstRegBase;
    // Reg candidates that can be reached after applying axis parts.
    llvm::BitVector possibleDstRegs;
    // Physical source (reg, lane) that produces this logical source element.
    unsigned srcReg;
    unsigned srcLane;
  };
  SmallVector<SrcCandidateInfo> srcCandidates;
  srcCandidates.reserve(srcRegsPerThread * numLanes);

  SmallVector<int> srcLogicalOutPos(srcType.getRank());
  for (unsigned d = 0; d < srcType.getRank(); ++d) {
    srcLogicalOutPos[d] = getOutPos(srcLayout, allDims[d]);
  }
  // Enumerate only the current warp-local physical fragment and map each
  // (srcReg, srcLane) directly to logical source coordinates.
  SmallVector<std::pair<StringAttr, int32_t>> srcPhysCoordsConst;
  srcPhysCoordsConst.reserve(srcLayout.getInDims().size());
  SmallVector<std::pair<StringAttr, int32_t>> dstOtherCoordsConst;
  dstOtherCoordsConst.reserve(otherDims.size());
  SmallVector<int32_t> logicalCoords(srcType.getRank(), 0);
  for (unsigned srcLane = 0; srcLane < numLanes; ++srcLane) {
    for (unsigned srcReg = 0; srcReg < srcRegsPerThread; ++srcReg) {
      srcPhysCoordsConst.clear();
      for (StringAttr inDim : srcLayout.getInDimNames()) {
        int32_t inVal = 0;
        if (inDim == kRegister) {
          inVal = static_cast<int32_t>(srcReg);
        } else if (inDim == kLane) {
          inVal = static_cast<int32_t>(srcLane);
        } else if (inDim != kWarp && inDim != kBlock) {
          // Warp-local source fragments should not vary on extra input dims.
          assert(srcLayout.getInDimSize(inDim) == 1 &&
                 "unexpected varying source in-dim in warp-local scatter");
        }
        srcPhysCoordsConst.push_back({inDim, inVal});
      }
      SmallVector<std::pair<StringAttr, int32_t>> srcLogicalCoordsConst =
          srcLayout.apply(srcPhysCoordsConst);
      for (unsigned d = 0; d < srcType.getRank(); ++d) {
        logicalCoords[d] = srcLogicalCoordsConst[srcLogicalOutPos[d]].second;
      }

      dstOtherCoordsConst.clear();
      for (unsigned d = 0, rank = dstType.getRank(); d < rank; ++d) {
        if (d == op.getAxis())
          continue;
        dstOtherCoordsConst.push_back({allDims[d], logicalCoords[d]});
      }
      SmallVector<std::pair<StringAttr, int32_t>> dstPhysOtherConst =
          invDstOtherLayout.apply(dstOtherCoordsConst);
      int32_t dstLaneBase = dstPhysOtherConst[dstOtherLaneOutPos].second;
      int32_t dstRegBase = dstPhysOtherConst[dstOtherRegOutPos].second;

      // Early prune: if no axis-part can yield a valid lane for this candidate,
      // it can never land in this warp and can be skipped.
      bool mayHitLane = false;
      for (int32_t lanePart : dstAxisLaneParts) {
        int32_t dstLane = dstLaneBase ^ lanePart;
        if (dstLane >= 0 && dstLane < static_cast<int32_t>(numLanes)) {
          mayHitLane = true;
          break;
        }
      }
      if (!mayHitLane)
        continue;

      // Precompute which destination regs are reachable by combining the
      // axis-reg parts with the other-dims base. This shrinks the per-candidate
      // inner loop from all regs to only the reachable ones.
      llvm::BitVector possibleDstRegs(dstRegsPerThread);
      for (int32_t regPart : dstAxisRegParts) {
        int32_t dstReg = dstRegBase ^ regPart;
        if (dstReg >= 0 && dstReg < static_cast<int32_t>(dstRegsPerThread))
          possibleDstRegs.set(static_cast<unsigned>(dstReg));
      }
      if (possibleDstRegs.none())
        continue;

      srcCandidates.push_back({dstLaneBase, dstRegBase,
                               std::move(possibleDstRegs), srcReg, srcLane});
    }
  }

  SmallVector<Value> results(dstValues.begin(), dstValues.end());
  SmallVector<Value> hasValue;
  if (hasCombine && !includeSelf) {
    hasValue.resize(dstRegsPerThread, b.false_val());
  }
  llvm::SmallDenseMap<unsigned, Value, 32> shuffledIdxCache;
  llvm::SmallDenseMap<unsigned, Value, 32> shuffledSrcCache;
  Value axisUpperBound = b.i32_val(dstType.getShape()[op.getAxis()]);
  Value numLanesVal = b.i32_val(numLanes);
  Value dstRegsPerThreadVal = b.i32_val(dstRegsPerThread);
  auto getShuffled = [&](const SmallVector<Value> &vals,
                         llvm::SmallDenseMap<unsigned, Value, 32> &cache,
                         unsigned reg, unsigned lane) {
    // Multiple candidates can map to the same (srcReg, srcLane). Cache the
    // shuffle to avoid emitting duplicate shuffle instructions.
    unsigned key = reg * numLanes + lane;
    auto it = cache.find(key);
    if (it != cache.end())
      return it->second;
    Value v = targetInfo.shuffleIdx(rewriter, loc, vals[reg], b.i32_val(lane));
    cache.try_emplace(key, v);
    return v;
  };

  // Group candidates by (srcReg, srcLane) so we compute the axis layout once
  // per unique shuffled index/value pair, instead of once per candidate.
  llvm::SmallDenseMap<unsigned, SmallVector<unsigned, 4>, 32> candidatesBySrc;
  candidatesBySrc.reserve(srcCandidates.size());
  for (unsigned i = 0; i < srcCandidates.size(); ++i) {
    const auto &cand = srcCandidates[i];
    unsigned key = cand.srcReg * numLanes + cand.srcLane;
    candidatesBySrc[key].push_back(i);
  }

  // Main scatter loop:
  // For each candidate logical source element (now represented by its physical
  // (srcReg, srcLane) plus precomputed destination base), do:
  // 1. Shuffle the dynamic index and source value into the current lane.
  // 2. Compute the axis-dependent (lane,reg) part of the destination from the
  //    dynamic index using invDstAxisLayout.
  // 3. Combine with the base (XOR) and update the destination if this lane/reg
  //    matches the current thread and is in bounds.
  auto updateDstReg = [&](unsigned dRegIdx, Value take,
                          Value shuffledSrcValue) {
    if (!hasCombine) {
      results[dRegIdx] = b.select(take, shuffledSrcValue, results[dRegIdx]);
      return;
    }

    if (includeSelf) {
      results[dRegIdx] = applyScatterCombine(
          loc, rewriter, op, results[dRegIdx], shuffledSrcValue, take);
      return;
    }

    Value hadValue = hasValue[dRegIdx];
    Value doCombine = b.and_(take, hadValue);
    Value doInit = b.and_(take, b.xor_(hadValue, b.i1_val(1)));
    Value combined = applyScatterCombine(loc, rewriter, op, results[dRegIdx],
                                         shuffledSrcValue, doCombine);
    Value updated = b.select(doInit, shuffledSrcValue, results[dRegIdx]);
    updated = b.select(doCombine, combined, updated);
    results[dRegIdx] = updated;
    hasValue[dRegIdx] = b.or_(hadValue, take);
  };

  for (auto &entry : candidatesBySrc) {
    unsigned key = entry.first;
    unsigned srcReg = key / numLanes;
    unsigned srcLane = key % numLanes;
    const SmallVector<unsigned, 4> &group = entry.second;

    Value shuffledIdx =
        getShuffled(idxValues, shuffledIdxCache, srcReg, srcLane);
    Value shuffledSrc =
        getShuffled(srcValues, shuffledSrcCache, srcReg, srcLane);
    Value dstAxis = convertIndexToI32(loc, shuffledIdx, rewriter);

    // Bounds check against the logical destination axis size.
    Value dstAxisInBounds = b.icmp_ult(dstAxis, axisUpperBound);

    // Apply axis-only inverse layout once per (srcReg, srcLane) group.
    SmallVector<std::pair<StringAttr, Value>> dstPhysAxis = applyLinearLayout(
        loc, rewriter, invDstAxisLayout, {{kScatterDim, dstAxis}});
    Value axisLanePart = dstPhysAxis[dstAxisLaneOutPos].second;
    Value axisRegPart = dstPhysAxis[dstAxisRegOutPos].second;

    for (unsigned idx : group) {
      const SrcCandidateInfo &cand = srcCandidates[idx];
      Value dstLane = b.xor_(b.i32_val(cand.dstLaneBase), axisLanePart);
      Value dstReg = b.xor_(b.i32_val(cand.dstRegBase), axisRegPart);
      Value laneInBounds = b.icmp_ult(dstLane, numLanesVal);
      Value regInBounds = b.icmp_ult(dstReg, dstRegsPerThreadVal);
      Value laneMatches = b.icmp_eq(dstLane, laneId);
      Value takeBase =
          b.and_(b.and_(b.and_(dstAxisInBounds, laneInBounds), regInBounds),
                 laneMatches);

      // Update only those regs that are reachable for this candidate.
      for (int dReg = cand.possibleDstRegs.find_first(); dReg >= 0;
           dReg = cand.possibleDstRegs.find_next(dReg)) {
        Value take = b.and_(takeBase, b.icmp_eq(dstReg, b.i32_val(dReg)));
        updateDstReg(static_cast<unsigned>(dReg), take, shuffledSrc);
      }
    }
  }

  rewriter.replaceOp(op, packLLElements(loc, getTypeConverter(), results,
                                        rewriter, op.getType()));
}

} // namespace

void triton::populateScatterOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                             RewritePatternSet &patterns,
                                             const TargetInfoBase &targetInfo,
                                             PatternBenefit benefit) {
  patterns.insert<ScatterOpConversion>(typeConverter, targetInfo, benefit);
}
