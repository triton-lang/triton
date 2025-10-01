#include "AsyncUtility.h"
#include "AtomicRMWOpsEmitter.h"
#include "BufferOpsEmitter.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"

using namespace mlir;
using namespace mlir::triton::gpu;

using ::mlir::LLVM::getSharedMemoryBase;
using ::mlir::LLVM::AMD::getVectorSize;
using ::mlir::LLVM::AMD::llLoad;
using ::mlir::LLVM::AMD::llStore;
using ::mlir::triton::AMD::ISAFamily;
using ::mlir::triton::gpu::getTotalElemsPerThread;

namespace {

std::optional<const char *> getAMDGPUMemScopeStr(MemSyncScope scope) {
  switch (scope) {
  case MemSyncScope::GPU:
    return "agent";
  case MemSyncScope::CTA:
    return "workgroup";
  // The default AMDHSA LLVM Sync Scope is "system", so no string is
  // provided here
  case MemSyncScope::SYSTEM:
  default:
    return "";
  }
}

std::pair<bool, bool> getOrderingFlags(MemSemantic memOrdering) {
  bool emitReleaseFence = false;
  bool emitAcquireFence = false;
  switch (memOrdering) {
  case MemSemantic::RELAXED:
    // In this case, no memory fences are needed
    break;
  case MemSemantic::RELEASE:
    emitReleaseFence = true;
    break;
  case MemSemantic::ACQUIRE:
    emitAcquireFence = true;
    break;
  case MemSemantic::ACQUIRE_RELEASE:
    emitAcquireFence = true;
    emitReleaseFence = true;
  default:
    // default == acq_rel, so we emit the same barriers
    emitAcquireFence = true;
    emitReleaseFence = true;
  }
  return {emitAcquireFence, emitReleaseFence};
}

LogicalResult emitFence(Operation *op, ConversionPatternRewriter &rewriter,
                        Location loc, MemSemantic memOrdering,
                        MemSyncScope memScope, bool preAtomic) {
  // This function emits an LLVM::FenceOp which will get lowered by the
  // LLVM backend to the right scope and ordering instructions, as
  // described in the "atomicrmw" entries for "global" address-space,
  // in the "AMDHSA Memory Model Code Sequences GFX942"
  // table in https://llvm.org/docs/AMDGPUUsage.html#memory-model-gfx942
  //
  // Triton supports three scopes for atomic access
  // 1. System
  // 2. GPU (default) ('Agent' for AMDGPU)
  // 3. CTA ('Workgroup' for AMDGPU)
  //
  // and 4 orderings
  // 1. Relaxed
  // 2. Acquire
  // 3. Release
  // 4. AcquireRelease
  //
  // The following table shows the scope and ordering instructions that
  // are emitted by this function for each combination of scope and ordering
  // for buffer-atomic instructions.
  //
  // Note: In the following comments, "[buffer-atomic_0.. buffer-atomic_n]"
  // represents a sequence of buffer-atomic instructions that are lowered from
  // a single tl.atomic_*
  //
  // Unordered(Relaxed):
  //   agent/workgroup: Instr seq: [buffer-atomic_0.. buffer-atomic_n]
  //                    No scope/ordering instrs are required.
  //   system: //TODO:
  // Acquire:
  //   workgroup: Instr seq: [buffer-atomic_0.. buffer-atomic_n]
  //              All waves in the workgroup use same L1 and L2.
  //              No scope/ordering instrs are required.
  //   agent: Instr seq: [buffer-atomic_0.. buffer-atomic_n],
  //                     s_waitcnt vmcnt(0), buffer_inv sc1=1
  //          Waves across an agent may use different L1 and L2.
  //          Atomic ops bypass L1 and operate on L2.
  //          s_waitcnt vmcnt(0) ensures that the atomicrmw has completed
  //          before invalidating the cache. buffer_inv sc1=1 will a) L1:
  //          invalidate cache b) L2: Invalidate non-coherently modified lines
  //          if multiple L2s are configured, NOP otherwise. This buffer_inv
  //          ensures that following loads do not see stale global values.
  //   system: //TODO:
  //
  // Release:
  //   workgroup: Instr seq: [buffer-atomic_0.. buffer-atomic_n]
  //              All waves in the workgroup use same L1 and L2 so all
  //              previous global writes of a waver are visible to all other
  //              waves in the workgroup. LDS operations for all waves are
  //              executed in a total global ordering and are observed by all
  //              waves in the workgroup. So LDS stores issued before the
  //              release will be visible to LDS loads after the read of the
  //              released buffer-atomic. So, swait_cnt lgkmcnt is not
  //              required.
  //   agent: Instr seq: buffer_wbl2 sc1=1, s_waitcnt vmcnt(0),
  //                     [buffer-atomic_0.. buffer-atomic_n]
  //          buffer_wbl2 sc1=1 ensures that dirtly L2 lines are visible to
  //          CUs that don't use the same L2.
  //          From SIMemoryLegalizer.cpp SIGfx940CacheControl::insertRelease:
  //            "Inserting a "S_WAITCNT vmcnt(0)" before is not required
  //             because the hardware does not reorder memory operations by
  //             the same wave with respect to a following "BUFFER_WBL2".
  //             The "BUFFER_WBL2" is guaranteed to initiate writeback of
  //             any dirty cache lines of earlier writes by the same wave.
  //             A "S_WAITCNT vmcnt(0)" is needed after to ensure the writeback
  //             has completed.""
  //   system: //TODO:
  //
  // AcquireRelease:
  //   Instr seq: Release scope/order insts,
  //              [buffer-atomic_0..buffer-atomic_n],
  //              Acquire scope/order instrs.
  //
  // LLVM::FenceOp lowering will emit the required cache ops and s_waitcnt
  // vmcnt(0) instrs

  auto [emitReleaseFence, emitAcquireFence] = getOrderingFlags(memOrdering);
  if (MemSyncScope::SYSTEM == memScope)
    return rewriter.notifyMatchFailure(
        op, "System memory scope is not supported for Buffer Atomic Ops");
  auto scopeStr = getAMDGPUMemScopeStr(memScope);
  if (!scopeStr)
    return rewriter.notifyMatchFailure(
        op, "Unsupported memory scope for Buffer Atomic Ops");

  StringAttr scope = mlir::StringAttr::get(loc.getContext(), *scopeStr);

  if (emitReleaseFence && preAtomic) {
    rewriter.create<LLVM::FenceOp>(loc, TypeRange{},
                                   LLVM::AtomicOrdering::release, scope);
  }

  if (emitAcquireFence && !preAtomic) {
    rewriter.create<LLVM::FenceOp>(loc, TypeRange{},
                                   LLVM::AtomicOrdering::acquire, scope);
  }
  return success();
}

// Return a predicate that is true only if the current thread holds unique data,
// according to freeVarsMask.
Value emitRedundantThreadPredicate(
    const llvm::MapVector<StringAttr, int32_t> &freeVarMasks,
    ConversionPatternRewriter &rewriter, Location loc,
    const AMD::TargetInfo &targetInfo) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto ctx = rewriter.getContext();
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  auto kBlock = str_attr("block");

  Value zero = b.i32_val(0);
  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  Value blockId = freeVarMasks.lookup(kBlock) == 0
                      ? zero
                      : targetInfo.getClusterCTAId(rewriter, loc);

  Value pred = b.true_val();
  auto dimNames = {kLane, kWarp, kBlock};
  auto dimIds = {laneId, warpId, blockId};
  for (auto [dimName, dimId] : llvm::zip(dimNames, dimIds)) {
    int32_t mask = freeVarMasks.lookup(dimName);
    if (mask != 0) {
      auto dimPred = b.icmp_eq(b.and_(dimId, b.i32_val(mask)), zero);
      pred = b.and_(pred, dimPred);
    }
  }
  return pred;
}

std::pair<Block *, Block *> emitBranch(RewriterBase &rewriter, Location loc,
                                       Value cond) {
  Block *currentBlock = rewriter.getInsertionBlock();
  Block *after =
      rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
  Block *body = rewriter.createBlock(after);
  rewriter.setInsertionPointToEnd(currentBlock);
  rewriter.create<LLVM::CondBrOp>(loc, cond, body, after);
  rewriter.setInsertionPointToStart(body);
  rewriter.create<LLVM::BrOp>(loc, after);
  rewriter.setInsertionPointToStart(body);
  return {body, after};
}

// Contains some helper functions for both Load and Store conversions.
struct LoadStoreConversionBase {
  explicit LoadStoreConversionBase(const AMD::TargetInfo &targetInfo,
                                   ModuleAxisInfoAnalysis &axisAnalysisPass)
      : targetInfo(targetInfo), axisAnalysisPass(axisAnalysisPass) {}

  // Create a LLVM vector of type `vecTy` containing all zeros
  Value createZeroVector(OpBuilder &builder, Location loc,
                         VectorType vecTy) const {
    mlir::Attribute zeroAttr = builder.getZeroAttr(vecTy.getElementType());
    auto denseValue =
        DenseElementsAttr::get(cast<mlir::ShapedType>(vecTy), zeroAttr);
    Value zeroVal = builder.create<LLVM::ConstantOp>(loc, vecTy, denseValue);
    return zeroVal;
  }

  // Given a vector of values `elems` and a starting point `start`, create a
  // LLVM vector of length `vec` whose elements are `elems[start, ...,
  // elems+vec-1]`
  Value packElementRangeIntoVector(RewriterBase &rewriter,
                                   const LLVMTypeConverter *typeConverter,
                                   Location loc, VectorType vecTy,
                                   ArrayRef<Value> elems, int64_t start) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    int64_t vec = vecTy.getNumElements();
    // If we need to mask the loaded value with other elements
    Value v = b.undef(vecTy);
    for (size_t s = 0; s < vec; ++s) {
      Value otherElem = elems[start + s];
      Value indexVal =
          LLVM::createIndexConstant(rewriter, loc, typeConverter, s);
      v = b.insert_element(vecTy, v, otherElem, indexVal);
    }
    return v;
  }

  // Return a tensor of pointers with the same type of `basePtr` and the same
  // shape of `offset`
  Type getPointerTypeWithShape(Value basePtr, Value offset) const {
    Type basePtrType = basePtr.getType();
    auto offsetType = cast<RankedTensorType>(offset.getType());
    return offsetType.cloneWith(std::nullopt, basePtrType);
  }

  // Unpack the elements contained in a `llvmStruct` into a `SmallVector` of
  // `Value`s. While you do that, check also the alignment of the mask and
  // update the vector length `vec` accordingly
  SmallVector<Value>
  getMaskElemsAndUpdateVeclen(ConversionPatternRewriter &rewriter, Location loc,
                              Value llMask, Value mask, unsigned &vec) const {
    SmallVector<Value> maskElems;
    if (llMask) {
      vec = std::min<size_t>(vec, getMaskAlignment(mask));
      maskElems = unpackLLElements(loc, llMask, rewriter);
    }
    return maskElems;
  }

  unsigned getMaskAlignment(Value mask) const {
    return axisAnalysisPass.getMaskAlignment(mask);
  }

protected:
  const AMD::TargetInfo &targetInfo;
  ModuleAxisInfoAnalysis &axisAnalysisPass;
};

// Contains some helper functions for direct to lds loads.
struct DirectToLdsLoadConversionBase : public LoadStoreConversionBase {
  explicit DirectToLdsLoadConversionBase(
      const AMD::TargetInfo &targetInfo,
      ModuleAxisInfoAnalysis &axisAnalysisPass)
      : LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  // direct to lds loads do not support per lane shared offsets. We need to
  // ensure that we write coalesced into shared memory. This means we cannot
  // exceed the supported load width because splitting them would cause strided
  // (non coalesced) writes. Additionally:
  //   1) For *non* swizzled shared encodings we check if they result in
  //      coalesced writes and can then lower them directly to the intrinsics.
  //   2) For swizzled shared encodings we need to transfer the swizzling to the
  //      source pointers. For now this is done by swizzling the pointers
  //      between the lane of a warp via permute. This only works if the swizzle
  //      pattern does not exchange elements between warps which holds for all
  //      our swizzle patterns. There is still a check performed to not silently
  //      produce wrong results if we invalidate the condition in the future
  LogicalResult canWriteCoalesced(RewriterBase &rewriter, Operation *op,
                                  RankedTensorType srcTy, MemDescType dstTy,
                                  unsigned vectorSize,
                                  bool hasSwizzling) const {
    int vecBits = vectorSize * dstTy.getElementTypeBitWidth();
    if (!targetInfo.supportsDirectToLdsLoadBitWidth(vecBits)) {
      LDBG(op << " results in unsupported load bitwidth: " << vecBits);
      return failure();
    }
    // Compute the blocked -> shared linear layout to check preconditions
    LinearLayout srcLayout = triton::gpu::toLinearLayout(srcTy);
    LinearLayout sharedLayout;
    if (auto paddedEnc = dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(
            dstTy.getEncoding())) {
      sharedLayout = paddedEnc.getLinearComponent();
    } else {
      sharedLayout = triton::gpu::toLinearLayout(dstTy);
    }
    LinearLayout srcToSharedLayout = srcLayout.invertAndCompose(sharedLayout);

    unsigned threadsPerWarp = lookupThreadsPerWarp(rewriter);
    if (!hasSwizzling &&
        !LLVM::AMD::canCoalesceWriteIntoSharedMemory(
            rewriter, srcToSharedLayout, threadsPerWarp, vectorSize)) {
      LDBG(op << " does not write coalesced into LDS and is not swizzled");
      return failure();
    }

    if (hasSwizzling && !LLVM::AMD::doesSwizzleInsideWarp(
                            rewriter, srcToSharedLayout, threadsPerWarp)) {
      LDBG(op << " does swizzle across warp boundaries");
      return failure();
    }
    return success();
  }

  // For each load emit the computation to get the lane id offset which holds
  // the source pointers/offsets we need to store to shared memory
  SmallVector<Value>
  emitSwizzledLaneOffsets(RewriterBase &rewriter, Operation *op,
                          RankedTensorType srcTy, MemDescType swizzledTy,
                          MemDescType flatTy, Value llDst, Type resElemTy,
                          unsigned vec) const {
    auto loc = op->getLoc();
    TritonLLVMOpBuilder b(loc, rewriter);

    // Create regToShared layout for the swizzled and flat encoding
    auto regLayout = triton::gpu::toLinearLayout(srcTy);

    auto sharedSwizz = triton::gpu::toLinearLayout(swizzledTy);
    auto sharedFlat = triton::gpu::toLinearLayout(flatTy);

    auto regToSharedSwizzled = regLayout.invertAndCompose(sharedSwizz);
    auto regToSharedFlat = regLayout.invertAndCompose(sharedFlat);

    MLIRContext *ctx = rewriter.getContext();
    StringAttr kBlock = str_attr("block");
    StringAttr kRegister = str_attr("register");
    StringAttr kLane = str_attr("lane");
    StringAttr kWarp = str_attr("warp");
    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    Value blockId = b.i32_val(0);

    int numberOfLoads = regToSharedSwizzled.getInDimSize(kRegister) / vec;

    // For each load compute the difference between the flat and the swizzled
    // linear offsets into shared memory
    // TODO (alex): this is only correct as long as the lds view is a contiguous
    // block. So this can break if we slice along the 2 minor dimensions
    SmallVector<Value> swizzledOffsets;
    swizzledOffsets.reserve(numberOfLoads);
    auto vecVal = b.i32_val(vec);
    for (int i = 0; i < numberOfLoads; i++) {
      auto regId = b.i32_val(i * vec);

      std::array<std::pair<StringAttr, Value>, 4> indices{{
          {kRegister, regId},
          {kLane, laneId},
          {kWarp, warpId},
          {kBlock, blockId},
      }};

      Value swizzledOffset =
          applyLinearLayout(loc, rewriter, regToSharedSwizzled, indices)[0]
              .second;
      Value flatOffset =
          applyLinearLayout(loc, rewriter, regToSharedFlat, indices)[0].second;

      // Normalize the offset by vecTy to obtain the offset in lanes
      auto laneOffet = b.sdiv(b.sub(swizzledOffset, flatOffset), vecVal);
      swizzledOffsets.push_back(laneOffet);
    }
    return swizzledOffsets;
  }

  // Swizzle the mask (1bit) based on selectLane via ballot
  Value shuffleMask(RewriterBase &rewriter, TritonLLVMOpBuilder &b,
                    Location loc, const TargetInfoBase &targetInfo,
                    Value selectLane, Value mask) const {
    auto warpMask =
        targetInfo.ballot(rewriter, loc, rewriter.getI64Type(), mask);
    // Extract the selectLane bit
    auto bitMask = b.lshr(warpMask, b.zext(rewriter.getI64Type(), selectLane));
    return b.trunc(i1_ty, bitMask);
  }

  SmallVector<Value> zipLoadValues(RewriterBase &rewriter, Location loc,
                                   unsigned vec, ArrayRef<Value> srcElems,
                                   Type srcTy, ArrayRef<Value> maskElems,
                                   ArrayRef<Value> otherElems, Type otherTy,
                                   ArrayRef<Value> swizzledLaneOffsets) const {
    TritonLLVMOpBuilder b(loc, rewriter);
    SmallVector<Value> loadVals;
    auto structTy = LLVM::LLVMStructType::getLiteral(
        rewriter.getContext(), ArrayRef<Type>{srcTy, i1_ty, otherTy, i32_ty});
    for (int i = 0; i < srcElems.size(); i++) {
      Value packedArr = rewriter.create<LLVM::UndefOp>(loc, structTy);
      // src
      packedArr = b.insert_val(packedArr, srcElems[i], 0);
      // mask
      auto maskElem = maskElems.empty() ? b.true_val() : maskElems[i];
      packedArr = b.insert_val(packedArr, maskElem, 1);
      // other
      if (!otherElems.empty())
        packedArr = b.insert_val(packedArr, otherElems[i], 2);
      // swizzleOffset are per vec so we need to duplicate values vec times
      auto swizzleOffset = swizzledLaneOffsets.empty()
                               ? b.i32_val(0)
                               : swizzledLaneOffsets[i / vec];
      packedArr = b.insert_val(packedArr, swizzleOffset, 3);

      loadVals.push_back(packedArr);
    }
    return loadVals;
  }

  auto unzipLoadValues(RewriterBase &rewriter, Location loc, int startIdx,
                       ArrayRef<Value> values, Type srcTy, Type otherTy,
                       bool hasOther, unsigned vec) const {
    TritonLLVMOpBuilder b(loc, rewriter);
    auto structElem = values[startIdx];
    Value offsetElem = b.extract_val(srcTy, structElem, 0);
    Value maskElem = b.extract_val(i1_ty, structElem, 1);
    // Gather other elements
    SmallVector<Value> otherElems;
    if (hasOther) {
      for (int i = 0; i < vec; i++) {
        otherElems.push_back(b.extract_val(otherTy, values[startIdx + i], 2));
      }
    }

    Value swizzleLaneOffset = b.extract_val(i32_ty, structElem, 3);

    return std::make_tuple(offsetElem, maskElem, std::move(otherElems),
                           swizzleLaneOffset);
  }

  void applySwizzling(RewriterBase &rewriter, Location loc, Value &srcOrOffset,
                      Value &mask, Value laneId,
                      Value swizzleLaneOffset) const {
    TritonLLVMOpBuilder b(loc, rewriter);
    // laneId + swizzleOffset will always stay inside the warp [0,
    // threadsPerWarp) because we only swizzle inside a warp
    Value swizzledLaneId = b.add(laneId, swizzleLaneOffset);
    // Shuffle based on swizzleLaneId to apply the swizzling
    srcOrOffset =
        targetInfo.shuffleIdx(rewriter, loc, srcOrOffset, swizzledLaneId);

    if (mask) {
      mask = shuffleMask(rewriter, b, loc, targetInfo, swizzledLaneId, mask);
    }
  }

  void lowerDirectToLDSLoad(
      RewriterBase &rewriter, Location loc, RankedTensorType srcTy,
      MemDescType dstTy, SmallVector<Value> loadVals, Value llDst,
      Type resElemTy, unsigned vec,
      std::function<SmallVector<Value>(RewriterBase &, Location,
                                       ArrayRef<Value>, Value, int, VectorType)>
          lowerInst) const {
    TritonLLVMOpBuilder b(loc, rewriter);
    auto *ctx = rewriter.getContext();

    // Build src to shared layout and remove broadcasted registers
    auto srcLayout = triton::gpu::toLinearLayout(srcTy);
    auto removeBroadcastSrc = actionRemoveBroadcastedRegs(srcLayout);
    srcLayout = removeBroadcastSrc.apply(srcLayout);
    loadVals = removeBroadcastSrc.apply(loadVals);

    LinearLayout sharedLayout;
    if (auto paddedEnc = dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(
            dstTy.getEncoding())) {
      sharedLayout = paddedEnc.getLinearComponent();
    } else {
      sharedLayout = triton::gpu::toLinearLayout(dstTy);
    }
    auto cvt = srcLayout.invertAndCompose(sharedLayout);
    cvt = cvt.sublayout(
        {str_attr("register"), str_attr("lane"), str_attr("warp")},
        {str_attr("offset")});

    auto smemObj =
        LLVM::getSharedMemoryObjectFromStruct(loc, llDst, resElemTy, rewriter);
    auto affineOffset = smemObj.getShmemOffset(loc, rewriter, dstTy);
    auto maskSpanAffineOffset = SharedMemoryObject::getMaskSpanOffsets(dstTy);
    auto [_, warpId] = getLaneAndWarpId(rewriter, loc);
    auto calcPaddedOffset = [&](Value smemOffset) {
      TritonLLVMOpBuilder b(loc, rewriter);
      auto bitwidth = dstTy.getElementTypeBitWidth();
      if (auto paddedEnc = dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(
              dstTy.getEncoding())) {
        // Apply the offset needed for padding.
        Value padOffset = emitPadding(loc, rewriter, paddedEnc, bitwidth,
                                      smemOffset, /*offsetInBytes=*/true);
        smemOffset = b.add(smemOffset, padOffset);
      }
      return smemOffset;
    };
    // We pass laneId==0 because GFX9 requires a scalar base pointer into LDS
    lowerLdSt(loc, ctx, cvt, loadVals, resElemTy, smemObj.getBase(),
              calcPaddedOffset, affineOffset, maskSpanAffineOffset,
              b.i32_val(0), warpId, rewriter, targetInfo, vec, lowerInst);
  }

  void emitOtherStore(RewriterBase &rewriter, Location loc,
                      const LLVMTypeConverter *typeConverter, VectorType vecTy,
                      Value mask, ArrayRef<Value> otherElems, Value shmemAddr,
                      Value laneId, bool hasSwizzling,
                      Value swizzleLaneOffset) const {
    TritonLLVMOpBuilder b(loc, rewriter);
    Value storeVal = packElementRangeIntoVector(rewriter, typeConverter, loc,
                                                vecTy, otherElems, 0);
    Type ptrTy = shmemAddr.getType();
    Value ldsAddr = b.gep(ptrTy, vecTy, shmemAddr, laneId);
    if (hasSwizzling)
      ldsAddr = b.gep(ptrTy, vecTy, ldsAddr, swizzleLaneOffset);
    llStore(rewriter, loc, ldsAddr, storeVal, b.icmp_ne(mask, b.true_val()),
            CacheModifier::NONE, /*forceNoAliasAsyncLoads=*/true);
  }
};

struct LoadOpConversion : public ConvertOpToLLVMPattern<triton::LoadOp>,
                          public LoadStoreConversionBase {
  LoadOpConversion(LLVMTypeConverter &converter,
                   const AMD::TargetInfo &targetInfo,
                   ModuleAxisInfoAnalysis &axisAnalysisPass,
                   PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // original values
    Value ptr = op.getPtr();
    Value mask = op.getMask();
    Value other = op.getOther();

    // adaptor values
    assert(!isTensorPointerType(ptr.getType()) &&
           "Cannot convert load with a tensor pointer into LLVM; "
           "this case should be transformed to normal load before lowering");
    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();

    // Determine the vectorization size
    Type valueTy = op.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));
    unsigned vec = getVectorSize(ptr, axisAnalysisPass);
    unsigned numElems = getTotalElemsPerThread(ptr.getType());

    // Get the LLVM values for pointers
    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    assert(ptrElems.size() == numElems);

    // Get the LLVM values for mask
    SmallVector<Value> maskElems =
        getMaskElemsAndUpdateVeclen(rewriter, loc, llMask, mask, vec);

    SmallVector<Value> otherElems;
    if (other)
      otherElems = unpackLLElements(loc, llOther, rewriter);

    // vectorized iteration through all the pointer/mask/other elements
    const int valueElemNBits =
        std::max(8u, valueElemTy.getIntOrFloatBitWidth());
    const size_t valueElemNBytes = valueElemNBits / 8;
    const int numVecs = numElems / vec;

    auto cacheMod = op.getCache();
    SmallVector<Value> loadedVals;
    Type vecTy = LLVM::getVectorType(valueElemTy, vec);
    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      const size_t maxWordWidth = std::max<size_t>(32, valueElemNBits);
      const size_t totalWidth = valueElemNBits * vec;
      const size_t width = std::min(totalWidth, maxWordWidth);
      const size_t nWords = std::max<size_t>(1, totalWidth / width);
      const size_t wordNElems = width / valueElemNBits;
      const size_t movWidth = width < 16 ? 16 : width;
      assert(wordNElems * nWords * numVecs == numElems);

      Value pred = mask ? maskElems[vecStart] : b.int_val(1, 1);
      Value ptr = ptrElems[vecStart];

      Value falseVal = createZeroVector(rewriter, loc, cast<VectorType>(vecTy));
      // If we need to mask the loaded value with other elements
      if (otherElems.size() != 0)
        falseVal = packElementRangeIntoVector(
            rewriter, this->getTypeConverter(), loc, cast<VectorType>(vecTy),
            otherElems, vecStart);

      Value loadVal =
          llLoad(rewriter, loc, ptr, vecTy, pred, falseVal, cacheMod);
      for (size_t ii = 0; ii < vec; ++ii) {
        Value vecIdx = createIndexAttrConstant(
            rewriter, loc, getTypeConverter()->getIndexType(), ii);
        Value loaded = b.extract_element(valueElemTy, loadVal, vecIdx);
        loadedVals.push_back(loaded);
      }
    } // end vec

    Type llvmResultStructTy = getTypeConverter()->convertType(valueTy);
    Value resultStruct = packLLElements(loc, getTypeConverter(), loadedVals,
                                        rewriter, llvmResultStructTy);

    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct BufferLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::BufferLoadOp>,
      public LoadStoreConversionBase {
  BufferLoadOpConversion(LLVMTypeConverter &converter,
                         const AMD::TargetInfo &targetInfo,
                         ModuleAxisInfoAnalysis &axisAnalysisPass,
                         PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::amdgpu::BufferLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    LLVM::AMD::BufferEmitter bufferEmitter(rewriter, loc, targetInfo);

    // original values
    Value ptr = op.getPtr();
    Value offset = op.getOffsets();
    Value mask = op.getMask();
    Value other = op.getOther();
    auto cacheMod = op.getCache();

    // Converted values
    Value llPtr = adaptor.getPtr();
    Value llOffset = adaptor.getOffsets();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();
    Value llStride = adaptor.getStride();

    // Determine the vectorization size
    Type valueTy = op.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));
    Type ptrType = getPointerTypeWithShape(ptr, offset);
    unsigned numElems = getTotalElemsPerThread(ptrType);
    unsigned vec = getVectorSize(ptr, offset, axisAnalysisPass);

    // Get the offset
    SmallVector<Value> offsetElems = unpackLLElements(loc, llOffset, rewriter);
    assert(offsetElems.size() == numElems);

    // Get the mask
    SmallVector<Value> maskElems =
        getMaskElemsAndUpdateVeclen(rewriter, loc, llMask, mask, vec);

    // Get the `other` value (if any)
    SmallVector<Value> otherElems;
    if (llOther)
      otherElems = unpackLLElements(loc, llOther, rewriter);

    // Create the resource descriptor and then emit the buffer_load intrinsic(s)
    Value rsrcDesc = bufferEmitter.createResourceDescriptor(llPtr, llStride);
    SmallVector<Value> loadedVals;
    Type vecTy = LLVM::getVectorType(valueElemTy, vec);
    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      Value pred = mask ? maskElems[vecStart] : b.int_val(1, 1);
      Value falseVal = createZeroVector(rewriter, loc, cast<VectorType>(vecTy));
      if (otherElems.size() != 0)
        falseVal = packElementRangeIntoVector(
            rewriter, this->getTypeConverter(), loc, cast<VectorType>(vecTy),
            otherElems, vecStart);
      Value loadVal = bufferEmitter.emitLoad(
          vecTy, rsrcDesc, offsetElems[vecStart], pred, falseVal, cacheMod);
      for (size_t ii = 0; ii < vec; ++ii) {
        Value vecIdx = createIndexAttrConstant(
            rewriter, loc, getTypeConverter()->getIndexType(), ii);
        Value loaded = b.extract_element(valueElemTy, loadVal, vecIdx);
        loadedVals.push_back(loaded);
      }
    } // end vec

    Type llvmResultStructTy = getTypeConverter()->convertType(valueTy);
    Value resultStruct = packLLElements(loc, getTypeConverter(), loadedVals,
                                        rewriter, llvmResultStructTy);

    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct BufferLoadToLocalOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::BufferLoadToLocalOp>,
      public DirectToLdsLoadConversionBase {
  BufferLoadToLocalOpConversion(LLVMTypeConverter &converter,
                                const AMD::TargetInfo &targetInfo,
                                ModuleAxisInfoAnalysis &axisAnalysisPass,
                                PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        DirectToLdsLoadConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::amdgpu::BufferLoadToLocalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    LLVM::AMD::BufferEmitter bufferEmitter(rewriter, loc, targetInfo);

    // Original values
    Value ptr = op.getPtr();
    Value offset = op.getOffsets();
    Value mask = op.getMask();

    // Converted values
    Value llPtr = adaptor.getPtr();
    Value llOffset = adaptor.getOffsets();
    Value llDst = adaptor.getDest();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();
    Value llStride = adaptor.getStride();

    RankedTensorType ptrType =
        cast<RankedTensorType>(getPointerTypeWithShape(ptr, offset));

    // We can load N elements at a time if:
    //  1. Every group of N source pointers are contiguous.  For example, if
    //     N=2, then the pointers should be [x, x+1, y, y+1, ...].
    //  2. The mask (if present) has "alignment" N, meaning that each group of N
    //     mask bits are the same.  For example if N=2, the mask must be
    //     [x, x, y, y, ...].
    unsigned vec = getVectorSize(ptr, offset, axisAnalysisPass);
    SmallVector<Value> maskElems =
        getMaskElemsAndUpdateVeclen(rewriter, loc, llMask, mask, vec);

    SmallVector<Value> offsetElems = unpackLLElements(loc, llOffset, rewriter);
    SmallVector<Value> otherElems;
    if (llOther)
      otherElems = unpackLLElements(loc, llOther, rewriter);

    auto dstTy = op.getDest().getType();
    auto resElemTy = getTypeConverter()->convertType(dstTy.getElementType());
    auto dstEnc = dstTy.getEncoding();

    // For padded encodings restrict vec by the min interval
    if (auto padEnc = dyn_cast<PaddedSharedEncodingAttr>(dstEnc)) {
      vec = std::min(vec, padEnc.getMinInterval());
    }

    auto maybeSwizzledEnc = dyn_cast<SwizzledSharedEncodingAttr>(dstEnc);
    bool hasSwizzling = maybeSwizzledEnc && maybeSwizzledEnc.getMaxPhase() != 1;
    if (failed(canWriteCoalesced(rewriter, op, ptrType, dstTy, vec,
                                 hasSwizzling))) {
      return failure();
    }

    // For swizzled layouts we need to use the non swizzled layout to compute
    // the LDS addresses since we gather into LDS
    auto flatDstTy = dstTy;
    SmallVector<Value> swizzledLaneOffsets;

    if (hasSwizzling) {
      // TODO (alex): this is only correct as long as the lds view is a
      // contiguous block. So this can break if we slice along the 2 minor
      // dimensions.
      auto flatSharedEnc = SwizzledSharedEncodingAttr::get(
          op->getContext(), maybeSwizzledEnc.getVec(), 1, 1,
          maybeSwizzledEnc.getOrder(), maybeSwizzledEnc.getCTALayout());
      flatDstTy = MemDescType::get(dstTy.getShape(), dstTy.getElementType(),
                                   flatSharedEnc, dstTy.getMemorySpace());
      swizzledLaneOffsets = emitSwizzledLaneOffsets(
          rewriter, op, ptrType, dstTy, flatDstTy, llDst, resElemTy, vec);
    }

    auto offsetTy = offsetElems[0].getType();
    bool hasOther = !otherElems.empty();
    auto otherTy = hasOther ? otherElems[0].getType() : i1_ty;
    // Zip buffer_offset, mask, other, swizzleOffsets for lowerLdSt
    auto loadVals =
        zipLoadValues(rewriter, loc, vec, offsetElems, offsetTy, maskElems,
                      otherElems, otherTy, swizzledLaneOffsets);

    // Create the resource descriptor and then emit the buffer_loads to lds
    // based on the collected shared addresses and vector size
    Value rsrcDesc = bufferEmitter.createResourceDescriptor(llPtr, llStride);

    Value threadPred = emitRedundantThreadPredicate(
        getFreeVariableMasks(ptrType), rewriter, loc, targetInfo);

    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    auto emitBufferLoadLds =
        [this, &op, &b, &bufferEmitter, &rsrcDesc, laneId = laneId, threadPred,
         offsetTy, otherTy, hasOther,
         hasSwizzling](RewriterBase &rewriter, Location loc,
                       ArrayRef<Value> loadVals, Value shmemAddr, int startIdx,
                       VectorType vecTy) -> SmallVector<Value> {
      auto [offsetElem, maskElem, otherElems, swizzleLaneOffset] =
          unzipLoadValues(rewriter, loc, startIdx, loadVals, offsetTy, otherTy,
                          hasOther, vecTy.getNumElements());
      int vecBits = vecTy.getNumElements() * vecTy.getElementTypeBitWidth();
      assert(targetInfo.supportsDirectToLdsLoadBitWidth(vecBits));
      Value vecBytesVal = b.i32_val(vecBits / 8);

      Value maybeSwizzledMaskElem = maskElem;
      if (hasSwizzling)
        applySwizzling(rewriter, loc, offsetElem, maybeSwizzledMaskElem, laneId,
                       swizzleLaneOffset);

      // If other=0.0 we remove other in canonicalizePointers and we can use out
      // of bounds to store 0 to LDS. So if we have other values we need to
      // predicate to not overwrite the other stores
      Value cond =
          hasOther ? b.and_(threadPred, maybeSwizzledMaskElem) : threadPred;

      auto [loadBlock, afterLoadBlock] = emitBranch(rewriter, loc, threadPred);

      auto bufferLoadToLds = bufferEmitter.emitLoadToLds(
          vecTy, vecBytesVal, rsrcDesc, offsetElem, shmemAddr,
          hasOther ? b.true_val() : maybeSwizzledMaskElem, op.getCache());
      AMD::addAsyncCopyAliasScope(bufferLoadToLds);

      if (hasOther) {
        emitOtherStore(rewriter, loc, this->getTypeConverter(), vecTy, maskElem,
                       otherElems, shmemAddr, laneId, hasSwizzling,
                       swizzleLaneOffset);
      }

      rewriter.setInsertionPointToStart(afterLoadBlock);

      return {};
    };

    lowerDirectToLDSLoad(rewriter, loc, ptrType, flatDstTy, loadVals, llDst,
                         resElemTy, vec, emitBufferLoadLds);

    // Drop the result token.
    Value zero = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), IntegerType::get(op.getContext(), 32),
        rewriter.getI32IntegerAttr(0));
    rewriter.replaceOp(op, zero);
    return success();
  }
};

struct AsyncCopyGlobalToLocalOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::AsyncCopyGlobalToLocalOp>,
      public DirectToLdsLoadConversionBase {
  AsyncCopyGlobalToLocalOpConversion(LLVMTypeConverter &converter,
                                     const AMD::TargetInfo &targetInfo,
                                     ModuleAxisInfoAnalysis &axisAnalysisPass,
                                     PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        DirectToLdsLoadConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncCopyGlobalToLocalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto srcTy = op.getSrc().getType();

    auto dstTy = op.getResult().getType();
    auto dstEnc = dstTy.getEncoding();
    auto resElemTy = getTypeConverter()->convertType(dstTy.getElementType());
    Value llDst = adaptor.getResult();

    // We can load N elements at a time if:
    //  1. Every group of N source pointers are contiguous.  For example, if
    //     N=2, then the pointers should be [x, x+1, y, y+1, ...].
    //  2. The mask (if present) has "alignment" N, meaning that each group of N
    //     mask bits are the same.  For example if N=2, the mask must be
    //     [x, x, y, y, ...].
    unsigned vec = getVectorSize(op.getSrc(), axisAnalysisPass);
    auto maskElements = getMaskElemsAndUpdateVeclen(
        rewriter, loc, adaptor.getMask(), op.getMask(), vec);

    auto srcElems = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    SmallVector<Value> otherElems;
    if (op.getOther())
      otherElems = unpackLLElements(loc, adaptor.getOther(), rewriter);

    // For padded encodings restrict vec by the min interval
    if (auto padEnc = dyn_cast<PaddedSharedEncodingAttr>(dstEnc)) {
      vec = std::min(vec, padEnc.getMinInterval());
    }

    auto maybeSwizzledEnc = dyn_cast<SwizzledSharedEncodingAttr>(dstEnc);
    bool hasSwizzling = maybeSwizzledEnc && maybeSwizzledEnc.getMaxPhase() != 1;
    if (failed(
            canWriteCoalesced(rewriter, op, srcTy, dstTy, vec, hasSwizzling))) {
      return failure();
    }

    // For swizzled layouts we need to use the non swizzled layout to compute
    // the LDS addresses since we gather into LDS
    auto flatDstTy = dstTy;
    SmallVector<Value> swizzledLaneOffsets;
    if (hasSwizzling) {
      auto flatSharedEnc = SwizzledSharedEncodingAttr::get(
          op->getContext(), maybeSwizzledEnc.getVec(), 1, 1,
          maybeSwizzledEnc.getOrder(), maybeSwizzledEnc.getCTALayout());
      flatDstTy = MemDescType::get(dstTy.getShape(), dstTy.getElementType(),
                                   flatSharedEnc, dstTy.getMemorySpace());
      swizzledLaneOffsets = emitSwizzledLaneOffsets(
          rewriter, op, srcTy, dstTy, flatDstTy, llDst, resElemTy, vec);
    }

    Type srcPtrTy = srcElems[0].getType();
    bool hasOther = !otherElems.empty();
    Type otherTy = hasOther ? otherElems[0].getType() : i1_ty;
    // Zip buffer_offset, mask, other, swizzleOffsets for lowerLdSt
    SmallVector<Value> loadVals =
        zipLoadValues(rewriter, loc, vec, srcElems, srcPtrTy, maskElements,
                      otherElems, otherTy, swizzledLaneOffsets);

    Value threadPred = emitRedundantThreadPredicate(getFreeVariableMasks(srcTy),
                                                    rewriter, loc, targetInfo);

    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    auto emitGlobalLoadLds =
        [this, &op, &b, laneId = laneId, threadPred, srcPtrTy, otherTy,
         hasOther, hasSwizzling](RewriterBase &rewriter, Location loc,
                                 ArrayRef<Value> loadValues, Value shmemAddr,
                                 int startIdx,
                                 VectorType vecTy) -> SmallVector<Value> {
      auto [srcElem, maskElem, otherElems, swizzleLaneOffset] =
          unzipLoadValues(rewriter, loc, startIdx, loadValues, srcPtrTy,
                          otherTy, hasOther, vecTy.getNumElements());
      int vecBits = vecTy.getNumElements() * vecTy.getElementTypeBitWidth();
      assert(targetInfo.supportsDirectToLdsLoadBitWidth(vecBits));
      Value maybeSwizzledMaskElem = maskElem;

      if (hasSwizzling)
        applySwizzling(rewriter, loc, srcElem, maybeSwizzledMaskElem, laneId,
                       swizzleLaneOffset);

      // Predicate load based on threadPred && swizzledMask
      auto cond = b.and_(threadPred, maybeSwizzledMaskElem);
      auto [loadBlock, afterLoadBlock] = emitBranch(rewriter, loc, cond);

      int32_t cacheModifiers =
          mlir::LLVM::AMD::getCtrlBitsForCacheModifierOnTarget(
              op.getCache(), /*isLoad=*/true, targetInfo);
      auto globalLoadLdsOp = rewriter.create<ROCDL::GlobalLoadLDSOp>(
          loc, srcElem, shmemAddr, vecBits / 8,
          /*offset=*/0, cacheModifiers, nullptr, nullptr, nullptr);
      AMD::addAsyncCopyAliasScope(globalLoadLdsOp);

      rewriter.setInsertionPointToStart(afterLoadBlock);

      if (hasOther) {
        emitOtherStore(rewriter, loc, this->getTypeConverter(), vecTy, maskElem,
                       otherElems, shmemAddr, laneId, hasSwizzling,
                       swizzleLaneOffset);
      }

      return {};
    };

    lowerDirectToLDSLoad(rewriter, loc, srcTy, flatDstTy, loadVals, llDst,
                         resElemTy, vec, emitGlobalLoadLds);

    // Drop the result token.
    Value zero = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), IntegerType::get(op.getContext(), 32),
        rewriter.getI32IntegerAttr(0));
    rewriter.replaceOp(op, zero);
    return success();
  }
};

struct AsyncTDMCopyGlobalToLocalOpConversion
    : public ConvertOpToLLVMPattern<
          triton::amdgpu::AsyncTDMCopyGlobalToLocalOp>,
      public LoadStoreConversionBase {
  AsyncTDMCopyGlobalToLocalOpConversion(
      LLVMTypeConverter &converter, const AMD::TargetInfo &targetInfo,
      ModuleAxisInfoAnalysis &axisAnalysisPass, PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  std::pair<Value, Value> createTDMDescriptors(
      RewriterBase &rewriter, Location loc,
      const LLVMTypeConverter *typeConverter, int64_t elementSizeInBytes,
      ArrayRef<Value> tensorShape, ArrayRef<int64_t> blockShape,
      ArrayRef<Value> tensorStride, Value srcPtr, Value dstPtr, Value pred,
      Value multicastMask, unsigned padIntervalInDwords,
      unsigned padAmountInDwords) const {
    assert(tensorShape.size() == 2 && tensorStride.size() == 2 &&
           blockShape.size() == 2);
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    Value ldsAddr = b.ptrtoint(i32_ty, dstPtr);
    Value globalAddr = b.ptrtoint(i64_ty, srcPtr);

    // group0 (128 bits / 4 dwords) effective bit encoding:
    // [1:0]:     pred
    // [63:32]:   lds address
    // [120:64]:  global address
    // [127:126]: type - currently always set to 0x2
    SmallVector<Value, 4> group0(4, b.i32_val(0));
    group0[0] = b.zext(i32_ty, pred);
    group0[1] = ldsAddr;
    group0[2] = b.trunc(i32_ty, globalAddr);
    group0[3] = b.trunc(i32_ty, b.lshr(globalAddr, b.i64_val(32)));
    group0[3] = b.or_(group0[3], b.i32_val(0x80000000));

    VectorType vecTy0 = vec_ty(i32_ty, 4);
    Value group0Vec = b.undef(vecTy0);
    for (unsigned ii = 0; ii < 4; ++ii) {
      Value vecIdx = createIndexAttrConstant(rewriter, loc,
                                             typeConverter->getIndexType(), ii);
      group0Vec = b.insert_element(vecTy0, group0Vec, group0[ii], vecIdx);
    }

    // group1 (256 bits / 8 dwords) effective bit encoding:
    // [15:0]:    multicast mask
    // [17:16]:   data size - log2(element size in bytes)
    // [20]:      enable padding
    // [24:22]:   pad interval - log2(pad interval in dwords) - 1
    // [31:25]:   pad amount - pad amount in dwords - 1
    // [79:48]:   tensor shape dim inner
    // [111:80]:  tensor shape dim outer
    // [127:112]: block shape dim inner
    // [143:128]: block shape dim outer
    // [207:160]: tensor stride dim outer (we only use 32 bits)
    SmallVector<Value, 8> group1(8, b.i32_val(0));
    int32_t dataSize = log2(elementSizeInBytes);
    group1[0] = multicastMask;
    group1[0] = b.or_(group1[0], b.i32_val(dataSize << 16));
    if (padIntervalInDwords > 0 && padAmountInDwords > 0) {
      assert(llvm::isPowerOf2_32(padIntervalInDwords));
      int32_t log2PadInterval = log2(padIntervalInDwords);
      group1[0] = b.or_(group1[0], b.i32_val(1 << 20));
      group1[0] = b.or_(group1[0], b.i32_val((log2PadInterval - 1) << 22));
      group1[0] = b.or_(group1[0], b.i32_val((padAmountInDwords - 1) << 25));
    }
    group1[1] = b.shl(tensorShape[1], b.i32_val(16));
    group1[2] = b.lshr(tensorShape[1], b.i32_val(16));
    group1[2] = b.or_(group1[2], b.shl(tensorShape[0], b.i32_val(16)));
    group1[3] = b.lshr(tensorShape[0], b.i32_val(16));
    group1[3] = b.or_(group1[3], b.i32_val(blockShape[1] << 16));
    group1[4] = b.i32_val(blockShape[0] & 0xFFFF);
    group1[5] = tensorStride[0];

    VectorType vecTy1 = vec_ty(i32_ty, 8);
    Value group1Vec = b.undef(vecTy1);
    for (unsigned ii = 0; ii < 8; ++ii) {
      Value vecIdx = createIndexAttrConstant(rewriter, loc,
                                             typeConverter->getIndexType(), ii);
      group1Vec = b.insert_element(vecTy1, group1Vec, group1[ii], vecIdx);
    }

    return {group0Vec, group1Vec};
  }

  LogicalResult
  matchAndRewrite(triton::amdgpu::AsyncTDMCopyGlobalToLocalOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto mod = op->getParentOfType<ModuleOp>();
    auto tensorDescTy = op.getDesc().getType();
    auto smemTy = op.getResult().getType();

    auto swizzledEnc =
        llvm::dyn_cast<SwizzledSharedEncodingAttr>(smemTy.getEncoding());
    if (swizzledEnc && swizzledEnc.getMaxPhase() != 1)
      return rewriter.notifyMatchFailure(op, "TDM does not support swizzling");

    auto paddedEnc =
        llvm::dyn_cast<PaddedSharedEncodingAttr>(smemTy.getEncoding());
    if (!paddedEnc && !swizzledEnc)
      return rewriter.notifyMatchFailure(
          op, "Invalid shared memory layout for TDM.");

    Type llvmElemTy = getTypeConverter()->convertType(smemTy.getElementType());
    auto elementBitWidth = llvmElemTy.getIntOrFloatBitWidth();

    unsigned padInterval = 0;
    unsigned padAmount = 0;
    if (paddedEnc) {
      if (paddedEnc.getIntervals().size() != 1 ||
          paddedEnc.getPaddings().size() != 1)
        return rewriter.notifyMatchFailure(
            op, "NYI: Multiple interval-padding pairs in TDM.");
      padInterval = paddedEnc.getIntervals()[0];
      padAmount = paddedEnc.getPaddings()[0];
    }
    unsigned dwordSize = 32;
    auto padIntervalInDwords = padInterval * elementBitWidth / dwordSize;
    auto padAmountInDwords = padAmount * elementBitWidth / dwordSize;
    if (padInterval > 0 && padIntervalInDwords < 2)
      return rewriter.notifyMatchFailure(
          op, "TDM padding interval must be at least 2 dwords");
    if (padAmount > 0 && padAmountInDwords < 1)
      return rewriter.notifyMatchFailure(
          op, "TDM padding amount must be at least 1 dword");

    // [base, shape0, shape1, stride0, stride1]
    SmallVector<Value> descriptorFields =
        unpackLLElements(loc, adaptor.getDesc(), rewriter);
    if (descriptorFields.size() != 5)
      return rewriter.notifyMatchFailure(op, "NYI: TDM > 2D cases.");

    Value base = descriptorFields[0];
    SmallVector<Value> tensorShape{descriptorFields[1], descriptorFields[2]};
    SmallVector<Value> tensorStride{descriptorFields[3], descriptorFields[4]};

    // Cast strides from i64 to i32
    tensorStride[0] = b.trunc(i32_ty, tensorStride[0]);
    tensorStride[1] = b.trunc(i32_ty, tensorStride[1]);

    SmallVector<Value> offset = adaptor.getIndices();
    SmallVector<int64_t> blockShape =
        llvm::to_vector(tensorDescTy.getBlockType().getShape());
    SmallVector<int64_t> blockShapePerCTA = blockShape;

    int numCTAs = TritonGPUDialect::getNumCTAs(mod);
    Value multicastMask = b.i32_val(0);
    if (numCTAs > 1) {
      return rewriter.notifyMatchFailure(op, "NYI: Support multicast.");
    }

    Type globalPtrTy = ptr_ty(ctx, 1);
    Type sharedPtrTy = ptr_ty(ctx, 3);

    // For block shape [M, N], each warp will handle shape [M/numWarps, N].
    auto numWarps = triton::gpu::lookupNumWarps(op);
    auto warpId = getLaneAndWarpId(rewriter, loc).second;

    int outerBlockShape = blockShapePerCTA[0];
    int outerBlockShapePerWarp = ceil(outerBlockShape, numWarps);
    int outerBlockStride = blockShapePerCTA[1];

    // Shift global pointer by offset
    Value outerOffset = b.mul(b.i32_val(outerBlockShapePerWarp), warpId);
    offset[0] = b.add(offset[0], outerOffset);

    Value baseOffset = b.add(b.mul(tensorStride[0], offset[0]),
                             b.mul(tensorStride[1], offset[1]));
    base = b.gep(globalPtrTy, llvmElemTy, base, baseOffset);

    // Shift shared pointer by offset
    auto dstMemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getResult(), llvmElemTy, rewriter);
    Value dstBase = dstMemObj.getBase();
    Value dstOffset = b.mul(b.i32_val(outerBlockStride), outerOffset);
    if (paddedEnc) {
      Value padding = emitPadding(loc, rewriter, paddedEnc, elementBitWidth,
                                  dstOffset, false);
      dstOffset = b.add(dstOffset, padding);
    }
    dstBase = b.gep(sharedPtrTy, llvmElemTy, dstBase, dstOffset);

    // Update tensor shape and block shape based on offset
    Value zero = b.i32_val(0);
    tensorShape[0] = b.smax(zero, b.sub(tensorShape[0], offset[0]));
    tensorShape[1] = b.smax(zero, b.sub(tensorShape[1], offset[1]));

    blockShapePerCTA[0] = outerBlockShapePerWarp;

    auto elementSizeInBytes = elementBitWidth / 8;
    auto [group0, group1] = createTDMDescriptors(
        rewriter, loc, getTypeConverter(), elementSizeInBytes, tensorShape,
        blockShapePerCTA, tensorStride, base, dstBase, op.getPred(),
        multicastMask, padIntervalInDwords, padAmountInDwords);
    LLVM::createLLVMIntrinsicCallOp(rewriter, loc,
                                    "llvm.amdgcn.tensor.load.to.lds.d2", {},
                                    {group0, group1, b.i32_val(0)});

    rewriter.eraseOp(op);
    return success();
  }
};

struct StoreOpConversion : public ConvertOpToLLVMPattern<triton::StoreOp>,
                           public LoadStoreConversionBase {
  StoreOpConversion(LLVMTypeConverter &converter,
                    const AMD::TargetInfo &targetInfo,
                    ModuleAxisInfoAnalysis &axisAnalysisPass,
                    PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value ptr = op.getPtr();
    Value value = op.getValue();
    Value mask = op.getMask();

    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llValue = adaptor.getValue();

    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    MLIRContext *ctx = rewriter.getContext();
    auto moduleOp = op->getParentOfType<ModuleOp>();

    auto valueTy = value.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));

    // Determine the vectorization size
    unsigned vec = getVectorSize(ptr, axisAnalysisPass);
    unsigned elemsPerThread = getTotalElemsPerThread(ptr.getType());

    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    auto valueElems = unpackLLElements(loc, llValue, rewriter);
    assert(ptrElems.size() == valueElems.size());

    SmallVector<Value> maskElems =
        getMaskElemsAndUpdateVeclen(rewriter, loc, llMask, mask, vec);

    const size_t valueElemNBits =
        std::max<int>(8, valueElemTy.getIntOrFloatBitWidth());
    const size_t valueElemNBytes = valueElemNBits / 8;

    auto cacheMod = op.getCache();
    const int numVecs = elemsPerThread / vec;
    auto freeVarMasks = getFreeVariableMasks(valueTy);
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    uint32_t regMask = freeVarMasks[str_attr("reg")];
    for (size_t vecStart = 0; vecStart < elemsPerThread; vecStart += vec) {
      if (!isCanonicalIndex(vecStart, regMask)) {
        // Don't emit store ops for redundant elements within a thread
        continue;
      }

      Value pred =
          llMask ? b.and_(threadPred, maskElems[vecStart]) : threadPred;

      auto vecTy = LLVM::getVectorType(valueElemTy, vec);

      const size_t maxWordWidth = std::max<size_t>(32, valueElemNBits);
      const size_t totalWidth = valueElemNBits * vec;
      const size_t width = std::min(totalWidth, maxWordWidth);
      const size_t nWords = std::max<size_t>(1, totalWidth / width);
      const size_t wordNElems = width / valueElemNBits;
      assert(wordNElems * nWords * numVecs == elemsPerThread);

      SmallVector<std::pair<Value, std::string>> asmArgs;
      Value elem = valueElems[vecStart];
      Value ptr = ptrElems[vecStart];

      // Create the store val
      Value storeVal = packElementRangeIntoVector(
          rewriter, this->getTypeConverter(), loc, cast<VectorType>(vecTy),
          valueElems, vecStart);
      llStore(rewriter, loc, ptr, storeVal, pred, cacheMod);
    } // end vec
    rewriter.eraseOp(op);
    return success();
  }
};

struct BufferAtomicRMWOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::BufferAtomicRMWOp>,
      public LoadStoreConversionBase {
  BufferAtomicRMWOpConversion(LLVMTypeConverter &converter,
                              const AMD::TargetInfo &targetInfo,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::amdgpu::BufferAtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    LLVM::AMD::BufferEmitter bufferEmitter(rewriter, loc, targetInfo);

    // original values
    Value ptr = op.getPtr();
    Value offset = op.getOffsets();
    Value mask = op.getMask();
    Value data = op.getValue();
    auto atomicRmwAttr = op.getAtomicRmwOp();

    Value llPtr = adaptor.getPtr();
    Value llOffset = adaptor.getOffsets();
    Value llMask = adaptor.getMask();
    Value llData = adaptor.getValue();
    Value llStride = adaptor.getStride();

    // Determine the vectorization size
    Type valueTy = data.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));
    Type ptrType = getPointerTypeWithShape(ptr, offset);

    unsigned numElems = getTotalElemsPerThread(ptrType);
    unsigned vec = getVectorSize(ptr, offset, axisAnalysisPass);

    // v4f16 and v4bf16 variants of buffer atomics do not exist.
    // only v2f16 and v2bf16.
    if (valueElemTy.isBF16() || valueElemTy.isF16()) {
      // We clamp to the only supported vectorization width here (2).
      // In ConvertToBufferOps we check that we have a large enough vector size
      assert(vec >= 2);
      vec = 2u;
      // The max width of a buffer atomic op is 64-bits
      // Some types like F32 don't have a 2x vectorized version
    } else if (valueElemTy.isF32() || valueElemTy.isF64() ||
               valueElemTy.isInteger(32) || valueElemTy.isInteger(64)) {
      vec = 1u;
    }

    // Get the offsets and value
    SmallVector<Value> offsetElems = unpackLLElements(loc, llOffset, rewriter);
    SmallVector<Value> valueElems = unpackLLElements(loc, llData, rewriter);

    // Get the mask
    SmallVector<Value> maskElems =
        getMaskElemsAndUpdateVeclen(rewriter, loc, llMask, mask, vec);

    Value rsrcDesc = bufferEmitter.createResourceDescriptor(llPtr, llStride);
    SmallVector<Value> loadedVals;

    // We need to manually emit memory fences (LLVM doesn't do this for buffer
    // ops) see: https://llvm.org/docs/AMDGPUUsage.html#memory-model-gfx942
    auto memOrdering = op.getSem();
    auto memScope = op.getScope();
    if (failed(emitFence(op, rewriter, loc, memOrdering, memScope,
                         true /*preAtomic*/))) {
      return failure();
    }

    mlir::Operation *lastRMWOp;
    MLIRContext *ctx = rewriter.getContext();
    GCNBuilder waitcntBuilder;

    //    We set GLC=1, to return the old value. Atomics in GFX942 execute with
    //    either device (default) or system scope (controlled by the sc1 flag).
    //    This is distinct from the memory scope of the atomic (i.e, the memory
    //    fences which appear before/after the ops).

    // Check if the op has users, if it does we set GLC=1, otherwise GLC=0
    auto opUsers = op.getResult().getUsers();
    auto hasUsers = std::distance(opUsers.begin(), opUsers.end()) > 0;
    auto moduleOp = op->getParentOfType<ModuleOp>();

    auto freeVarMasks = getFreeVariableMasks(valueTy);
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    uint32_t regMask = freeVarMasks[str_attr("reg")];
    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      if (!isCanonicalIndex(vecStart, regMask)) {
        // Don't emit store ops for redundant elements within a thread
        continue;
      }

      Value pred =
          llMask ? b.and_(threadPred, maskElems[vecStart]) : threadPred;

      Type vecTy = LLVM::getVectorType(valueElemTy, vec);
      Value falseVal = createZeroVector(rewriter, loc, cast<VectorType>(vecTy));
      // Create the store val
      Value storeVal = packElementRangeIntoVector(
          rewriter, this->getTypeConverter(), loc, cast<VectorType>(vecTy),
          valueElems, vecStart);

      Value loadVal = bufferEmitter.emitAtomicRMW(
          atomicRmwAttr, vecTy, rsrcDesc, offsetElems[vecStart], storeVal, pred,
          hasUsers);
      // Track the last op, so we can emit a fenceop after the loop
      lastRMWOp = loadVal.getDefiningOp();

      for (size_t ii = 0; ii < vec; ++ii) {
        Value vecIdx = createIndexAttrConstant(
            rewriter, loc, getTypeConverter()->getIndexType(), ii);
        Value loaded = b.extract_element(valueElemTy, loadVal, vecIdx);
        loadedVals.push_back(loaded);
      }
    } // end vec

    // Acquire Fence post-atomic
    if (failed(emitFence(op, rewriter, lastRMWOp->getLoc(), memOrdering,
                         memScope, false /*preAtomic*/))) {
      return failure();
    }

    finalizeTensorAtomicResults(op, dyn_cast<RankedTensorType>(valueTy),
                                rewriter, loadedVals, valueElemTy, b,
                                threadPred, targetInfo, getTypeConverter());
    return success();
  }
};

struct BufferAtomicCASOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::BufferAtomicCASOp>,
      public LoadStoreConversionBase {
  BufferAtomicCASOpConversion(LLVMTypeConverter &converter,
                              const AMD::TargetInfo &targetInfo,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::amdgpu::BufferAtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    LLVM::AMD::BufferEmitter bufferEmitter(rewriter, loc, targetInfo);

    // original values
    Value ptr = op.getPtr();
    Value offset = op.getOffsets();
    Value cmp = op.getCmp();
    Value val = op.getVal();

    Value llPtr = adaptor.getPtr();
    Value llOffset = adaptor.getOffsets();
    Value llVal = adaptor.getVal();
    Value llCmp = adaptor.getCmp();
    Value llStride = adaptor.getStride();

    // Determine the vectorization size
    Type valueTy = val.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));
    Type ptrType = getPointerTypeWithShape(ptr, offset);

    unsigned numElems = getTotalElemsPerThread(ptrType);
    // Max supported vectorization for i32 and i64 is 1x
    // on CDNA3 and CDNA4
    // BUFFER_ATOMIC_CMPSWAP(i32) and BUFFER_ATOMIC_CMPSWAP_X2(i64)
    unsigned vec = 1u;

    // Get the offsets, val, and cmp
    SmallVector<Value> offsetElems = unpackLLElements(loc, llOffset, rewriter);
    SmallVector<Value> valElems = unpackLLElements(loc, llVal, rewriter);
    SmallVector<Value> cmpElems = unpackLLElements(loc, llCmp, rewriter);

    Value rsrcDesc = bufferEmitter.createResourceDescriptor(llPtr, llStride);
    SmallVector<Value> loadedVals;

    // We need to manually emit memory fences (LLVM doesn't do this for buffer
    // ops)
    auto memOrdering = op.getSem();
    auto memScope = op.getScope();
    // Release Fence pre-atomic
    if (failed(emitFence(op, rewriter, loc, memOrdering, memScope,
                         true /*preAtomic*/))) {
      return failure();
    }

    mlir::Operation *lastCASOp;
    MLIRContext *ctx = rewriter.getContext();
    GCNBuilder waitcntBuilder;

    // Check if the op has users, if it does we set GLC=1, otherwise GLC=0
    auto opUsers = op.getResult().getUsers();
    auto hasUsers = !op.getResult().getUsers().empty();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto freeVarMasks = getFreeVariableMasks(valueTy);
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);

    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      Type vecTy = LLVM::getVectorType(valueElemTy, vec);
      Value pred = threadPred;
      // Create the store val
      Value casStoreVal = packElementRangeIntoVector(
          rewriter, this->getTypeConverter(), loc, cast<VectorType>(vecTy),
          valElems, vecStart);
      // Create the cmp val
      Value casCmpVal = packElementRangeIntoVector(
          rewriter, this->getTypeConverter(), loc, cast<VectorType>(vecTy),
          cmpElems, vecStart);

      Value loadVal =
          bufferEmitter.emitAtomicCAS(vecTy, rsrcDesc, offsetElems[vecStart],
                                      casCmpVal, casStoreVal, pred, hasUsers);
      // Track the last op, so we can emit a fenceop after the loop
      lastCASOp = loadVal.getDefiningOp();

      for (size_t ii = 0; ii < vec; ++ii) {
        Value vecIdx = createIndexAttrConstant(
            rewriter, loc, getTypeConverter()->getIndexType(), ii);
        Value loaded = b.extract_element(valueElemTy, loadVal, vecIdx);
        loadedVals.push_back(loaded);
      }
    } // end vec

    // Emit post-atomic acquire fence
    if (failed(emitFence(op, rewriter, lastCASOp->getLoc(), memOrdering,
                         memScope, false /*preAtomic*/))) {
      return failure();
    }

    finalizeTensorAtomicResults(op, dyn_cast<RankedTensorType>(valueTy),
                                rewriter, loadedVals, valueElemTy, b,
                                threadPred, targetInfo, getTypeConverter());
    return success();
  }
};

struct BufferStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::BufferStoreOp>,
      public LoadStoreConversionBase {
  BufferStoreOpConversion(LLVMTypeConverter &converter,
                          const AMD::TargetInfo &targetInfo,
                          ModuleAxisInfoAnalysis &axisAnalysisPass,
                          PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::amdgpu::BufferStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    LLVM::AMD::BufferEmitter bufferEmitter(rewriter, loc, targetInfo);

    // original values
    Value ptr = op.getPtr();
    Value offset = op.getOffsets();
    Value mask = op.getMask();
    Value data = op.getValue();
    auto cacheMod = op.getCache();

    Value llPtr = adaptor.getPtr();
    Value llOffset = adaptor.getOffsets();
    Value llMask = adaptor.getMask();
    Value llData = adaptor.getValue();
    Value llStride = adaptor.getStride();

    // Determine the vectorization size
    Type valueTy = data.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));
    Type ptrType = getPointerTypeWithShape(ptr, offset);

    unsigned numElems = getTotalElemsPerThread(ptrType);
    unsigned vec = getVectorSize(ptr, offset, axisAnalysisPass);

    // Get the offsets and value
    SmallVector<Value> offsetElems = unpackLLElements(loc, llOffset, rewriter);
    SmallVector<Value> valueElems = unpackLLElements(loc, llData, rewriter);

    // Get the mask
    SmallVector<Value> maskElems =
        getMaskElemsAndUpdateVeclen(rewriter, loc, llMask, mask, vec);

    Value rsrcDesc = bufferEmitter.createResourceDescriptor(llPtr, llStride);
    MLIRContext *ctx = rewriter.getContext();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto freeVarMasks = getFreeVariableMasks(valueTy);
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    uint32_t regMask = freeVarMasks[str_attr("reg")];
    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      if (!isCanonicalIndex(vecStart, regMask)) {
        // Don't emit store ops for redundant elements within a thread
        continue;
      }

      Value pred =
          llMask ? b.and_(threadPred, maskElems[vecStart]) : threadPred;

      Type vecTy = LLVM::getVectorType(valueElemTy, vec);
      // Create the store val
      Value storeVal = packElementRangeIntoVector(
          rewriter, this->getTypeConverter(), loc, cast<VectorType>(vecTy),
          valueElems, vecStart);
      bufferEmitter.emitStore(rsrcDesc, offsetElems[vecStart], storeVal, pred,
                              cacheMod);
    } // end vec

    rewriter.eraseOp(op);
    return success();
  }
};

struct AtomicCASOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicCASOp>,
      public LoadStoreConversionBase {
  AtomicCASOpConversion(LLVMTypeConverter &converter,
                        const AMD::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // extract relevant info from Module
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    MLIRContext *ctx = rewriter.getContext();
    Value ptr = op.getPtr();

    Value llPtr = adaptor.getPtr();
    Value llCmp = adaptor.getCmp();
    Value llVal = adaptor.getVal();

    // prep data by unpacking to get data ready
    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    auto cmpElements = unpackLLElements(loc, llCmp, rewriter);
    auto valElements = unpackLLElements(loc, llVal, rewriter);

    auto memOrdering = op.getSem();
    auto atomicMemOrdering = getMemoryOrdering(memOrdering);
    if (!atomicMemOrdering)
      return rewriter.notifyMatchFailure(op, "Unknown AMDGPU memory ordering");
    auto scope = op.getScope();
    auto scopeStr = getAMDGPUMemScopeStr(scope);
    if (!scopeStr)
      return rewriter.notifyMatchFailure(op, "Unknown AMDGPU memory scope");

    // deal with tensor or scalar
    auto valueTy = op.getResult().getType();
    auto tensorTy = dyn_cast<RankedTensorType>(valueTy);
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : valueTy;
    auto valueElemNBits = valueElemTy.getIntOrFloatBitWidth();
    auto elemsPerThread = getTotalElemsPerThread(op.getVal().getType());
    SmallVector<Value> resultVals(elemsPerThread);

    // atomic ops
    for (size_t i = 0; i < elemsPerThread; i += 1) {
      Value casVal = valElements[i];
      Value casCmp = cmpElements[i];
      Value casPtr = ptrElements[i];
      // use op
      if (tensorTy) { // for tensor
        auto retType = valueElemTy;
        // TODO: USE ATOMIC CAS OP on Tensor
        auto successOrdering = *atomicMemOrdering;
        auto failureOrdering = LLVM::AtomicOrdering::monotonic;
        auto cmpxchg = rewriter.create<LLVM::AtomicCmpXchgOp>(
            loc, casPtr, casCmp, casVal, successOrdering, failureOrdering,
            StringRef(scopeStr.value()));

        // Extract the new_loaded value from the pair.
        Value ret = b.extract_val(valueElemTy, cmpxchg, i);
        resultVals[i] = ret;
      } else { // for scalar
        // Build blocks to bypass the atomic instruction for ~rmwMask.
        auto *curBlock = rewriter.getInsertionBlock();
        auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
        auto *atomicBlock = rewriter.createBlock(
            curBlock->getParent(), std::next(Region::iterator(curBlock)));

        // Fill entry block with global memory barrier and conditional branch.
        rewriter.setInsertionPointToEnd(curBlock);
        auto tid = getThreadId(rewriter, loc);
        Value pred = b.icmp_eq(tid, b.i32_val(i));
        rewriter.create<LLVM::CondBrOp>(loc, pred, atomicBlock, endBlock);

        // Build main block with atomic_cmpxchg.
        rewriter.setInsertionPointToEnd(atomicBlock);

        auto successOrdering = LLVM::AtomicOrdering::acq_rel;
        auto failureOrdering = LLVM::AtomicOrdering::monotonic;
        auto cmpxchg = rewriter.create<LLVM::AtomicCmpXchgOp>(
            loc, casPtr, casCmp, casVal, successOrdering, failureOrdering,
            StringRef("agent"));

        if (!op.getResult().use_empty()) {
          // Extract the new_loaded value from the pair.
          Value newLoaded = b.extract_val(valueElemTy, cmpxchg, 0);
          Value atomPtr =
              getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
          b.store(newLoaded, atomPtr);
        }

        rewriter.create<LLVM::BrOp>(loc, ValueRange(), endBlock);

        // Build the last block: synced load from shared memory, exit.
        rewriter.setInsertionPointToStart(endBlock);

        if (op.getResult().use_empty()) {
          rewriter.eraseOp(op);
          return success();
        }

        GCNBuilder BuilderMemfenceLDS;
        BuilderMemfenceLDS.create<>("s_waitcnt lgkmcnt(0)")->operator()();
        BuilderMemfenceLDS.launch(rewriter, loc, void_ty(ctx));
        b.barrier();
        Value atomPtr =
            getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
        Value ret = b.load(valueElemTy, atomPtr);
        rewriter.replaceOp(op, {ret});
        return success();
      }
    }

    // FIXME: threadPred = b.true_val() is buggy
    finalizeTensorAtomicResults(op, tensorTy, rewriter, resultVals, valueElemTy,
                                b, b.true_val(), targetInfo,
                                getTypeConverter());
    return success();
  }
};

bool supportsGlobalAtomicF16PackedAndDpp(ISAFamily isaFamily) {
  switch (isaFamily) {
  case ISAFamily::CDNA1:
  case ISAFamily::CDNA2:
  case ISAFamily::CDNA3:
  case ISAFamily::CDNA4:
    return true;
  default:
    break;
  }
  return false;
}

struct AtomicRMWOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicRMWOp>,
      public LoadStoreConversionBase {
  AtomicRMWOpConversion(LLVMTypeConverter &converter,
                        const AMD::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto binOp = matchAtomicOp(op.getAtomicRmwOp());
    if (!binOp)
      return rewriter.notifyMatchFailure(op, "Unsupported RMW operation");

    auto memOrder = getMemoryOrdering(op.getSem());
    if (!memOrder)
      return rewriter.notifyMatchFailure(op, "Unsupported RMW memory order");

    auto scopeStr = getAMDGPUMemScopeStr(op.getScope());
    if (!scopeStr)
      return rewriter.notifyMatchFailure(op, "Unsupported RMW scope");

    auto emitter =
        LLVM::AMD::AtomicRMWEmitter(targetInfo, *binOp, *memOrder, *scopeStr);

    Value val = op.getVal();
    Value ptr = op.getPtr();
    Value opResult = op.getResult();
    auto atomicRmwAttr = op.getAtomicRmwOp();

    Value llPtr = adaptor.getPtr();
    Value llVal = adaptor.getVal();
    Value llMask = adaptor.getMask();

    auto valElements = unpackLLElements(loc, llVal, rewriter);
    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    SmallVector<Value> maskElements;
    if (llMask)
      maskElements = unpackLLElements(loc, llMask, rewriter);

    auto tensorTy = dyn_cast<RankedTensorType>(opResult.getType());
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : opResult.getType();

    int numElems = 1;
    // In the case of unpaired f16 elements utilize dpp instructions to
    // accelerate atomics. Here is an algorithm of lowering
    // tt::atomicRmwOp(%ptr, %val, %mask):
    // 0. Group thread by pairs. Master thread is (tid % 2 == 0);
    // 1. All the threads send %val to (tid - 1) thread via dppUpdateOp shl, so
    //    all the masters receive value from secondary threads;
    // 2. Take into account parity in the %mask value, build control flow
    //    structures according to it;
    // 3. Generate llvm::atomicRmwOp in the threads enabled by %mask value;
    // 4. All the threads send result of generated operation to (tid + 1) thread
    //    via dppUpdateOp shl, so all secondary thread also receive their
    //    result.
    //
    // This approach enables us to use half the active threads committing atomic
    // requests to avoid generating of code providing unified access to f16
    // element and reduce contention.
    bool applyPackingF16 = false;
    auto vec = getVectorSize(ptr, axisAnalysisPass);

    // CDNA3/CDNA4 arch allows to accelerate its atomics with LDS reduction
    // algorithm, which is only applicable for atomics with no return. Otherwise
    // we have to deal with an additional overhead.
    bool enableIntraWaveReduce =
        llvm::is_contained({ISAFamily::CDNA3, ISAFamily::CDNA4},
                           targetInfo.getISAFamily()) &&
        tensorTy && opResult.use_empty();

    // TODO: support data types less than 32 bits
    enableIntraWaveReduce &= valueElemTy.getIntOrFloatBitWidth() >= 32;

    if (tensorTy) {
      bool isF16Ty = valueElemTy.isF16() || valueElemTy.isBF16();
      unsigned availableVecSize = isF16Ty ? 2 : 1;
      vec = std::min<unsigned>(vec, availableVecSize);
      // Force F16 packing in the case it's not coming in as packed, but the
      // ISA can support packed atomic instructions.
      applyPackingF16 =
          supportsGlobalAtomicF16PackedAndDpp(targetInfo.getISAFamily()) &&
          vec == 1 && isF16Ty && atomicRmwAttr == RMWOp::FADD &&
          !enableIntraWaveReduce;
      numElems = tensorTy.getNumElements();

      auto threadOrder = getThreadOrder(tensorTy);
      unsigned contigWithinLanes =
          axisAnalysisPass.getAxisInfo(ptr)->getContiguity(threadOrder.front());
      enableIntraWaveReduce &= contigWithinLanes == 1;
    }

    auto vecTy = vec_ty(valueElemTy, vec);
    auto elemsPerThread = getTotalElemsPerThread(val.getType());

    auto freeVarMasks = getFreeVariableMasks(op.getPtr().getType());
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    auto tid = getThreadId(rewriter, loc);

    std::optional<Value> atomicSharedMemBase =
        op->hasAttr("allocation.offset")
            ? std::optional<Value>(getSharedMemoryBase(
                  loc, rewriter, targetInfo, op.getOperation()))
            : std::nullopt;

    SmallVector<Value> resultVals(elemsPerThread);
    for (size_t i = 0; i < elemsPerThread; i += vec) {
      // TODO: in case llMask is zero we can create only one branch for all
      // elemsPerThread.
      Value rmwMask = llMask ? b.and_(threadPred, maskElements[i]) : threadPred;
      if (applyPackingF16) {
        resultVals[i] = emitter.emitPairedAtomicForEvenTID(
            rewriter, ptrElements[i], valElements[i], rmwMask);
      } else {
        Value valElement;
        if (vec == 1) {
          valElement = valElements[i];
        } else {
          Value vecVal = b.undef(vecTy);
          for (size_t ii = 0; ii < vec; ++ii)
            vecVal = b.insert_element(vecTy, vecVal, valElements[i + ii],
                                      b.i32_val(ii));
          valElement = vecVal;
        }

        // If we have a single tl.atomic_rmw that is lowered into multiple
        // llvm.atomic_rmw, and we set the ordering for each to aql_rel (the
        // default if no sem value is explicitly set in the DSL level
        // tl.atomic_add. The llvm backend will insert extra buffer invalidates
        // and L2 write backs causing a perforance degration. To avoid this we
        // set the ordering to release for the first, acquire for the last, and
        // relaxed for anything in between so that only a single set of
        // buffer_inv and buffer_wbl2 instructions are inserted by the backend
        // for any "cluster" of atomic ops.
        if ((vec > 1 || elemsPerThread > 1) &&
            op.getSem() == MemSemantic::ACQUIRE_RELEASE) {
          if (i == 0) {
            // First
            emitter.setAtomicOrdering(LLVM::AtomicOrdering::release);
          } else if (i == elemsPerThread - vec) {
            // Last
            emitter.setAtomicOrdering(LLVM::AtomicOrdering::acquire);
          } else {
            // Middle
            emitter.setAtomicOrdering(LLVM::AtomicOrdering::monotonic);
          }
        }

        Value retVal =
            emitter.emitAtomicRMW(rewriter, ptrElements[i], valElement, rmwMask,
                                  atomicSharedMemBase, enableIntraWaveReduce);

        if (tensorTy) {
          for (int ii = 0; ii < vec; ++ii) {
            resultVals[i + ii] =
                vec == 1
                    ? retVal
                    : b.extract_element(valueElemTy, retVal, b.i32_val(ii));
          }
        } else {
          if (!atomicSharedMemBase.has_value()) {
            rewriter.eraseOp(op);
            return success();
          }
          Value atomPtr = *atomicSharedMemBase;
          b.barrier();
          Value ret = b.load(valueElemTy, atomPtr);

          rewriter.replaceOp(op, {ret});
          return success();
        }
      }
    }
    finalizeTensorAtomicResults(op, tensorTy, rewriter, resultVals, valueElemTy,
                                b, threadPred, targetInfo, getTypeConverter());
    return success();
  }
};

struct AsyncWaitOpConversion : public ConvertOpToLLVMPattern<AsyncWaitOp> {
  AsyncWaitOpConversion(LLVMTypeConverter &converter,
                        const AMD::TargetInfo &targetInfo,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(AsyncWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    switch (targetInfo.getISAFamily()) {
    case ISAFamily::CDNA1:
    case ISAFamily::CDNA2:
    case ISAFamily::CDNA3:
    case ISAFamily::CDNA4:
      break;
    default:
      return rewriter.notifyMatchFailure(
          op, "Only supported on CDNA target architecture");
    }

    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    // global.load.lds uses vmcnt to synchronize
    // The rocdl op stores all available counters in a single int32 value (v).
    // The vmcnt (6 bits) is split into a lower 3:0 and higher 5:4 parts.
    // The lower part is stored in bits 3:0 of v and the higher part in bits
    // 15:14. We have to set all other bits in v to 1 to signal we are not
    // interested in those.

    // Clamp vmcnt to 6bits; a lower vmcnt will produce a conservative wait
    unsigned vmCnt = std::min(63u, op.getNum());

    // Extract low and high bits and combine while setting all other bits to 1
    unsigned lowBits = vmCnt & 0xF;
    unsigned highBits = vmCnt >> 4 << 14;
    unsigned otherCnts = ~0xC00F; // C00F has bits 15:14 and 3:0 set
    unsigned waitValue = lowBits | highBits | otherCnts;

    rewriter.create<ROCDL::SWaitcntOp>(loc, waitValue);

    // Drop the result AsyncToken
    rewriter.replaceOp(op, b.i32_val(0));
    return success();
  }

private:
  const AMD::TargetInfo &targetInfo;
};

struct AsyncTDMWaitConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::AsyncTDMWait> {
  AsyncTDMWaitConversion(LLVMTypeConverter &converter, PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit) {}

  LogicalResult
  matchAndRewrite(triton::amdgpu::AsyncTDMWait op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    LLVM::createLLVMIntrinsicCallOp(rewriter, loc,
                                    "llvm.amdgcn.s.wait.tensorcnt", {},
                                    {b.i16_val(op.getNum())});
    rewriter.eraseOp(op);
    return success();
  }
};

struct AsyncCommitGroupOpConversion
    : public ConvertOpToLLVMPattern<AsyncCommitGroupOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(AsyncCommitGroupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Drop the result AsyncToken
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    rewriter.replaceOp(op, b.i32_val(0));
    return success();
  }
};

} // namespace

namespace mlir::triton::AMD {
void populateLoadStoreOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                       const TargetInfo &targetInfo,
                                       RewritePatternSet &patterns,
                                       ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                       PatternBenefit benefit) {
  patterns
      .add<AtomicCASOpConversion, AtomicRMWOpConversion, LoadOpConversion,
           StoreOpConversion, BufferLoadOpConversion,
           BufferLoadToLocalOpConversion, BufferStoreOpConversion,
           BufferAtomicRMWOpConversion, AsyncCopyGlobalToLocalOpConversion,
           AsyncTDMCopyGlobalToLocalOpConversion, BufferAtomicCASOpConversion>(
          typeConverter, targetInfo, axisInfoAnalysis, benefit);
  patterns.add<AsyncWaitOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<AsyncTDMWaitConversion>(typeConverter, benefit);
  patterns.add<AsyncCommitGroupOpConversion>(typeConverter, benefit);
}
} // namespace mlir::triton::AMD
