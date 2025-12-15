#include "AsyncUtility.h"
#include "AtomicRMWOpsEmitter.h"
#include "BufferOpsEmitter.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TDMUtility.h"
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
    LLVM::FenceOp::create(rewriter, loc, TypeRange{},
                          LLVM::AtomicOrdering::release, scope);
  }

  if (emitAcquireFence && !preAtomic) {
    LLVM::FenceOp::create(rewriter, loc, TypeRange{},
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
  LLVM::CondBrOp::create(rewriter, loc, cond, body, after);
  rewriter.setInsertionPointToStart(body);
  LLVM::BrOp::create(rewriter, loc, after);
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
    Value zeroVal = LLVM::ConstantOp::create(builder, loc, vecTy, denseValue);
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
                                  bool requiresSrcPtrSwizzling) const {
    if (targetInfo.supportsDirectToLDSScattering()) {
      return success();
    }

    int vecBits = vectorSize * dstTy.getElementTypeBitWidth();
    if (!targetInfo.supportsDirectToLdsLoadBitWidth(vecBits)) {
      LDBG(*op << " results in unsupported load bitwidth: " << vecBits);
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
    if (!requiresSrcPtrSwizzling &&
        !LLVM::AMD::canCoalesceWriteIntoSharedMemory(
            rewriter, srcToSharedLayout, threadsPerWarp, vectorSize)) {
      LDBG(*op << " does not write coalesced into LDS and is not swizzled");
      return failure();
    }

    if (requiresSrcPtrSwizzling &&
        !LLVM::AMD::doesSwizzleInsideWarp(rewriter, srcToSharedLayout,
                                          threadsPerWarp)) {
      LDBG(*op << " does swizzle across warp boundaries");
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
      Value packedArr = LLVM::UndefOp::create(rewriter, loc, structTy);
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

  LogicalResult lowerDirectToLDSLoad(
      RewriterBase &rewriter, Location loc, RankedTensorType srcTy,
      MemDescType dstTy, SmallVector<Value> loadVals, Value llDst,
      Type resElemTy, unsigned vec, int numCTAs,
      triton::AMD::ISAFamily isaFamily,
      std::function<SmallVector<Value>(RewriterBase &, Location,
                                       ArrayRef<Value>, Value, int, VectorType,
                                       Value)>
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
    if (!cvt.isTrivialOver({str_attr("block")})) {
      return emitError(
          loc,
          "direct to lds loads do not support non-trivial block dimension");
    }
    cvt = cvt.sublayout(
        {str_attr("register"), str_attr("lane"), str_attr("warp")},
        {str_attr("offset")});

    Value ctaMulticastMask;
    if (numCTAs > 1 && isaFamily == ISAFamily::GFX1250) {
      ctaMulticastMask = LLVM::AMD::emitCtaMulticastMask(
          rewriter, loc, targetInfo.getClusterCTAId(rewriter, loc), srcLayout);
    }

    auto smemObj =
        LLVM::getSharedMemoryObjectFromStruct(loc, llDst, resElemTy, rewriter);
    auto affineOffset = smemObj.getShmemOffset(loc, rewriter, dstTy);
    auto maskSpanAffineOffset = SharedMemoryObject::getMaskSpanOffsets(dstTy);

    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);

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

    auto lowerInstForwardMulticastMask =
        [&](RewriterBase &rewriter, Location loc, ArrayRef<Value> vals,
            Value shmemAddr, int idx, VectorType vecTy) {
          return lowerInst(rewriter, loc, vals, shmemAddr, idx, vecTy,
                           ctaMulticastMask);
        };

    // If we do not support scattering (GFX9) the address should be the start
    // address (scalar) of the warp
    laneId = targetInfo.supportsDirectToLDSScattering() ? laneId : b.i32_val(0);
    lowerLdSt(loc, ctx, cvt, loadVals, resElemTy, smemObj.getBase(),
              calcPaddedOffset, affineOffset, maskSpanAffineOffset, laneId,
              warpId, rewriter, targetInfo, vec, lowerInstForwardMulticastMask);
    return success();
  }

  void emitOtherStore(RewriterBase &rewriter, Location loc,
                      const LLVMTypeConverter *typeConverter, VectorType vecTy,
                      Value mask, ArrayRef<Value> otherElems, Value shmemAddr,
                      Value laneId, bool requiresSrcPtrSwizzling,
                      Value swizzleLaneOffset) const {
    TritonLLVMOpBuilder b(loc, rewriter);
    Value storeVal = packElementRangeIntoVector(rewriter, typeConverter, loc,
                                                vecTy, otherElems, 0);
    Type ptrTy = shmemAddr.getType();
    Value ldsAddr = shmemAddr;
    // When scattering is unsupported, shmemAddr is the warp base address.
    // Use shmemAddr + lane_id [+ swizzleOffset] to compute each lane's address.
    if (!targetInfo.supportsDirectToLDSScattering()) {
      ldsAddr = b.gep(ptrTy, vecTy, shmemAddr, laneId);
      if (requiresSrcPtrSwizzling)
        ldsAddr = b.gep(ptrTy, vecTy, ldsAddr, swizzleLaneOffset);
    }
    llStore(rewriter, loc, ldsAddr, storeVal, b.icmp_ne(mask, b.true_val()),
            CacheModifier::NONE, targetInfo.requiresAliasInfoForAsyncOps());
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

    Value multicastMask;
    auto mod = op->getParentOfType<ModuleOp>();
    int numCTAs = TritonGPUDialect::getNumCTAs(mod);
    if (numCTAs > 1) {
      Value clusterCTAId = targetInfo.getClusterCTAId(rewriter, loc);
      auto regLayout =
          triton::gpu::toLinearLayout(cast<RankedTensorType>(ptr.getType()));
      multicastMask = LLVM::AMD::emitCtaMulticastMask(rewriter, loc,
                                                      clusterCTAId, regLayout);
    }

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

      Value loadVal = llLoad(rewriter, loc, ptr, vecTy, pred, falseVal,
                             multicastMask, cacheMod);
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

    // If the op has a contiguity hint use it to increase the vector size.
    vec = std::max(vec, op.getContiguity());

    // For padded encodings restrict vec by the min interval
    if (auto padEnc = dyn_cast<PaddedSharedEncodingAttr>(dstEnc)) {
      vec = std::min(vec, padEnc.getMinInterval());
    }

    auto maybeSwizzledEnc = dyn_cast<SwizzledSharedEncodingAttr>(dstEnc);
    bool requiresSrcPtrSwizzling =
        !targetInfo.supportsDirectToLDSScattering() && maybeSwizzledEnc &&
        maybeSwizzledEnc.getMaxPhase() != 1;
    if (failed(canWriteCoalesced(rewriter, op, ptrType, dstTy, vec,
                                 requiresSrcPtrSwizzling))) {
      return failure();
    }

    // For swizzled layouts we need to use the non swizzled layout to compute
    // the LDS addresses since we gather into LDS
    auto flatDstTy = dstTy;
    SmallVector<Value> swizzledLaneOffsets;

    if (requiresSrcPtrSwizzling) {
      // TODO (alex): this is only correct as long as the lds view is a
      // contiguous block. So this can break if we slice along the 2 minor
      // dimensions.
      auto flatSharedEnc = SwizzledSharedEncodingAttr::get(
          op->getContext(), maybeSwizzledEnc.getVec(), 1, 1,
          maybeSwizzledEnc.getOrder(), maybeSwizzledEnc.getCGALayout());
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
         offsetTy, otherTy, hasOther, requiresSrcPtrSwizzling](
            RewriterBase &rewriter, Location loc, ArrayRef<Value> loadVals,
            Value shmemAddr, int startIdx, VectorType vecTy,
            Value multicastMask) -> SmallVector<Value> {
      auto [offsetElem, maskElem, otherElems, swizzleLaneOffset] =
          unzipLoadValues(rewriter, loc, startIdx, loadVals, offsetTy, otherTy,
                          hasOther, vecTy.getNumElements());
      int vecBits = vecTy.getNumElements() * vecTy.getElementTypeBitWidth();
      assert(targetInfo.supportsDirectToLdsLoadBitWidth(vecBits));
      Value vecBytesVal = b.i32_val(vecBits / 8);

      Value maybeSwizzledMaskElem = maskElem;
      if (requiresSrcPtrSwizzling)
        applySwizzling(rewriter, loc, offsetElem, maybeSwizzledMaskElem, laneId,
                       swizzleLaneOffset);

      // If other=0.0 we remove other in canonicalizePointers and we can use out
      // of bounds to store 0 to LDS. So if we have other values we need to
      // predicate to not overwrite the other stores
      Value cond =
          hasOther ? b.and_(threadPred, maybeSwizzledMaskElem) : threadPred;

      auto [loadBlock, afterLoadBlock] = emitBranch(rewriter, loc, cond);

      auto bufferLoadToLds = bufferEmitter.emitLoadToLds(
          vecTy, vecBytesVal, rsrcDesc, offsetElem, shmemAddr,
          hasOther ? b.true_val() : maybeSwizzledMaskElem, op.getCache());
      if (targetInfo.requiresAliasInfoForAsyncOps())
        AMD::addAsyncCopyAliasScope(bufferLoadToLds);

      if (hasOther) {
        emitOtherStore(rewriter, loc, this->getTypeConverter(), vecTy, maskElem,
                       otherElems, shmemAddr, laneId, requiresSrcPtrSwizzling,
                       swizzleLaneOffset);
      }

      rewriter.setInsertionPointToStart(afterLoadBlock);

      return {};
    };

    int numCTAs = TritonGPUDialect::getNumCTAs(op->getParentOfType<ModuleOp>());
    auto res = lowerDirectToLDSLoad(
        rewriter, loc, ptrType, flatDstTy, loadVals, llDst, resElemTy, vec,
        numCTAs, targetInfo.getISAFamily(), emitBufferLoadLds);
    if (failed(res)) {
      return failure();
    }

    // Drop the result token.
    Value zero = LLVM::ConstantOp::create(rewriter, op.getLoc(),
                                          IntegerType::get(op.getContext(), 32),
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

    // If the op has a contiguity hint use it to increase the vector size.
    vec = std::max(vec, op.getContiguity());

    // For padded encodings restrict vec by the min interval
    if (auto padEnc = dyn_cast<PaddedSharedEncodingAttr>(dstEnc)) {
      vec = std::min(vec, padEnc.getMinInterval());
    }

    auto maybeSwizzledEnc = dyn_cast<SwizzledSharedEncodingAttr>(dstEnc);
    bool requiresSrcPtrSwizzling =
        !targetInfo.supportsDirectToLDSScattering() && maybeSwizzledEnc &&
        maybeSwizzledEnc.getMaxPhase() != 1;

    if (failed(canWriteCoalesced(rewriter, op, srcTy, dstTy, vec,
                                 requiresSrcPtrSwizzling))) {
      return failure();
    }

    // For swizzled layouts we need to use the non swizzled layout to compute
    // the LDS addresses since we gather into LDS
    auto flatDstTy = dstTy;
    SmallVector<Value> swizzledLaneOffsets;
    if (requiresSrcPtrSwizzling) {
      auto flatSharedEnc = SwizzledSharedEncodingAttr::get(
          op->getContext(), maybeSwizzledEnc.getVec(), 1, 1,
          maybeSwizzledEnc.getOrder(), maybeSwizzledEnc.getCGALayout());
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

    auto freeVarMasks = getFreeVariableMasks(srcTy);
    // We load redundant data on different CTAs so each CTA has a copy in its
    // shared memory; the multicast mask will be used by the hardware to
    // efficiently broadcast to different CTAs.
    freeVarMasks[rewriter.getStringAttr("block")] = 0;
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);

    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    auto emitGlobalLoadLds =
        [this, &op, &b, laneId = laneId, threadPred, srcPtrTy, otherTy,
         hasOther, requiresSrcPtrSwizzling](
            RewriterBase &rewriter, Location loc, ArrayRef<Value> loadValues,
            Value shmemAddr, int startIdx, VectorType vecTy,
            Value multicastMask) -> SmallVector<Value> {
      auto [srcElem, maskElem, otherElems, swizzleLaneOffset] =
          unzipLoadValues(rewriter, loc, startIdx, loadValues, srcPtrTy,
                          otherTy, hasOther, vecTy.getNumElements());
      int vecBits = vecTy.getNumElements() * vecTy.getElementTypeBitWidth();
      assert(targetInfo.supportsDirectToLdsLoadBitWidth(vecBits));
      Value maybeSwizzledMaskElem = maskElem;

      if (requiresSrcPtrSwizzling)
        applySwizzling(rewriter, loc, srcElem, maybeSwizzledMaskElem, laneId,
                       swizzleLaneOffset);

      // Predicate load based on threadPred && swizzledMask
      auto cond = b.and_(threadPred, maybeSwizzledMaskElem);
      auto [loadBlock, afterLoadBlock] = emitBranch(rewriter, loc, cond);

      emitAsyncLoad(rewriter, loc, targetInfo, vecBits, srcElem, shmemAddr,
                    op.getCache(), multicastMask);

      rewriter.setInsertionPointToStart(afterLoadBlock);

      if (hasOther) {
        emitOtherStore(rewriter, loc, this->getTypeConverter(), vecTy, maskElem,
                       otherElems, shmemAddr, laneId, requiresSrcPtrSwizzling,
                       swizzleLaneOffset);
      }

      return {};
    };

    int numCTAs = TritonGPUDialect::getNumCTAs(op->getParentOfType<ModuleOp>());
    auto res = lowerDirectToLDSLoad(
        rewriter, loc, srcTy, flatDstTy, loadVals, llDst, resElemTy, vec,
        numCTAs, targetInfo.getISAFamily(), emitGlobalLoadLds);
    if (failed(res)) {
      return failure();
    }

    // Drop the result token.
    Value zero = LLVM::ConstantOp::create(rewriter, op.getLoc(),
                                          IntegerType::get(op.getContext(), 32),
                                          rewriter.getI32IntegerAttr(0));
    rewriter.replaceOp(op, zero);
    return success();
  }

  void emitAsyncLoad(RewriterBase &rewriter, Location loc,
                     AMD::TargetInfo targetInfo, int vecBits, Value srcPtr,
                     Value shmemAddr, triton::CacheModifier cacheMod,
                     Value multicastMask) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    int32_t cacheModifiers =
        mlir::LLVM::AMD::getCtrlBitsForCacheModifierOnTarget(
            cacheMod, /*isLoad=*/true, targetInfo);

    if (llvm::is_contained({ISAFamily::CDNA3, ISAFamily::CDNA4},
                           targetInfo.getISAFamily())) {
      auto globalLoadLdsOp = ROCDL::GlobalLoadLDSOp::create(
          rewriter, loc, srcPtr, shmemAddr, vecBits / 8,
          /*offset=*/0, cacheModifiers, nullptr, nullptr, nullptr);
      if (targetInfo.requiresAliasInfoForAsyncOps())
        AMD::addAsyncCopyAliasScope(globalLoadLdsOp);
    } else if (targetInfo.getISAFamily() == ISAFamily::GFX1250) {
      if (cacheMod != triton::CacheModifier::NONE) {
        emitRemark(loc) << "cache modifiers not yet implemented on gfx1250";
      }
      if (multicastMask) {
        std::string intrinsic =
            "llvm.amdgcn.cluster.load.async.to.lds.b" + std::to_string(vecBits);
        auto globalLoadLdsOp = LLVM::createLLVMIntrinsicCallOp(
            rewriter, loc, intrinsic, {},
            {srcPtr, shmemAddr, b.i32_val(0), b.i32_val(cacheModifiers),
             multicastMask});
      } else {
        std::string intrinsic =
            "llvm.amdgcn.global.load.async.to.lds.b" + std::to_string(vecBits);
        auto globalLoadLdsOp = LLVM::createLLVMIntrinsicCallOp(
            rewriter, loc, intrinsic, {},
            {srcPtr, shmemAddr, b.i32_val(0), b.i32_val(cacheModifiers)});
      }
    }
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

  LogicalResult
  matchAndRewrite(triton::amdgpu::AsyncTDMCopyGlobalToLocalOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto tensorDescTy = op.getDesc().getType();
    auto smemTy = op.getResult().getType();
    auto paddedEnc =
        llvm::dyn_cast<PaddedSharedEncodingAttr>(smemTy.getEncoding());
    Type elementType = getTypeConverter()->convertType(smemTy.getElementType());
    int numCTAs = TritonGPUDialect::getNumCTAs(op->getParentOfType<ModuleOp>());

    triton::LinearLayout sharedLayout;
    unsigned padInterval = 0;
    unsigned padAmount = 0;
    if (paddedEnc) {
      assert(paddedEnc.getIntervals().size() == 1 &&
             paddedEnc.getPaddings().size() == 1);
      sharedLayout = paddedEnc.getLinearComponent();
      padInterval = paddedEnc.getIntervals()[0];
      padAmount = paddedEnc.getPaddings()[0];
    } else {
      sharedLayout = triton::gpu::toLinearLayout(smemTy);
    }
    Value multicastMask;
    if (numCTAs > 1) {
      multicastMask = LLVM::AMD::emitCtaMulticastMask(
          rewriter, loc, targetInfo.getClusterCTAId(rewriter, loc),
          sharedLayout);
    }

    SmallVector<Value> desc =
        unpackLLElements(loc, adaptor.getDesc(), rewriter);

    SmallVector<int64_t> blockShape =
        llvm::to_vector(tensorDescTy.getBlockType().getShape());

    // 2D tensors: 12 dwords (group0: 4, group1: 8)
    // 3D-5D tensors: 20 dwords (group0: 4, group1: 8, group2: 4, group3: 4)
    assert((blockShape.size() <= 2 && desc.size() == 12) ||
           (blockShape.size() > 2 && desc.size() == 20));

    auto dstMemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getResult(), elementType, rewriter);
    Value dstPtr = dstMemObj.getBase();
    SmallVector<Value> offset = adaptor.getIndices();
    int numWarps = triton::gpu::lookupNumWarps(op);

    Value barrierPtr = nullptr;
    if (op.getBarrier()) {
      auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
          loc, adaptor.getBarrier(),
          typeConverter->convertType(
              op.getBarrier().getType().getElementType()),
          rewriter);
      barrierPtr = smemObj.getBase();
    }

    auto kBlock = rewriter.getStringAttr("block");
    auto cgaLayout = sharedLayout.sublayout(
        {kBlock}, to_vector(sharedLayout.getOutDimNames()));
    auto ctaId =
        numCTAs > 1 ? targetInfo.getClusterCTAId(rewriter, loc) : b.i32_val(0);

    auto shapePerCTA = triton::gpu::getShapePerCTA(smemTy);
    mlir::LLVM::AMD::emitTDMOperation(
        rewriter, loc, getTypeConverter(), desc, shapePerCTA, numWarps,
        padInterval, padAmount, offset, dstPtr, op.getPred(), multicastMask,
        elementType, barrierPtr, /*isLoad=*/true, cgaLayout, ctaId);

    rewriter.eraseOp(op);
    return success();
  }
};

struct AsyncTDMCopyLocalToGlobalOpConversion
    : public ConvertOpToLLVMPattern<
          triton::amdgpu::AsyncTDMCopyLocalToGlobalOp>,
      public LoadStoreConversionBase {
  AsyncTDMCopyLocalToGlobalOpConversion(
      LLVMTypeConverter &converter, const AMD::TargetInfo &targetInfo,
      ModuleAxisInfoAnalysis &axisAnalysisPass, PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::amdgpu::AsyncTDMCopyLocalToGlobalOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto tensorDescTy = op.getDesc().getType();
    auto smemTy = op.getSrc().getType();
    Type elementType = getTypeConverter()->convertType(smemTy.getElementType());
    int numCTAs = TritonGPUDialect::getNumCTAs(op->getParentOfType<ModuleOp>());

    SmallVector<Value> desc =
        unpackLLElements(loc, adaptor.getDesc(), rewriter);

    SmallVector<int64_t> blockShape =
        llvm::to_vector(tensorDescTy.getBlockType().getShape());

    // 2D tensors: 12 dwords (group0: 4, group1: 8)
    // 3D-5D tensors: 20 dwords (group0: 4, group1: 8, group2: 4, group3: 4)
    assert((blockShape.size() <= 2 && desc.size() == 12) ||
           (blockShape.size() > 2 && desc.size() == 20));

    auto dstMemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getSrc(), elementType, rewriter);
    Value dstPtr = dstMemObj.getBase();
    SmallVector<Value> offset = adaptor.getIndices();
    int numWarps = triton::gpu::lookupNumWarps(op);

    Value barrierPtr = nullptr;
    if (op.getBarrier()) {
      auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
          loc, adaptor.getBarrier(),
          typeConverter->convertType(
              op.getBarrier().getType().getElementType()),
          rewriter);
      barrierPtr = smemObj.getBase();
    }

    // Verifier ensures smem is not usind a PaddedSharedEncodingAttr
    auto sharedLayout = triton::gpu::toLinearLayout(smemTy);
    auto kBlock = rewriter.getStringAttr("block");
    auto cgaLayout = sharedLayout.sublayout(
        {kBlock}, to_vector(sharedLayout.getOutDimNames()));
    auto ctaId =
        numCTAs > 1 ? targetInfo.getClusterCTAId(rewriter, loc) : b.i32_val(0);

    auto shapePerCTA = triton::gpu::getShapePerCTA(smemTy);
    mlir::LLVM::AMD::emitTDMOperation(
        rewriter, loc, getTypeConverter(), desc, shapePerCTA, numWarps,
        /*padInterval=*/0, /*padAmount=*/0, offset, dstPtr, b.true_val(),
        /*multicastMask=*/{}, elementType, barrierPtr,
        /*isLoad=*/false, cgaLayout, ctaId);

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
        auto cmpxchg = LLVM::AtomicCmpXchgOp::create(
            rewriter, loc, casPtr, casCmp, casVal, successOrdering,
            failureOrdering, StringRef(scopeStr.value()));

        // Extract the new_loaded value from the pair.
        Value ret = b.extract_val(valueElemTy, cmpxchg, 0);
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
        LLVM::CondBrOp::create(rewriter, loc, pred, atomicBlock, endBlock);

        // Build main block with atomic_cmpxchg.
        rewriter.setInsertionPointToEnd(atomicBlock);

        auto successOrdering = LLVM::AtomicOrdering::acq_rel;
        auto failureOrdering = LLVM::AtomicOrdering::monotonic;
        auto cmpxchg = LLVM::AtomicCmpXchgOp::create(
            rewriter, loc, casPtr, casCmp, casVal, successOrdering,
            failureOrdering, StringRef("agent"));

        if (!op.getResult().use_empty()) {
          // Extract the new_loaded value from the pair.
          Value newLoaded = b.extract_val(valueElemTy, cmpxchg, 0);
          Value atomPtr =
              getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
          b.store(newLoaded, atomPtr);
        }

        LLVM::BrOp::create(rewriter, loc, ValueRange(), endBlock);

        // Build the last block: synced load from shared memory, exit.
        rewriter.setInsertionPointToStart(endBlock);

        if (op.getResult().use_empty()) {
          rewriter.eraseOp(op);
          return success();
        }

        auto dsCount = rewriter.getI32IntegerAttr(0);
        amdgpu::MemoryCounterWaitOp::create(rewriter, op->getLoc(),
                                            /*load=*/nullptr, /*store=*/nullptr,
                                            /*ds=*/dsCount);
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

    bool needLdsStaging = !tensorTy && !opResult.use_empty();
    std::optional<Value> atomicSharedMemBase =
        op->hasAttr("allocation.offset") && needLdsStaging
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

struct AsyncWaitOpConversion
    : public ConvertOpToLLVMPattern<amdgpu::AsyncWaitOp> {
  AsyncWaitOpConversion(LLVMTypeConverter &converter,
                        const AMD::TargetInfo &targetInfo,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(amdgpu::AsyncWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    switch (targetInfo.getISAFamily()) {
    case ISAFamily::CDNA1:
    case ISAFamily::CDNA2:
    case ISAFamily::CDNA3:
    case ISAFamily::CDNA4: {
      // global.load.lds uses vmcnt to synchronize
      // The rocdl op stores all available counters in a single int32 value (v).
      // The vmcnt (6 bits) is split into a lower 3:0 and higher 5:4 parts.
      // The lower part is stored in bits 3:0 of v and the higher part in bits
      // 15:14. We have to set all other bits in v to 1 to signal we are not
      // interested in those.

      // Clamp vmcnt to 6bits; a lower vmcnt will produce a conservative wait
      unsigned vmCnt = std::min(63u, op.getNumInst());

      // Extract low and high bits and combine while setting all other bits to 1
      unsigned lowBits = vmCnt & 0xF;
      unsigned highBits = vmCnt >> 4 << 14;
      unsigned otherCnts = ~0xC00F; // C00F has bits 15:14 and 3:0 set
      unsigned waitValue = lowBits | highBits | otherCnts;

      ROCDL::SWaitcntOp::create(rewriter, loc, waitValue);
      break;
    }
    case ISAFamily::GFX1250: {
      // Clamp asyncCnt to 6bits(hw imit); lower means conservative
      unsigned asyncCnt = std::min(63u, op.getNumInst());
      ROCDL::WaitAsynccntOp::create(rewriter, loc, asyncCnt);
      break;
    }
    default:
      return rewriter.notifyMatchFailure(
          op, "Only supported on CDNA target architecture");
    }

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
    ROCDL::WaitTensorcntOp::create(rewriter, loc, op.getNum());
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

struct AsyncCopyMbarrierArriveOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::AsyncCopyMbarrierArriveOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::amdgpu::AsyncCopyMbarrierArriveOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    TritonLLVMOpBuilder b(loc, rewriter);
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getBarrier(),
        typeConverter->convertType(op.getBarrier().getType().getElementType()),
        rewriter);
    LLVM::createLLVMIntrinsicCallOp(
        rewriter, loc, "llvm.amdgcn.ds.atomic.async.barrier.arrive.b64",
        void_ty(getContext()), smemObj.getBase());
    rewriter.eraseOp(op);
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
           BufferAtomicCASOpConversion, AsyncTDMCopyGlobalToLocalOpConversion,
           AsyncTDMCopyLocalToGlobalOpConversion>(typeConverter, targetInfo,
                                                  axisInfoAnalysis, benefit);
  patterns.add<AsyncWaitOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<AsyncTDMWaitConversion>(typeConverter, benefit);
  patterns.add<AsyncCommitGroupOpConversion>(typeConverter, benefit);
  patterns.add<AsyncCopyMbarrierArriveOpConversion>(typeConverter, benefit);
}
} // namespace mlir::triton::AMD
