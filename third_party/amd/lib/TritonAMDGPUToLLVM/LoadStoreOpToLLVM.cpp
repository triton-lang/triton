#include "BufferOpsEmitter.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "SchedInstructions.h"
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

using namespace mlir;
using namespace mlir::triton::gpu;

using ::mlir::LLVM::delinearize;
using ::mlir::LLVM::getSharedMemoryBase;
using ::mlir::LLVM::AMD::getVectorSize;
using ::mlir::LLVM::AMD::llLoad;
using ::mlir::LLVM::AMD::llStore;
using ::mlir::triton::gpu::getTotalElemsPerThread;

namespace {
// Return the mask for the unique data accessed by given tensor type.
// Used to mask out the redundant data accessed by threads.
Value redundantDataMask(Type valueTy, ConversionPatternRewriter &rewriter,
                        Location loc, const AMD::TargetInfo &targetInfo) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto tensorTy = dyn_cast<RankedTensorType>(valueTy);
  Value mask = b.int_val(1, 1);
  auto tid = b.tid_val();
  auto clusterCTAId = targetInfo.getClusterCTAId(rewriter, loc);
  if (tensorTy) {
    auto layout = tensorTy.getEncoding();
    auto shape = tensorTy.getShape();
    unsigned rank = shape.size();
    auto sizePerThread = triton::gpu::getSizePerThread(layout);
    auto threadsPerWarp = triton::gpu::getThreadsPerWarp(layout);
    auto warpsPerCTA = triton::gpu::getWarpsPerCTA(layout);
    auto threadOrder = triton::gpu::getThreadOrder(layout);
    SmallVector<unsigned> warpOrder(rank);
    if (auto enc = dyn_cast<DotOperandEncodingAttr>(layout)) {
      warpOrder =
          triton::gpu::getMatrixOrder(rank, /*rowMajor=*/enc.getOpIdx() == 1);
    } else {
      warpOrder = triton::gpu::getWarpOrder(layout);
    }
    auto shapePerCTATile = triton::gpu::getShapePerCTATile(layout);
    Value warpSize = b.i32_val(triton::gpu::getWarpSize(layout));
    Value laneId = b.urem(tid, warpSize);
    Value warpId = b.udiv(tid, warpSize);
    // TODO: [DOT LL]
    // The delinearize function is not entirely correct for certain layouts,
    // such as wgmma. The correct approach is to convert a legacy layout to its
    // corresponding linear layout and use the linear layout's
    // getFreeVariableMasks to identify redundant elements.
    SmallVector<Value> multiDimWarpId =
        delinearize(rewriter, loc, warpId, warpsPerCTA, warpOrder);
    SmallVector<Value> multiDimThreadId =
        delinearize(rewriter, loc, laneId, threadsPerWarp, threadOrder);
    for (unsigned dim = 0; dim < rank; ++dim) {
      // if there is no data replication across threads on this dimension
      if (shape[dim] >= shapePerCTATile[dim])
        continue;
      // Otherwise, we need to mask threads that will replicate data on this
      // dimension. Calculate the thread index on this dimension for the CTA
      Value threadDim =
          b.add(b.mul(multiDimWarpId[dim], b.i32_val(threadsPerWarp[dim])),
                multiDimThreadId[dim]);
      mask = b.and_(mask,
                    b.icmp_slt(b.mul(threadDim, b.i32_val(sizePerThread[dim])),
                               b.i32_val(shape[dim])));
    }
    // Do not write duplicated data when multicast is enabled
    if (triton::gpu::getNumCTAs(layout) > 1) {
      auto _0 = b.i32_val(0);
      auto CTAsPerCGA = triton::gpu::getCTAsPerCGA(layout);
      auto CTASplitNum = triton::gpu::getCTASplitNum(layout);
      auto CTAOrder = triton::gpu::getCTAOrder(layout);

      auto multiDimClusterCTAId =
          delinearize(rewriter, loc, clusterCTAId, CTAsPerCGA, CTAOrder);

      for (unsigned dim = 0; dim < rank; ++dim) {
        // Skip when multicast is not enabled in this dimension
        if (CTAsPerCGA[dim] == CTASplitNum[dim])
          continue;
        // This wrapping rule must be consistent with emitCTAOffsetForLayout
        unsigned splitNum = std::min<unsigned>(shape[dim], CTASplitNum[dim]);
        Value repId = b.udiv(multiDimClusterCTAId[dim], b.i32_val(splitNum));
        // Consider the example where CTAsPerCGA = [4] and CTASplitNum = [2]:
        //     CTA0 and CTA2 holds data of block0,
        //     CTA1 and CTA3 holds data of block1.
        // Only CTA0 and CTA1 are expected to write while CTA2 and CTA3 should
        // be masked. We add the following mask:
        //     multiDimClusterCTAId[dim] / splitNum == 0
        // Actually in all existing cases of multicast, splitNum is always 1.
        // The mask is equivalent to:
        //     multiDimClusterCTAId[dim] == 0
        mask = b.and_(mask, b.icmp_eq(repId, _0));
      }
    }
  } else {
    // If the tensor is not ranked, then it is a scalar and only thread 0 of
    // CTA0 can write
    mask = b.and_(mask, b.icmp_eq(clusterCTAId, b.i32_val(0)));
    mask = b.and_(mask, b.icmp_eq(tid, b.i32_val(0)));
  }
  return mask;
}

// Contains some helper functions for both Load and Store conversions.
struct LoadStoreConversionBase {
  explicit LoadStoreConversionBase(const AMD::TargetInfo &targetInfo,
                                   ModuleAxisInfoAnalysis &axisAnalysisPass)
      : targetInfo(targetInfo), axisAnalysisPass(axisAnalysisPass) {}

  // Createa a LLVM vector of type `vecTy` containing all zeros
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
  Value packElementRangeIntoVector(ConversionPatternRewriter &rewriter,
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

  unsigned getPtrAlignment(Value ptr) const {
    return axisAnalysisPass.getPtrAlignment(ptr);
  }

  std::optional<const std::string>
  getAMDGPUMemScopeStr(MemSyncScope scope) const {
    // See: https://llvm.org/docs/AMDGPUUsage.html#memory-scopes
    auto scopeStr = "";
    switch (scope) {
    case MemSyncScope::SYSTEM:
      // The default AMDHSA LLVM Sync Scope is "system", so no string is
      // provided here
      scopeStr = "";
      break;
    case MemSyncScope::GPU:
      scopeStr = "agent";
      break;
    case MemSyncScope::CTA:
      scopeStr = "workgroup";
      break;
    default:
      return std::nullopt;
    }

    return scopeStr;
  }

protected:
  const AMD::TargetInfo &targetInfo;
  ModuleAxisInfoAnalysis &axisAnalysisPass;
};

struct LoadOpConversion : public ConvertOpToLLVMPattern<triton::LoadOp>,
                          public LoadStoreConversionBase {
  using ConvertOpToLLVMPattern<triton::LoadOp>::ConvertOpToLLVMPattern;

  LoadOpConversion(LLVMTypeConverter &converter,
                   const AMD::TargetInfo &targetInfo,
                   ModuleAxisInfoAnalysis &axisAnalysisPass,
                   PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::LoadOp>(converter, benefit),
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
    int64_t ptrAlignmentBytes = getPtrAlignment(ptr) * valueElemNBytes;

    auto cacheMod = op.getCache();
    SmallVector<Value> loadedVals;
    Type vecTy = LLVM::getFixedVectorType(valueElemTy, vec);
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
                             ptrAlignmentBytes, cacheMod);
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

    setNumGeneratedGlobalLoads(op, numVecs, vecTy);

    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct BufferLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::BufferLoadOp>,
      public LoadStoreConversionBase {
  using ConvertOpToLLVMPattern<
      triton::amdgpu::BufferLoadOp>::ConvertOpToLLVMPattern;

  BufferLoadOpConversion(LLVMTypeConverter &converter,
                         const AMD::TargetInfo &targetInfo,
                         ModuleAxisInfoAnalysis &axisAnalysisPass,
                         PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::amdgpu::BufferLoadOp>(converter,
                                                             benefit),
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
    Type vecTy = LLVM::getFixedVectorType(valueElemTy, vec);
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

    const int numVecs = numElems / vec;
    setNumGeneratedGlobalLoads(op, numVecs, vecTy);

    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct AsyncCopyGlobalToLocalOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::AsyncCopyGlobalToLocalOp>,
      public LoadStoreConversionBase {
  AsyncCopyGlobalToLocalOpConversion(LLVMTypeConverter &converter,
                                     const AMD::TargetInfo &targetInfo,
                                     ModuleAxisInfoAnalysis &axisAnalysisPass,
                                     PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  bool supportsLoadWidth(unsigned bits,
                         const AMD::TargetInfo &targetInfo) const {
    llvm::SmallSetVector<unsigned, 10> supportedWidths;
    using mlir::triton::AMD::ISAFamily;
    switch (targetInfo.getISAFamily()) {
    case ISAFamily::CDNA1:
    case ISAFamily::CDNA2:
    case ISAFamily::CDNA3:
      supportedWidths.insert(8);
      supportedWidths.insert(16);
      supportedWidths.insert(32);
      if (targetInfo.getGPUKind() == llvm::AMDGPU::GPUKind::GK_GFX950) {
        supportedWidths.insert(96);
        supportedWidths.insert(128);
      }
      break;
    default:
      return false;
    }

    return supportedWidths.contains(bits);
  }

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncCopyGlobalToLocalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto srcTy = op.getSrc().getType();
    auto srcEncoding = srcTy.getEncoding();

    if (!isa<BlockedEncodingAttr, SliceEncodingAttr>(srcEncoding))
      return rewriter.notifyMatchFailure(
          op, "requires Blocked or Slice encoding for src");
    if (srcTy.getShape().size() != 2)
      return rewriter.notifyMatchFailure(op, "only supports 2d tensors");

    auto dstTy = op.getResult().getType();
    auto resElemTy = getTypeConverter()->convertType(dstTy.getElementType());

    Value llSrc = adaptor.getSrc();

    auto srcElems = unpackLLElements(loc, llSrc, rewriter);

    Value llDst = adaptor.getResult();
    auto smemObj = mlir::LLVM::getSharedMemoryObjectFromStruct(
        loc, llDst, resElemTy, rewriter);

    // We can load N elements at a time if:
    //  1. Every group of N source pointers are contiguous.  For example, if
    //     N=2, then the pointers should be [x, x+1, y, y+1, ...].
    //  2. The mask (if present) has "alignment" N, meaning that each group of N
    //     mask bits are the same.  For example if N=2, the mask must be
    //     [x, x, y, y, ...].
    unsigned maxVec =
        mlir::LLVM::AMD::getContiguity(op.getSrc(), axisAnalysisPass);
    Value mask = op.getMask();
    if (mask) {
      maxVec = std::min(maxVec, getMaskAlignment(mask));
    }

    // global.load.lds does not support per lane offsets.
    // We need to ensure that we write coalesced into shared memory. This means
    // that the kLane dim needs to be contigeous based on the vectorization
    // size.
    auto shape = dstTy.getShape();
    LinearLayout srcLayout =
        triton::gpu::toLinearLayout(shape, srcTy.getEncoding());
    LinearLayout sharedLayout =
        triton::gpu::toLinearLayout(shape, dstTy.getEncoding());
    LinearLayout srcToSharedLayout = srcLayout.invertAndCompose(sharedLayout);

    StringAttr kLane = rewriter.getStringAttr("lane");
    for (int inLane : llvm::seq(srcToSharedLayout.getInDimSizeLog2(kLane))) {
      auto basis = srcToSharedLayout.getBasis(kLane, inLane)[0];
      unsigned expected = maxVec * (1 << inLane);
      if (basis != expected) {
        LDBG("detected uncoalesced layout from blocked to shared in async copy "
             "for lane "
             << 1 + inLane << "; given " << basis << " but expected "
             << expected);
        return rewriter.notifyMatchFailure(op,
                                           "does not write coalesced into LDS");
      }
    }

    // Addresses to store into, one per `vecTy`.
    VectorType vecTy;
    SmallVector<Value> shmemAddrs;
    bool ok = emitTransferBetweenRegistersAndShared(
        srcTy, dstTy, resElemTy, {}, smemObj, loc, rewriter, targetInfo,
        [&](VectorType vecTy_, Value shmemAddr) {
          vecTy = vecTy_;
          shmemAddrs.push_back(shmemAddr);
        });
    assert(ok);

    int vecBits = vecTy.getNumElements() * vecTy.getElementTypeBitWidth();
    if (!supportsLoadWidth(vecBits, targetInfo)) {
      return rewriter.notifyMatchFailure(
          op, "Async copy does not support the required load vectorization");
    }

    int vecBytes = vecBits / 8;
    assert(llvm::isPowerOf2_32(vecBytes));
    Value vecBytesVal = b.i32_val(vecBytes);

    Value cacheModifiers =
        b.i32_val(mlir::LLVM::AMD::getCtrlBitsForCacheModifierOnTarget(
            op.getCache(), false, targetInfo));

    Value llMask = adaptor.getMask();
    SmallVector<Value> maskElems;
    if (llMask) {
      maskElems = unpackLLElements(loc, llMask, rewriter);
      assert(srcElems.size() == maskElems.size());
    }

    Value other = op.getOther();
    SmallVector<Value> otherElems;
    if (other) {
      otherElems = unpackLLElements(loc, adaptor.getOther(), rewriter);
      assert(srcElems.size() == otherElems.size());
    }

    for (int i = 0; i < shmemAddrs.size(); i++) {
      auto srcIdx = i * maxVec;
      auto srcPtr = srcElems[srcIdx];

      if (!mask) {
        rewriter.create<ROCDL::GlobalLoadLDSOp>(
            loc, srcPtr, shmemAddrs[i], vecBytesVal, /*offset=*/b.i32_val(0),
            cacheModifiers);
        continue;
      }

      Block *currentBlock = rewriter.getInsertionBlock();
      Block *afterLoad =
          rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
      Block *loadBlock = rewriter.createBlock(afterLoad);
      rewriter.setInsertionPointToEnd(currentBlock);
      rewriter.create<LLVM::CondBrOp>(loc, maskElems[srcIdx], loadBlock,
                                      afterLoad);
      rewriter.setInsertionPointToStart(loadBlock);
      rewriter.create<ROCDL::GlobalLoadLDSOp>(
          loc, srcPtr, shmemAddrs[i], vecBytesVal, /*offset=*/b.i32_val(0),
          cacheModifiers);

      rewriter.create<LLVM::BrOp>(loc, afterLoad);
      rewriter.setInsertionPointToStart(afterLoad);
      if (other) {
        Value storeVal = packElementRangeIntoVector(
            rewriter, this->getTypeConverter(), loc, vecTy, otherElems, srcIdx);
        llStore(rewriter, loc, shmemAddrs[i], storeVal,
                b.icmp_ne(maskElems[srcIdx], b.true_val()), 0, op.getCache());
      }
    }

    // Drop the result token.
    Value zero = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), IntegerType::get(op.getContext(), 32),
        rewriter.getI32IntegerAttr(0));
    rewriter.replaceOp(op, zero);
    return success();
  }
};

struct StoreOpConversion : public ConvertOpToLLVMPattern<triton::StoreOp>,
                           public LoadStoreConversionBase {
  using ConvertOpToLLVMPattern<triton::StoreOp>::ConvertOpToLLVMPattern;

  StoreOpConversion(LLVMTypeConverter &converter,
                    const AMD::TargetInfo &targetInfo,
                    ModuleAxisInfoAnalysis &axisAnalysisPass,
                    PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::StoreOp>(converter, benefit),
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
    int64_t ptrAlignmentBytes = getPtrAlignment(ptr) * valueElemNBytes;

    auto cacheMod = op.getCache();
    const int numVecs = elemsPerThread / vec;
    Value rDataMask = redundantDataMask(valueTy, rewriter, loc, targetInfo);
    for (size_t vecStart = 0; vecStart < elemsPerThread; vecStart += vec) {
      Value pred = mask ? b.and_(maskElems[vecStart], rDataMask) : rDataMask;
      auto vecTy = LLVM::getFixedVectorType(valueElemTy, vec);

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
      llStore(rewriter, loc, ptr, storeVal, pred, ptrAlignmentBytes, cacheMod);
    } // end vec
    rewriter.eraseOp(op);
    return success();
  }
};

static LLVM::AtomicOrdering getMemoryOrdering(MemSemantic memOrdering) {
  switch (memOrdering) {
  case MemSemantic::RELAXED:
    return LLVM::AtomicOrdering::monotonic;
  case MemSemantic::ACQUIRE:
    return LLVM::AtomicOrdering::acquire;
  case MemSemantic::RELEASE:
    return LLVM::AtomicOrdering::release;
  case MemSemantic::ACQUIRE_RELEASE:
    return LLVM::AtomicOrdering::acq_rel;
  default:
    return LLVM::AtomicOrdering::acq_rel;
  }
}

struct BufferAtomicRMWOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::BufferAtomicRMWOp>,
      public LoadStoreConversionBase {
  using ConvertOpToLLVMPattern<
      triton::amdgpu::BufferAtomicRMWOp>::ConvertOpToLLVMPattern;

  BufferAtomicRMWOpConversion(LLVMTypeConverter &converter,
                              const AMD::TargetInfo &targetInfo,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::amdgpu::BufferAtomicRMWOp>(converter,
                                                                  benefit),
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

    // We need to manually emit memory fences (LLVM doesn't do this for buffer
    // ops) see: https://llvm.org/docs/AMDGPUUsage.html#memory-model-gfx942
    auto memOrdering = op.getSem();
    auto atomicMemOrdering = getMemoryOrdering(memOrdering);
    auto rel = LLVM::AtomicOrdering::release;
    auto acq = LLVM::AtomicOrdering::acquire;

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

    Value rsrcDesc = bufferEmitter.createResourceDescriptor(llPtr);
    Value rDataMask = redundantDataMask(valueTy, rewriter, loc, targetInfo);
    SmallVector<Value> loadedVals;

    // set the scope
    auto memScope = op.getScope();
    auto scopeStr = "";
    switch (memScope) {
    // System scope is not supported yet
    case MemSyncScope::SYSTEM:
      return rewriter.notifyMatchFailure(
          op, "System memory scope is not supported for Buffer Atomic RMW");
    case MemSyncScope::GPU:
      scopeStr = "agent";
      break;
    case MemSyncScope::CTA:
      scopeStr = "workgroup";
      break;
    default:
      return rewriter.notifyMatchFailure(
          op, "Unsupported memory scope for Buffer Atomic RMW");
    }

    StringAttr scope = mlir::StringAttr::get(loc.getContext(), scopeStr);

    if (emitReleaseFence)
      rewriter.create<LLVM::FenceOp>(loc, TypeRange{}, rel, scope);

    mlir::Operation *lastRMWOp;
    MLIRContext *ctx = rewriter.getContext();
    GCNBuilder waitcntBuilder;

    // Triton supports three scopes for atomic access
    // 1. System
    // 2. GPU (default)
    // 3. CTA (i.e., threadblock or warp-group)
    //
    // Currently, the AMD backend emits atomics with agent-scope.
    //
    // The following properties are used to emit proper synchronization
    // primitives between sequential buffer atomics See: Memory Model GFX942
    // (MI300 series)
    // https://llvm.org/docs/AMDGPUUsage.html#memory-model-gfx942:
    //
    // buffer/global/flat_load/store/atomic instructions to global memory are
    // termed vector memory operations.
    //
    // 1. Vector memory operations access a single vector L1 cache shared by
    // all SIMDs a CU.
    //    No special action is required for coherence between wavefronts in the
    //    same work-group since they execute on the same CU.
    //
    // 2. Each CU has a separate request queue per channel for its associated
    // L2.
    //    Therefore, the vector and scalar memory operations performed by
    //    wavefronts executing with different L1 caches and the same L2 cache
    //    can be reordered relative to each other. A `s_waitcnt vmcnt(0)` is
    //    required to ensure synchronization between vector memory operations of
    //    different CUs. It ensures a previous vector memory operation has
    //    completed before executing a subsequent vector memory or LDS operation
    //    and so can be used to meet the requirements of acquire and release.
    //
    // 3. Atomic read-modify-write instructions implicitly bypass the L1 cache
    //    (specific to gfx942)
    //    Therefore, they do not use the sc0 bit for coherence and instead use
    //    it to indicate if the instruction returns the original value being
    //    updated. They do use sc1 to indicate system or agent scope coherence.
    //    See the cache modifiers word in BufferEmitter::fillCommonArgs for
    //    more details.
    //
    // In summary:
    // 1. We have to emit memory fences (i.e., acq/rel/acq_rel) before and after
    //    our buffer atomics.
    // 2. Because buffer atomic rmw ops skip the l1 cache, s_waitcnt vmcnt(0) is
    //    sufficient for synchronization between instructions.
    //    We don't need to invalidate L1 between these ops on GFX942, just after
    //    (i.e., we can skip `buffer_wbinvl1_vol`)
    // 3. We don't have to explicitly write to the l2 cache because
    //    `s_waitcnt vmcnt(0)` already does this as-per the MI300/CDNA3 ISA
    //    docs: "Decremented for reads when the data has been written back to
    //    the VGPRs, and for writes when the data has been written to the L2
    //    cache. Ordering: Memory reads and writes return in the order they were
    //    issued, including mixing reads and writes"
    // 4. We set GLC=1, to return the old value. Atomics in GFX942 execute with
    //    either device (default) or system scope (controlled by the sc1 flag).
    //    This is distinct from the memory scope of the atomic (i.e, the memory
    //    fences which appear before/after the ops).

    if (memScope == MemSyncScope::GPU) {
      waitcntBuilder.create<>("s_waitcnt vmcnt(0)")->operator()();
    } else if (memScope == MemSyncScope::CTA) {
      // TODO: Within a CTA we can possibly relax this?
      waitcntBuilder.create<>("s_waitcnt vmcnt(0)")->operator()();
    }

    // Check if the op has users, if it does we set GLC=1, otherwise GLC=0
    auto opUsers = op.getResult().getUsers();
    auto hasUsers = std::distance(opUsers.begin(), opUsers.end()) > 0;

    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      Type vecTy = LLVM::getFixedVectorType(valueElemTy, vec);
      Value pred = mask ? b.and_(maskElems[vecStart], rDataMask) : rDataMask;
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

      // To sync vector memory ops between CUs within an agent, we need an
      // s_waitcnt skip doing this on the last iteration of the loop
      // In the relaxed memory ordering, we don't need this barrier
      if (vecStart < numElems - vec && (emitReleaseFence || emitAcquireFence)) {
        Value inst =
            waitcntBuilder.launch(rewriter, lastRMWOp->getLoc(), void_ty(ctx));
        lastRMWOp = inst.getDefiningOp();
      }
      for (size_t ii = 0; ii < vec; ++ii) {
        Value vecIdx = createIndexAttrConstant(
            rewriter, loc, getTypeConverter()->getIndexType(), ii);
        Value loaded = b.extract_element(valueElemTy, loadVal, vecIdx);
        loadedVals.push_back(loaded);
      }
    } // end vec

    // Acquire Fence post-atomic
    if (emitAcquireFence)
      rewriter.create<LLVM::FenceOp>(lastRMWOp->getLoc(), TypeRange{}, acq,
                                     scope);

    Type llvmResultStructTy = getTypeConverter()->convertType(valueTy);
    Value resultStruct = packLLElements(loc, getTypeConverter(), loadedVals,
                                        rewriter, llvmResultStructTy);

    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct BufferStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::BufferStoreOp>,
      public LoadStoreConversionBase {
  using ConvertOpToLLVMPattern<
      triton::amdgpu::BufferStoreOp>::ConvertOpToLLVMPattern;

  BufferStoreOpConversion(LLVMTypeConverter &converter,
                          const AMD::TargetInfo &targetInfo,
                          ModuleAxisInfoAnalysis &axisAnalysisPass,
                          PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::amdgpu::BufferStoreOp>(converter,
                                                              benefit),
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
    Value rDataMask = redundantDataMask(valueTy, rewriter, loc, targetInfo);
    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      Type vecTy = LLVM::getFixedVectorType(valueElemTy, vec);
      Value pred = mask ? b.and_(maskElems[vecStart], rDataMask) : rDataMask;
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
  using ConvertOpToLLVMPattern<triton::AtomicCASOp>::ConvertOpToLLVMPattern;

  AtomicCASOpConversion(LLVMTypeConverter &converter,
                        const AMD::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::AtomicCASOp>(converter, benefit),
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
    auto scope = op.getScope();
    auto scopeStr = getAMDGPUMemScopeStr(scope);
    if (!scopeStr)
      return rewriter.notifyMatchFailure(op, "Unknown AMDGPU memory scope");

    // deal with tensor or scalar
    auto valueTy = op.getResult().getType();
    auto TensorTy = dyn_cast<RankedTensorType>(valueTy);
    Type valueElemTy =
        TensorTy ? getTypeConverter()->convertType(TensorTy.getElementType())
                 : valueTy;
    auto valueElemNBits = valueElemTy.getIntOrFloatBitWidth();
    auto elemsPerThread = getTotalElemsPerThread(op.getVal().getType());
    // vec = 1 for scalar
    auto vec = getVectorSize(op.getPtr(), axisAnalysisPass);
    // tensor
    if (TensorTy) {
      auto valTy = cast<RankedTensorType>(op.getVal().getType());
      vec = std::min<unsigned>(vec, valTy.getElementType().isF16() ? 2 : 1);
    }

    Value mask = redundantDataMask(valueTy, rewriter, loc, targetInfo);
    auto vecTy = vec_ty(valueElemTy, vec);
    SmallVector<Value> resultVals(elemsPerThread);

    // atomic ops
    for (size_t i = 0; i < elemsPerThread; i += vec) {
      Value casVal = b.undef(vecTy);
      for (int ii = 0; ii < vec; ++ii) {
        Value iiVal = createIndexAttrConstant(
            rewriter, loc, getTypeConverter()->getIndexType(), ii);
        casVal = b.insert_element(vecTy, casVal, valElements[i + ii], iiVal);
      }

      Value casPtr = ptrElements[i];
      Value casCmp = cmpElements[i];
      casVal = valElements[i];

      // use op
      if (TensorTy) { // for tensor
        auto retType = vec == 1 ? valueElemTy : vecTy;
        // TODO: USE ATOMIC CAS OP on Tensor
        auto successOrdering = atomicMemOrdering;
        auto failureOrdering = LLVM::AtomicOrdering::monotonic;
        auto cmpxchg = rewriter.create<LLVM::AtomicCmpXchgOp>(
            loc, casPtr, casCmp, casVal, successOrdering, failureOrdering,
            StringRef(scopeStr.value()));

        // Extract the new_loaded value from the pair.
        Value ret = b.extract_val(valueElemTy, cmpxchg, i);

        for (int ii = 0; ii < vec; ++ii) {
          resultVals[i + ii] =
              vec == 1 ? ret
                       : b.extract_element(valueElemTy, ret, b.i32_val(ii));
        }
      } else { // for scalar
        // Build blocks to bypass the atomic instruction for ~rmwMask.
        auto *curBlock = rewriter.getInsertionBlock();
        auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
        auto *atomicBlock = rewriter.createBlock(
            curBlock->getParent(), std::next(Region::iterator(curBlock)));

        // Fill entry block with global memory barrier and conditional branch.
        rewriter.setInsertionPointToEnd(curBlock);
        auto tid = b.tid_val();
        Value pred = b.icmp_eq(tid, b.i32_val(i));
        rewriter.create<LLVM::CondBrOp>(loc, pred, atomicBlock, endBlock);

        // Build main block with atomic_cmpxchg.
        rewriter.setInsertionPointToEnd(atomicBlock);

        auto successOrdering = LLVM::AtomicOrdering::acq_rel;
        auto failureOrdering = LLVM::AtomicOrdering::monotonic;
        auto cmpxchg = rewriter.create<LLVM::AtomicCmpXchgOp>(
            loc, casPtr, casCmp, casVal, successOrdering, failureOrdering,
            StringRef("agent"));

        if (atomicNeedsSharedMemory(op.getResult())) {
          // Extract the new_loaded value from the pair.
          Value newLoaded = b.extract_val(valueElemTy, cmpxchg, 0);
          Value atomPtr =
              getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
          b.store(newLoaded, atomPtr);
        }

        rewriter.create<LLVM::BrOp>(loc, ValueRange(), endBlock);

        // Build the last block: synced load from shared memory, exit.
        rewriter.setInsertionPointToStart(endBlock);

        if (!atomicNeedsSharedMemory(op.getResult())) {
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
      }
    }

    // replace op
    if (TensorTy) {
      Type structTy = getTypeConverter()->convertType(TensorTy);
      Value resultStruct = packLLElements(loc, getTypeConverter(), resultVals,
                                          rewriter, structTy);
      rewriter.replaceOp(op, {resultStruct});
    }
    return success();
  }
};

bool supportsGlobalAtomicF16PackedAndDpp(triton::AMD::ISAFamily isaFamily) {
  return isaFamily == triton::AMD::ISAFamily::CDNA1 ||
         isaFamily == triton::AMD::ISAFamily::CDNA2 ||
         isaFamily == triton::AMD::ISAFamily::CDNA3;
}

Value generateI32DppMove(PatternRewriter &rewriter, Value val, int dppCtrl) {
  assert(val.getType().isInteger(32));
  auto loc = val.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value old = b.i32_val(0);
  int rowMask = 0b1111;  // enable all rows
  int bankMask = 0b1111; // enable all banks
  bool boundCtrl = false;
  auto dppMovOp = rewriter.create<ROCDL::DPPUpdateOp>(
      loc, i32_ty, old, val, dppCtrl, rowMask, bankMask, boundCtrl);
  return dppMovOp.getResult();
}

Value shiftLeftI32ByDpp(PatternRewriter &rewriter, Value val) {
  return generateI32DppMove(rewriter, val, 0x101); // shift left 1 lane
}

Value shiftRightI32ByDpp(PatternRewriter &rewriter, Value val) {
  return generateI32DppMove(rewriter, val, 0x111); // shift right 1 lane
}

Value generatePopcount64(PatternRewriter &rewriter, Value val) {
  auto loc = val.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value m1 = b.i64_val(0x5555555555555555); // binary: 0101 0101..
  Value m2 = b.i64_val(0x3333333333333333); // binary: 0011 0011..
  Value m4 = b.i64_val(0x0f0f0f0f0f0f0f0f); // binary: 0000 1111..
  // binary: 0000 0001 0000 0001..
  Value h01 = b.i64_val(0x0101010101010101);
  // put count of each 2 bits into those 2 bits
  val = b.sub(val, b.and_(m1, b.lshr(val, b.i64_val(1))));
  // put count of each 4 bits into those 4 bits
  val = b.add(b.and_(val, m2), b.and_(b.lshr(val, b.i64_val(2)), m2));
  // put count of each 8 bits into those 8 bits
  val = b.and_(b.add(val, b.lshr(val, b.i64_val(4))), m4);
  // left 8 bits of x + (x<<8) + (x<<16) + (x<<24) + ...
  return b.lshr(b.mul(val, h01), b.i64_val(56));
}

Value genReadFirstLane(PatternRewriter &rewriter, Value v) {
  auto loc = v.getLoc();
  std::string intrinsic = "llvm.amdgcn.readfirstlane";
  return LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsic, i32_ty, v)
      ->getResult(0);
}

Value genPermute(PatternRewriter &rewriter, Value v, Value dst) {
  auto loc = v.getLoc();
  std::string intrinsic = "llvm.amdgcn.ds.permute";
  return LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsic, i32_ty,
                                         ValueRange{dst, v})
      ->getResult(0);
}

Value genBPermute(PatternRewriter &rewriter, Value v, Value dst) {
  auto loc = v.getLoc();
  std::string intrinsic = "llvm.amdgcn.ds.bpermute";
  return LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsic, i32_ty,
                                         ValueRange{dst, v})
      ->getResult(0);
}

template <typename Generator, typename... Values>
Value genI32TiledOp(PatternRewriter &rewriter, Generator genCall,
                    Value argToSplit, Values... args) {
  auto loc = argToSplit.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type ty = argToSplit.getType();
  size_t tySize = ty.getIntOrFloatBitWidth();
  size_t i32Size = i32_ty.getIntOrFloatBitWidth();
  size_t count = tySize / i32Size;
  assert(tySize % i32Size == 0 && count > 0 &&
         "Unalligned types are not supported yet.");
  Type i32VecValTy = vec_ty(i32_ty, count);
  Value vec = b.undef(i32VecValTy);
  Value valCasted = b.bitcast(argToSplit, i32VecValTy);
  for (int i = 0; i < count; i++) {
    Value subVal = b.extract_element(i32_ty, valCasted, b.i32_val(i));
    Value result = genCall(rewriter, subVal, args...);
    vec = b.insert_element(i32VecValTy, vec, result, b.i32_val(i));
  }
  return b.bitcast(vec, ty);
}

struct AtomicRMWOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicRMWOp>,
      public LoadStoreConversionBase {
  using ConvertOpToLLVMPattern<triton::AtomicRMWOp>::ConvertOpToLLVMPattern;

  AtomicRMWOpConversion(LLVMTypeConverter &converter,
                        const AMD::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::AtomicRMWOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  /// Try to match the mlir::triton::RMWOp to LLVM::AtomicBinOp.
  static std::optional<LLVM::AtomicBinOp> matchAtomicOp(RMWOp atomicOp) {
    switch (atomicOp) {
    case RMWOp::AND:
      return LLVM::AtomicBinOp::_and;
    case RMWOp::OR:
      return LLVM::AtomicBinOp::_or;
    case RMWOp::XOR:
      return LLVM::AtomicBinOp::_xor;
    case RMWOp::ADD:
      return LLVM::AtomicBinOp::add;
    case RMWOp::FADD:
      return LLVM::AtomicBinOp::fadd;
    case RMWOp::MAX:
      return LLVM::AtomicBinOp::max;
    case RMWOp::MIN:
      return LLVM::AtomicBinOp::min;
    case RMWOp::UMAX:
      return LLVM::AtomicBinOp::umax;
    case RMWOp::UMIN:
      return LLVM::AtomicBinOp::umin;
    case RMWOp::XCHG:
      return LLVM::AtomicBinOp::xchg;
    default:
      return std::nullopt;
    }
    llvm_unreachable("Invalid RMWOp");
  }

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    MLIRContext *ctx = rewriter.getContext();

    auto atomicRmwAttr = op.getAtomicRmwOp();
    Value ptr = op.getPtr();
    Value val = op.getVal();

    Value llPtr = adaptor.getPtr();
    Value llVal = adaptor.getVal();
    Value llMask = adaptor.getMask();

    auto valElements = unpackLLElements(loc, llVal, rewriter);
    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    SmallVector<Value> maskElements;
    if (llMask)
      maskElements = unpackLLElements(loc, llMask, rewriter);

    Value opResult = op.getResult();
    auto tensorTy = dyn_cast<RankedTensorType>(opResult.getType());
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : opResult.getType();
    const size_t valueElemNbits = valueElemTy.getIntOrFloatBitWidth();
    auto elemsPerThread = getTotalElemsPerThread(val.getType());
    // vec = 1, numElements = 1 for scalar
    auto vec = getVectorSize(ptr, axisAnalysisPass);
    int numElems = 1;
    Type packF16Ty = vec_ty(valueElemTy, 2);

    // CDNA3 arch allows to accelerate its atomics with LDS reduction algorithm,
    // which is only applicable for atomics with no return. Otherwise we have to
    // deal with an additional overhead.
    bool enableIntraWaveReduce =
        targetInfo.getISAFamily() == triton::AMD::ISAFamily::CDNA3 &&
        tensorTy && opResult.use_empty();

    // TODO: support data types less than 32 bits
    enableIntraWaveReduce &= valueElemNbits >= 32;

    // In the case of unpaired f16 elements utilize dpp instructions to
    // accelerate atomics. Here is an algorithm of lowering
    // tt::atomicRmwOp(%ptr, %val, %mask):
    // 0. Group thread by pairs. Master thread is (tid % 2 == 0);
    // 1. All the threads send %val to (tid - 1) thread via dppUpdateOp shl, so
    //    all the masters recieve value from secondary threads;
    // 2. Take into account parity in the %mask value, build control flow
    //    structures according to it;
    // 3. Generate llvm::atomicRmwOp in the threads enabled by %mask value;
    // 4. All the threads send result of generated operation to (tid + 1) thread
    //    via dppUpdateOp shl, so all secondary thread also recieve their
    //    result.
    //
    // This approach enables us to use half the active threads committing atomic
    // requests to avoid generating of code providing unified access to f16
    // element and reduce contantion.
    bool useDppForPackedF16 = false;
    // tensor
    if (tensorTy) {
      auto valTy = cast<RankedTensorType>(val.getType());
      bool isF16Ty = valueElemTy.isF16() || valueElemTy.isBF16();
      unsigned availableVecSize = isF16Ty ? 2 : 1;
      vec = std::min<unsigned>(vec, availableVecSize);
      // Force F16 packing  in the case it's not comming in as packed, but the
      // ISA can support packed atomic instructions.
      useDppForPackedF16 =
          supportsGlobalAtomicF16PackedAndDpp(targetInfo.getISAFamily()) &&
          vec == 1 && isF16Ty && atomicRmwAttr == RMWOp::FADD;
      // mask
      numElems = tensorTy.getNumElements();
    }
    Value mask = b.int_val(1, 1);
    auto tid = b.tid_val();
    mask = b.and_(mask, b.icmp_slt(b.mul(tid, b.i32_val(elemsPerThread)),
                                   b.i32_val(numElems)));
    if (useDppForPackedF16)
      mask = b.and_(mask, b.icmp_eq(b.urem(tid, b.i32_val(2)), b.i32_val(0)));

    auto memOrdering = op.getSem();
    auto scope = op.getScope();
    auto atomicMemOrdering = getMemoryOrdering(memOrdering);

    auto scopeStr = getAMDGPUMemScopeStr(scope);
    if (!scopeStr)
      return rewriter.notifyMatchFailure(op, "Unknown AMDGPU memory scope");

    auto vecTy = vec_ty(valueElemTy, vec);
    auto retType = vec == 1 ? valueElemTy : vecTy;
    retType = useDppForPackedF16 ? packF16Ty : retType;
    SmallVector<Value> resultVals(elemsPerThread);
    for (size_t i = 0; i < elemsPerThread; i += vec) {
      Value rmwPtr = ptrElements[i];
      // TODO: in case llMask is zero we can create only one branch for all
      // elemsPerThread.
      Value rmwMask = llMask ? b.and_(mask, maskElements[i]) : mask;

      Value operand;
      if (useDppForPackedF16) {
        // Move %val to left neighbour to proceed packed atomic further.
        Value packedVal = b.null(packF16Ty);
        packedVal = b.insert_element(packF16Ty, packedVal, valElements[i],
                                     b.i32_val(0));
        // Pack to i32 type to simplify transaction
        packedVal = b.bitcast(packedVal, i32_ty);
        Value dppMoveRes = shiftLeftI32ByDpp(rewriter, packedVal);
        // Unpack results back
        Value unpackedDppRes = b.bitcast(dppMoveRes, packF16Ty);
        operand = b.undef(packF16Ty);
        operand =
            b.insert_element(packF16Ty, operand, valElements[i], b.i32_val(0));
        operand = b.insert_element(
            packF16Ty, operand,
            b.extract_element(valueElemTy, unpackedDppRes, b.i32_val(0)),
            b.i32_val(1));
      } else if (vec == 1) {
        operand = valElements[i];
      } else {
        operand = b.undef(vecTy);
        for (size_t ii = 0; ii < vec; ++ii)
          operand = b.insert_element(vecTy, operand, valElements[i + ii],
                                     b.i32_val(ii));
      }

      Value undefVal = b.undef(retType);
      // Build blocks to bypass the atomic instruction for ~rmwMask.
      auto *curBlock = rewriter.getInsertionBlock();
      auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
      auto *atomicBlock = rewriter.createBlock(
          curBlock->getParent(), std::next(Region::iterator(curBlock)));
      endBlock->addArgument({retType}, {loc});

      rewriter.setInsertionPointToEnd(curBlock);
      rewriter.create<LLVM::CondBrOp>(loc, rmwMask, atomicBlock, endBlock,
                                      undefVal);

      rewriter.setInsertionPointToEnd(atomicBlock);
      auto maybeKind = matchAtomicOp(atomicRmwAttr);
      Value atom;
      if (enableIntraWaveReduce) {
        atom = atomicIntraWaveReduce(rewriter, rmwPtr, operand, *maybeKind,
                                     atomicMemOrdering, scopeStr.value());
      } else {
        atom = rewriter
                   .create<LLVM::AtomicRMWOp>(loc, *maybeKind, rmwPtr, operand,
                                              atomicMemOrdering,
                                              StringRef(scopeStr.value()))
                   .getResult();
      }

      if (!tensorTy) {
        if (atomicNeedsSharedMemory(op.getResult())) {
          Value atomPtr =
              getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
          b.store(atom, atomPtr);
        }
      }
      rewriter.create<LLVM::BrOp>(loc, atom, endBlock);

      rewriter.setInsertionPointToStart(endBlock);
      Value retVal = endBlock->getArgument(0);
      if (tensorTy) {
        if (useDppForPackedF16) {
          // Return packed to i32 result after atomic operation back from master
          // lane.
          auto packedRet = b.bitcast(retVal, i32_ty);
          Value dppMovRes = shiftRightI32ByDpp(rewriter, packedRet);
          // Unpack results back
          Value unpackedDppRes = b.bitcast(dppMovRes, packF16Ty);
          retVal = b.insert_element(
              packF16Ty, retVal,
              b.extract_element(valueElemTy, unpackedDppRes, b.i32_val(1)),
              b.i32_val(1));
          resultVals[i] =
              b.extract_element(valueElemTy, retVal, b.urem(tid, b.i32_val(2)));
        } else {
          for (int ii = 0; ii < vec; ++ii) {
            resultVals[i + ii] =
                vec == 1
                    ? retVal
                    : b.extract_element(valueElemTy, retVal, b.i32_val(ii));
          }
        }
      } else {
        if (!atomicNeedsSharedMemory(op.getResult())) {
          rewriter.eraseOp(op);
          return success();
        }
        Value atomPtr =
            getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
        b.barrier();
        Value ret = b.load(valueElemTy, atomPtr);
        rewriter.replaceOp(op, {ret});
      }
    }
    if (tensorTy) {
      Type structTy = getTypeConverter()->convertType(tensorTy);
      Value resultStruct = packLLElements(loc, getTypeConverter(), resultVals,
                                          rewriter, structTy);
      rewriter.replaceOp(op, {resultStruct});
    }
    return success();
  }

private:
  Value atomicIntraWaveReduce(PatternRewriter &rewriter, Value rmwPtr,
                              Value operand, LLVM::AtomicBinOp opKind,
                              LLVM::AtomicOrdering memOrdering,
                              StringRef scope) const {
    // This approach minimizes intra-warp thread contention when accessing
    // global memory pointers. It is particularly advantageous for certain ISA
    // families, such as CDNA3. The algorithm follows these steps:
    // 1. Analyze thread groups and their relative positions:
    // 1.1. Consider groups of threads sharing identical pointers using
    //      `readfirstlane` and ballot `intrinsics`.
    // 1.2. Compute parameters to form contiguous groups and further optimize
    //      them.
    // 1.3. Disable threads that have already been processed.
    // 1.4. If thread was not considered, jump to `1.1.`.
    // 2. Form contiguous groups:
    //    Use `permute` instructions to organize threads within the wavefront
    //    into continuous groups.
    // 4. Reduce Groups to Leader threads:
    //    Apply `bpermute` and operation-specific arithmetic based on the opKind
    //    to consolidate group data into leader threads.
    // 5. Perform global atomic operations by leader threads.
    auto loc = operand.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Type operandElemType = operand.getType();
    Type origPtrType = rmwPtr.getType();

    rmwPtr = b.ptrtoint(i64_ty, rmwPtr);

    auto *curBlock = rewriter.getInsertionBlock();
    auto *afterLoopBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
    afterLoopBlock->addArgument(i32_ty, loc);    // idx
    afterLoopBlock->addArgument(i32_ty, loc);    // cnt
    afterLoopBlock->addArgument(int_ty(1), loc); // isLeader
    auto *loopBody = rewriter.createBlock(
        curBlock->getParent(), std::next(Region::iterator(curBlock)));
    loopBody->addArgument(i32_ty, loc); // base
    rewriter.setInsertionPointToEnd(curBlock);
    rewriter.create<LLVM::BrOp>(loc, b.i32_val(0), loopBody);

    // Greed search of same addr within wavefront. Also collect auxiliary
    // information about relative position:
    // - idx in a group + base laneId. This param is required to form continuous
    //   groups further;
    // - cnt of remaining threads in a group after current thread;
    // - leadership status of the current thread.
    rewriter.setInsertionPointToEnd(loopBody);
    // `readfirstlane` considers only enabled threads
    Value chosen = genI32TiledOp(rewriter, genReadFirstLane, rmwPtr);
    // this flag is required to disable thread if we have already checked its
    // pointer
    Value done = b.icmp_eq(chosen, rmwPtr);
    Value mask = targetInfo.ballot(rewriter, loc, i64_ty, done);
    Value start = loopBody->getArgument(0);
    Value cnt = b.trunc(i32_ty, generatePopcount64(rewriter, mask));
    Value mbcntLoRes = rewriter
                           .create<ROCDL::MbcntLoOp>(
                               loc, i32_ty, b.trunc(i32_ty, mask), b.i32_val(0))
                           ->getResult(0);
    Value idx = rewriter.create<ROCDL::MbcntHiOp>(
        loc, i32_ty, b.trunc(i32_ty, b.lshr(mask, b.i64_val(32))), mbcntLoRes);
    Value base = b.add(start, cnt);
    Value leader = b.icmp_eq(idx, b.i32_val(0));
    cnt = b.sub(cnt, idx);
    idx = b.add(idx, start);
    rewriter.create<LLVM::CondBrOp>(loc, done, afterLoopBlock,
                                    ValueRange({idx, cnt, leader}), loopBody,
                                    ValueRange({base}));

    rewriter.setInsertionPointToEnd(afterLoopBlock);

    Value idxRes = afterLoopBlock->getArgument(0);
    Value cntRes = afterLoopBlock->getArgument(1);
    Value leaderRes = afterLoopBlock->getArgument(2);
    Value idxScaledForPermute = b.mul(idxRes, b.i32_val(4));

    // Make groups continuous
    rmwPtr = genI32TiledOp(rewriter, genPermute, rmwPtr, idxScaledForPermute);
    operand = genI32TiledOp(rewriter, genPermute, operand, idxScaledForPermute);
    // Actualize auxiliary info as well
    Value packedRoleInfo = genI32TiledOp(
        rewriter, genPermute,
        b.or_(b.zext(i32_ty, leaderRes),
              b.or_(idxScaledForPermute, b.shl(cntRes, b.i32_val(8)))),
        idxScaledForPermute);
    idxScaledForPermute = packedRoleInfo;
    cntRes = b.and_(b.lshr(packedRoleInfo, b.i32_val(8)), b.i32_val(0xff));
    leaderRes = b.icmp_ne(b.and_(packedRoleInfo, b.i32_val(1)), b.i32_val(0));

    auto *afterRedBlock =
        afterLoopBlock->splitBlock(rewriter.getInsertionPoint());
    afterRedBlock->addArgument(operandElemType, loc);
    auto *partialReductionBlock =
        rewriter.createBlock(afterLoopBlock->getParent(),
                             std::next(Region::iterator(afterLoopBlock)));
    rewriter.setInsertionPointToEnd(afterLoopBlock);
    Value reductionCond =
        b.icmp_ne(targetInfo.ballot(rewriter, loc, i64_ty,
                                    b.icmp_ne(cntRes, b.i32_val(1))),
                  b.i64_val(0));
    rewriter.create<LLVM::CondBrOp>(loc, reductionCond, partialReductionBlock,
                                    afterRedBlock, operand);
    rewriter.setInsertionPointToEnd(partialReductionBlock);

    auto performOpIfCond = [&](Value res, Value v, Value cond) -> Value {
      Type ty = v.getType();
      assert(ty == res.getType());
      Value notCond = b.icmp_eq(cond, b.false_val());
      switch (opKind) {
      case LLVM::AtomicBinOp::_and:
        // res &= cond ? v : 1111..
        return b.and_(res,
                      b.or_(v, b.sub(b.int_val(ty.getIntOrFloatBitWidth(), 0),
                                     b.zext(ty, notCond))));
      case LLVM::AtomicBinOp::_or:
        // res |= cond ? v : 0
        return b.or_(res, b.mul(v, b.zext(ty, cond)));
      case LLVM::AtomicBinOp::_xor:
        // res ^= cond ? v : 0
        return b.xor_(res, b.mul(v, b.zext(ty, cond)));
      case LLVM::AtomicBinOp::add:
        // res += cond ? v : 0
        return b.add(res, b.mul(v, b.zext(ty, cond)));
      case LLVM::AtomicBinOp::fadd:
        // res += cond ? v : 0
        return b.fadd(
            res, b.fmul(v, b.inttofloat(
                               ty, b.zext(int_ty(ty.getIntOrFloatBitWidth()),
                                          cond))));
      case LLVM::AtomicBinOp::max:
      case LLVM::AtomicBinOp::umax:
        // res = cond ? umax(v, res) : res
        return b.or_(b.mul(res, b.zext(ty, notCond)),
                     b.mul(b.umax(v, res), b.zext(ty, cond)));
      case LLVM::AtomicBinOp::min:
      case LLVM::AtomicBinOp::umin:
        // res = cond ? umin(v, res) : res
        return b.or_(b.mul(res, b.zext(ty, notCond)),
                     b.mul(b.umin(v, res), b.zext(ty, cond)));
      case LLVM::AtomicBinOp::xchg:
        // res = cond ? v : res
        return b.or_(b.mul(res, b.zext(ty, notCond)),
                     b.mul(v, b.zext(ty, cond)));
      default:
        llvm_unreachable("Unsupported atomic binary operation.");
      }
    };
    Value acc = operand;
    // Reduce to leader thread
    for (int i = 32; i != 0; i /= 2) {
      Value tmp = genI32TiledOp(rewriter, genBPermute, acc,
                                b.add(idxScaledForPermute, b.i32_val(i * 4)));
      acc = performOpIfCond(acc, tmp, b.icmp_ult(b.i32_val(i), cntRes));
    }

    rewriter.create<LLVM::BrOp>(loc, acc, afterRedBlock);
    rewriter.setInsertionPointToEnd(afterRedBlock);

    auto *endBlock = afterRedBlock->splitBlock(rewriter.getInsertionPoint());
    endBlock->addArgument(operandElemType, loc);
    auto *leaderBlock = rewriter.createBlock(
        afterRedBlock->getParent(), std::next(Region::iterator(afterRedBlock)));
    rewriter.setInsertionPointToEnd(afterRedBlock);
    Value leaderCond = leaderRes;
    Value defaultRes = b.undef(operandElemType);
    rewriter.create<LLVM::CondBrOp>(loc, leaderCond, leaderBlock, endBlock,
                                    defaultRes);
    rewriter.setInsertionPointToEnd(leaderBlock);
    // Utilize global atomic only by leader threads
    rmwPtr = b.inttoptr(origPtrType, rmwPtr);
    Value atom = rewriter
                     .create<LLVM::AtomicRMWOp>(loc, opKind, rmwPtr,
                                                afterRedBlock->getArgument(0),
                                                memOrdering, scope)
                     .getResult();
    rewriter.create<LLVM::BrOp>(loc, atom, endBlock);
    rewriter.setInsertionPointToStart(endBlock);

    return endBlock->getArgument(0);
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

    using mlir::triton::AMD::ISAFamily;

    switch (targetInfo.getISAFamily()) {
    case ISAFamily::CDNA1:
    case ISAFamily::CDNA2:
    case ISAFamily::CDNA3:
      break;
    default:
      return rewriter.notifyMatchFailure(
          op, "Only supported on target architecture");
    }

    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    // global.load.lds uses vmcnt to synchronize
    // The rocdl op stores all available counters in a single int32 value (v).
    // The vmcnt (6 bits) is split into a lower 3:0 and higher 5:4 parts.
    // The lower part is stored in bits 3:0 of v and the higher part in bits
    // 15:14. We have to set all other bits in v to 1 to signal we are not
    // interested in those.

    int vmCnt = op.getNum();
    if (vmCnt >= 64) {
      return emitError(loc, "AsyncWait does not support values >= 64");
    }

    // Extract low and high bits and combine while setting all other bits to 1
    unsigned lowBits = vmCnt & 0xF;
    unsigned highBits = vmCnt >> 4 << 14;
    unsigned otherCnts = ~0xC00F; // C00F has bits 15:14 and 3:0 set
    unsigned waitValue = lowBits | highBits | otherCnts;

    rewriter.create<ROCDL::WaitcntOp>(loc, waitValue);

    // Drop the result AsyncToken
    rewriter.replaceOp(op, b.i32_val(0));
    return success();
  }

private:
  const AMD::TargetInfo &targetInfo;
};

struct AsyncCommitGroupOpConversion
    : public ConvertOpToLLVMPattern<AsyncCommitGroupOp> {
  using ConvertOpToLLVMPattern<AsyncCommitGroupOp>::ConvertOpToLLVMPattern;

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
                                       int numWarps,
                                       ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                       PatternBenefit benefit) {
  patterns
      .add<AtomicCASOpConversion, AtomicRMWOpConversion, LoadOpConversion,
           StoreOpConversion, BufferLoadOpConversion, BufferStoreOpConversion,
           BufferAtomicRMWOpConversion, AsyncCopyGlobalToLocalOpConversion>(
          typeConverter, targetInfo, axisInfoAnalysis, benefit);
  patterns.add<AsyncWaitOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<AsyncCommitGroupOpConversion>(typeConverter, benefit);
}
} // namespace mlir::triton::AMD
