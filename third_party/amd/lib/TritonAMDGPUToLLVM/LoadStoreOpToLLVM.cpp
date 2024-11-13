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
using ::mlir::LLVM::AMD::llLoad;
using ::mlir::LLVM::AMD::llStore;
using ::mlir::triton::gpu::getTotalElemsPerThread;

namespace {
// Return the mask for the unique data accessed by given tensor type.
// Used to mask out the redundant data accessed by threads.
Value redundantDataMask(Type valueTy, ConversionPatternRewriter &rewriter,
                        Location loc, const AMD::TargetInfo &targetInfo) {
  auto tensorTy = dyn_cast<RankedTensorType>(valueTy);
  Value mask = int_val(1, 1);
  auto tid = tid_val();
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
    auto shapePerCTATile = triton::gpu::getShapePerCTATile(layout, shape);
    Value warpSize = i32_val(triton::gpu::getWarpSize(layout));
    Value laneId = urem(tid, warpSize);
    Value warpId = udiv(tid, warpSize);
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
          add(mul(multiDimWarpId[dim], i32_val(threadsPerWarp[dim])),
              multiDimThreadId[dim]);
      mask = and_(mask, icmp_slt(mul(threadDim, i32_val(sizePerThread[dim])),
                                 i32_val(shape[dim])));
    }
    // Do not write duplicated data when multicast is enabled
    if (triton::gpu::getNumCTAs(layout) > 1) {
      auto _0 = i32_val(0);
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
        Value repId = udiv(multiDimClusterCTAId[dim], i32_val(splitNum));
        // Consider the example where CTAsPerCGA = [4] and CTASplitNum = [2]:
        //     CTA0 and CTA2 holds data of block0,
        //     CTA1 and CTA3 holds data of block1.
        // Only CTA0 and CTA1 are expected to write while CTA2 and CTA3 should
        // be masked. We add the following mask:
        //     multiDimClusterCTAId[dim] / splitNum == 0
        // Actually in all existing cases of multicast, splitNum is always 1.
        // The mask is equivalent to:
        //     multiDimClusterCTAId[dim] == 0
        mask = and_(mask, icmp_eq(repId, _0));
      }
    }
  } else {
    // If the tensor is not ranked, then it is a scalar and only thread 0 of
    // CTA0 can write
    mask = and_(mask, icmp_eq(clusterCTAId, i32_val(0)));
    mask = and_(mask, icmp_eq(tid, i32_val(0)));
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
    int64_t vec = vecTy.getNumElements();
    // If we need to mask the loaded value with other elements
    Value v = undef(vecTy);
    for (size_t s = 0; s < vec; ++s) {
      Value otherElem = elems[start + s];
      Value indexVal =
          LLVM::createIndexConstant(rewriter, loc, typeConverter, s);
      v = insert_element(vecTy, v, otherElem, indexVal);
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

  // Get contiguity for a tensor pointer `ptr`
  unsigned getContiguity(Value ptr) const {
    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return 1;
    return axisAnalysisPass.getPtrContiguity(ptr);
  }

  // Get contiguity for a scalar pointer `ptr` and a tensor `offset`
  unsigned getContiguity(Value ptr, Value offset) const {
    // Get contiguity from the offset
    Type type = getPointerTypeWithShape(ptr, offset);
    RankedTensorType tensorTy = cast<RankedTensorType>(type);
    auto layout = tensorTy.getEncoding();
    auto order = triton::gpu::getOrder(layout);
    auto uniqueContigPerThread =
        triton::gpu::getUniqueContigPerThread(layout, tensorTy.getShape());
    assert(order[0] < uniqueContigPerThread.size() &&
           "Unexpected uniqueContigPerThread size");
    unsigned contiguity = uniqueContigPerThread[order[0]];

    // Get alignment from the pointer. Since this is a scalar pointer
    // we should not take the pointer contiguity to consider alignment
    auto *axisInfo = axisAnalysisPass.getAxisInfo(ptr);
    auto maxMultipleBytes = axisInfo->getDivisibility(0);
    auto elemNumBits = triton::getPointeeBitWidth(tensorTy);
    auto elemNumBytes = std::max<unsigned>(elemNumBits / 8, 1);
    auto align = std::max<int64_t>(maxMultipleBytes / elemNumBytes, 1);

    // Final contiguity is a min of the offset contiguity and pointer alignment
    contiguity = std::min<int64_t>(align, contiguity);
    return contiguity;
  }

  // Determine the vector size of a tensor of pointers
  unsigned getVectorSize(Value ptr) const {
    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return 1;
    auto contiguity = getContiguity(ptr);
    auto pointeeBitWidth = triton::getPointeeBitWidth(tensorTy);
    return std::min<unsigned>(128 / pointeeBitWidth, contiguity);
  }

  // Given a scalar pointer and a tensor of offsets, determine the vector size
  unsigned getVectorSize(Value ptr, Value offset) const {
    auto contiguity = getContiguity(ptr, offset);
    auto pointeeBitWidth = triton::getPointeeBitWidth(ptr.getType());
    return std::min<unsigned>(128 / pointeeBitWidth, contiguity);
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
    unsigned vec = getVectorSize(ptr);
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

      Value pred = mask ? maskElems[vecStart] : int_val(1, 1);
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
            rewriter, loc, this->getTypeConverter()->getIndexType(), ii);
        Value loaded = extract_element(valueElemTy, loadVal, vecIdx);
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
    LLVM::AMD::BufferEmitter bufferEmitter(rewriter, loc, targetInfo);

    // original values
    Value ptr = op.getPtr();
    Value offset = op.getOffsets();
    Value mask = op.getMask();
    Value other = op.getOther();

    // Converted values
    Value llPtr = adaptor.getPtr();
    Value llOffset = adaptor.getOffsets();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();

    // Determine the vectorization size
    Type valueTy = op.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));
    Type ptrType = getPointerTypeWithShape(ptr, offset);
    unsigned numElems = getTotalElemsPerThread(ptrType);
    unsigned vec = getVectorSize(ptr, offset);

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
    Value rsrcDesc = bufferEmitter.createResourceDescriptor(llPtr);
    SmallVector<Value> loadedVals;
    Type vecTy = LLVM::getFixedVectorType(valueElemTy, vec);
    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      Value pred = mask ? maskElems[vecStart] : int_val(1, 1);
      Value falseVal = createZeroVector(rewriter, loc, cast<VectorType>(vecTy));
      if (otherElems.size() != 0)
        falseVal = packElementRangeIntoVector(
            rewriter, this->getTypeConverter(), loc, cast<VectorType>(vecTy),
            otherElems, vecStart);
      Value loadVal = bufferEmitter.emitLoad(
          vecTy, rsrcDesc, offsetElems[vecStart], pred, falseVal);
      for (size_t ii = 0; ii < vec; ++ii) {
        Value vecIdx = createIndexAttrConstant(
            rewriter, loc, this->getTypeConverter()->getIndexType(), ii);
        Value loaded = extract_element(valueElemTy, loadVal, vecIdx);
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
    MLIRContext *ctx = rewriter.getContext();

    auto valueTy = value.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));

    // Determine the vectorization size
    unsigned vec = getVectorSize(ptr);
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
      Value pred = mask ? and_(maskElems[vecStart], rDataMask) : rDataMask;
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
    LLVM::AMD::BufferEmitter bufferEmitter(rewriter, loc, targetInfo);

    // original values
    Value ptr = op.getPtr();
    Value offset = op.getOffsets();
    Value mask = op.getMask();
    Value data = op.getValue();

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
    unsigned vec = getVectorSize(ptr, offset);

    // Get the offsets and value
    SmallVector<Value> offsetElems = unpackLLElements(loc, llOffset, rewriter);
    SmallVector<Value> valueElems = unpackLLElements(loc, llData, rewriter);

    // Get the mask
    SmallVector<Value> maskElems =
        getMaskElemsAndUpdateVeclen(rewriter, loc, llMask, mask, vec);

    Value rsrcDesc = bufferEmitter.createResourceDescriptor(llPtr);
    Value rDataMask = redundantDataMask(valueTy, rewriter, loc, targetInfo);
    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      Type vecTy = LLVM::getFixedVectorType(valueElemTy, vec);
      Value pred = mask ? and_(maskElems[vecStart], rDataMask) : rDataMask;
      // Create the store val
      Value storeVal = packElementRangeIntoVector(
          rewriter, this->getTypeConverter(), loc, cast<VectorType>(vecTy),
          valueElems, vecStart);
      bufferEmitter.emitStore(rsrcDesc, offsetElems[vecStart], storeVal, pred);
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

    // deal with tensor or scalar
    auto valueTy = op.getResult().getType();
    auto TensorTy = dyn_cast<RankedTensorType>(valueTy);
    Type valueElemTy =
        TensorTy ? getTypeConverter()->convertType(TensorTy.getElementType())
                 : valueTy;
    auto valueElemNBits = valueElemTy.getIntOrFloatBitWidth();
    auto elemsPerThread = getTotalElemsPerThread(op.getVal().getType());
    // vec = 1 for scalar
    auto vec = getVectorSize(op.getPtr());
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
      Value casVal = undef(vecTy);
      for (int ii = 0; ii < vec; ++ii) {
        Value iiVal = createIndexAttrConstant(
            rewriter, loc, getTypeConverter()->getIndexType(), ii);
        casVal = insert_element(vecTy, casVal, valElements[i + ii], iiVal);
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
            StringRef("agent"));

        // Extract the new_loaded value from the pair.
        Value ret = extract_val(valueElemTy, cmpxchg, i);

        for (int ii = 0; ii < vec; ++ii) {
          resultVals[i + ii] =
              vec == 1 ? ret : extract_element(valueElemTy, ret, i32_val(ii));
        }
      } else { // for scalar
        // Build blocks to bypass the atomic instruction for ~rmwMask.
        auto *curBlock = rewriter.getInsertionBlock();
        auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
        auto *atomicBlock = rewriter.createBlock(
            curBlock->getParent(), std::next(Region::iterator(curBlock)));

        // Fill entry block with global memory barrier and conditional branch.
        rewriter.setInsertionPointToEnd(curBlock);
        auto tid = tid_val();
        Value pred = icmp_eq(tid, i32_val(i));
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
          Value newLoaded = extract_val(valueElemTy, cmpxchg, 0);
          Value atomPtr =
              getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
          store(newLoaded, atomPtr);
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
        barrier();
        Value atomPtr =
            getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
        Value ret = load(valueElemTy, atomPtr);
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
  Value old = i32_val(0);
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
    auto vec = getVectorSize(ptr);
    int numElems = 1;
    Type packF16Ty = vec_ty(valueElemTy, 2);

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
    Value mask = int_val(1, 1);
    auto tid = tid_val();
    mask = and_(mask,
                icmp_slt(mul(tid, i32_val(elemsPerThread)), i32_val(numElems)));
    if (useDppForPackedF16)
      mask = and_(mask, icmp_eq(urem(tid, i32_val(2)), i32_val(0)));

    auto memOrdering = op.getSem();
    auto atomicMemOrdering = getMemoryOrdering(memOrdering);

    auto vecTy = vec_ty(valueElemTy, vec);
    auto retType = vec == 1 ? valueElemTy : vecTy;
    retType = useDppForPackedF16 ? packF16Ty : retType;
    SmallVector<Value> resultVals(elemsPerThread);
    for (size_t i = 0; i < elemsPerThread; i += vec) {
      Value rmwPtr = ptrElements[i];
      // TODO: in case llMask is zero we can create only one branch for all
      // elemsPerThread.
      Value rmwMask = llMask ? and_(mask, maskElements[i]) : mask;

      Value operand;
      if (useDppForPackedF16) {
        // Move %val to left neighbour to proceed packed atomic further.
        Value packedVal = null(packF16Ty);
        packedVal =
            insert_element(packF16Ty, packedVal, valElements[i], i32_val(0));
        // Pack to i32 type to simplify transaction
        packedVal = bitcast(packedVal, i32_ty);
        Value dppMoveRes = shiftLeftI32ByDpp(rewriter, packedVal);
        // Unpack results back
        Value unpackedDppRes = bitcast(dppMoveRes, packF16Ty);
        operand = undef(packF16Ty);
        operand =
            insert_element(packF16Ty, operand, valElements[i], i32_val(0));
        operand = insert_element(
            packF16Ty, operand,
            extract_element(valueElemTy, unpackedDppRes, i32_val(0)),
            i32_val(1));
      } else if (vec == 1) {
        operand = valElements[i];
      } else {
        operand = undef(vecTy);
        for (size_t ii = 0; ii < vec; ++ii)
          operand =
              insert_element(vecTy, operand, valElements[i + ii], i32_val(ii));
      }

      Value undefVal = undef(retType);
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
      // TODO: use rocdl.raw.buffer.atomic from ROCDL dialect to use efficient
      // atomics for MI-* series of AMD GPU.
      Value atom =
          rewriter
              .create<LLVM::AtomicRMWOp>(loc, *maybeKind, rmwPtr, operand,
                                         atomicMemOrdering, StringRef("agent"))
              .getResult();
      if (!tensorTy) {
        if (atomicNeedsSharedMemory(op.getResult())) {
          Value atomPtr =
              getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
          store(atom, atomPtr);
        }
      }
      rewriter.create<LLVM::BrOp>(loc, atom, endBlock);

      rewriter.setInsertionPointToStart(endBlock);
      Value retVal = endBlock->getArgument(0);
      if (tensorTy) {
        if (useDppForPackedF16) {
          // Return packed to i32 result after atomic operation back from master
          // lane.
          auto packedRet = bitcast(retVal, i32_ty);
          Value dppMovRes = shiftRightI32ByDpp(rewriter, packedRet);
          // Unpack results back
          Value unpackedDppRes = bitcast(dppMovRes, packF16Ty);
          retVal = insert_element(
              packF16Ty, retVal,
              extract_element(valueElemTy, unpackedDppRes, i32_val(1)),
              i32_val(1));
          resultVals[i] =
              extract_element(valueElemTy, retVal, urem(tid, i32_val(2)));
        } else {
          for (int ii = 0; ii < vec; ++ii) {
            resultVals[i + ii] =
                vec == 1 ? retVal
                         : extract_element(valueElemTy, retVal, i32_val(ii));
          }
        }
      } else {
        if (!atomicNeedsSharedMemory(op.getResult())) {
          rewriter.eraseOp(op);
          return success();
        }
        Value atomPtr =
            getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
        barrier();
        Value ret = load(valueElemTy, atomPtr);
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
           StoreOpConversion, BufferLoadOpConversion, BufferStoreOpConversion>(
          typeConverter, targetInfo, axisInfoAnalysis, benefit);
}
} // namespace mlir::triton::AMD
