#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

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
    auto order = triton::gpu::getOrder(layout);
    auto shapePerCTATile = triton::gpu::getShapePerCTATile(layout, shape);
    Value warpSize = i32_val(triton::gpu::getWarpSize(layout));
    Value laneId = urem(tid, warpSize);
    Value warpId = udiv(tid, warpSize);
    SmallVector<Value> multiDimWarpId =
        delinearize(rewriter, loc, warpId, warpsPerCTA, order);
    SmallVector<Value> multiDimThreadId =
        delinearize(rewriter, loc, laneId, threadsPerWarp, order);
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
                                   ModuleAxisInfoAnalysis &axisAnalysisPass,
                                   const DenseSet<Value> &assumptions)
      : targetInfo(targetInfo), axisAnalysisPass(axisAnalysisPass),
        assumptions(assumptions) {}

  unsigned getContiguity(Value ptr) const {
    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return 1;
    return axisAnalysisPass.getPtrContiguity(ptr);
  }

  unsigned getVectorSize(Value ptr) const {
    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return 1;
    auto contiguity = getContiguity(ptr);
    auto pointeeBitWidth = triton::getPointeeBitWidth(tensorTy);
    // The maximum vector size is 128 bits on NVIDIA GPUs.
    return std::min<unsigned>(128 / pointeeBitWidth, contiguity);
  }

  // Look through the available assumption to verify if the expression has been
  // assumed positive
  bool verifyNonNegativeByAssumption(Value expr) const {
    for (Value assume : assumptions) {
      if (auto cmpOp = dyn_cast<arith::CmpIOp>(assume.getDefiningOp())) {
        bool isGreaterThan =
            (cmpOp.getPredicate() == arith::CmpIPredicate::sge ||
             cmpOp.getPredicate() == arith::CmpIPredicate::sgt);
        APInt cst;
        if (isGreaterThan && (cmpOp.getLhs() == expr) &&
            matchPattern(cmpOp.getRhs(), m_ConstantInt(&cst))) {
          return cst.isNonNegative();
        }
      }
    }
    return false;
  }

  // Look if the expression is a block argument with a "tt.non_negative"
  // property
  bool verifyNonNegativeByFunctionProperty(Value expr) const {
    if (!expr.getDefiningOp()) {
      BlockArgument blockArg = dyn_cast<BlockArgument>(expr);
      if (blockArg && blockArg.getOwner()->isEntryBlock()) {
        Operation *op = blockArg.getOwner()->getParentOp();
        if (auto fun = dyn_cast<FunctionOpInterface>(op))
          if (fun.getArgAttr(blockArg.getArgNumber(), "tt.non_negative"))
            return true;
      }
    }
    return false;
  }

  bool verifyNonNegativeExpr(Value expr) const {

    // Base case 1: check if the expression is contained in any assumption
    if (verifyNonNegativeByAssumption(expr))
      return true;

    // Base case 2: check if the expression is a BlockArgument and if there
    // is a property that states its non-negativity
    if (verifyNonNegativeByFunctionProperty(expr))
      return true;

    // Recurse if the operation is defined
    Operation *op = expr.getDefiningOp();
    if (!op)
      return false;

    bool nonNegative =
        llvm::TypeSwitch<Operation *, bool>(expr.getDefiningOp())
            .Case<triton::BroadcastOp>([&](auto broadcastOp) {
              return verifyNonNegativeExpr(broadcastOp.getSrc());
            })
            .Case<triton::ExpandDimsOp>([&](auto expandOp) {
              return verifyNonNegativeExpr(expandOp.getSrc());
            })
            .Case<triton::SplatOp>([&](auto splatOp) {
              return verifyNonNegativeExpr(splatOp.getSrc());
            })
            .Case<triton::MakeRangeOp>([&](auto makeRangeOp) {
              return makeRangeOp.getStart() >= 0 && makeRangeOp.getEnd() >= 0;
            })
            .Case<arith::ConstantIntOp>(
                [&](auto constIntOp) { return constIntOp.value() >= 0; })
            .Case<arith::ConstantOp>([&](arith::ConstantOp constOp) {
              Value val = constOp.getResult();
              DenseIntElementsAttr constVal;
              if (matchPattern(val, m_Constant(&constVal)) &&
                  constVal.isSplat())
                return constVal.getSplatValue<APInt>().isNonNegative();
              return false;
            })
            .Case<triton::GetProgramIdOp>([&](auto pidOp) { return true; })
            .Case<arith::MaxSIOp>([&](auto maxOp) {
              // max(a,b) >= 0 iff a>=0 || b>=0
              bool nnLhs = verifyNonNegativeExpr(maxOp.getLhs());
              bool nnRhs = verifyNonNegativeExpr(maxOp.getRhs());
              return nnLhs || nnRhs;
            })
            .Case<arith::RemSIOp>([&](auto remsiOp) {
              // a % b >= 0 iff a>=0
              return (verifyNonNegativeExpr(remsiOp.getLhs()));
            })
            .Case<arith::AddIOp, arith::MinSIOp, arith::MulIOp, arith::DivSIOp>(
                // Generally speaking, a OP b >= 0  iff  a >= 0 && b >= 0 when
                // OP != sub
                [&](Operation *binOp) {
                  bool nnLhs = verifyNonNegativeExpr(binOp->getOperand(0));
                  bool nnRhs = verifyNonNegativeExpr(binOp->getOperand(1));
                  return nnLhs && nnRhs;
                })
            .Default([&](Operation *op) {
              // Base case 3: unknown operation
              return false;
            });
    return nonNegative;
  }

  // Quick analysis on the Triton IR to decide if we can safely use
  // buffer operations
  bool canUseBufferOps(Value ptr) const {
    // 1. Check if the pointer is uniform: i.e., if it comes from a scalar
    // pointer(splatted) and non-uniform offset addition
    DenseSet<triton::AddPtrOp> nonUniformUpdates;
    SmallVector<Operation *> queue{ptr.getDefiningOp()};
    while (!queue.empty()) {
      Operation *curOp = queue.pop_back_val();
      if (!curOp)
        continue;
      if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(curOp))
        if (isa<RankedTensorType>(addPtrOp.getPtr().getType()))
          nonUniformUpdates.insert(addPtrOp);
      for (Value operand : curOp->getOperands())
        queue.push_back(operand.getDefiningOp());
    }

    // 2. Check the that pointer is not a block argument. We cannot
    // be sure if the block argument has been already non-uniformly
    // updated by the caller
    bool useBufferOps = (nonUniformUpdates.size() == 1);

    if (useBufferOps) {
      triton::AddPtrOp addPtrOp = (*nonUniformUpdates.begin());
      // 2. Check that the tensor pointer is not coming from a function
      // argument. We have no way to know if that pointer has been
      // already updated by the caller
      Value basePtr = addPtrOp.getPtr();
      auto maybeBufferArg = dyn_cast<BlockArgument>(basePtr);
      useBufferOps =
          !maybeBufferArg ||
          !isa<FunctionOpInterface>(maybeBufferArg.getOwner()->getParentOp());

      // 3. Check if the offset can be expressed ad 32-bits
      Value offset = addPtrOp.getOffset();
      useBufferOps =
          useBufferOps &&
          (cast<RankedTensorType>(offset.getType()).getElementTypeBitWidth() ==
           32);

      // 4. Check if the offset is non-negative
      useBufferOps = useBufferOps && verifyNonNegativeExpr(offset);
    }
    return useBufferOps;
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
  DenseSet<Value> assumptions;
};

struct LoadOpConversion : public ConvertOpToLLVMPattern<triton::LoadOp>,
                          public LoadStoreConversionBase {
  using ConvertOpToLLVMPattern<triton::LoadOp>::ConvertOpToLLVMPattern;

  LoadOpConversion(LLVMTypeConverter &converter,
                   const AMD::TargetInfo &targetInfo,
                   ModuleAxisInfoAnalysis &axisAnalysisPass,
                   const DenseSet<Value> &assumptions, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::LoadOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass, assumptions) {}

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
    bool useBufferOps =
        tools::getBoolEnv("AMDGCN_USE_BUFFER_OPS") && canUseBufferOps(ptr);
    if (llMask)
      vec = std::min<size_t>(vec, getMaskAlignment(mask));

    // Get the LLVM values for pointers
    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    assert(ptrElems.size() == numElems);

    // Get the LLVM values for mask
    SmallVector<Value> maskElems;
    if (llMask) {
      maskElems = unpackLLElements(loc, llMask, rewriter);
      assert(maskElems.size() == numElems);
    }

    // Get the LLVM values for `other`
    // TODO: (goostavz) handle when other is const but not splat, which
    //       should be rarely seen
    bool otherIsSplatConstInt = false;
    DenseElementsAttr constAttr;
    int64_t splatVal = 0;
    if (other && isa<IntegerType>(valueElemTy) &&
        matchPattern(other, m_Constant(&constAttr)) && constAttr.isSplat() &&
        isa<IntegerType>(constAttr.getElementType())) {
      otherIsSplatConstInt = true;
      splatVal = constAttr.getSplatValue<APInt>().getSExtValue();
    }
    SmallVector<Value> otherElems;
    if (other) {
      otherElems = unpackLLElements(loc, llOther, rewriter);
    }

    // vectorized iteration through all the pointer/mask/other elements
    const int valueElemNBits =
        std::max(8u, valueElemTy.getIntOrFloatBitWidth());
    const size_t valueElemNBytes = valueElemNBits / 8;
    const int numVecs = numElems / vec;
    int64_t ptrAlignmentBytes = getPtrAlignment(ptr) * valueElemNBytes;

    auto cacheMod = op.getCache();
    SmallVector<Value> loadedVals;
    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      size_t in_off = 0;

      const size_t maxWordWidth = std::max<size_t>(32, valueElemNBits);
      const size_t totalWidth = valueElemNBits * vec;
      const size_t width = std::min(totalWidth, maxWordWidth);
      const size_t nWords = std::max<size_t>(1, totalWidth / width);
      const size_t wordNElems = width / valueElemNBits;
      const size_t movWidth = width < 16 ? 16 : width;
      assert(wordNElems * nWords * numVecs == numElems);

      Value pred = mask ? maskElems[vecStart] : int_val(1, 1);
      auto vecTy = LLVM::getFixedVectorType(valueElemTy, vec);
      Value ptr = addrspacecast(ptr_ty(getContext()), ptrElems[vecStart]);

      mlir::Attribute zeroAttr = rewriter.getZeroAttr(valueElemTy);
      auto denseValue =
          DenseElementsAttr::get(cast<mlir::ShapedType>(vecTy), zeroAttr);
      Value zeroVal = rewriter.create<LLVM::ConstantOp>(loc, vecTy, denseValue);

      Value falseVal = zeroVal;
      // If we need to mask the loaded value with other elements
      if (otherElems.size() != 0) {
        Value v = undef(vecTy);
        for (size_t s = 0; s < vec; ++s) {
          Value otherElem = otherElems[vecStart + s];
          Value indexVal = LLVM::createIndexConstant(
              rewriter, loc, this->getTypeConverter(), s);
          v = insert_element(vecTy, v, otherElem, indexVal);
        }
        falseVal = v;
      }

      Value loadVal =
          llLoad(rewriter, loc, ptr, vecTy, pred, falseVal, targetInfo,
                 ptrAlignmentBytes, cacheMod, useBufferOps);
      for (size_t ii = 0; ii < vec; ++ii) {
        Value vecIdx = createIndexAttrConstant(
            rewriter, loc, this->getTypeConverter()->getIndexType(), ii % vec);
        Value loaded = extract_element(valueElemTy, loadVal, vecIdx);
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

struct StoreOpConversion : public ConvertOpToLLVMPattern<triton::StoreOp>,
                           public LoadStoreConversionBase {
  using ConvertOpToLLVMPattern<triton::StoreOp>::ConvertOpToLLVMPattern;

  StoreOpConversion(LLVMTypeConverter &converter,
                    const AMD::TargetInfo &targetInfo,
                    ModuleAxisInfoAnalysis &axisAnalysisPass,
                    const DenseSet<Value> &assumptions, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::StoreOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass, assumptions) {}

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
    bool useBufferOps =
        tools::getBoolEnv("AMDGCN_USE_BUFFER_OPS") && canUseBufferOps(ptr);

    auto valueTy = value.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));

    unsigned vec = getVectorSize(ptr);
    unsigned elemsPerThread = getTotalElemsPerThread(ptr.getType());

    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    auto valueElems = unpackLLElements(loc, llValue, rewriter);
    assert(ptrElems.size() == valueElems.size());

    // Determine the vectorization size
    SmallVector<Value> maskElems;
    if (llMask) {
      maskElems = unpackLLElements(loc, llMask, rewriter);
      assert(valueElems.size() == maskElems.size());

      unsigned maskAlign = getMaskAlignment(mask);
      vec = std::min(vec, maskAlign);
    }

    const size_t valueElemNBits =
        std::max<int>(8, valueElemTy.getIntOrFloatBitWidth());
    const size_t valueElemNBytes = valueElemNBits / 8;
    int64_t ptrAlignmentBytes = getPtrAlignment(ptr) * valueElemNBytes;

    auto cacheMod = op.getCache();
    const int numVecs = elemsPerThread / vec;
    Value rDataMask = redundantDataMask(valueTy, rewriter, loc, targetInfo);
    for (size_t vecStart = 0; vecStart < elemsPerThread; vecStart += vec) {
      size_t in_off = 0;
      Value pred = mask ? and_(maskElems[vecStart], rDataMask) : rDataMask;
      auto vecTy = LLVM::getFixedVectorType(valueElemTy, vec);

      const size_t maxWordWidth = std::max<size_t>(32, valueElemNBits);
      const size_t totalWidth = valueElemNBits * vec;
      const size_t width = std::min(totalWidth, maxWordWidth);
      const size_t nWords = std::max<size_t>(1, totalWidth / width);
      const size_t wordNElems = width / valueElemNBits;
      assert(wordNElems * nWords * numVecs == elemsPerThread);

      Type valArgTy = IntegerType::get(ctx, width);
      auto wordTy = vec_ty(valueElemTy, wordNElems);

      SmallVector<std::pair<Value, std::string>> asmArgs;
      Value elem = valueElems[vecStart];
      Value ptr = addrspacecast(ptr_ty(getContext()), ptrElems[vecStart]);

      // Create the store val
      Value storeVal = undef(vecTy);
      for (size_t s = 0; s < vec; ++s) {
        Value otherElem = valueElems[vecStart + s];
        Value indexVal = createIndexAttrConstant(
            rewriter, loc, this->getTypeConverter()->getIndexType(), s);
        storeVal = insert_element(vecTy, storeVal, otherElem, indexVal);
      }
      llStore(rewriter, loc, ptr, storeVal, pred, targetInfo, ptrAlignmentBytes,
              cacheMod, useBufferOps);
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
                        const DenseSet<Value> &assumptions,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::AtomicCASOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass, assumptions) {}

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

struct AtomicRMWOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicRMWOp>,
      public LoadStoreConversionBase {
  using ConvertOpToLLVMPattern<triton::AtomicRMWOp>::ConvertOpToLLVMPattern;

  AtomicRMWOpConversion(LLVMTypeConverter &converter,
                        const AMD::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        const DenseSet<Value> &assumptions,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::AtomicRMWOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass, assumptions) {}

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
    // tensor
    if (tensorTy) {
      auto valTy = cast<RankedTensorType>(val.getType());
      vec = std::min<unsigned>(vec, valTy.getElementType().isF16() ? 2 : 1);
      // mask
      numElems = tensorTy.getNumElements();
    }
    Value mask = int_val(1, 1);
    auto tid = tid_val();
    mask = and_(mask,
                icmp_slt(mul(tid, i32_val(elemsPerThread)), i32_val(numElems)));

    auto memOrdering = op.getSem();
    auto atomicMemOrdering = getMemoryOrdering(memOrdering);

    auto vecTy = vec_ty(valueElemTy, vec);
    auto retType = vec == 1 ? valueElemTy : vecTy;
    SmallVector<Value> resultVals(elemsPerThread);
    const bool f16v2 = vec == 2 && valueElemTy.isF16();
    for (size_t i = 0; i < elemsPerThread; i += vec) {
      Value rmwPtr = ptrElements[i];
      // TODO: in case llMask is zero we can create only one branch for all
      // elemsPerThread.
      Value rmwMask = llMask ? and_(mask, maskElements[i]) : mask;

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
      Value atom = rewriter
                       .create<LLVM::AtomicRMWOp>(
                           loc, *maybeKind, rmwPtr, valElements[i],
                           atomicMemOrdering, StringRef("agent"))
                       .getResult();

      // NV for the f16v2 case generates one packed instruction. We have to
      // create two separate instructions since LLVM::AtomicRMWOp doesn't
      // support this. Can be optimized out with rocdl.raw.buffer.atomic.
      if (f16v2) {
        Value atom2 =
            rewriter
                .create<LLVM::AtomicRMWOp>(
                    loc, *maybeKind, ptrElements[i + 1], valElements[i + 1],
                    atomicMemOrdering, StringRef("agent"))
                .getResult();
        auto tmp = insert_element(vecTy, undef(vecTy), atom, i32_val(0));
        atom = insert_element(vecTy, tmp, atom2, i32_val(1)).getResult();
      }
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
        for (int ii = 0; ii < vec; ++ii) {
          resultVals[i + ii] =
              vec == 1 ? retVal
                       : extract_element(valueElemTy, retVal, i32_val(ii));
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
                                       const DenseSet<Value> &assumptions,
                                       PatternBenefit benefit) {
  patterns.add<AtomicCASOpConversion, AtomicRMWOpConversion, LoadOpConversion,
               StoreOpConversion>(typeConverter, targetInfo, axisInfoAnalysis,
                                  assumptions, benefit);
}
} // namespace mlir::triton::AMD
