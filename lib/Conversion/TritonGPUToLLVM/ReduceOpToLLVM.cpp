#include "ReduceOpToLLVM.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::shflSync;
using ::mlir::LLVM::storeShared;
using ::mlir::triton::gpu::getElemsPerThread;
using ::mlir::triton::gpu::getOrder;

struct ReduceOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::ReduceOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      triton::ReduceOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (ReduceOpHelper(op).isFastReduction())
      return matchAndRewriteFast(op, adaptor, rewriter);
    return matchAndRewriteBasic(op, adaptor, rewriter);
  }

private:
  void accumulate(ConversionPatternRewriter &rewriter, Location loc,
                  RedOp redOp, Value &acc, Value cur, bool isFirst) const {
    if (isFirst) {
      acc = cur;
      return;
    }
    switch (redOp) {
    case RedOp::ADD:
      acc = add(acc, cur);
      break;
    case RedOp::FADD:
      acc = fadd(acc.getType(), acc, cur);
      break;
    case RedOp::MIN:
      acc = smin(acc, cur);
      break;
    case RedOp::MAX:
      acc = smax(acc, cur);
      break;
    case RedOp::UMIN:
      acc = umin(acc, cur);
      break;
    case RedOp::UMAX:
      acc = umax(acc, cur);
      break;
    case RedOp::FMIN:
      acc = fmin(acc, cur);
      break;
    case RedOp::FMAX:
      acc = fmax(acc, cur);
      break;
    case RedOp::XOR:
      acc = xor_(acc, cur);
      break;
    case RedOp::ARGMIN:
    case RedOp::ARGMAX:
    case RedOp::ARGUMIN:
    case RedOp::ARGUMAX:
    case RedOp::ARGFMIN:
    case RedOp::ARGFMAX:
      llvm::report_fatal_error(
          "This accumulate implementation is not for argmin / argmax");
    default:
      llvm::report_fatal_error("Unsupported reduce op");
    }
  }

  void accumulateWithIndex(ConversionPatternRewriter &rewriter, Location loc,
                           RedOp redOp, Value &acc, Value &accIndex, Value cur,
                           Value curIndex, bool isFirst) const {
    if (isFirst) {
      acc = cur;
      accIndex = curIndex;
      return;
    }
    switch (redOp) {
    case RedOp::ARGMIN:
      accIndex = select(
          icmp_slt(acc, cur), accIndex,
          select(icmp_sgt(acc, cur), curIndex, smin(accIndex, curIndex)));
      acc = smin(acc, cur);
      break;
    case RedOp::ARGMAX:
      accIndex = select(
          icmp_sgt(acc, cur), accIndex,
          select(icmp_slt(acc, cur), curIndex, smin(accIndex, curIndex)));
      acc = smax(acc, cur);
      break;
    case RedOp::ARGUMIN:
      accIndex = select(
          icmp_ult(acc, cur), accIndex,
          select(icmp_ugt(acc, cur), curIndex, smin(accIndex, curIndex)));
      acc = umin(acc, cur);
      break;
    case RedOp::ARGUMAX:
      accIndex = select(
          icmp_ugt(acc, cur), accIndex,
          select(icmp_ult(acc, cur), curIndex, smin(accIndex, curIndex)));
      acc = umax(acc, cur);
      break;
    case RedOp::ARGFMIN:
      accIndex = select(
          fcmp_olt(acc, cur), accIndex,
          select(fcmp_ogt(acc, cur), curIndex, smin(accIndex, curIndex)));
      acc = fmin(acc, cur);
      break;
    case RedOp::ARGFMAX:
      accIndex = select(
          fcmp_ogt(acc, cur), accIndex,
          select(fcmp_olt(acc, cur), curIndex, smin(accIndex, curIndex)));
      acc = fmax(acc, cur);
      break;
    case RedOp::ADD:
    case RedOp::FADD:
    case RedOp::MIN:
    case RedOp::MAX:
    case RedOp::UMIN:
    case RedOp::UMAX:
    case RedOp::FMIN:
    case RedOp::FMAX:
    case RedOp::XOR:
      llvm::report_fatal_error(
          "This accumulate implementation is only for argmin / argmax");
    default:
      llvm::report_fatal_error("Unsupported reduce op");
    }
  }

  // Use shared memory for reduction within warps and across warps
  LogicalResult
  matchAndRewriteBasic(triton::ReduceOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    unsigned axis = op.getAxis();
    bool withIndex = triton::ReduceOp::withIndex(op.getRedOp());

    auto srcTy = op.getOperand().getType().cast<RankedTensorType>();
    auto srcLayout = srcTy.getEncoding().cast<BlockedEncodingAttr>();
    auto srcOrd = srcLayout.getOrder();
    auto srcShape = srcTy.getShape();

    auto llvmElemTy = getTypeConverter()->convertType(srcTy.getElementType());
    auto llvmIndexTy = getTypeConverter()->getIndexType();
    auto elemPtrTy = LLVM::LLVMPointerType::get(llvmElemTy, 3);
    auto indexPtrTy = LLVM::LLVMPointerType::get(llvmIndexTy, 3);
    Value smemBase = getSharedMemoryBase(loc, rewriter, op.getOperation());
    smemBase = bitcast(smemBase, elemPtrTy);

    ReduceOpHelper helper(op);
    auto smemShape = helper.getScratchConfigBasic();
    unsigned elems = product<unsigned>(smemShape);
    Value indexSmemBase = gep(elemPtrTy, smemBase, i32_val(elems));
    indexSmemBase = bitcast(indexSmemBase, indexPtrTy);

    unsigned srcElems = getElemsPerThread(srcTy);
    auto srcIndices = emitIndices(loc, rewriter, srcLayout, srcTy);
    auto srcValues = getTypeConverter()->unpackLLElements(
        loc, adaptor.getOperand(), rewriter, srcTy);

    SmallVector<SmallVector<unsigned>> offset =
        emitOffsetForLayout(srcLayout, srcTy);

    std::map<SmallVector<unsigned>, Value> accs;
    std::map<SmallVector<unsigned>, Value> accIndices;
    std::map<SmallVector<unsigned>, SmallVector<Value>> indices;

    // reduce within threads
    for (unsigned i = 0; i < srcElems; ++i) {
      SmallVector<unsigned> key = offset[i];
      key[axis] = 0;
      bool isFirst = accs.find(key) == accs.end();
      if (!withIndex) {
        accumulate(rewriter, loc, op.getRedOp(), accs[key], srcValues[i],
                   isFirst);
      } else {
        Value curIndex = srcIndices[i][axis];
        accumulateWithIndex(rewriter, loc, op.getRedOp(), accs[key],
                            accIndices[key], srcValues[i], curIndex, isFirst);
      }
      if (isFirst)
        indices[key] = srcIndices[i];
    }

    // cached int32 constants
    std::map<int, Value> ints;
    ints[0] = i32_val(0);
    for (int N = smemShape[axis] / 2; N > 0; N >>= 1)
      ints[N] = i32_val(N);
    Value sizePerThread = i32_val(srcLayout.getSizePerThread()[axis]);

    // reduce across threads
    for (auto it : accs) {
      const SmallVector<unsigned> &key = it.first;
      Value acc = it.second;
      Value accIndex;
      if (withIndex)
        accIndex = accIndices[key];
      SmallVector<Value> writeIdx = indices[key];

      writeIdx[axis] = udiv(writeIdx[axis], sizePerThread);
      Value writeOffset = linearize(rewriter, loc, writeIdx, smemShape, srcOrd);
      Value writePtr = gep(elemPtrTy, smemBase, writeOffset);
      Value indexWritePtr = gep(indexPtrTy, indexSmemBase, writeOffset);
      store(acc, writePtr);
      if (withIndex)
        store(accIndex, indexWritePtr);

      SmallVector<Value> readIdx(writeIdx.size(), ints[0]);
      for (int N = smemShape[axis] / 2; N > 0; N >>= 1) {
        readIdx[axis] = ints[N];
        Value readMask = icmp_slt(writeIdx[axis], ints[N]);
        Value readOffset = select(
            readMask, linearize(rewriter, loc, readIdx, smemShape, srcOrd),
            ints[0]);
        Value readPtr = gep(elemPtrTy, writePtr, readOffset);
        barrier();
        if (!withIndex) {
          Value cur = load(readPtr);
          accumulate(rewriter, loc, op.getRedOp(), acc, cur, false);
          barrier();
          store(acc, writePtr);
        } else {
          Value cur = load(readPtr);
          Value indexReadPtr = gep(indexPtrTy, indexWritePtr, readOffset);
          Value curIndex = load(indexReadPtr);
          accumulateWithIndex(rewriter, loc, op.getRedOp(), acc, accIndex, cur,
                              curIndex, false);
          barrier();
          store(acc, writePtr);
          store(accIndex, indexWritePtr);
        }
      }
    }

    barrier();

    // set output values
    if (auto resultTy = op.getType().dyn_cast<RankedTensorType>()) {
      // nd-tensor where n >= 1
      auto resultLayout = resultTy.getEncoding();
      auto resultShape = resultTy.getShape();

      unsigned resultElems = getElemsPerThread(resultTy);
      auto resultIndices = emitIndices(loc, rewriter, resultLayout, resultTy);
      assert(resultIndices.size() == resultElems);

      SmallVector<Value> resultVals(resultElems);
      for (unsigned i = 0; i < resultElems; ++i) {
        SmallVector<Value> readIdx = resultIndices[i];
        readIdx.insert(readIdx.begin() + axis, ints[0]);
        Value readOffset = linearize(rewriter, loc, readIdx, smemShape, srcOrd);
        Value readPtr = gep(elemPtrTy, smemBase, readOffset);
        Value indexReadPtr = gep(indexPtrTy, indexSmemBase, readOffset);
        resultVals[i] = withIndex ? load(indexReadPtr) : load(readPtr);
      }
      Value ret = getTypeConverter()->packLLElements(loc, resultVals, rewriter,
                                                     resultTy);
      rewriter.replaceOp(op, ret);
    } else {
      // 0d-tensor -> scalar
      Value resultVal = withIndex ? load(indexSmemBase) : load(smemBase);
      rewriter.replaceOp(op, resultVal);
    }

    return success();
  }

  // Use warp shuffle for reduction within warps and shared memory for data
  // exchange across warps
  LogicalResult matchAndRewriteFast(triton::ReduceOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    unsigned axis = adaptor.getAxis();
    bool withIndex = triton::ReduceOp::withIndex(op.getRedOp());

    auto srcTy = op.getOperand().getType().cast<RankedTensorType>();
    auto srcLayout = srcTy.getEncoding();
    auto srcShape = srcTy.getShape();
    auto order = getOrder(srcLayout);

    auto threadsPerWarp = triton::gpu::getThreadsPerWarp(srcLayout);
    auto warpsPerCTA = triton::gpu::getWarpsPerCTA(srcLayout);

    auto llvmElemTy = getTypeConverter()->convertType(srcTy.getElementType());
    auto llvmIndexTy = getTypeConverter()->getIndexType();
    auto elemPtrTy = LLVM::LLVMPointerType::get(llvmElemTy, 3);
    auto indexPtrTy = LLVM::LLVMPointerType::get(llvmIndexTy, 3);
    Value smemBase = getSharedMemoryBase(loc, rewriter, op.getOperation());
    smemBase = bitcast(smemBase, elemPtrTy);

    ReduceOpHelper helper(op);
    auto smemShapes = helper.getScratchConfigsFast();
    unsigned elems = product<unsigned>(smemShapes[0]);
    unsigned maxElems = std::max(elems, product<unsigned>(smemShapes[1]));
    Value indexSmemBase = gep(elemPtrTy, smemBase, i32_val(maxElems));
    indexSmemBase = bitcast(indexSmemBase, indexPtrTy);

    unsigned sizeIntraWarps = helper.getIntraWarpSize();
    unsigned sizeInterWarps = helper.getInterWarpSize();

    unsigned srcElems = getElemsPerThread(srcTy);
    auto srcIndices = emitIndices(loc, rewriter, srcLayout, srcTy);
    auto srcValues = getTypeConverter()->unpackLLElements(
        loc, adaptor.getOperand(), rewriter, srcTy);

    SmallVector<SmallVector<unsigned>> offset =
        emitOffsetForLayout(srcLayout, srcTy);

    std::map<SmallVector<unsigned>, Value> accs;
    std::map<SmallVector<unsigned>, Value> accIndices;
    std::map<SmallVector<unsigned>, SmallVector<Value>> indices;

    // reduce within threads
    for (unsigned i = 0; i < srcElems; ++i) {
      SmallVector<unsigned> key = offset[i];
      key[axis] = 0;
      bool isFirst = accs.find(key) == accs.end();
      if (!withIndex) {
        accumulate(rewriter, loc, op.getRedOp(), accs[key], srcValues[i],
                   isFirst);
      } else {
        Value curIndex = srcIndices[i][axis];
        accumulateWithIndex(rewriter, loc, op.getRedOp(), accs[key],
                            accIndices[key], srcValues[i], curIndex, isFirst);
      }
      if (isFirst)
        indices[key] = srcIndices[i];
    }

    Value threadId = getThreadId(rewriter, loc);
    Value warpSize = i32_val(32);
    Value warpId = udiv(threadId, warpSize);
    Value laneId = urem(threadId, warpSize);

    SmallVector<Value> multiDimLaneId =
        delinearize(rewriter, loc, laneId, threadsPerWarp, order);
    SmallVector<Value> multiDimWarpId =
        delinearize(rewriter, loc, warpId, warpsPerCTA, order);

    Value laneIdAxis = multiDimLaneId[axis];
    Value warpIdAxis = multiDimWarpId[axis];

    Value zero = i32_val(0);
    Value laneZero = icmp_eq(laneIdAxis, zero);

    for (auto it : accs) {
      const SmallVector<unsigned> &key = it.first;
      Value acc = it.second;
      Value accIndex;
      if (withIndex)
        accIndex = accIndices[key];

      // Reduce within warps
      for (unsigned N = sizeIntraWarps / 2; N > 0; N >>= 1) {
        Value shfl = shflSync(loc, rewriter, acc, N);
        if (!withIndex) {
          accumulate(rewriter, loc, op.getRedOp(), acc, shfl, false);
        } else {
          Value shflIndex = shflSync(loc, rewriter, accIndex, N);
          accumulateWithIndex(rewriter, loc, op.getRedOp(), acc, accIndex, shfl,
                              shflIndex, false);
        }
      }

      SmallVector<Value> writeIdx = indices[key];
      writeIdx[axis] = (sizeInterWarps == 1) ? zero : warpIdAxis;
      Value writeOffset =
          linearize(rewriter, loc, writeIdx, smemShapes[0], order);
      Value writePtr = gep(elemPtrTy, smemBase, writeOffset);
      storeShared(rewriter, loc, writePtr, acc, laneZero);
      if (withIndex) {
        Value indexWritePtr = gep(indexPtrTy, indexSmemBase, writeOffset);
        storeShared(rewriter, loc, indexWritePtr, accIndex, laneZero);
      }
    }

    barrier();

    // The second round of shuffle reduction
    //   now the problem size: sizeInterWarps, s1, s2, .. , sn
    //   where sizeInterWarps is 2^m
    //
    // Each thread needs to process:
    //   elemsPerThread = sizeInterWarps * s1 * s2 .. Sn / numThreads
    unsigned numThreads =
        product<unsigned>(triton::gpu::getWarpsPerCTA(srcLayout)) * 32;
    unsigned elemsPerThread = std::max<unsigned>(elems / numThreads, 1);
    Value readOffset = threadId;
    for (unsigned round = 0; round < elemsPerThread; ++round) {
      Value readPtr = gep(elemPtrTy, smemBase, readOffset);
      // FIXME(Qingyi): need predicate icmp_slt(threadId,
      // i32_val(sizeInerWarps))
      Value acc = load(readPtr);
      Value accIndex;
      if (withIndex) {
        Value readIndexPtr = gep(indexPtrTy, indexSmemBase, readOffset);
        accIndex = load(readIndexPtr);
      }

      for (unsigned N = sizeInterWarps / 2; N > 0; N >>= 1) {
        Value shfl = shflSync(loc, rewriter, acc, N);
        if (!withIndex) {
          accumulate(rewriter, loc, op.getRedOp(), acc, shfl, false);
        } else {
          Value shflIndex = shflSync(loc, rewriter, accIndex, N);
          accumulateWithIndex(rewriter, loc, op.getRedOp(), acc, accIndex, shfl,
                              shflIndex, false);
        }
      }

      // only the first thread in each sizeInterWarps is writing
      Value writeOffset = readOffset;
      Value writePtr = gep(elemPtrTy, smemBase, writeOffset);
      Value threadIsNeeded = icmp_slt(threadId, i32_val(elems));
      Value laneIdModSizeInterWarps = urem(laneId, i32_val(sizeInterWarps));
      Value laneIdModSizeInterWarpsIsZero =
          icmp_eq(laneIdModSizeInterWarps, zero);
      Value pred = and_(threadIsNeeded, laneIdModSizeInterWarpsIsZero);
      storeShared(rewriter, loc, writePtr, acc, pred);
      if (withIndex) {
        Value writeIndexPtr = gep(indexPtrTy, indexSmemBase, writeOffset);
        storeShared(rewriter, loc, writeIndexPtr, accIndex, pred);
      }

      if (round != elemsPerThread - 1) {
        readOffset = add(readOffset, i32_val(numThreads));
      }
    }

    // We could avoid this barrier in some of the layouts, however this is not
    // the general case.
    // TODO: optimize the barrier in case the layouts are accepted.
    barrier();

    // set output values
    if (auto resultTy = op.getType().dyn_cast<RankedTensorType>()) {
      // nd-tensor where n >= 1
      auto resultLayout = resultTy.getEncoding().cast<SliceEncodingAttr>();
      auto resultShape = resultTy.getShape();
      unsigned resultElems = getElemsPerThread(resultTy);
      auto resultIndices = emitIndices(loc, rewriter, resultLayout, resultTy);
      assert(resultIndices.size() == resultElems);

      SmallVector<Value> resultVals(resultElems);
      for (size_t i = 0; i < resultElems; ++i) {
        SmallVector<Value> readIdx = resultIndices[i];
        readIdx.insert(readIdx.begin() + axis, i32_val(0));
        Value readOffset =
            linearize(rewriter, loc, readIdx, smemShapes[0], order);
        Value readPtr = gep(elemPtrTy, smemBase, readOffset);
        Value indexReadPtr = gep(indexPtrTy, indexSmemBase, readOffset);
        resultVals[i] = withIndex ? load(indexReadPtr) : load(readPtr);
      }
      Value ret = getTypeConverter()->packLLElements(loc, resultVals, rewriter,
                                                     resultTy);
      rewriter.replaceOp(op, ret);
    } else {
      // 0d-tensor -> scalar
      Value resultVal = withIndex ? load(indexSmemBase) : load(smemBase);
      rewriter.replaceOp(op, resultVal);
    }

    return success();
  }
};

void populateReduceOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, AxisInfoAnalysis &axisInfoAnalysis,
    const Allocation *allocation, Value smem,
    ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo &indexCacheInfo,
    PatternBenefit benefit) {
  patterns.add<ReduceOpConversion>(typeConverter, allocation, smem,
                                   indexCacheInfo, benefit);
}
