#include "GenericReduceOpToLLVM.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::shflSync;
using ::mlir::LLVM::storeShared;
using ::mlir::triton::gpu::getElemsPerThread;
using ::mlir::triton::gpu::getOrder;

struct GenericReduceOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::GenericReduceOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      triton::GenericReduceOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GenericReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (ReduceOpHelper(op).isFastReduction())
      return matchAndRewriteFast(op, adaptor, rewriter);
    return matchAndRewriteBasic(op, adaptor, rewriter);
  }

private:

  void accumulate(ConversionPatternRewriter &rewriter,
                  Region &reduceOp, Value &acc, Value cur, bool isFirst) const {
    if (isFirst) {
      acc = cur;
      return;
    }

    // Create a new copy of the reduce block, and inline it
    Block *currentBlock = rewriter.getBlock();
    Region &parent = *currentBlock->getParent();
    rewriter.cloneRegionBefore(reduceOp, &parent.front());
    auto &newReduce = parent.front();
    auto returnOp = dyn_cast<triton::GenericReduceReturnOp>(newReduce.getTerminator());
    rewriter.mergeBlockBefore(&newReduce, &*rewriter.getInsertionPoint(), {acc, cur});
    acc = returnOp.getResult();
    // Delete the terminator, which is no longer used
    rewriter.eraseOp(returnOp);
  }

  // Use shared memory for reduction within warps and across warps
  LogicalResult
  matchAndRewriteBasic(triton::GenericReduceOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    unsigned axis = op.getAxis();

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


    Region *reduceOp = &op.getRegion();

    // reduce within threads
    for (unsigned i = 0; i < srcElems; ++i) {
      SmallVector<unsigned> key = offset[i];
      key[axis] = 0;
      bool isFirst = accs.find(key) == accs.end();
      accumulate(rewriter, *reduceOp, accs[key], srcValues[i], isFirst);
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
      SmallVector<Value> writeIdx = indices[key];

      writeIdx[axis] = udiv(writeIdx[axis], sizePerThread);
      Value writeOffset = linearize(rewriter, loc, writeIdx, smemShape, srcOrd);
      Value writePtr = gep(elemPtrTy, smemBase, writeOffset);
      Value indexWritePtr = gep(indexPtrTy, indexSmemBase, writeOffset);
      store(acc, writePtr);

      SmallVector<Value> readIdx(writeIdx.size(), ints[0]);
      for (int N = smemShape[axis] / 2; N > 0; N >>= 1) {
        readIdx[axis] = ints[N];
        Value readMask = icmp_slt(writeIdx[axis], ints[N]);
        Value readOffset = select(
            readMask, linearize(rewriter, loc, readIdx, smemShape, srcOrd),
            ints[0]);
        Value readPtr = gep(elemPtrTy, writePtr, readOffset);
        barrier();
        Value cur = load(readPtr);
        accumulate(rewriter, *reduceOp, acc, cur, false);
        barrier();
        store(acc, writePtr);
      }
    }

    barrier();

    // set output values
    if (auto resultTy = op.getType().dyn_cast<RankedTensorType>()) {
      // nd-tensor where n >= 1
      auto resultLayout = resultTy.getEncoding();

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
        resultVals[i] = load(readPtr);
      }
      Value ret = getTypeConverter()->packLLElements(loc, resultVals, rewriter,
                                                     resultTy);
      rewriter.replaceOp(op, ret);
    } else {
      // 0d-tensor -> scalar
      Value resultVal = load(smemBase);
      rewriter.replaceOp(op, resultVal);
    }

    return success();
  }

  // Use warp shuffle for reduction within warps and shared memory for data
  // exchange across warps
  LogicalResult matchAndRewriteFast(triton::GenericReduceOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {

    Location loc = op->getLoc();
    unsigned axis = adaptor.getAxis();

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

    auto &currentBlock = *rewriter.getBlock();
    auto *reduceOp = &op.getRegion();

    // reduce within threads
    for (unsigned i = 0; i < srcElems; ++i) {
      SmallVector<unsigned> key = offset[i];
      key[axis] = 0;
      bool isFirst = accs.find(key) == accs.end();
      accumulate(rewriter, *reduceOp, accs[key], srcValues[i], isFirst);
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

      // Reduce within warps
      for (unsigned N = sizeIntraWarps / 2; N > 0; N >>= 1) {
        Value shfl = shflSync(loc, rewriter, acc, N);
        accumulate(rewriter, *reduceOp, acc, shfl, false);
      }

      SmallVector<Value> writeIdx = indices[key];
      writeIdx[axis] = (sizeInterWarps == 1) ? zero : warpIdAxis;
      Value writeOffset =
          linearize(rewriter, loc, writeIdx, smemShapes[0], order);
      Value writePtr = gep(elemPtrTy, smemBase, writeOffset);
      storeShared(rewriter, loc, writePtr, acc, laneZero);
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

      for (unsigned N = sizeInterWarps / 2; N > 0; N >>= 1) {
        Value shfl = shflSync(loc, rewriter, acc, N);
        accumulate(rewriter, *reduceOp, acc, shfl, false);
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

      if (round != elemsPerThread - 1) {
        readOffset = add(readOffset, i32_val(numThreads));
      }
    }

    // We could avoid this barrier in some of the layouts, however this is not
    // the general case.
    // TODO: optimize the barrier incase the layouts are accepted.
    barrier();

    // set output values
    if (auto resultTy = op.getType().dyn_cast<RankedTensorType>()) {
      // nd-tensor where n >= 1
      auto resultLayout = resultTy.getEncoding().cast<SliceEncodingAttr>();
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
        resultVals[i] = load(readPtr);
      }

      Value ret = getTypeConverter()->packLLElements(loc, resultVals, rewriter,
                                                     resultTy);
      rewriter.replaceOp(op, ret);
    } else {
      // 0d-tensor -> scalar
      Value resultVal = load(smemBase);
      rewriter.replaceOp(op, resultVal);
    }

    return success();
  }
};

void populateGenericReduceOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, AxisInfoAnalysis &axisInfoAnalysis,
    const Allocation *allocation, Value smem,
    ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo &indexCacheInfo,
    PatternBenefit benefit) {
  patterns.add<GenericReduceOpConversion>(typeConverter, allocation, smem,
                                          indexCacheInfo, benefit);
}
