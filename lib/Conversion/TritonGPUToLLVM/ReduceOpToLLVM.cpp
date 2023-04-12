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

  // Calculates the write index in the shared memory where we would be writing
  // the within-thread accumulations before we start doing across-threads
  // accumulations. `index` is the index of the within-thread accumulations in
  // the full tensor, whereas `writeIdx` is the mapped-to index in the shared
  // memory
  void getWriteIndexBasic(ConversionPatternRewriter &rewriter, Location loc,
                          Attribute layout, SmallVector<Value> &index,
                          SmallVector<Value> &writeIdx,
                          std::map<int, Value> &ints, unsigned axis) const {
    writeIdx = index;
    auto sizePerThread = triton::gpu::getSizePerThread(layout);
    Value axisSizePerThread = ints[sizePerThread[axis]];
    Value _8 = ints[8];
    Value _16 = ints[16];
    if (layout.isa<BlockedEncodingAttr>()) {
      // A single thread owns axisSizePerThread contiguous values
      // on the reduction axis. After within thread reduction,
      // we would have a single accumulation every `axisSizePerThread`
      // contiguous values in the original tensor, so we would need
      // to map every `axisSizePerThread` to 1 value in smem as:
      // writeIdx[axis] = index[axis] / axisSizePerThread
      writeIdx[axis] = udiv(index[axis], axisSizePerThread);
    }
    auto mmaLayout = layout.dyn_cast<MmaEncodingAttr>();
    if (mmaLayout && mmaLayout.isAmpere()) {
      if (axis == 0) {
        // Because warpTileSize = [16, 8] and threadsPerWarp = [8, 4], each 8
        // rows in smem would correspond to a warp. The mapping
        // is: (warp_index) x 8 + (row index within warp)
        writeIdx[axis] =
            add(mul(udiv(index[axis], _16), _8), urem(index[axis], _8));
      } else {
        // Same as BlockedEncodingAttr case
        writeIdx[axis] = udiv(index[axis], axisSizePerThread);
      }
    }
    if (mmaLayout && !mmaLayout.isAmpere()) {
      llvm::report_fatal_error("Unsupported layout");
    }
  }

  // Use shared memory for reduction within warps and across warps
  LogicalResult
  matchAndRewriteBasic(triton::ReduceOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    ReduceOpHelper helper(op);
    Location loc = op->getLoc();
    unsigned axis = op.getAxis();
    // Specifies whether the reduce operation returns an index
    // rather than a value, e.g. argmax, argmin, .. etc
    bool withIndex = triton::ReduceOp::withIndex(op.getRedOp());

    auto srcTy = op.getOperand().getType().cast<RankedTensorType>();
    auto srcLayout = srcTy.getEncoding();
    if (!helper.isSupportedLayout()) {
      assert(false && "Unexpected srcLayout in ReduceOpConversion");
    }
    // The order of the axes for the the threads within the warp
    auto srcOrd = triton::gpu::getOrder(srcLayout);
    auto sizePerThread = triton::gpu::getSizePerThread(srcLayout);
    auto srcShape = srcTy.getShape();

    auto llvmElemTy = getTypeConverter()->convertType(srcTy.getElementType());
    auto llvmIndexTy = getTypeConverter()->getIndexType();
    auto elemPtrTy = LLVM::LLVMPointerType::get(llvmElemTy, 3);
    auto indexPtrTy = LLVM::LLVMPointerType::get(llvmIndexTy, 3);

    Value smemBase = getSharedMemoryBase(loc, rewriter, op.getOperation());
    smemBase = bitcast(smemBase, elemPtrTy);

    auto smemShape = helper.getScratchConfigBasic();
    unsigned elems = product<unsigned>(smemShape);
    Value indexSmemBase = gep(elemPtrTy, smemBase, i32_val(elems));
    indexSmemBase = bitcast(indexSmemBase, indexPtrTy);

    unsigned srcElems = getElemsPerThread(srcTy);
    // Emits indices of the original tensor that each thread
    // would own
    auto srcIndices = emitIndices(loc, rewriter, srcLayout, srcTy);
    auto srcValues = getTypeConverter()->unpackLLElements(
        loc, adaptor.getOperand(), rewriter, srcTy);
    // Emits offsets (the offset from the base index)
    // of the original tensor that each thread would own
    SmallVector<SmallVector<unsigned>> offset =
        emitOffsetForLayout(srcLayout, srcTy);
    // Keep track of accumulations and their indices
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
    ints[sizePerThread[axis]] = i32_val(sizePerThread[axis]);
    ints[8] = i32_val(8);
    ints[16] = i32_val(16);

    // reduce across threads
    for (auto it : accs) {
      const SmallVector<unsigned> &key = it.first;
      Value acc = it.second;
      Value accIndex;
      if (withIndex)
        accIndex = accIndices[key];
      // get the writeIdx at which to write in smem
      SmallVector<Value> writeIdx;
      getWriteIndexBasic(rewriter, loc, srcLayout, indices[key], writeIdx, ints,
                         axis);
      // calculate the offset in smem for that writeIdx
      Value writeOffset = linearize(rewriter, loc, writeIdx, smemShape, srcOrd);
      // Get element pointers for the value and index
      Value writePtr = gep(elemPtrTy, smemBase, writeOffset);
      Value indexWritePtr = gep(indexPtrTy, indexSmemBase, writeOffset);
      // Store the within-thread accumulated value at writePtr
      store(acc, writePtr);
      // Store the index of within-thread accumulation at indexWritePtr
      if (withIndex)
        store(accIndex, indexWritePtr);

      SmallVector<Value> readIdx(writeIdx.size(), ints[0]);
      // Perform parallel reduction with sequential addressing
      // E.g. We reduce `smemShape[axis]` elements into `smemShape[axis]/2`
      // elements using `smemShape[axis]/2` threads where each thread
      // would accumalte values that are `smemShape[axis]/2` apart
      // to avoid bank conflicts. Then we repeat with `smemShape[axis]/4`
      // threads, .. etc.
      for (int N = smemShape[axis] / 2; N > 0; N >>= 1) {
        // The readIdx will be N elements away on the reduction axis
        readIdx[axis] = ints[N];
        // If the writeIdx is greater or equal to N, do nothing
        Value readMask = icmp_slt(writeIdx[axis], ints[N]);
        // Calculate the readOffset, if readMask is False, readOffset=0
        // meaning we reduce the value at writeIdx with itself
        Value readOffset = select(
            readMask, linearize(rewriter, loc, readIdx, smemShape, srcOrd),
            ints[0]);
        // The readPtr is readOffset away from writePtr
        Value readPtr = gep(elemPtrTy, writePtr, readOffset);
        barrier();
        // If we do not care about the index, i.e. this is not an argmax,
        // argmin, .. etc
        if (!withIndex) {
          // The value at the readPtr, whereas acc is the value at writePtr
          Value cur = load(readPtr);
          accumulate(rewriter, loc, op.getRedOp(), acc, cur, false);
          barrier();
          // Update writePtr value
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
    ReduceOpHelper helper(op);
    Location loc = op->getLoc();
    unsigned axis = adaptor.getAxis();
    bool withIndex = triton::ReduceOp::withIndex(op.getRedOp());

    auto srcTy = op.getOperand().getType().cast<RankedTensorType>();
    auto srcLayout = srcTy.getEncoding();
    if (!helper.isSupportedLayout()) {
      assert(false && "Unexpected srcLayout in ReduceOpConversion");
    }
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
