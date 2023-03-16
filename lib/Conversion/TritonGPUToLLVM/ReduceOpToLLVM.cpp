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
  void accumulate(ConversionPatternRewriter &rewriter, Region &combineOp,
                  llvm::SmallVectorImpl<Value> &acc, ValueRange cur,
                  bool isFirst) const {
    if (isFirst) {
      acc.resize(cur.size());
      for (unsigned i = 0; i < cur.size(); ++i) {
        acc[i] = cur[i];
      }
      return;
    }

    // Create a new copy of the reduce block, and inline it
    Block *currentBlock = rewriter.getBlock();
    Region &parent = *currentBlock->getParent();
    rewriter.cloneRegionBefore(combineOp, &parent.front());
    auto &newReduce = parent.front();
    auto returnOp = dyn_cast<triton::ReduceReturnOp>(newReduce.getTerminator());

    llvm::SmallVector<Value> combineArgs(2 * acc.size());
    for (unsigned i = 0; i < acc.size(); ++i) {
      combineArgs[i] = acc[i];
      combineArgs[acc.size() + i] = cur[i];
    }

    rewriter.mergeBlockBefore(&newReduce, &*rewriter.getInsertionPoint(),
                              combineArgs);

    auto results = returnOp.getResult();
    for (unsigned i = 0; i < acc.size(); ++i) {
      acc[i] = results[i];
    }

    // Delete the terminator, which is no longer used
    rewriter.eraseOp(returnOp);
  }

  SmallVector<SmallVector<Value>>
  unpackInputs(Location loc, triton::ReduceOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const {
    auto types = op.getInputTypes();
    auto operands = adaptor.getOperands();
    unsigned srcElems = getElemsPerThread(types[0]);
    SmallVector<SmallVector<Value>> srcValues(srcElems);
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto values = getTypeConverter()->unpackLLElements(loc, operands[i],
                                                         rewriter, types[i]);

      assert(values.size() == srcValues.size());
      for (unsigned j = 0; j < srcValues.size(); ++j) {
        srcValues[j].push_back(values[j]);
      }
    }
    return srcValues;
  }

  // Use shared memory for reduction within warps and across warps
  LogicalResult
  matchAndRewriteBasic(triton::ReduceOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    unsigned axis = op.getAxis();

    ReduceOpHelper helper(op);
    auto srcTys = op.getInputTypes();
    auto srcLayout = helper.getSrcLayout().cast<BlockedEncodingAttr>();
    auto srcOrd = srcLayout.getOrder();
    auto srcShape = helper.getSrcShape();

    SmallVector<Type> elemPtrTys(srcTys.size());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto ty = srcTys[i].getElementType();
      auto llvmElemTy = getTypeConverter()->convertType(ty);
      elemPtrTys[i] = LLVM::LLVMPointerType::get(llvmElemTy, 3);
    }
    auto llvmIndexTy = getTypeConverter()->getIndexType();
    auto indexPtrTy = LLVM::LLVMPointerType::get(llvmIndexTy, 3);

    auto smemShape = helper.getScratchConfigBasic();
    unsigned elems = product<unsigned>(smemShape);

    SmallVector<Value> smemBases(op.getNumOperands());
    smemBases[0] = bitcast(
        getSharedMemoryBase(loc, rewriter, op.getOperation()), elemPtrTys[0]);
    for (unsigned i = 1; i < op.getNumOperands(); ++i) {
      smemBases[i] =
          bitcast(gep(elemPtrTys[i - 1], smemBases[i - 1], i32_val(elems)),
                  elemPtrTys[i]);
    }

    unsigned srcElems = getElemsPerThread(srcTys[0]);
    auto srcIndices = emitIndices(loc, rewriter, srcLayout, srcTys[0]);
    auto srcValues = unpackInputs(loc, op, adaptor, rewriter);

    // Assumes offsets don't actually depend on type
    SmallVector<SmallVector<unsigned>> offset =
        emitOffsetForLayout(srcLayout, srcTys[0]);

    std::map<SmallVector<unsigned>, SmallVector<Value>> accs;
    std::map<SmallVector<unsigned>, SmallVector<Value>> indices;

    Region *combineOp = &op.getCombineOp();

    // reduce within threads
    for (unsigned i = 0; i < srcElems; ++i) {
      SmallVector<unsigned> key = offset[i];
      key[axis] = 0;
      bool isFirst = accs.find(key) == accs.end();
      accumulate(rewriter, *combineOp, accs[key], srcValues[i], isFirst);
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
      auto &acc = it.second;
      SmallVector<Value> writeIdx = indices[key];

      writeIdx[axis] = udiv(writeIdx[axis], sizePerThread);
      Value writeOffset = linearize(rewriter, loc, writeIdx, smemShape, srcOrd);
      SmallVector<Value> writePtrs(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        writePtrs[i] = gep(elemPtrTys[i], smemBases[i], writeOffset);
        store(acc[i], writePtrs[i]);
      }

      SmallVector<Value> readIdx(writeIdx.size(), ints[0]);
      for (int N = smemShape[axis] / 2; N > 0; N >>= 1) {
        readIdx[axis] = ints[N];
        Value readMask = icmp_slt(writeIdx[axis], ints[N]);
        Value readOffset = select(
            readMask, linearize(rewriter, loc, readIdx, smemShape, srcOrd),
            ints[0]);
        SmallVector<Value> readPtrs(op.getNumOperands());
        for (unsigned i = 0; i < op.getNumOperands(); ++i) {
          readPtrs[i] = gep(elemPtrTys[i], writePtrs[i], readOffset);
        }

        barrier();
        SmallVector<Value> cur(op.getNumOperands());
        for (unsigned i = 0; i < op.getNumOperands(); ++i) {
          cur[i] = load(readPtrs[i]);
        }
        accumulate(rewriter, *combineOp, acc, cur, false);
        barrier();
        for (unsigned i = 0; i < op.getNumOperands(); ++i) {
          store(acc[i], writePtrs[i]);
        }
      }
    }

    barrier();

    // set output values
    SmallVector<Value> results(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      if (auto resultTy =
              op.getResult()[i].getType().dyn_cast<RankedTensorType>()) {
        // nd-tensor where n >= 1

        auto resultLayout = resultTy.getEncoding();

        unsigned resultElems = getElemsPerThread(resultTy);
        auto resultIndices = emitIndices(loc, rewriter, resultLayout, resultTy);
        assert(resultIndices.size() == resultElems);

        SmallVector<Value> resultVals(resultElems);
        for (unsigned j = 0; j < resultElems; ++j) {
          SmallVector<Value> readIdx = resultIndices[j];
          readIdx.insert(readIdx.begin() + axis, ints[0]);
          Value readOffset =
              linearize(rewriter, loc, readIdx, smemShape, srcOrd);
          Value readPtr = gep(elemPtrTys[i], smemBases[i], readOffset);
          resultVals[j] = load(readPtr);
        }
        results[i] = getTypeConverter()->packLLElements(loc, resultVals,
                                                        rewriter, resultTy);
      } else {
        // 0d-tensor -> scalar
        results[i] = load(smemBases[i]);
      }
    }

    auto parentBlock = op.getOperation()->getBlock();
    rewriter.replaceOp(op, results);
    return success();
  }

  // Use warp shuffle for reduction within warps and shared memory for data
  // exchange across warps
  LogicalResult matchAndRewriteFast(triton::ReduceOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {

    Location loc = op->getLoc();
    unsigned axis = adaptor.getAxis();

    ReduceOpHelper helper(op);
    auto srcTys = op.getInputTypes();
    auto srcLayout = helper.getSrcLayout().cast<BlockedEncodingAttr>();
    auto srcOrd = srcLayout.getOrder();
    auto srcShape = helper.getSrcShape();

    SmallVector<Type> elemPtrTys(srcTys.size());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto ty = srcTys[i].getElementType();
      auto llvmElemTy = getTypeConverter()->convertType(ty);
      elemPtrTys[i] = LLVM::LLVMPointerType::get(llvmElemTy, 3);
    }
    auto llvmIndexTy = getTypeConverter()->getIndexType();
    auto indexPtrTy = LLVM::LLVMPointerType::get(llvmIndexTy, 3);

    auto smemShapes = helper.getScratchConfigsFast();
    unsigned elems = product<unsigned>(smemShapes[0]);
    unsigned maxElems = std::max(elems, product<unsigned>(smemShapes[1]));

    SmallVector<Value> smemBases(op.getNumOperands());
    smemBases[0] = bitcast(
        getSharedMemoryBase(loc, rewriter, op.getOperation()), elemPtrTys[0]);
    for (unsigned i = 1; i < op.getNumOperands(); ++i) {
      smemBases[i] =
          bitcast(gep(elemPtrTys[i - 1], smemBases[i - 1], i32_val(maxElems)),
                  elemPtrTys[i]);
    }

    unsigned sizeIntraWarps = helper.getIntraWarpSize();
    unsigned sizeInterWarps = helper.getInterWarpSize();

    unsigned srcElems = getElemsPerThread(srcTys[0]);
    auto srcIndices = emitIndices(loc, rewriter, srcLayout, srcTys[0]);
    auto srcValues = unpackInputs(loc, op, adaptor, rewriter);

    std::map<SmallVector<unsigned>, SmallVector<Value>> accs;
    std::map<SmallVector<unsigned>, SmallVector<Value>> indices;

    // Assumes offsets don't actually depend on type
    SmallVector<SmallVector<unsigned>> offset =
        emitOffsetForLayout(srcLayout, srcTys[0]);

    auto *combineOp = &op.getCombineOp();

    // reduce within threads
    for (unsigned i = 0; i < srcElems; ++i) {
      SmallVector<unsigned> key = offset[i];
      key[axis] = 0;
      bool isFirst = accs.find(key) == accs.end();
      accumulate(rewriter, *combineOp, accs[key], srcValues[i], isFirst);
      if (isFirst)
        indices[key] = srcIndices[i];
    }

    Value threadId = getThreadId(rewriter, loc);
    Value warpSize = i32_val(32);
    Value warpId = udiv(threadId, warpSize);
    Value laneId = urem(threadId, warpSize);

    auto threadsPerWarp = triton::gpu::getThreadsPerWarp(srcLayout);
    auto warpsPerCTA = triton::gpu::getWarpsPerCTA(srcLayout);
    auto order = getOrder(srcLayout);
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
      SmallVector<Value> acc = it.second;

      // Reduce within warps
      for (unsigned N = sizeIntraWarps / 2; N > 0; N >>= 1) {
        SmallVector<Value> shfl(op.getNumOperands());
        for (unsigned i = 0; i < op.getNumOperands(); ++i) {
          shfl[i] = shflSync(loc, rewriter, acc[i], N);
        }
        accumulate(rewriter, *combineOp, acc, shfl, false);
      }

      SmallVector<Value> writeIdx = indices[key];
      writeIdx[axis] = (sizeInterWarps == 1) ? zero : warpIdAxis;
      Value writeOffset =
          linearize(rewriter, loc, writeIdx, smemShapes[0], order);
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        Value writePtr = gep(elemPtrTys[i], smemBases[i], writeOffset);
        storeShared(rewriter, loc, writePtr, acc[i], laneZero);
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
      // FIXME(Qingyi): need predicate icmp_slt(threadId,
      // i32_val(sizeInerWarps))
      SmallVector<Value> acc(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        Value readPtr = gep(elemPtrTys[i], smemBases[i], readOffset);
        acc[i] = load(readPtr);
      }

      for (unsigned N = sizeInterWarps / 2; N > 0; N >>= 1) {
        SmallVector<Value> shfl(op.getNumOperands());
        for (unsigned i = 0; i < op.getNumOperands(); ++i) {
          shfl[i] = shflSync(loc, rewriter, acc[i], N);
        }
        accumulate(rewriter, *combineOp, acc, shfl, false);
      }

      // only the first thread in each sizeInterWarps is writing
      Value writeOffset = readOffset;
      SmallVector<Value> writePtrs(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        writePtrs[i] = gep(elemPtrTys[i], smemBases[i], writeOffset);
      }
      Value threadIsNeeded = icmp_slt(threadId, i32_val(elems));
      Value laneIdModSizeInterWarps = urem(laneId, i32_val(sizeInterWarps));
      Value laneIdModSizeInterWarpsIsZero =
          icmp_eq(laneIdModSizeInterWarps, zero);
      Value pred = and_(threadIsNeeded, laneIdModSizeInterWarpsIsZero);

      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        storeShared(rewriter, loc, writePtrs[i], acc[i], pred);
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
    SmallVector<Value> results(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      if (auto resultTy =
              op.getResult()[i].getType().dyn_cast<RankedTensorType>()) {
        // nd-tensor where n >= 1
        auto resultLayout = resultTy.getEncoding().cast<SliceEncodingAttr>();
        unsigned resultElems = getElemsPerThread(resultTy);
        auto resultIndices = emitIndices(loc, rewriter, resultLayout, resultTy);
        assert(resultIndices.size() == resultElems);

        SmallVector<Value> resultVals(resultElems);
        for (size_t j = 0; j < resultElems; ++j) {
          SmallVector<Value> readIdx = resultIndices[j];
          readIdx.insert(readIdx.begin() + axis, i32_val(0));
          Value readOffset =
              linearize(rewriter, loc, readIdx, smemShapes[0], order);
          Value readPtr = gep(elemPtrTys[i], smemBases[i], readOffset);
          resultVals[j] = load(readPtr);
        }

        results[i] = getTypeConverter()->packLLElements(loc, resultVals,
                                                        rewriter, resultTy);
      } else {
        // 0d-tensor -> scalar
        results[i] = load(smemBases[i]);
      }
    }
    rewriter.replaceOp(op, results);

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
