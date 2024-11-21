#include "ReduceScanCommon.h"
#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::delinearize;
using ::mlir::LLVM::linearize;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getThreadOrder;
using ::mlir::triton::gpu::getTotalElemsPerThread;

namespace {
struct ReduceOpConversion
    : public ConvertTritonGPUReduceScanToLLVMPattern<triton::ReduceOp> {
public:
  ReduceOpConversion(LLVMTypeConverter &typeConverter,
                     const TargetInfoBase &targetInfo, PatternBenefit benefit)
      : ConvertTritonGPUReduceScanToLLVMPattern<triton::ReduceOp>(typeConverter,
                                                                  benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ReduceOpHelper helper(op);
    assert(helper.isSupportedLayout() &&
           "Unexpected srcLayout in ReduceOpConversion");
    Location loc = op->getLoc();

    auto srcValues = unpackInputs(loc, op, adaptor, rewriter);
    std::map<SmallVector<unsigned>, SmallVector<Value>> accs;
    std::map<SmallVector<unsigned>, SmallVector<Value>> indices;
    // First reduce all the values along axis within each thread.
    reduceWithinThreads(helper, srcValues, accs, indices, rewriter);

    // Then reduce across threads within a warp.
    reduceWithinWarps(helper, accs, rewriter);

    if (helper.isWarpSynchronous()) {
      // If all the values to be reduced are within the same warp there is
      // nothing left to do.
      packResults(helper, accs, rewriter);
      return success();
    }

    // Compute a shared memory base per operand.
    auto smemShape = helper.getScratchRepShape();

    SmallVector<Value> smemBases =
        getSmemBases(op, product<unsigned>(smemShape), rewriter, targetInfo);

    storeWarpReduceToSharedMemory(helper, accs, indices, smemBases, rewriter);

    sync(rewriter, loc, op);

    // The second round of shuffle reduction
    //   now the problem size: sizeInterWarps, s1, s2, .. , sn
    //   where sizeInterWarps is 2^m
    //
    // Each thread needs to process:
    //   elemsPerThread = sizeInterWarps * s1 * s2 .. Sn / numThreads
    accumulatePartialReductions(helper, smemBases, rewriter);

    // We could avoid this barrier in some of the layouts, however this is not
    // the general case.
    // TODO: optimize the barrier in case the layouts are accepted.
    sync(rewriter, loc, op);

    // set output values
    loadReductionAndPackResult(helper, smemShape, smemBases, rewriter);

    return success();
  }

private:
  const TargetInfoBase &targetInfo;

  void accumulate(Location loc, ConversionPatternRewriter &rewriter,
                  Region &combineOp, SmallVector<Value> &acc, ValueRange cur,
                  Value pred = {}) const {
    auto results = applyCombineOp(loc, rewriter, combineOp, acc, cur, pred);
    if (acc.size() < results.size()) {
      acc.resize(results.size());
    }
    for (unsigned i = 0; i < acc.size(); ++i) {
      acc[i] = results[i];
    }
  }

  SmallVector<SmallVector<Value>>
  unpackInputs(Location loc, triton::ReduceOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const {
    auto types = op.getInputTypes();
    auto operands = adaptor.getOperands();
    unsigned srcElems = getTotalElemsPerThread(types[0]);
    SmallVector<SmallVector<Value>> srcValues(srcElems);
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto values = unpackLLElements(loc, operands[i], rewriter);

      assert(values.size() == srcValues.size());
      for (unsigned j = 0; j < srcValues.size(); ++j) {
        srcValues[j].push_back(values[j]);
      }
    }
    return srcValues;
  }

  void sync(ConversionPatternRewriter &rewriter, Location loc,
            triton::ReduceOp op) const {
    barrier();
  }

  // Reduce along op axis for elements that are in the same thread. The
  // accumulated value is stored in accs.
  void reduceWithinThreads(
      ReduceOpHelper &helper, SmallVector<SmallVector<Value>> &srcValues,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &indices,
      ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    RankedTensorType operandType = op.getInputTypes()[0];
    // Assumes offsets don't actually depend on type
    SmallVector<SmallVector<unsigned>> offsets =
        emitOffsetForLayout(helper.getSrcLayout(), operandType);

    // Thread X might hold the same input value in two registers.  Get the
    // indices in `offsets` that hold unique values, and only accumualte over
    // those.
    llvm::MapVector<ArrayRef<unsigned>, int> uniqueOffsets;
    for (int i = 0; i < offsets.size(); ++i) {
      uniqueOffsets.insert({offsets[i], i});
    }

    unsigned srcElems = getTotalElemsPerThread(operandType);
    auto *combineOp = &op.getCombineOp();
    auto srcIndices = emitIndices(op.getLoc(), rewriter, targetInfo,
                                  helper.getSrcLayout(), operandType, true);
    // reduce within threads
    for (const auto &[_, i] : uniqueOffsets) {
      SmallVector<unsigned> key = offsets[i];
      key[op.getAxis()] = 0;
      bool isFirst = accs.find(key) == accs.end();
      accumulate(op.getLoc(), rewriter, *combineOp, accs[key], srcValues[i]);
      if (isFirst)
        indices[key] = srcIndices[i];
    }
  }

  // Apply warp reduction across the given number of contiguous lanes using op
  // region and the accumulator values as source.
  void warpReduce(ConversionPatternRewriter &rewriter, Location loc,
                  SmallVector<Value> &acc, triton::ReduceOp op,
                  unsigned numLaneToReduce, unsigned interleave,
                  Value pred = {}) const {
    auto success = targetInfo.warpReduce(rewriter, loc, acc, op,
                                         numLaneToReduce, interleave);
    if (success)
      return;

    auto mod = op->getParentOfType<ModuleOp>();
    unsigned iWarpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);

    for (unsigned N = numLaneToReduce / 2; N > 0; N >>= 1) {
      SmallVector<Value> shfl(acc.size());
      for (unsigned i = 0; i < acc.size(); ++i) {
        shfl[i] = targetInfo.shuffleXor(rewriter, loc, acc[i], N * interleave);
      }
      accumulate(op.getLoc(), rewriter, op.getCombineOp(), acc, shfl, pred);
    }
  }

  // Reduce across threads within each warp.
  void
  reduceWithinWarps(ReduceOpHelper &helper,
                    std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
                    ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    unsigned sizeIntraWarps = helper.getIntraWarpSizeWithUniqueData();
    unsigned threadOffsetOnReductionAxis =
        helper.getThreadOffsetOnReductionAxis();
    for (auto it : accs) {
      const SmallVector<unsigned> &key = it.first;
      SmallVector<Value> &acc = accs[key];
      warpReduce(rewriter, op.getLoc(), acc, op, sizeIntraWarps,
                 threadOffsetOnReductionAxis);
    }
  }

  // Pack the accumulator values and replace the reduce op with the result.
  void packResults(ReduceOpHelper &helper,
                   std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
                   ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    Location loc = op.getLoc();
    unsigned axis = op.getAxis();
    SmallVector<Value> results(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      if (auto resultTy =
              dyn_cast<RankedTensorType>(op.getResult()[i].getType())) {
        auto resultLayout = cast<SliceEncodingAttr>(resultTy.getEncoding());
        unsigned resultElems = getTotalElemsPerThread(resultTy);
        SmallVector<SmallVector<unsigned>> resultOffset =
            emitOffsetForLayout(resultLayout, resultTy);
        SmallVector<Value> resultVals;
        for (int j = 0; j < resultElems; j++) {
          auto key = resultOffset[j];
          key.insert(key.begin() + axis, 0);
          resultVals.push_back(accs[key][i]);
        }
        results[i] = packLLElements(loc, getTypeConverter(), resultVals,
                                    rewriter, resultTy);
      } else
        results[i] = accs.begin()->second[i];
    }
    rewriter.replaceOp(op, results);
  }

  // For slice layout some ids are duplicated on multiple lanes, so we need to
  // handle the delinearization of laneId in a special way. We need to
  // generalize this part of the logic to work on any kind of linear layout
  // uniformely.
  SmallVector<Value>
  getMultiDimLaneId(ReduceOpHelper &helper, Value &laneId, Location &loc,
                    ConversionPatternRewriter &rewriter) const {
    auto srcLayout = helper.getSrcLayout();
    auto srcShape = helper.getSrcShape();
    auto order = triton::gpu::getThreadOrder(srcLayout);
    SmallVector<Value> multiDimLaneId;

    if (auto sliceLayout = mlir::dyn_cast<SliceEncodingAttr>(srcLayout)) {
      auto parentLayout = sliceLayout.getParent();
      SmallVector<unsigned> dims = {sliceLayout.getDim()};
      while (auto parentSliceLayout =
                 mlir::dyn_cast<SliceEncodingAttr>(parentLayout)) {
        dims.push_back(parentSliceLayout.getDim());
        parentLayout = parentSliceLayout.getParent();
      }

      auto parentThreadsPerWarps = triton::gpu::getThreadsPerWarp(parentLayout);
      auto parentOrder = triton::gpu::getThreadOrder(parentLayout);
      multiDimLaneId = delinearize(rewriter, loc, laneId, parentThreadsPerWarps,
                                   parentOrder);
      for (unsigned dim : llvm::reverse(dims)) {
        multiDimLaneId.erase(multiDimLaneId.begin() + dim);
      }
    } else {
      SmallVector<unsigned> threadsPerWarps =
          triton::gpu::getThreadsPerWarp(srcLayout);
      threadsPerWarps[helper.getAxis()] =
          triton::gpu::getThreadsPerWarpWithUniqueData(
              srcLayout, srcShape)[helper.getAxis()];
      multiDimLaneId =
          delinearize(rewriter, loc, laneId, threadsPerWarps, order);
    }
    return multiDimLaneId;
  }

  SmallVector<Value>
  getMultiDimWarpId(ReduceOpHelper &helper, Value &warpId, Location &loc,
                    ConversionPatternRewriter &rewriter) const {
    auto srcLayout = helper.getSrcLayout();
    auto srcShape = helper.getSrcShape();
    auto order = triton::gpu::getWarpOrder(srcLayout);
    SmallVector<Value> multiDimWarpId;

    // 2x2 warps with slice dim = 0, warpId = 2 ends up writing at the same
    // address as warpId = 0 since the warpsPerCTA is [1, 2], need to figure out
    // a way to properly delinearize warpId in the slice case
    if (auto sliceLayout = mlir::dyn_cast<SliceEncodingAttr>(srcLayout)) {
      auto parentLayout = sliceLayout.getParent();
      SmallVector<unsigned> dims = {sliceLayout.getDim()};
      while (auto parentSliceLayout =
                 mlir::dyn_cast<SliceEncodingAttr>(parentLayout)) {
        dims.push_back(parentSliceLayout.getDim());
        parentLayout = parentSliceLayout.getParent();
      }

      auto parentWarpsPerCTA = triton::gpu::getWarpsPerCTA(parentLayout);
      auto parentOrder = triton::gpu::getWarpOrder(parentLayout);
      multiDimWarpId =
          delinearize(rewriter, loc, warpId, parentWarpsPerCTA, parentOrder);
      for (unsigned dim : llvm::reverse(dims)) {
        multiDimWarpId.erase(multiDimWarpId.begin() + dim);
      }
    } else {
      SmallVector<unsigned> warpsPerCTA =
          triton::gpu::getWarpsPerCTA(srcLayout);
      warpsPerCTA[helper.getAxis()] = triton::gpu::getWarpsPerCTAWithUniqueData(
          srcLayout, srcShape)[helper.getAxis()];
      multiDimWarpId = delinearize(rewriter, loc, warpId, warpsPerCTA, order);
    }
    return multiDimWarpId;
  }

  void storeWarpReduceToSharedMemory(
      ReduceOpHelper &helper,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &indices,
      SmallVector<Value> &smemBases,
      ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    Location loc = op.getLoc();
    Value threadId = getThreadId(rewriter, loc);
    auto srcLayout = helper.getSrcLayout();
    Value warpSize = i32_val(triton::gpu::getWarpSize(srcLayout));
    Value warpId = udiv(threadId, warpSize);
    Value laneId = urem(threadId, warpSize);
    auto srcShape = helper.getSrcShape();
    unsigned axis = op.getAxis();
    auto smemShape = helper.getScratchRepShape();

    SmallVector<Value> multiDimLaneId =
        getMultiDimLaneId(helper, laneId, loc, rewriter);
    Value laneIdAxis = multiDimLaneId[axis];
    Value zero = i32_val(0);
    Value laneZero = icmp_eq(laneIdAxis, zero);

    SmallVector<Value> multiDimWarpId =
        getMultiDimWarpId(helper, warpId, loc, rewriter);
    Value warpIdAxis = multiDimWarpId[axis];

    auto smemOrder = helper.getOrderWithAxisAtBeginning();
    for (auto it : accs) {
      const SmallVector<unsigned> &key = it.first;
      SmallVector<Value> &acc = it.second;

      SmallVector<Value> writeIdx = indices[key];
      writeIdx[axis] = warpIdAxis;
      Value writeOffset =
          linearize(rewriter, loc, writeIdx, smemShape, smemOrder);
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto elemTy = getElementType(op, i);
        Value writePtr =
            gep(smemBases[i].getType(), elemTy, smemBases[i], writeOffset);
        targetInfo.storeShared(rewriter, loc, writePtr, acc[i], laneZero);
      }
    }
  }

  // Load the reduction of each warp and accumulate them to a final value and
  // store back to shared memory.
  void accumulatePartialReductions(ReduceOpHelper &helper,
                                   SmallVector<Value> &smemBases,
                                   ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    auto srcLayout = helper.getSrcLayout();
    auto smemShape = helper.getScratchRepShape();
    unsigned elems = product<unsigned>(smemShape);
    unsigned sizeInterWarps = helper.getInterWarpSizeWithUniqueData();
    Location loc = op.getLoc();

    Value threadId = getThreadId(rewriter, loc);
    Value warpSize = i32_val(triton::gpu::getWarpSize(srcLayout));
    Value laneId = urem(threadId, warpSize);
    Value zero = i32_val(0);

    auto mod = op.getOperation()->getParentOfType<ModuleOp>();
    unsigned numThreads =
        product<unsigned>(triton::gpu::getWarpsPerCTA(srcLayout)) *
        triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    unsigned elemsPerThread = std::max<unsigned>(elems / numThreads, 1);
    Value threadIsNeeded = icmp_slt(threadId, i32_val(elems));
    Value readOffset = threadId;
    for (unsigned round = 0; round < elemsPerThread; ++round) {
      SmallVector<Value> acc(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto elemTy = getElementType(op, i);
        Value readPtr =
            gep(smemBases[i].getType(), elemTy, smemBases[i], readOffset);
        acc[i] = targetInfo.loadShared(rewriter, loc, readPtr, elemTy,
                                       threadIsNeeded);
      }
      warpReduce(rewriter, loc, acc, op, sizeInterWarps, 1 /* interleave */,
                 threadIsNeeded);
      // only the first thread in each sizeInterWarps is writing
      Value writeOffset = readOffset;
      SmallVector<Value> writePtrs(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto elemTy = getElementType(op, i);
        writePtrs[i] =
            gep(smemBases[i].getType(), elemTy, smemBases[i], writeOffset);
      }

      Value laneIdModSizeInterWarps = urem(laneId, i32_val(sizeInterWarps));
      Value laneIdModSizeInterWarpsIsZero =
          icmp_eq(laneIdModSizeInterWarps, zero);
      Value pred = and_(threadIsNeeded, laneIdModSizeInterWarpsIsZero);

      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        targetInfo.storeShared(rewriter, loc, writePtrs[i], acc[i], pred);
      }

      if (round != elemsPerThread - 1) {
        readOffset = add(readOffset, i32_val(numThreads));
      }
    }
  }

  // Load the final reduction from shared memory and replace the reduce result
  // with it.
  void loadReductionAndPackResult(ReduceOpHelper &helper,
                                  SmallVector<unsigned> smemShape,
                                  SmallVector<Value> &smemBases,
                                  ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    Location loc = op.getLoc();
    auto srcLayout = helper.getSrcLayout();
    auto axis = op.getAxis();
    auto smemOrder = helper.getOrderWithAxisAtBeginning();
    SmallVector<Value> results(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto elemTy = getElementType(op, i);
      if (auto resultTy =
              dyn_cast<RankedTensorType>(op.getResult()[i].getType())) {
        // nd-tensor where n >= 1
        auto resultLayout = cast<SliceEncodingAttr>(resultTy.getEncoding());
        unsigned resultElems = getTotalElemsPerThread(resultTy);
        auto resultIndices = emitIndices(loc, rewriter, targetInfo,
                                         resultLayout, resultTy, true);
        auto resultShape = resultTy.getShape();
        auto resultCTATile = getShapePerCTATile(resultLayout);
        assert(resultIndices.size() == resultElems);

        SmallVector<Value> resultVals(resultElems);
        for (size_t j = 0; j < resultElems; ++j) {
          SmallVector<Value> readIdx = resultIndices[j];
          readIdx.insert(readIdx.begin() + op.getAxis(), i32_val(0));
          for (size_t resultIdx = 0, resultDim = resultShape.size();
               resultIdx < resultDim; ++resultIdx) {
            auto smemIdx = resultIdx < op.getAxis() ? resultIdx : resultIdx + 1;
            if (resultCTATile[resultIdx] > smemShape[smemIdx] ||
                resultShape[resultIdx] > smemShape[smemIdx]) {
              // When srcShape smaller then src sizePerThread, only srcShape
              // elements is accumulated in smem. Modulo smemShape effectively
              // replicates srcShape elements to src sizePerThread.
              readIdx[smemIdx] =
                  urem(readIdx[smemIdx], i32_val(smemShape[smemIdx]));
            }
          }
          Value readOffset =
              linearize(rewriter, loc, readIdx, smemShape, smemOrder);
          Value readPtr =
              gep(smemBases[i].getType(), elemTy, smemBases[i], readOffset);
          resultVals[j] = load(elemTy, readPtr);
        }

        results[i] = packLLElements(loc, getTypeConverter(), resultVals,
                                    rewriter, resultTy);
      } else {
        // 0d-tensor -> scalar
        results[i] = load(elemTy, smemBases[i]);
      }
    }
    rewriter.replaceOp(op, results);
  }
};
} // namespace

void mlir::triton::populateReduceOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<ReduceOpConversion>(typeConverter, targetInfo, benefit);
}
