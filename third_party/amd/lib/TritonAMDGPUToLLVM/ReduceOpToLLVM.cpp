#include "ReduceOpToLLVM.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Utility.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::delinearize;
using ::mlir::LLVM::linearize;
using ::mlir::LLVM::AMD::loadShared;
using ::mlir::LLVM::AMD::shflSync;
using ::mlir::LLVM::AMD::storeShared;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::AMD::TritonGPUToLLVMTypeConverter;
using ::AMD::ConvertTritonGPUOpToLLVMPatternBase;
using ::AMD::ConvertTritonGPUOpToLLVMPattern;

namespace AMD{
namespace {
struct ReduceOpConversion
    : public ConvertTritonGPUReduceScanToLLVMPattern<triton::ReduceOp> {
public:
  ReduceOpConversion(
      TritonGPUToLLVMTypeConverter &typeConverter, ModuleAllocation &allocation,
      ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo &indexCacheInfo,
      int computeCapability, PatternBenefit benefit)
      : ConvertTritonGPUReduceScanToLLVMPattern<triton::ReduceOp>(
            typeConverter, allocation, indexCacheInfo, benefit),
        computeCapability(computeCapability) {}

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
    auto smemShape = helper.getScratchConfig();

    SmallVector<Value> smemBases =
        getSmemBases(op, product<unsigned>(smemShape), rewriter);

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
  int computeCapability;

  void accumulate(ConversionPatternRewriter &rewriter, Region &combineOp,
                  SmallVector<Value> &acc, ValueRange cur, bool isFirst) const {
    if (isFirst) {
      acc = SmallVector<Value>(cur.begin(), cur.end());
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

    rewriter.inlineBlockBefore(&newReduce, &*rewriter.getInsertionPoint(),
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
    unsigned srcElems = getTotalElemsPerThread(types[0]);
    SmallVector<SmallVector<Value>> srcValues(srcElems);
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto values = getTypeConverter()->unpackLLElements(loc, operands[i], rewriter);

      assert(values.size() == srcValues.size());
      for (unsigned j = 0; j < srcValues.size(); ++j) {
        srcValues[j].push_back(values[j]);
      }
    }
    return srcValues;
  }

  void sync(ConversionPatternRewriter &rewriter, Location loc,
            triton::ReduceOp op) const {
    // TODO[shuhaoj]: change hard code style of numThreads. Hide async_agent
    // attr.
    if (getWSAgentId(op)) {
      barSync(rewriter, op, getAgentIds(op).front(), 128);
    } else {
      barrier();
    }
  }

  // Check if the reduction can use a redux op and return the kind.
  std::optional<NVVM::ReduxKind> matchReduxKind(triton::ReduceOp op) const {
    #ifdef USE_ROCM
      return std::nullopt;
    #endif
    if (computeCapability < 80)
      return std::nullopt;
    if (op.getNumOperands() != 1 || op.getNumResults() != 1)
      return std::nullopt;
    Block *block = &(*op.getCombineOp().begin());
    Operation *yield = block->getTerminator();
    Operation *reduceOp = yield->getOperand(0).getDefiningOp();
    if (!reduceOp || reduceOp->getNumOperands() != 2 ||
        reduceOp->getNumResults() != 1)
      return std::nullopt;
    auto intType = reduceOp->getResultTypes()[0].dyn_cast<IntegerType>();
    if (!intType || intType.getWidth() > 32)
      return std::nullopt;
    if (reduceOp->getOperand(0) != block->getArgument(0) ||
        reduceOp->getOperand(1) != block->getArgument(1))
      return std::nullopt;
    if (isa<arith::AddIOp>(reduceOp))
      return NVVM::ReduxKind::ADD;
    if (isa<arith::AndIOp>(reduceOp))
      return NVVM::ReduxKind::AND;
    if (isa<arith::OrIOp>(reduceOp))
      return NVVM::ReduxKind::OR;
    if (isa<arith::XOrIOp>(reduceOp))
      return NVVM::ReduxKind::XOR;
    if (isa<arith::MinSIOp>(reduceOp))
      return NVVM::ReduxKind::MIN;
    if (isa<arith::MinUIOp>(reduceOp))
      return NVVM::ReduxKind::UMIN;
    if (isa<arith::MaxSIOp>(reduceOp))
      return NVVM::ReduxKind::MAX;
    if (isa<arith::MaxUIOp>(reduceOp))
      return NVVM::ReduxKind::UMAX;
    return std::nullopt;
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
    SmallVector<SmallVector<unsigned>> offset =
        emitOffsetForLayout(helper.getSrcLayout(), operandType);
    unsigned srcElems = getTotalElemsPerThread(operandType);
    auto *combineOp = &op.getCombineOp();
    auto srcIndices =
        emitIndices(op.getLoc(), rewriter, helper.getSrcLayout(), operandType);
    // reduce within threads
    for (unsigned i = 0; i < srcElems; ++i) {
      SmallVector<unsigned> key = offset[i];
      key[op.getAxis()] = 0;
      bool isFirst = accs.find(key) == accs.end();
      accumulate(rewriter, *combineOp, accs[key], srcValues[i], isFirst);
      if (isFirst)
        indices[key] = srcIndices[i];
    }
  }

  // Apply warp reduction across the given number of contiguous lanes using op
  // region and the accumulator values as source.
  void warpReduce(ConversionPatternRewriter &rewriter, Location loc,
                  SmallVector<Value> &acc, triton::ReduceOp op,
                  unsigned numLaneToReduce, unsigned interleave) const {
    if (auto kind = matchReduxKind(op)) {
      // Based on benchmarking on A100 redux op gives a speed up only when doing
      // a single reduction (not partioned) and when the mask is static.
      // Therefore we currently only enable it to reduce across all the lanes.
      if (numLaneToReduce == 32) {
        assert(acc.size() == 1);
        Value mask = i32_val(0xFFFFFFFF);
        // Even though we currently don't use redux for partitioned reduction
        // the code below supports it in case we want to tweak the heuristic.
        if (numLaneToReduce < 32) {
          // For partitioned reduction we need to caluclate the mask so that
          // each group of numLaneToReduce threads has the correct mask.
          unsigned bitmask = (1 << numLaneToReduce) - 1;
          Value threadId = getThreadId(rewriter, loc);
          Value laneId = urem(threadId, i32_val(32));
          mask = shl(i32_val(bitmask),
                     and_(laneId, i32_val(~(numLaneToReduce - 1))));
        }
        for (unsigned i = 0; i < acc.size(); ++i) {
          unsigned bitwidth = acc[i].getType().cast<IntegerType>().getWidth();
          if (bitwidth < 32) {
            if (*kind == NVVM::ReduxKind::MIN || *kind == NVVM::ReduxKind::MAX)
              acc[i] = sext(i32_ty, acc[i]);
            else
              acc[i] = zext(i32_ty, acc[i]);
          }
          acc[i] = rewriter.create<NVVM::ReduxOp>(loc, acc[i].getType(), acc[0],
                                                  *kind, mask);
          if (bitwidth < 32)
            acc[i] = trunc(int_ty(bitwidth), acc[i]);
        }
        return;
      }
    }

    for (unsigned N = numLaneToReduce / 2; N > 0; N >>= 1) {
      SmallVector<Value> shfl(acc.size());
      unsigned shuffleIdx = N;
#ifdef USE_ROCM
      auto srcTys = op.getInputTypes();
      auto inputTy = srcTys[0].cast<RankedTensorType>();
      auto inMfma =
        inputTy.getEncoding().dyn_cast<triton::gpu::MfmaEncodingAttr>();
      if (inMfma && inMfma.getIsTransposed()) {
        assert(numLaneToReduce == 2 || numLaneToReduce == 4);
        // for mfma 32x32 adjacent threads in y dimension in transposed MFMA
        // layout are 32 apart: [[0 0 0 0 32 32 32 32 ...] [1 1 1 1 33 33 33 33
        // ...] ...]. for mfma 16x16 adjacent threads in y dimension in
        // transposed MFMA layout are 16 apart: [[0 0 0 0 16 16 16 16 32 32 32
        // 32 ...] [1 1 1 1 33 33 33 33 ...] ...].
        const int warpSize = 64;
        shuffleIdx = warpSize / N / 2;
      }
#endif
      for (unsigned i = 0; i < acc.size(); ++i) {
        shfl[i] = shflSync(loc, rewriter, acc[i], shuffleIdx * interleave);
      }
      accumulate(rewriter, op.getCombineOp(), acc, shfl, false);
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

  // Pack the accumualtor values and replace the reduce op with the result.
  void packResults(ReduceOpHelper &helper,
                   std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
                   ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    Location loc = op.getLoc();
    unsigned axis = op.getAxis();
    SmallVector<Value> results(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      if (auto resultTy =
              op.getResult()[i].getType().dyn_cast<RankedTensorType>()) {
        auto resultLayout = resultTy.getEncoding().cast<SliceEncodingAttr>();
        unsigned resultElems = getTotalElemsPerThread(resultTy);
        SmallVector<SmallVector<unsigned>> resultOffset =
            emitOffsetForLayout(resultLayout, resultTy);
        SmallVector<Value> resultVals;
        for (int j = 0; j < resultElems; j++) {
          auto key = resultOffset[j];
          key.insert(key.begin() + axis, 0);
          resultVals.push_back(accs[key][i]);
        }
        results[i] = getTypeConverter()->packLLElements(loc, resultVals,
                                                        rewriter, resultTy);
      } else
        results[i] = accs.begin()->second[i];
    }
    rewriter.replaceOp(op, results);
  }

  SmallVector<Value>
  getMultiDimWarpId(ReduceOpHelper &helper, Value &warpId, Location &loc,
                    ConversionPatternRewriter &rewriter) const {
    auto srcLayout = helper.getSrcLayout();
    auto srcShape = helper.getSrcShape();
    auto order = getOrder(srcLayout);
    SmallVector<Value> multiDimWarpId;

    // 2x2 warps with slice dim = 0, warpId = 2 ends up writing at the same
    // address as warpId = 0 since the warpsPerCTA is [1, 2], need to figure out
    // a way to properly delinearize warpId in the slice case
    if (auto sliceLayout = srcLayout.dyn_cast<SliceEncodingAttr>()) {
      auto parentLayout = sliceLayout.getParent();
      auto parentWarpsPerCTA = triton::gpu::getWarpsPerCTA(parentLayout);
      auto parentOrder = triton::gpu::getOrder(parentLayout);
      multiDimWarpId =
          delinearize(rewriter, loc, warpId, parentWarpsPerCTA, parentOrder);
      multiDimWarpId.erase(multiDimWarpId.begin() + sliceLayout.getDim());
    } else {
      auto warpsPerCTA =
          triton::gpu::getWarpsPerCTAWithUniqueData(srcLayout, srcShape);
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
    Value warpSize = i32_val(32);
    Value warpId = udiv(threadId, warpSize);
    Value laneId = urem(threadId, warpSize);
    auto srcLayout = helper.getSrcLayout();
    auto srcShape = helper.getSrcShape();
    unsigned axis = op.getAxis();
    auto smemShape = helper.getScratchConfig();

    auto threadsPerWarp =
        triton::gpu::getThreadsPerWarpWithUniqueData(srcLayout, srcShape);
    auto order = getOrder(srcLayout);
    SmallVector<Value> multiDimLaneId =
        delinearize(rewriter, loc, laneId, threadsPerWarp, order);
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
        Value writePtr = gep(ptr_ty(rewriter.getContext(), 3), elemTy,
                             smemBases[i], writeOffset);
        storeShared(rewriter, loc, writePtr, acc[i], laneZero);
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
    auto smemShape = helper.getScratchConfig();
    unsigned elems = product<unsigned>(smemShape);
    unsigned sizeInterWarps = helper.getInterWarpSizeWithUniqueData();
    Location loc = op.getLoc();

    Value threadId = getThreadId(rewriter, loc);
    Value warpSize = i32_val(32);
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
        Value readPtr = gep(ptr_ty(rewriter.getContext(), 3), elemTy,
                            smemBases[i], readOffset);
        acc[i] = loadShared(rewriter, loc, readPtr, elemTy, threadIsNeeded);
      }
      warpReduce(rewriter, loc, acc, op, sizeInterWarps, 1 /* interleave */);
      // only the first thread in each sizeInterWarps is writing
      Value writeOffset = readOffset;
      SmallVector<Value> writePtrs(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto elemTy = getElementType(op, i);
        writePtrs[i] = gep(ptr_ty(rewriter.getContext(), 3), elemTy,
                           smemBases[i], writeOffset);
      }

      Value laneIdModSizeInterWarps = urem(laneId, i32_val(sizeInterWarps));
      Value laneIdModSizeInterWarpsIsZero =
          icmp_eq(laneIdModSizeInterWarps, zero);
      Value pred = and_(threadIsNeeded, laneIdModSizeInterWarpsIsZero);
      unsigned wavefront_size = triton::gpu::getWarpSize(srcLayout);

      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
#if USE_ROCM
        // This barrier is known to be critical for Navi 2x/3x
        if (i > 0 && wavefront_size == 32) {
            GCNBuilder BuilderMemfenceLDS;
            BuilderMemfenceLDS.create<>("s_waitcnt lgkmcnt(0)")->operator()();
            BuilderMemfenceLDS.launch(rewriter, loc, void_ty(rewriter.getContext()));
        }
#endif
        storeShared(rewriter, loc, writePtrs[i], acc[i], pred);
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
              op.getResult()[i].getType().dyn_cast<RankedTensorType>()) {
        // nd-tensor where n >= 1
        auto resultLayout = resultTy.getEncoding().cast<SliceEncodingAttr>();
        unsigned resultElems = getTotalElemsPerThread(resultTy);
        auto resultIndices = emitIndices(loc, rewriter, resultLayout, resultTy);
        assert(resultIndices.size() == resultElems);

        SmallVector<Value> resultVals(resultElems);
        for (size_t j = 0; j < resultElems; ++j) {
          SmallVector<Value> readIdx = resultIndices[j];
          readIdx.insert(readIdx.begin() + op.getAxis(), i32_val(0));
          Value readOffset =
              linearize(rewriter, loc, readIdx, smemShape, smemOrder);
          Value readPtr = gep(ptr_ty(rewriter.getContext(), 3), elemTy,
                              smemBases[i], readOffset);
          resultVals[j] = load(elemTy, readPtr);
        }

        results[i] = getTypeConverter()->packLLElements(loc, resultVals,
                                                        rewriter, resultTy);
      } else {
        // 0d-tensor -> scalar
        results[i] = load(elemTy, smemBases[i]);
      }
    }
    rewriter.replaceOp(op, results);
  }
};
}

void populateReduceOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    ModuleAllocation &allocation,
    ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo &indexCacheInfo,
    int computeCapability, PatternBenefit benefit) {
  patterns.add<ReduceOpConversion>(typeConverter, allocation, indexCacheInfo,
                                   computeCapability, benefit);
}
}
