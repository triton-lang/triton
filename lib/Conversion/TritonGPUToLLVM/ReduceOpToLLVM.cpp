#include "ReduceOpToLLVM.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Utility.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::delinearize;
using ::mlir::LLVM::linearize;
using ::mlir::LLVM::shflSync;
using ::mlir::LLVM::storeShared;
using ::mlir::triton::gpu::getCTASplitNum;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getTotalElemsPerThread;

struct ReduceOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::ReduceOp> {
public:
  ReduceOpConversion(
      TritonGPUToLLVMTypeConverter &typeConverter, ModuleAllocation &allocation,
      ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo &indexCacheInfo,
      int computeCapability, PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::ReduceOp>(
            typeConverter, allocation, indexCacheInfo, benefit),
        computeCapability(computeCapability) {}

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // When cross-CTA reduction is implemented in the future, this assertion can
    // be removed
    assert(isReduceWithinCTA(op) &&
           "Layout optimization passes such as PlanCTAPass and "
           "RemoveLayoutConversionPass should avoid cross-CTA reduction");

    if (ReduceOpHelper(op).isFastReduction())
      return matchAndRewriteFast(op, adaptor, rewriter);
    return matchAndRewriteBasic(op, adaptor, rewriter);
  }

private:
  int computeCapability;

  bool isReduceWithinCTA(triton::ReduceOp op) const {
    auto axis = op.getAxis();
    ReduceOpHelper helper(op);
    auto srcLayout = helper.getSrcLayout();
    auto CTASplitNum = getCTASplitNum(srcLayout);
    assert(axis < CTASplitNum.size());
    return CTASplitNum[axis] == 1;
  }

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
      auto values = getTypeConverter()->unpackLLElements(loc, operands[i],
                                                         rewriter, types[i]);

      assert(values.size() == srcValues.size());
      for (unsigned j = 0; j < srcValues.size(); ++j) {
        srcValues[j].push_back(values[j]);
      }
    }
    return srcValues;
  }

  // Calculates the write index in the shared memory where we would be writing
  // the within-thread accumulations before we start doing across-threads
  // accumulations. `index` is the index of the within-thread accumulations in
  // the full tensor, whereas `writeIdx` is the mapped-to index in the shared
  // memory
  void getWriteIndexBasic(ConversionPatternRewriter &rewriter, Location loc,
                          Attribute layout, SmallVector<Value> &index,
                          SmallVector<Value> &writeIdx,
                          std::map<int, Value> &ints, unsigned originalAxis,
                          unsigned axis) const {
    if (auto sliceLayout = layout.dyn_cast<SliceEncodingAttr>()) {
      // Recover the axis in the parent layout
      auto parentAxis = axis < sliceLayout.getDim() ? axis : axis + 1;
      auto parentLayout = sliceLayout.getParent();
      getWriteIndexBasic(rewriter, loc, parentLayout, index, writeIdx, ints,
                         originalAxis, parentAxis);
      return;
    }

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
      // writeIdx[originalAxis] = index[originalAxis] / axisSizePerThread
      writeIdx[originalAxis] = udiv(index[originalAxis], axisSizePerThread);
    } else if (auto mmaLayout = layout.dyn_cast<MmaEncodingAttr>()) {
      if (!mmaLayout.isAmpere() && !mmaLayout.isHopper()) {
        llvm::report_fatal_error("Unsupported layout");
      }
      if (originalAxis == 0) {
        // Because warpTileSize = [16, 8] and threadsPerWarp = [8, 4], each 8
        // rows in smem would correspond to a warp. The mapping
        // is: (warp_index) x 8 + (row index within warp)
        writeIdx[originalAxis] = add(mul(udiv(index[originalAxis], _16), _8),
                                     urem(index[originalAxis], _8));
      } else {
        // Same as BlockedEncodingAttr case
        writeIdx[originalAxis] = udiv(index[originalAxis], axisSizePerThread);
      }
    } else {
      llvm::report_fatal_error("Unsupported layout");
    }
  }

  // Use shared memory for reduction within warps and across warps
  LogicalResult
  matchAndRewriteBasic(triton::ReduceOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    ReduceOpHelper helper(op);
    Location loc = op.getLoc();
    unsigned axis = op.getAxis();

    auto srcTys = op.getInputTypes();
    auto srcLayout = helper.getSrcLayout();
    if (!helper.isSupportedLayout()) {
      assert(false && "Unexpected srcLayout in ReduceOpConversion");
    }
    // The order of the axes for the the threads within the warp
    auto srcOrd = triton::gpu::getOrder(srcLayout);
    auto sizePerThread = triton::gpu::getSizePerThread(srcLayout);
    auto srcShape = helper.getSrcShape();

    SmallVector<Type> elemPtrTys(srcTys.size());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto ty = srcTys[i].getElementType();
      auto llvmElemTy = getTypeConverter()->convertType(ty);
      elemPtrTys[i] = LLVM::LLVMPointerType::get(llvmElemTy, 3);
    }
    auto llvmIndexTy = getTypeConverter()->getIndexType();

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

    auto srcValues = unpackInputs(loc, op, adaptor, rewriter);
    std::map<SmallVector<unsigned>, SmallVector<Value>> accs;
    std::map<SmallVector<unsigned>, SmallVector<Value>> indices;
    reduceWithinThreads(helper, srcValues, accs, indices, rewriter);

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
      auto &acc = it.second;
      // get the writeIdx at which to write in smem
      SmallVector<Value> writeIdx;
      getWriteIndexBasic(rewriter, loc, srcLayout, indices[key], writeIdx, ints,
                         axis, axis);

      // calculate the offset in smem for that writeIdx
      Value writeOffset = linearize(rewriter, loc, writeIdx, smemShape, srcOrd);
      SmallVector<Value> writePtrs(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        // Store the within-thread accumulated value into shared memory
        writePtrs[i] = gep(elemPtrTys[i], smemBases[i], writeOffset);
        store(acc[i], writePtrs[i]);
      }

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
        SmallVector<Value> readPtrs(op.getNumOperands());
        for (unsigned i = 0; i < op.getNumOperands(); ++i) {
          // The readPtr is readOffset away from writePtr
          readPtrs[i] = gep(elemPtrTys[i], writePtrs[i], readOffset);
        }

        sync(rewriter, loc, op);

        // Combine accumulator value from another thread
        SmallVector<Value> cur(op.getNumOperands());
        for (unsigned i = 0; i < op.getNumOperands(); ++i) {
          cur[i] = load(readPtrs[i]);
        }
        accumulate(rewriter, op.getCombineOp(), acc, cur, false);

        sync(rewriter, loc, op);

        // Publish our new accumulator value to shared memory
        for (unsigned i = 0; i < op.getNumOperands(); ++i) {
          store(acc[i], writePtrs[i]);
        }
      }
    }

    sync(rewriter, loc, op);

    // set output values
    SmallVector<Value> results(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      if (auto resultTy =
              op.getResult()[i].getType().dyn_cast<RankedTensorType>()) {
        // nd-tensor where n >= 1

        auto resultLayout = resultTy.getEncoding();

        unsigned resultElems = getTotalElemsPerThread(resultTy);
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
                  unsigned numLaneToReduce) const {
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
      for (unsigned i = 0; i < acc.size(); ++i) {
        shfl[i] = shflSync(loc, rewriter, acc[i], N);
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
    for (auto it : accs) {
      const SmallVector<unsigned> &key = it.first;
      SmallVector<Value> &acc = accs[key];
      warpReduce(rewriter, op.getLoc(), acc, op, sizeIntraWarps);
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

  // Return the type of the shared memory pointer for operand i.
  Type getElementPtrType(triton::ReduceOp op, int i) const {
    auto ty = op.getInputTypes()[i].getElementType();
    auto llvmElemTy = getTypeConverter()->convertType(ty);
    return LLVM::LLVMPointerType::get(llvmElemTy, 3);
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
    auto smemShapes = helper.getScratchConfigsFast();

    auto threadsPerWarp =
        triton::gpu::getThreadsPerWarpWithUniqueData(srcLayout, srcShape);
    auto warpsPerCTA =
        triton::gpu::getWarpsPerCTAWithUniqueData(srcLayout, srcShape);
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
      SmallVector<Value> &acc = it.second;

      SmallVector<Value> writeIdx = indices[key];
      writeIdx[axis] = warpIdAxis;
      Value writeOffset =
          linearize(rewriter, loc, writeIdx, smemShapes[0], order);
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto elemPtrTy = getElementPtrType(op, i);
        Value writePtr = gep(elemPtrTy, smemBases[i], writeOffset);
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
    auto smemShapes = helper.getScratchConfigsFast();
    unsigned elems = product<unsigned>(smemShapes[0]);
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
    Value readOffset = threadId;
    for (unsigned round = 0; round < elemsPerThread; ++round) {
      // FIXME(Qingyi): need predicate icmp_slt(threadId,
      // i32_val(sizeInerWarps))
      SmallVector<Value> acc(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto elemPtrTy = getElementPtrType(op, i);
        Value readPtr = gep(elemPtrTy, smemBases[i], readOffset);
        acc[i] = load(readPtr);
      }
      warpReduce(rewriter, loc, acc, op, sizeInterWarps);

      // only the first thread in each sizeInterWarps is writing
      Value writeOffset = readOffset;
      SmallVector<Value> writePtrs(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto elemPtrTy = getElementPtrType(op, i);
        writePtrs[i] = gep(elemPtrTy, smemBases[i], writeOffset);
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
  }

  // Load the final reduction from shared memory and replace the reduce result
  // with it.
  void loadReductionAndPackResult(ReduceOpHelper &helper,
                                  SmallVector<Value> &smemBases,
                                  ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    Location loc = op.getLoc();
    auto smemShapes = helper.getScratchConfigsFast();
    auto order = getOrder(helper.getSrcLayout());
    SmallVector<Value> results(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
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
              linearize(rewriter, loc, readIdx, smemShapes[0], order);
          Value readPtr =
              gep(getElementPtrType(op, i), smemBases[i], readOffset);
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
  }

  // Use warp shuffle for reduction within warps and shared memory for data
  // exchange across warps
  LogicalResult matchAndRewriteFast(triton::ReduceOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
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
    auto smemShapes = helper.getScratchConfigsFast();
    unsigned elems = product<unsigned>(smemShapes[0]);
    unsigned maxElems = std::max(elems, product<unsigned>(smemShapes[1]));
    SmallVector<Value> smemBases(op.getNumOperands());
    smemBases[0] =
        bitcast(getSharedMemoryBase(loc, rewriter, op.getOperation()),
                getElementPtrType(op, 0));
    for (unsigned i = 1; i < op.getNumOperands(); ++i) {
      smemBases[i] = bitcast(gep(getElementPtrType(op, i - 1), smemBases[i - 1],
                                 i32_val(maxElems)),
                             getElementPtrType(op, i));
    }
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
    loadReductionAndPackResult(helper, smemBases, rewriter);

    return success();
  }
};

void populateReduceOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    ModuleAllocation &allocation,
    ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo &indexCacheInfo,
    int computeCapability, PatternBenefit benefit) {
  patterns.add<ReduceOpConversion>(typeConverter, allocation, indexCacheInfo,
                                   computeCapability, benefit);
}
