//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "xpu/lib/Conversion/TritonXPUToLLVM/PatternTritonXPUOpToLLVM.h"
#include "llvm/ADT/TypeSwitch.h"
#include <cfloat>
namespace {

using ::mlir::triton::gpu::getTotalElemsPerThread;

struct XPUReduceOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::ReduceOp> {

  XPUReduceOpConversion(LLVMTypeConverter &converter,
                        const xpu::TargetInfo &targetInfo,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::ReduceOp>(converter, benefit),
        targetInfo(targetInfo) {}

  inline SmallVector<SmallVector<Value>>
  emitIndices(Location loc, RewriterBase &rewriter,
              const TargetInfoBase &target, Attribute layout,
              RankedTensorType type, const Value &loopIdx) const {
    SmallVector<int64_t> shape(type.getShape());

    // for sliceEncoding, we need to set layout as its parent
    if (auto slice = dyn_cast<SliceEncodingAttr>(layout)) {
      layout = slice.getParent();
      shape.insert(shape.begin() + slice.getDim(), 1);
    }

    auto clusterLayout = mlir::cast<triton::xpu::ClusterLayoutAttr>(layout);
    unsigned rank = shape.size();
    unsigned elemsPerCore = clusterLayout.getTotalElemsPerThread(shape, type);
    SmallVector<SmallVector<Value>> indices(elemsPerCore,
                                            SmallVector<Value>(rank));

    // const int nthreads = core_num() * cluster_num();
    // const int tid = cluster_id() * core_num() + core_id();
    // for (int i = 0; i < iterCount; ++i) {
    //     const int idx = tid + nthreads * i;
    //     const int indice = idx * buf_len;
    Value coreNum = mlir::LLVM::XPU::getBlockDim(rewriter, loc);
    Value coreId = getThreadId(rewriter, loc);
    Value bufLen = i32_val(elemsPerCore);
    Value base = mul(add(coreId, mul(loopIdx, coreNum)), bufLen);

    for (unsigned n = 0; n < elemsPerCore; ++n) {
      for (unsigned k = 0; k < rank; ++k) {
        indices[n][k] = add(base, idx_val(n));
      }
    }
    return indices;
  }

  // Return the pointee type of the shared memory pointer for operand i.
  Type getElementType(triton::xpu::ReduceOp op, int i) const {
    auto ty = getElementTypeOrSelf(op.getInputTypes()[i].getElementType());
    return getTypeConverter()->convertType(ty);
  }

  // Helper to compute the smem bases in both reductions and scans
  std::pair<SmallVector<Value>, SmallVector<Value>>
  getSmemBases(triton::xpu::ReduceOp op, unsigned elems,
               ConversionPatternRewriter &rewriter) const {
    ReduceOpHelper helper(op);
    SmallVector<int64_t> offsets;
    // auto curIdx = helper.getReduceId();
    auto prevSMOffset =
        helper.getReduceId() == 0
            ? 0
            : helper.getSMOffsets(helper.getReduceId() - 1)->endOffset;
    // op->dump();
    // LLVM_DEBUG(llvm::dbgs() << "\nprevSMOffset = " << prevSMOffset << "\n");

    auto loc = op.getLoc();
    // indices will store the index of the op operands in descending order
    // of their bitwidths
    std::vector<unsigned> indices(op.getNumOperands() - 1); // skip loopIndex
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](unsigned i, unsigned j) {
      if (i == op.getNumOperands() - 1 ||
          j == op.getNumOperands() - 1) { // skip loopIndex
        return false;
      }
      return op.getElementTypes()[i].getIntOrFloatBitWidth() >
             op.getElementTypes()[j].getIntOrFloatBitWidth();
    });

    // Assign base index to each operand in their order in indices
    std::map<unsigned, Value> indexToBase;
    indexToBase[indices[0]] =
        LLVM::getSharedMemoryBase(loc, rewriter, op.getOperation());
    // add prev reduceOp used sm bytes offset
    indexToBase[indices[0]] =
        gep(ptr_ty(rewriter.getContext(), 2), getElementType(op, indices[0]),
            indexToBase[indices[0]], i32_val(prevSMOffset));

    offsets.push_back(
        (getElementType(op, indices[0]).getIntOrFloatBitWidth() * elems) / 8);
    for (unsigned i = 1; i < (op.getNumOperands() - 1); ++i) { // skip loopIndex
      indexToBase[indices[i]] = gep(
          ptr_ty(rewriter.getContext(), 2), getElementType(op, indices[i - 1]),
          indexToBase[indices[i - 1]], i32_val(elems));
      offsets.push_back(
          (getElementType(op, indices[i - 1]).getIntOrFloatBitWidth() * elems) /
          8);
    }

    // smemBases[k] is the base pointer for the k-th operand
    SmallVector<Value> smemBases(op.getNumOperands() - 1);     // skip loopIndex
    for (unsigned i = 0; i < (op.getNumOperands() - 1); ++i) { // skip loopIndex
      smemBases[i] = indexToBase[i];
    }

    // loopResCacheSmemBases[k] is the base pointer for the k-th operand which
    // is the prev loop reduce result
    SmallVector<Value> loopResCacheSmemBases(op.getNumOperands() -
                                             1); // skip loopIndex
    std::map<unsigned, Value> indexToBaseForLoopCache;
    indexToBaseForLoopCache[indices[0]] = gep(
        ptr_ty(rewriter.getContext(), 2), getElementType(op, indices.back()),
        smemBases.back(), i32_val(elems));
    offsets.push_back(
        (getElementType(op, indices[0]).getIntOrFloatBitWidth() * 1) / 8);
    for (unsigned i = 1; i < (op.getNumOperands() - 1); ++i) { // skip loopIndex
      indexToBaseForLoopCache[indices[i]] = gep(
          ptr_ty(rewriter.getContext(), 2), getElementType(op, indices[i - 1]),
          indexToBaseForLoopCache[indices[i - 1]], i32_val(1));
      offsets.push_back(
          (getElementType(op, indices[0]).getIntOrFloatBitWidth() * 1) / 8);
    }

    for (unsigned i = 0; i < (op.getNumOperands() - 1); ++i) { // skip loopIndex
      loopResCacheSmemBases[i] = indexToBaseForLoopCache[i];
    }

    helper.setSMOffsets(helper.getReduceId(), offsets);
    return {smemBases, loopResCacheSmemBases};
  }

  LogicalResult
  matchAndRewrite(triton::xpu::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ReduceOpHelper helper(op);
    assert(cast<triton::xpu::ClusterLayoutAttr>(helper.getSrcLayout()) &&
           "Unexpected srcLayout in ReduceOpConversion");
    Location loc = op->getLoc();
    auto srcValues = unpackInputs(loc, op, adaptor, rewriter);

    // Init shared memory for mask.
    initSharedMemory(helper, rewriter, loc);

    std::map<SmallVector<unsigned>, SmallVector<Value>> accs;
    std::map<SmallVector<unsigned>, SmallVector<Value>> indices;
    // First reduce all the values along axis within each thread.
    reduceWithinThreads(helper, srcValues, accs, indices, rewriter);

    if (helper.isCoreSynchronous()) {
      LLVM_DEBUG(llvm::dbgs() << "\nisCoreSynchronous=True\n");

      // If all the values to be reduced are within the same warp there is
      // nothing left to do.
      packResults(helper, accs, rewriter);
      return success();
    }
    LLVM_DEBUG(llvm::dbgs() << "\nisCoreSynchronous=False\n");

    // Then reduce across threads within a group.
    reduceWithinGroups(helper, accs, rewriter);
    // LLVM_DEBUG(llvm::dbgs() << "\n After matchAndRewrite xpu::ReduceOp:"
    //              << op->getParentOfType<ModuleOp>() << "\n");
    // helper.dumpSMOffsets();
    return success();
  }

private:
  const TargetInfoBase &targetInfo;

  inline bool isNeedLoopCacheResult(triton::xpu::ReduceOp op) const {
    if (op.getAxis() == 1) {
      LLVM_DEBUG(llvm::dbgs() << "\nDont Need SM Loop Result Cache");
      return false;
    }

    if (auto resultTy =
            dyn_cast<RankedTensorType>(op.getResult()[0].getType())) {
      auto resultShape = resultTy.getShape();
      auto rank = resultShape.size();

      if (rank == 1) { // TODO[dyq]: add [op.getAxis() == None] logic
        LLVM_DEBUG(llvm::dbgs() << "\nNeed SM Loop Result Cache");
        return true;
      }

      LLVM_DEBUG(llvm::dbgs() << "\nDont Need SM Loop Result Cache");
      return false;
    }

    // scalar
    LLVM_DEBUG(llvm::dbgs() << "\nNeed SM Loop Result Cache");
    return true;
  }

  SmallVector<Value> initVals(triton::xpu::ReduceOp &op,
                              ConversionPatternRewriter &rewriter,
                              Location loc) const {

    auto naiveInit = [&](Type elemTy, int value) {
      Value val;
      if (elemTy.isInteger(1)) {
        val = int_val(1, value);
      } else if (elemTy.isInteger(8)) {
        val = int_val(8, value);
      } else if (elemTy.isInteger(16)) {
        val = int_val(16, value);
      } else if (elemTy.isInteger(32)) {
        val = i32_val(value);
      } else if (elemTy.isInteger(64)) {
        val = int_val(64, value);
      } else if (elemTy.isF16()) {
        val = f16_val(value);
      } else if (elemTy.isF32()) {
        val = f32_val(value);
      } else if (elemTy.isF64()) {
        val = f64_val(value);
      } else {
        LLVM_DEBUG(elemTy.dump());
        llvm_unreachable("[Reduce Init]: Unsupported ElemTy in Naive Init");
      }
      return val;
    };

    auto maxInit = [&](Type elemTy) {
      Value val;
      if (elemTy.isInteger(1)) {
        val = int_val(1, 1);
      } else if (elemTy.isInteger(8)) {
        val = int_val(8, INT8_MAX);
      } else if (elemTy.isInteger(16)) {
        val = int_val(16, INT16_MAX);
      } else if (elemTy.isInteger(32)) {
        val = i32_val(INT32_MAX);
      } else if (elemTy.isInteger(64)) {
        val = int_val(64, INT64_MAX);
      } else if (elemTy.isF16()) {
        val = f16_val(65504);
      } else if (elemTy.isF32()) {
        val = f32_val(FLT_MAX);
      } else if (elemTy.isF64()) {
        val = f64_val(DBL_MAX);
      } else {
        LLVM_DEBUG(elemTy.dump());
        llvm_unreachable("[Reduce Init]: Unsupported ElemTy in Max Init");
      }
      return val;
    };

    auto minInit = [&](Type elemTy) {
      Value val;
      if (elemTy.isInteger(1)) {
        val = int_val(1, 0);
      } else if (elemTy.isInteger(8)) {
        val = int_val(8, -INT8_MAX);
      } else if (elemTy.isInteger(16)) {
        val = int_val(16, -INT16_MAX);
      } else if (elemTy.isInteger(32)) {
        val = i32_val(-INT32_MAX);
      } else if (elemTy.isInteger(64)) {
        val = int_val(64, -INT64_MAX);
      } else if (elemTy.isF16()) {
        val = f16_val(-65504);
      } else if (elemTy.isF32()) {
        val = f32_val(-FLT_MAX);
      } else if (elemTy.isF64()) {
        val = f64_val(-DBL_MAX);
      } else {
        LLVM_DEBUG(elemTy.dump());
        llvm_unreachable("[Reduce Init]: Unsupported ElemTy in Min Init");
      }
      return val;
    };

    auto &combineBlock = op.getCombineOp().getBlocks().front();
    SmallVector<Operation *> blockArgDefOps;
    for (int i = 0; i < combineBlock.getArguments().size() / 2; ++i) {
      auto arg = combineBlock.getArgument(i);
      bool isBreak = false;
      for (auto user : arg.getUsers()) {
        TypeSwitch<Operation *>(user)
            .Case<arith::AndIOp>([&](auto andIOp) {
              blockArgDefOps.emplace_back(andIOp);
              isBreak = true;
            })
            .Case<arith::OrIOp>([&](auto orIOp) {
              blockArgDefOps.emplace_back(orIOp);
              isBreak = true;
            })
            .Case<arith::AddFOp>([&](auto addFOp) {
              blockArgDefOps.emplace_back(addFOp);
              isBreak = true;
            })
            .Case<arith::AddIOp>([&](auto addIOp) {
              blockArgDefOps.emplace_back(addIOp);
              isBreak = true;
            })
            .Case<arith::SubFOp>([&](auto subFOp) {
              blockArgDefOps.emplace_back(subFOp);
              isBreak = true;
            })
            .Case<arith::SubIOp>([&](auto subIOp) {
              blockArgDefOps.emplace_back(subIOp);
              isBreak = true;
            })
            .Case<arith::MulFOp>([&](auto mulFOp) {
              blockArgDefOps.emplace_back(mulFOp);
              isBreak = true;
            })
            .Case<arith::MulIOp>([&](auto mulIOp) {
              blockArgDefOps.emplace_back(mulIOp);
              isBreak = true;
            })
            .Case<arith::DivFOp>([&](auto divFOp) {
              blockArgDefOps.emplace_back(divFOp);
              isBreak = true;
            })
            .Case<arith::DivSIOp>([&](auto divSIOp) {
              blockArgDefOps.emplace_back(divSIOp);
              isBreak = true;
            })
            .Case<arith::DivUIOp>([&](auto divUIOp) {
              blockArgDefOps.emplace_back(divUIOp);
              isBreak = true;
            })
            .Case<arith::MaxNumFOp>([&](auto maxNumFOp) {
              blockArgDefOps.emplace_back(maxNumFOp);
              isBreak = true;
            })
            .Case<arith::MaxSIOp>([&](auto maxSIOp) {
              blockArgDefOps.emplace_back(maxSIOp);
              isBreak = true;
            })
            .Case<arith::MaxUIOp>([&](auto maxUIOp) {
              blockArgDefOps.emplace_back(maxUIOp);
              isBreak = true;
            })
            .Case<arith::MinNumFOp>([&](auto minNumFOp) {
              blockArgDefOps.emplace_back(minNumFOp);
              isBreak = true;
            })
            .Case<arith::MinSIOp>([&](auto minSIOp) {
              blockArgDefOps.emplace_back(minSIOp);
              isBreak = true;
            })
            .Case<arith::MinUIOp>([&](auto minUIOp) {
              blockArgDefOps.emplace_back(minUIOp);
              isBreak = true;
            })
            .Case<arith::MaximumFOp>([&](auto maximumFOp) {
              blockArgDefOps.emplace_back(maximumFOp);
              isBreak = true;
            })
            .Case<arith::MinimumFOp>([&](auto minimumFOp) {
              blockArgDefOps.emplace_back(minimumFOp);
              isBreak = true;
            })
            .Case<arith::CmpFOp>([&](auto cmpFOp) {
              if (cmpFOp.getPredicate() == arith::CmpFPredicate::OGT ||
                  cmpFOp.getPredicate() == arith::CmpFPredicate::OGE ||
                  cmpFOp.getPredicate() == arith::CmpFPredicate::OLT ||
                  cmpFOp.getPredicate() == arith::CmpFPredicate::OLE ||
                  cmpFOp.getPredicate() == arith::CmpFPredicate::UGT ||
                  cmpFOp.getPredicate() == arith::CmpFPredicate::UGE ||
                  cmpFOp.getPredicate() == arith::CmpFPredicate::ULT ||
                  cmpFOp.getPredicate() == arith::CmpFPredicate::ULE) {
                blockArgDefOps.emplace_back(cmpFOp);
                isBreak = true;
              }
            })
            .Case<arith::CmpIOp>([&](auto cmpIOp) {
              if (cmpIOp.getPredicate() == arith::CmpIPredicate::slt ||
                  cmpIOp.getPredicate() == arith::CmpIPredicate::sle ||
                  cmpIOp.getPredicate() == arith::CmpIPredicate::sgt ||
                  cmpIOp.getPredicate() == arith::CmpIPredicate::sge ||
                  cmpIOp.getPredicate() == arith::CmpIPredicate::ult ||
                  cmpIOp.getPredicate() == arith::CmpIPredicate::ule ||
                  cmpIOp.getPredicate() == arith::CmpIPredicate::ugt ||
                  cmpIOp.getPredicate() == arith::CmpIPredicate::uge) {
                blockArgDefOps.emplace_back(cmpIOp);
                isBreak = true;
              }
            });
        if (isBreak) {
          break;
        }
      }
    }

    auto types = op.getInputTypes();
    assert(blockArgDefOps.size() == types.size() &&
           "[Reduce Init]: BlockArgDefOps Size() != Types Size");
    SmallVector<Value> vals;
    for (int i = 0; i < blockArgDefOps.size(); ++i) {
      auto elemTy = getElementTypeOrSelf(types[i]);
      if (auto vecTy = dyn_cast<mlir::VectorType>(elemTy)) {
        elemTy = vecTy.getElementType();
      }
      Value val = naiveInit(elemTy, 0);
      auto blockArgDefOp = blockArgDefOps[i];
      TypeSwitch<Operation *>(blockArgDefOp)
          .Case<arith::AndIOp>([&](auto andIOp) { val = naiveInit(elemTy, 1); })
          .Case<arith::OrIOp>([&](auto orIOp) { val = naiveInit(elemTy, 0); })
          .Case<arith::AddFOp>([&](auto addFOp) { val = naiveInit(elemTy, 0); })
          .Case<arith::AddIOp>([&](auto addIOp) { val = naiveInit(elemTy, 0); })
          .Case<arith::SubFOp>([&](auto subFOp) { val = naiveInit(elemTy, 0); })
          .Case<arith::SubIOp>([&](auto subIOp) { val = naiveInit(elemTy, 0); })
          .Case<arith::MulFOp>([&](auto mulFOp) { val = naiveInit(elemTy, 1); })
          .Case<arith::MulIOp>([&](auto mulIOp) { val = naiveInit(elemTy, 1); })
          .Case<arith::DivFOp>([&](auto divFOp) { val = naiveInit(elemTy, 1); })
          .Case<arith::DivSIOp>(
              [&](auto divSIOp) { val = naiveInit(elemTy, 1); })
          .Case<arith::DivUIOp>(
              [&](auto divUIOp) { val = naiveInit(elemTy, 1); })
          .Case<arith::MaxNumFOp>(
              [&](auto maxNumFOp) { val = minInit(elemTy); })
          .Case<arith::MaxSIOp>([&](auto maxSIOp) { val = minInit(elemTy); })
          .Case<arith::MaxUIOp>(
              [&](auto maxUIOp) { val = naiveInit(elemTy, 0); })
          .Case<arith::MinNumFOp>(
              [&](auto minNumFOp) { val = maxInit(elemTy); })
          .Case<arith::MinSIOp>([&](auto minSIOp) { val = maxInit(elemTy); })
          .Case<arith::MinUIOp>([&](auto minUIOp) { val = maxInit(elemTy); })
          .Case<arith::MaximumFOp>(
              [&](auto maximumFOp) { val = minInit(elemTy); })
          .Case<arith::MinimumFOp>(
              [&](auto minimumFOp) { val = maxInit(elemTy); })
          .Case<arith::CmpFOp>([&](auto cmpFOp) {
            if (cmpFOp.getPredicate() == arith::CmpFPredicate::OGT ||
                cmpFOp.getPredicate() == arith::CmpFPredicate::OGE ||
                cmpFOp.getPredicate() == arith::CmpFPredicate::UGT ||
                cmpFOp.getPredicate() == arith::CmpFPredicate::UGE) {
              val = minInit(elemTy);
            } else if (cmpFOp.getPredicate() == arith::CmpFPredicate::OLT ||
                       cmpFOp.getPredicate() == arith::CmpFPredicate::OLE ||
                       cmpFOp.getPredicate() == arith::CmpFPredicate::ULT ||
                       cmpFOp.getPredicate() == arith::CmpFPredicate::ULE) {
              val = maxInit(elemTy);
            } else {
              llvm_unreachable(
                  "[Reduce Init]: Unsupported CmpFPredicate in CmpFOp");
            }
          })
          .Case<arith::CmpIOp>([&](auto cmpIOp) {
            if (cmpIOp.getPredicate() == arith::CmpIPredicate::sgt ||
                cmpIOp.getPredicate() == arith::CmpIPredicate::sge ||
                cmpIOp.getPredicate() == arith::CmpIPredicate::ugt ||
                cmpIOp.getPredicate() == arith::CmpIPredicate::uge) {
              val = minInit(elemTy);
            } else if (cmpIOp.getPredicate() == arith::CmpIPredicate::slt ||
                       cmpIOp.getPredicate() == arith::CmpIPredicate::sle ||
                       cmpIOp.getPredicate() == arith::CmpIPredicate::ult ||
                       cmpIOp.getPredicate() == arith::CmpIPredicate::ule) {
              val = maxInit(elemTy);
            } else {
              llvm_unreachable(
                  "[Reduce Init]: Unsupported CmpFPredicate in CmpIOp");
            }
          })
          .Default([&](auto defaultOp) {
            LLVM_DEBUG(defaultOp->dump());
            llvm_unreachable(
                "[Reduce Init]: Unsupported Operation in Reduce Output");
          });
      vals.emplace_back(val);
    }

    return vals;
  }

  void initSharedMemory(ReduceOpHelper &helper,
                        ConversionPatternRewriter &rewriter,
                        Location loc) const {
    // Init shared memory
    ConversionPatternRewriter::InsertionGuard guard(
        rewriter); // save reduceOpPtr to restore
    auto reduceOpPtr = rewriter.saveInsertionPoint();
    triton::xpu::ReduceOp op = helper.getXPUOperation();
    auto func = op->template getParentOfType<LLVM::LLVMFuncOp>();
    rewriter.setInsertionPointToStart(&(func.front()));
    // Compute a shared memory base per operand.
    Value coreId = getThreadId(rewriter, loc);
    SmallVector<Value> vals = initVals(op, rewriter, loc);

    auto smemShape = helper.getXPUScratchConfig();
    auto [smemBases, loopResCacheSmemBases] =
        getSmemBases(op, product<unsigned>(smemShape), rewriter);
    for (unsigned i = 0; i < (op.getNumOperands() - 1); ++i) { // skip loopIndex
      auto elemTy = getElementType(op, i);
      Value initPtr =
          gep(ptr_ty(rewriter.getContext(), 2), elemTy, smemBases[i], coreId);
      store_sm(vals[i], initPtr);
    }
    xpu_barrier();
    rewriter.restoreInsertionPoint(reduceOpPtr); // restore reduceOpPtr
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
    auto returnOp =
        dyn_cast<triton::xpu::ReduceReturnOp>(newReduce.getTerminator());

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

  void calculate(ConversionPatternRewriter &rewriter, const Location &loc,
                 Operation *op, Value &acc, const Value &cur) const {
    TypeSwitch<Operation *>(op)
        .Case<arith::AddFOp>([&](auto addfOp) { acc = fadd(acc, cur); })
        .Case<arith::MulFOp>([&](auto mulfOp) { acc = fmul(acc, cur); })
        .Case<arith::MaxNumFOp>([&](auto maxfOp) { acc = fmax(acc, cur); })
        .Case<arith::MinNumFOp>([&](auto minfOp) { acc = fmin(acc, cur); })
        .Case<arith::OrIOp>([&](auto oriOp) { acc = or_(acc, cur); })
        .Case<arith::XOrIOp>([&](auto xoriOp) { acc = xor_(acc, cur); })
        .Case<arith::AndIOp>([&](auto andOp) { acc = and_(acc, cur); })
        .Default([&](auto defaultOp) {
          LLVM_DEBUG(defaultOp->dump());
          llvm_unreachable("[Vectorization]: Unsupported Operation Type "
                           "To VecType in Reduce");
        });
  }

  void accmulateWithinVector(ConversionPatternRewriter &rewriter,
                             const Location &loc, Operation *op,
                             Value &accVec) const {
    auto accTy = cast<VectorType>(accVec.getType());
    size_t vecSize = accTy.getNumElements();
    Type elemTy = getElementTypeOrSelf(accTy);
    Value acc = extract_element(elemTy, accVec, i32_val(0));
    for (size_t i = 1; i < vecSize; ++i) {
      auto cur = extract_element(elemTy, accVec, i32_val(i));
      calculate(rewriter, loc, op, acc, cur);
    }
    accVec = acc;
  }

  void accmulateNaive(ConversionPatternRewriter &rewriter, const Location &loc,
                      SmallVector<Operation *> &ops, SmallVector<Value> &accs,
                      SmallVector<Value> &curs, bool isFirst) const {
    for (unsigned i = 0; i < accs.size(); ++i) {
      if (isFirst) {
        accs[i] = curs[i];
      } else {
        calculate(rewriter, loc, ops[i], accs[i], curs[i]);
      }
    }
  }

  SmallVector<SmallVector<Value>>
  unpackInputs(Location loc, triton::xpu::ReduceOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const {
    auto types = op.getInputTypes();
    auto operands = adaptor.getOperands();
    unsigned srcElems = getTotalElemsPerThread(types[0]);
    SmallVector<SmallVector<Value>> srcValues(srcElems);
    for (unsigned i = 0; i < (op.getNumOperands() - 1); ++i) { // skip loopIndex
      auto values = unpackLLElements(loc, operands[i], rewriter);

      assert(values.size() == srcValues.size());
      for (unsigned j = 0; j < srcValues.size(); ++j) {
        srcValues[j].push_back(values[j]);
      }
    }

    return srcValues;
  }

  // Reduce along op axis for elements that are in the same thread. The
  // accumulated value is stored in accs.
  void reduceWithinThreads(
      ReduceOpHelper &helper, SmallVector<SmallVector<Value>> &srcValues,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &indices,
      ConversionPatternRewriter &rewriter) const {
    triton::xpu::ReduceOp op = helper.getXPUOperation();
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
    auto srcIndices =
        emitIndices(op.getLoc(), rewriter, targetInfo, helper.getSrcLayout(),
                    operandType, op.getLoopIndex());

    // reduce within threads
    for (const auto &[_, i] : uniqueOffsets) {
      SmallVector<unsigned> key = offsets[i];
      key[op.getAxis()] = 0;
      bool isFirst = accs.find(key) == accs.end();
      accumulate(rewriter, *combineOp, accs[key], srcValues[i], isFirst);
      if (isFirst)
        indices[key] = srcIndices[i];
    }

    // Accumulate within Vector
    if (helper.isVectorized()) {
      SmallVector<Operation *> returnDefOps = helper.getReturnDefOps();
      for (auto &it : accs) {
        SmallVector<Value> &accVecs = it.second;
        assert(accVecs.size() == returnDefOps.size() &&
               "accVecs.size() !=returnDefOps.size()");
        for (unsigned i = 0; i < returnDefOps.size(); ++i) {
          accmulateWithinVector(rewriter, op.getLoc(), returnDefOps[i],
                                accVecs[i]);
        }
      }
    }
  }

  void storeCoreReduceToSharedMemory(
      ReduceOpHelper &helper,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
      SmallVector<Value> &smemBases,
      ConversionPatternRewriter &rewriter) const {
    triton::xpu::ReduceOp op = helper.getXPUOperation();
    Location loc = op.getLoc();
    Value coreId = getThreadId(rewriter, loc);

    for (auto it : accs) {
      const SmallVector<unsigned> &key = it.first;
      SmallVector<Value> &acc = it.second;

      for (unsigned i = 0; i < (op.getNumOperands() - 1);
           ++i) { // skip loopIndex
        auto elemTy = getElementType(op, i);
        Value writePtr =
            gep(ptr_ty(rewriter.getContext(), 2), elemTy, smemBases[i], coreId);
        store_sm(acc[i], writePtr);
      }
    }
  }

  mlir::RewriterBase::InsertPoint
  getPreviousInsertionPoint(PatternRewriter &rewriter) const {
    auto currentInsertionPoint = rewriter.getInsertionPoint();
    auto oldInsertionPoint = currentInsertionPoint;
    Block *currentBlock_0 = rewriter.getInsertionBlock();
    // Move the iterator one step backward
    if (currentInsertionPoint != currentBlock_0->begin()) {
      --currentInsertionPoint;
    }
    // Set the insertion point to the adjusted position and save it
    rewriter.setInsertionPoint(currentBlock_0, currentInsertionPoint);
    auto startInsertionPoint = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(currentBlock_0, oldInsertionPoint);
    return startInsertionPoint;
  }

  void createCondBr(PatternRewriter &rewriter, Location loc, Value condition,
                    Block *&trueDest, Block *&falseDest) const {
    Block *currentBlock = rewriter.getInsertionBlock();
    falseDest = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    trueDest = rewriter.createBlock(falseDest);

    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<LLVM::CondBrOp>(loc, condition, trueDest, falseDest);
    rewriter.setInsertionPointToStart(trueDest);

    rewriter.create<LLVM::BrOp>(loc, falseDest);
    rewriter.setInsertionPointToStart(falseDest);
  }

  void moveOpsBetweenInsertionPoints(ReduceOpHelper &helper,
                                     PatternRewriter &rewriter, Block *trueDest,
                                     Block *falseDest, Block::iterator start,
                                     Block::iterator end) const {
    triton::xpu::ReduceOp op = helper.getXPUOperation();

    auto terminator = &*(trueDest->begin());

    // LLVM_DEBUG(llvm::dbgs() << "\n Before moveBefore:" <<
    // op->getParentOfType<ModuleOp>()
    //              << "\n");

    // LLVM_DEBUG(llvm::dbgs() << "\n movedOp: " << "\n");
    start++; // skip the previous op
    while (start != end) {
      Operation *op = &*start++;
      //   op->dump();
      op->moveBefore(trueDest, trueDest->end());
    }

    // deal the previous op
    Operation *endOp = &*end;
    endOp->moveBefore(trueDest, trueDest->end());
    terminator->moveBefore(trueDest, trueDest->end());
  }

  void accumulatePartialReductions(ReduceOpHelper &helper,
                                   SmallVector<Value> &smemBases,
                                   SmallVector<Value> &loopResCacheSmemBases,
                                   ConversionPatternRewriter &rewriter) const {
    triton::xpu::ReduceOp op = helper.getXPUOperation();
    auto srcLayout = helper.getSrcLayout();
    Location loc = op.getLoc();
    unsigned groupSizeInt = helper.getIntraGroupSizeWithUniqueData();
    unsigned operandNum = op.getInputTypes().size();

    Value coreId = getThreadId(rewriter, loc);
    Value groupSize = i32_val(groupSizeInt);
    Value coreIdInGroup = urem(coreId, groupSize);
    Value groupId = udiv(coreId, groupSize);
    Value groupSkip = mul(groupId, groupSize);
    // Value laneId = add(mul(groupId, groupSize), coreIdInGroup);
    Value zero = i32_val(0);
    Value coreIdInGroupZero = icmp_eq(coreIdInGroup, zero);

    auto startInsertionPoint = getPreviousInsertionPoint(rewriter);

    SmallVector<Value> acc(operandNum); // skip loopIndex
    SmallVector<SmallVector<Value>> readValues(groupSizeInt);
    for (unsigned i = 0; i < operandNum; ++i) { // skip loopIndex
      auto elemTy = getElementType(op, i);

      for (unsigned readOffset = 0; readOffset < groupSizeInt; ++readOffset) {
        Value laneId = add(groupSkip, i32_val(readOffset));
        Value readPtr =
            gep(ptr_ty(rewriter.getContext(), 2), elemTy, smemBases[i], laneId);
        readValues[readOffset].push_back(load_sm(elemTy, readPtr));
      }
    }

    SmallVector<Operation *> returnDefOps = helper.getReturnDefOps();
    for (auto [i, v] : llvm::enumerate(readValues)) {
      if (helper.isVectorized()) {
        accmulateNaive(rewriter, loc, returnDefOps, acc, v, i == 0);
      } else {
        accumulate(rewriter, op.getCombineOp(), acc, v, i == 0);
      }
    }

    SmallVector<Value> writePtrs(operandNum);   // skip loopIndex
    for (unsigned i = 0; i < operandNum; ++i) { // skip loopIndex
      auto elemTy = getElementType(op, i);
      // TODO[dyq]: check writeOffset == i32_val(0)
      Value writeOffset = groupSkip;
      writePtrs[i] = gep(ptr_ty(rewriter.getContext(), 2), elemTy, smemBases[i],
                         /*writeOffset*/ writeOffset);
    }

    for (unsigned i = 0; i < operandNum; ++i) { // skip loopIndex
      store_sm(acc[i], writePtrs[i]);
    }

    // reduce calcution(cur loop) finsh, now move it to the trueDest
    auto endInsertionPoint = getPreviousInsertionPoint(rewriter);

    Block *trueDest = nullptr;
    Block *falseDest = nullptr;
    createCondBr(rewriter, op->getLoc(), coreIdInGroupZero, trueDest,
                 falseDest);

    moveOpsBetweenInsertionPoints(helper, rewriter, trueDest, falseDest,
                                  startInsertionPoint.getPoint(),
                                  endInsertionPoint.getPoint());

    if (isNeedLoopCacheResult(op)) {
      // reduce with prev loopResult
      Value laneId = coreId;
      Value zero = i32_val(0);
      Value laneZero = icmp_eq(laneId, zero);
      Value loopIndex = op.getLoopIndex();
      Value loopNonZero = icmp_ne(loopIndex, zero);
      Value cond = and_(laneZero, loopNonZero);

      Block *loopReduceTrueDest = nullptr;
      Block *loopReduceFalseDest = nullptr;
      createCondBr(rewriter, op->getLoc(), cond, loopReduceTrueDest,
                   loopReduceFalseDest);

      auto curInsertionPoint = rewriter.getInsertionPoint();
      Block *curBlock = rewriter.getInsertionBlock();

      rewriter.setInsertionPointToStart(loopReduceTrueDest);
      SmallVector<Value> curResSmemValues(operandNum);
      SmallVector<Value> loopResCacheSmemValues(operandNum);
      for (unsigned i = 0; i < (operandNum); ++i) { // skip loopIndex
        auto elemTy = getElementType(op, i);
        curResSmemValues[i] = load_sm(elemTy, smemBases[i]);
        loopResCacheSmemValues[i] = load_sm(elemTy, loopResCacheSmemBases[i]);
      }

      SmallVector<Operation *> returnDefOps = helper.getReturnDefOps();
      if (helper.isVectorized()) {
        accmulateNaive(rewriter, loc, returnDefOps, loopResCacheSmemValues,
                       loopResCacheSmemValues, false);
      } else {
        accumulate(rewriter, op.getCombineOp(), curResSmemValues,
                   loopResCacheSmemValues, false);
      }

      // store the final result
      for (unsigned i = 0; i < operandNum; ++i) { // skip loopIndex
        store_sm(curResSmemValues[i], smemBases[i]);
      }

      rewriter.setInsertionPoint(curBlock, curInsertionPoint);
    }
  }

  // Load the final reduction from shared memory and replace the reduce result
  // with it.
  void loadReductionAndPackResult(ReduceOpHelper &helper,
                                  SmallVector<unsigned> smemShape,
                                  SmallVector<Value> &smemBases,
                                  SmallVector<Value> &loopResCacheSmemBases,
                                  ConversionPatternRewriter &rewriter) const {
    triton::xpu::ReduceOp op = helper.getXPUOperation();
    Location loc = op.getLoc();
    auto srcLayout = helper.getSrcLayout();
    auto axis = op.getAxis();
    auto smemOrder = helper.getOrderWithAxisAtBeginning();
    unsigned groupSizeInt = helper.getIntraGroupSizeWithUniqueData();

    Value coreId = getThreadId(rewriter, loc);
    Value groupSize = i32_val(groupSizeInt);
    Value groupId = udiv(coreId, groupSize);
    Value groupSkip = mul(groupId, groupSize);
    SmallVector<Value> results(op.getNumOperands() - 1);       // skip loopIndex
    for (unsigned i = 0; i < (op.getNumOperands() - 1); ++i) { // skip loopIndex
      auto elemTy = getElementType(op, i);
      if (auto resultTy =
              dyn_cast<RankedTensorType>(op.getResult()[i].getType())) {
        // nd-tensor where n >= 1
        auto resultLayout = cast<SliceEncodingAttr>(resultTy.getEncoding());
        unsigned resultElems = getTotalElemsPerThread(resultTy);
        auto resultIndices =
            emitIndices(loc, rewriter, targetInfo, resultLayout, resultTy,
                        op.getLoopIndex());
        auto resultShape = resultTy.getShape();
        auto resultCTATile = getShapePerCTATile(resultLayout, resultShape);
        assert(resultIndices.size() == resultElems);

        SmallVector<Value> resultVals(resultElems);
        for (size_t j = 0; j < resultElems; ++j) {
          SmallVector<Value> readIdx = resultIndices[j];
          //   readIdx.insert(readIdx.begin() + op.getAxis(), i32_val(0));
          //   for (size_t resultIdx = 0, resultDim = resultShape.size();
          //        resultIdx < resultDim; ++resultIdx) {
          //     auto smemIdx = resultIdx < op.getAxis() ? resultIdx :
          //     resultIdx
          //     + 1; if (resultCTATile[resultIdx] > smemShape[smemIdx] ||
          //         resultShape[resultIdx] > smemShape[smemIdx]) {
          //       // When srcShape smaller then src sizePerThread, only
          //       srcShape
          //       // elements is accumulated in smem. Modulo smemShape
          //       effectively
          //       // replicates srcShape elements to src sizePerThread.
          //       readIdx[smemIdx] =
          //           urem(readIdx[smemIdx], i32_val(smemShape[smemIdx]));
          //     }
          //   }

          Value readOffset = groupSkip;
          //   Value readOffset = linearize(rewriter, loc, readIdx, smemShape,
          //   smemOrder);
          Value readPtr = gep(ptr_ty(rewriter.getContext(), 2), elemTy,
                              smemBases[i], readOffset);
          resultVals[j] = load_sm(elemTy, readPtr);
        }

        results[i] = packLLElements(loc, getTypeConverter(), resultVals,
                                    rewriter, resultTy);
      } else {
        // 0d-tensor -> scalar
        results[i] = load_sm(elemTy, smemBases[i]);
        // save reduce result in cur loop
        if (isNeedLoopCacheResult(op))
          store_sm(results[i], loopResCacheSmemBases[i]);
      }
    }

    rewriter.replaceOp(op, results);
  }

  // Reduce across threads within each group.
  void
  reduceWithinGroups(ReduceOpHelper &helper,
                     std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
                     ConversionPatternRewriter &rewriter) const {
    triton::xpu::ReduceOp op = helper.getXPUOperation();
    Location loc = op.getLoc();
    unsigned axis = op.getAxis();

    // unsigned sizeIntraGroups = helper.getIntraGroupSizeWithUniqueData();
    // unsigned threadOffsetOnReductionAxis =
    // helper.getThreadOffsetOnReductionAxis(); LLVM_DEBUG(llvm::dbgs() <<
    // "\nsizeIntraGroups=" << sizeIntraGroups << "\n"); LLVM_DEBUG(llvm::dbgs()
    // <<
    // "\nthreadOffsetOnReductionAxis="
    //              << threadOffsetOnReductionAxis << "\n");

    // Compute a shared memory base per operand.
    auto smemShape = helper.getXPUScratchConfig();

    auto [smemBases, loopResCacheSmemBases] =
        getSmemBases(op, product<unsigned>(smemShape), rewriter);

    storeCoreReduceToSharedMemory(helper, accs, smemBases, rewriter);
    // LLVM_DEBUG(llvm::dbgs() << "\n After storeCoreReduceToSharedMemory:"
    //              << op->getParentOfType<ModuleOp>() << "\n");

    xpu_barrier();

    accumulatePartialReductions(helper, smemBases, loopResCacheSmemBases,
                                rewriter);
    // LLVM_DEBUG(llvm::dbgs() << "\n After accumulatePartialReductions:"
    //              << op->getParentOfType<ModuleOp>() << "\n");

    xpu_barrier();

    loadReductionAndPackResult(helper, smemShape, smemBases,
                               loopResCacheSmemBases, rewriter);

    // llvm_unreachable("Not Supported");
  }

  // Pack the accumulator values and replace the reduce op with the result.
  void packResults(ReduceOpHelper &helper,
                   std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
                   ConversionPatternRewriter &rewriter) const {
    triton::xpu::ReduceOp op = helper.getXPUOperation();
    Location loc = op.getLoc();
    unsigned axis = op.getAxis();
    SmallVector<Value> results((op.getNumOperands() - 1));     // skip loopIndex
    for (unsigned i = 0; i < (op.getNumOperands() - 1); ++i) { // skip loopIndex
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
};

} // namespace

void mlir::triton::xpu::populateReduceOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<XPUReduceOpConversion>(typeConverter, targetInfo, benefit);
}
