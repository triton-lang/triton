/*
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

#include <memory>

#define GEN_PASS_CLASSES
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

#define DEBUG_TYPE "nvws-lower-aref"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
using namespace triton::nvidia_gpu;
using namespace triton::nvws;

struct ArefUseNode {
  ArefUseNode *parent;
  Operation *op;
  SmallVector<ArefUseNode *> subOps;
  bool containsAsync = false;
};

struct ArefValue {
  Value emptyMbars;
  Value fullMbars;
  int depth;
  SmallVector<ArefUseNode *> users;
  SmallVector<Value> updatedValues;
};

struct ArefUseGraph {
  llvm::MapVector<Operation *, ArefUseNode> nodes;
  llvm::MapVector<Value, ArefValue> arefs;
};

MemDescType getMemDesc(PatternRewriter &rewriter, llvm::ArrayRef<int64_t> shape,
                       Type intType) {
  auto ctx = rewriter.getContext();
  Attribute sharedMemorySpace = triton::gpu::SharedMemorySpaceAttr::get(ctx);
  auto barrierCTALayout = CTALayoutAttr::get(
      /*context=*/ctx, /*CTAsPerCGA=*/{1},
      /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
  auto barrierEncoding =
      SwizzledSharedEncodingAttr::get(ctx, 1, 1, 1, {0}, barrierCTALayout);
  return MemDescType::get(shape, intType, barrierEncoding, sharedMemorySpace,
                          /*mutableMemory=*/true);
}
MemDescType getBarrierMemDesc(PatternRewriter &rewriter,
                              llvm::ArrayRef<int64_t> shape) {
  return getMemDesc(rewriter, shape, rewriter.getI64Type());
}

Value getBarrierAt(PatternRewriter &rewriter, Location loc, Value mbars,
                   Value idx) {
  auto memDesc = getBarrierMemDesc(rewriter, {1});
  return rewriter.create<triton::gpu::MemDescSubviewOp>(
      loc, memDesc, mbars, SmallVector<Value>{idx});
}

Value getStageIndex(PatternRewriter &rewriter, Location loc, Value idx,
                    int depth) {
  Value stageIdx = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
  if (depth > 1) {
    Value depthVal = rewriter.create<arith::ConstantIntOp>(loc, depth, 32);
    Value stage = rewriter.create<arith::RemSIOp>(loc, idx, depthVal);
    stageIdx = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIntegerType(32), stage);
  }
  return stageIdx;
}

Value getPhase(PatternRewriter &rewriter, Location loc, Value k, Value phase,
               int depth) {
  // parity = (k % numStage) % 2 == 0 ? parity : parity ^ 1
  Value D = rewriter.create<arith::ConstantIntOp>(loc, depth, 32);
  Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);
  Value two = rewriter.create<arith::ConstantIntOp>(loc, 2, 32);
  Value kmodD = rewriter.create<arith::RemSIOp>(loc, k, D);
  Value kmodDmod2 = rewriter.create<arith::RemSIOp>(loc, k, two);
  Value pred = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, kmodDmod2,
      rewriter.create<arith::ConstantIntOp>(loc, 0, 32));
  Value parityXor = rewriter.create<arith::XOrIOp>(loc, phase, one);
  return rewriter.create<arith::SelectOp>(loc, pred, phase, parityXor);
}

Value getBarrier(PatternRewriter &rewriter, Location loc, Value mBars,
                 Value idx, int depth) {
  auto stageIdx = getStageIndex(rewriter, loc, idx, depth);
  return getBarrierAt(rewriter, loc, mBars, stageIdx);
}

void waitOnMbar(PatternRewriter &rewriter, Location loc, Value mBars, Value idx,
                Value phase0, int depth) {
  auto ctx = rewriter.getContext();
  auto barrier = getBarrier(rewriter, loc, mBars, idx, depth);
  auto waitPhase = getPhase(rewriter, loc, idx, phase0, depth);
  // wait on empty/full
  auto waitOp = rewriter.create<triton::nvidia_gpu::WaitBarrierOp>(loc, barrier,
                                                                   waitPhase);
}

void signalArrival(PatternRewriter &rewriter, Location loc, Value mBars,
                   Value idx, int depth) {
  // And a barrier arrive to signal completion of the region if none of the
  // underlying ops are already async.
  auto stageIdx = getStageIndex(rewriter, loc, idx, depth);
  auto barrier = getBarrierAt(rewriter, loc, mBars, stageIdx);
  // wait on empty. Use a default phase for now, will rewrite later
  rewriter.create<triton::nvidia_gpu::ArriveBarrierOp>(loc, barrier, 1);
}

class LowerArefCreate : public OpRewritePattern<ArefCreateOp> {
  ArefUseGraph &graph;

public:
  using OpRewritePattern::OpRewritePattern;

  LowerArefCreate(mlir::MLIRContext *context, ArefUseGraph &graph)
      : OpRewritePattern<ArefCreateOp>(context), graph(graph) {}

  LogicalResult matchAndRewrite(ArefCreateOp op,
                                PatternRewriter &rewriter) const override {
    if (graph.arefs.contains(op.getResult())) {
      // We've already added the barriers for this op
      return failure();
    }
    auto loc = op.getLoc();
    auto ctx = op.getContext();
    auto uses = op.getResult().getUses();
    auto numBatches = op.getResult().getType().getNumBatchAxes();

    int rank = 0;
    if (numBatches)
      rank = *numBatches;
    if (rank > 1)
      op.emitError("TODO: Implement multi-axis slicing");

    int depth = 1;
    if (rank == 1) {
      if (auto mType = dyn_cast<MemDescType>(op.getOperand(0).getType()))
        depth = mType.getShape()[0];
      if (auto rType = dyn_cast<RankedTensorType>(op.getOperand(0).getType()))
        depth = rType.getShape()[0];
    }

    rewriter.setInsertionPointAfter(op);
    Attribute sharedMemorySpace = triton::gpu::SharedMemorySpaceAttr::get(ctx);
    auto barrierCTALayout = CTALayoutAttr::get(
        /*context=*/ctx, /*CTAsPerCGA=*/{1},
        /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
    auto barrierEncoding =
        SwizzledSharedEncodingAttr::get(ctx, 1, 1, 1, {0}, barrierCTALayout);
    auto barrierType = MemDescType::get(
        SmallVector<int64_t>{depth}, rewriter.getI64Type(), barrierEncoding,
        sharedMemorySpace, /*mutableMemory=*/true);
    // Create two mbarriers
    auto emptyMbars = rewriter.create<LocalAllocOp>(loc, barrierType, Value());
    emptyMbars->setAttr("aref_empty_mbarriers", rewriter.getUnitAttr());

    auto fullMbars = rewriter.create<LocalAllocOp>(loc, barrierType, Value());
    fullMbars->setAttr("aref_full_mbarriers", rewriter.getUnitAttr());

    auto lb = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    auto ub = rewriter.create<arith::ConstantIntOp>(loc, depth, 32);
    auto step = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);
    auto dLoop = rewriter.create<scf::ForOp>(loc, lb, ub, step);
    rewriter.setInsertionPointToStart(dLoop.getBody());

    for (int i = 0; i < 2; ++i) {
      auto memDesc = getBarrierMemDesc(rewriter, {1});
      auto singleBarrier = rewriter.create<triton::gpu::MemDescSubviewOp>(
          loc, memDesc, i == 0 ? emptyMbars.getResult() : fullMbars.getResult(),
          SmallVector<Value>{dLoop.getInductionVar()});

      int count = i == 0 ? 0 : 1;
      rewriter.create<InitBarrierOp>(loc, singleBarrier, count);
    }

    graph.arefs[op] =
        ArefValue{emptyMbars.getResult(), fullMbars.getResult(), depth};
    for (auto user : op.getResult().getUsers())
      graph.arefs[op].users.push_back(&graph.nodes[user]);

    return success();
  }
};

template <bool put, typename T>
LogicalResult lowerRegion(T op, PatternRewriter &rewriter,
                          ArefUseGraph &graph) {
  // Lowering rules for Put/Get
  // 1) they wait on empty/full barriers respectively at start
  // 2) if full/empty barriers are signaled by enclosed ops, they will
  // have already been lowered, so we can skip arrives. If not, add an arrive
  // j3) If a Get op encloses a Put op, the get op's empty barrier depends
  // on the put op's full barrier
  auto ctx = op.getContext();
  auto loc = op.getLoc();
  auto aref = op.getOperand();
  if (!graph.arefs.contains(aref)) {
    // we don't have barriers yet
    return failure();
  }
  auto emptyMbars = graph.arefs[aref].emptyMbars;
  auto fullMbars = graph.arefs[aref].fullMbars;
  auto depth = graph.arefs[aref].depth;

  if (put)
    graph.arefs[aref].updatedValues.clear();

  rewriter.setInsertionPoint(op);

  auto zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
  auto one = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);
  Value idx = depth > 1 ? op.getIndexes()[0] : zero;

  auto stageIdx = getStageIndex(rewriter, loc, idx, depth);

  waitOnMbar(rewriter, loc, put ? emptyMbars : fullMbars, idx, put ? one : zero,
             depth);

  SmallVector<Value> views;

  // slice aref values
  if (depth > 1) {
    for (auto value :
         aref.template getDefiningOp<ArefCreateOp>().getOperands()) {
      if (auto mType = dyn_cast<MemDescType>(value.getType())) {
        auto shape = mType.getShape();
        SmallVector<int64_t> tensorShape(shape.begin() + 1, shape.end());
        auto memDescTypeNew =
            MemDescType::get(tensorShape, mType.getElementType(),
                             mType.getEncoding(), mType.getMemorySpace(), true);
        SmallVector<Value> offsets(mType.getShape().size(), zero);
        offsets[0] = stageIdx;
        Value singleBuffer = rewriter.create<triton::gpu::MemDescSubviewOp>(
            loc, memDescTypeNew, value, offsets);
        views.push_back(singleBuffer);
      } else if (auto rType = dyn_cast<RankedTensorType>(value.getType())) {
        return op.emitError("FIXME: In-register Tensors not yet supported");
      } else {
        return op.emitError("Aref input type not supported for slicing");
      }
    }
  } else {
    if (graph.arefs[aref].updatedValues.size() > 0) {
      views.append(graph.arefs[aref].updatedValues.begin(),
                   graph.arefs[aref].updatedValues.end());
    } else {
      for (auto value :
           aref.template getDefiningOp<ArefCreateOp>().getOperands())
        views.push_back(value);
    }
  }

  auto opBody = &op->getRegion(0).front();
  if (opBody->getArguments().size() != views.size())
    return op.emitError("number of views and arguments mismatch.");
  for (auto [arg, view] : zip(opBody->getArguments(), views))
    arg.replaceAllUsesWith(view);

  for (auto [ret, val] :
       llvm::zip(op.getResults(), opBody->back().getOperands())) {
    ret.replaceAllUsesWith(val);
  }

  if (put) {
    for (auto result : opBody->back().getOperands())
      if (isa<RankedTensorType>(result.getType()))
        graph.arefs[op.getOperand()].updatedValues.push_back(result);
  }

  for (auto &bodyOp : llvm::make_early_inc_range(opBody->without_terminator()))
    bodyOp.moveBefore(op);

  if (!graph.nodes[op].containsAsync) {
    signalArrival(rewriter, loc, put ? fullMbars : emptyMbars, idx, depth);
  }

  rewriter.eraseOp(op);
  return success();
}

Value getNumIterations(PatternRewriter &rewriter, scf::ForOp loop) {
  auto l = loop.getLowerBound();
  auto u = loop.getUpperBound();
  auto s = loop.getStep();
  auto n = rewriter.create<arith::SubIOp>(u.getLoc(), u, l);
  return rewriter.create<arith::DivSIOp>(u.getLoc(), n, s);
}

class LowerArefGet : public OpRewritePattern<ArefGetOp> {
  ArefUseGraph &graph;

public:
  using OpRewritePattern::OpRewritePattern;

  LowerArefGet(mlir::MLIRContext *context, ArefUseGraph &graph)
      : OpRewritePattern<ArefGetOp>(context), graph(graph) {}

  LogicalResult matchAndRewrite(ArefGetOp op,
                                PatternRewriter &rewriter) const override {
    return lowerRegion<0>(op, rewriter, graph);
  }
};

class LowerArefPut : public OpRewritePattern<ArefPutOp> {
  ArefUseGraph &graph;

public:
  using OpRewritePattern::OpRewritePattern;

  LowerArefPut(mlir::MLIRContext *context, ArefUseGraph &graph)
      : OpRewritePattern<ArefPutOp>(context), graph(graph) {}

  LogicalResult matchAndRewrite(ArefPutOp op,
                                PatternRewriter &rewriter) const override {
    return lowerRegion<1>(op, rewriter, graph);
  }
};

class TMAStoreLowering : public OpRewritePattern<DescriptorStoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DescriptorStoreOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();
    Attribute sharedMemorySpace = triton::gpu::SharedMemorySpaceAttr::get(ctx);
    auto loc = op.getLoc();
    auto tensorType = op.getSrc().getType();
    auto order = getOrder(tensorType);
    auto ctaLayout = getCTALayout(tensorType.getEncoding());
    auto m = op->getParentOfType<ModuleOp>();
    auto numWarps =
        mlir::cast<mlir::IntegerAttr>(m->getAttr("ttg.num-warps")).getInt();
    Attribute encoding = SwizzledSharedEncodingAttr::get(
        tensorType.getContext(), 1, 1, 1, order, ctaLayout);
    if (tensorType.getRank() > 1) {
      encoding = NVMMASharedEncodingAttr::get(
          tensorType.getContext(), tensorType.getShape(), order, ctaLayout,
          tensorType.getElementType(), false);
    }
    MemDescType memDescType =
        MemDescType::get(tensorType.getShape(), tensorType.getElementType(),
                         encoding, sharedMemorySpace, /*mutableMemory=*/true);
    Value alloc = rewriter.create<LocalAllocOp>(loc, memDescType, op.getSrc());
    rewriter.create<triton::nvidia_gpu::FenceAsyncSharedOp>(loc, false);
    // use id 2 for named barrier
    auto barId = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 2, 32);
    auto numThreads =
        rewriter.create<arith::ConstantIntOp>(op.getLoc(), numWarps * 32, 32);
    rewriter.create<NVVM::BarrierOp>(op.getLoc(), barId, numThreads);
    auto tensorOp =
        dyn_cast<ReinterpretTensorDescOp>(op.getDesc().getDefiningOp());
    rewriter.create<triton::nvidia_gpu::AsyncTMACopyLocalToGlobalOp>(
        loc, tensorOp.getRawDesc(), op.getIndices(), alloc);
    rewriter.create<triton::nvidia_gpu::TMAStoreWaitOp>(loc, 0);
    // Ensure all threads arrive at this point to avoid race conditions between
    // two TMA stores in Blackwell tests with sub-tiling enabled. Without this
    // barrier, TMAStoreWaitOp might be executed by another warp that is
    // slightly ahead of the warp issuing AsyncTMACopyLocalToGlobal. The barrier
    // ensures that all warps proceed simultaneously after the data is copied.
    rewriter.create<NVVM::BarrierOp>(op.getLoc(), barId, numThreads);
    rewriter.eraseOp(op);
    return success();
  }
};

class ParallelizeDescriptorLoad : public OpRewritePattern<DescriptorLoadOp> {
  ArefUseGraph &graph;

public:
  using OpRewritePattern::OpRewritePattern;

  ParallelizeDescriptorLoad(mlir::MLIRContext *context, ArefUseGraph &graph)
      : OpRewritePattern<DescriptorLoadOp>(context), graph(graph) {}

  LogicalResult matchAndRewrite(DescriptorLoadOp op,
                                PatternRewriter &rewriter) const override {

    auto ctx = op.getContext();
    auto loc = op.getLoc();
    SmallVector<Operation *> users(op->user_begin(), op->user_end());
    if (users.size() != 1)
      return failure();
    if (!isa<LocalStoreOp>(users[0]))
      return failure();

    auto putOp = dyn_cast<ArefPutOp>(op->getBlock()->getParentOp());
    if (!putOp)
      return failure();

    auto aref = putOp.getOperand();
    if (!graph.arefs.contains(aref))
      return failure();

    auto descOp =
        dyn_cast<ReinterpretTensorDescOp>(op.getDesc().getDefiningOp());
    if (!descOp)
      return failure();

    auto fullBarrier =
        getBarrier(rewriter, loc, graph.arefs[aref].fullMbars,
                   putOp.getIndexes()[0], graph.arefs[aref].depth);

    auto buf = users[0]->getOperand(1);
    Value pred = rewriter.create<arith::ConstantIntOp>(loc, 1, 1);
    rewriter.create<triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp>(
        loc, descOp.getRawDesc(), op.getIndices(), fullBarrier, buf, pred);
    int sizeInBytes = 0;
    auto memDesc = cast<MemDescType>(buf.getType());
    auto bufShape = memDesc.getShape();
    auto elementType = memDesc.getElementType();
    SmallVector<int64_t> tensorShape(bufShape.begin() + 1, bufShape.end());
    sizeInBytes +=
        product(tensorShape) * elementType.getIntOrFloatBitWidth() / 8;

    rewriter.create<triton::nvidia_gpu::BarrierExpectOp>(loc, fullBarrier,
                                                         sizeInBytes, pred);
    graph.nodes[putOp].containsAsync = true;

    rewriter.eraseOp(users[0]);
    rewriter.eraseOp(op);
    return success();
  }
};

class ParallelizeTCGen5MMA : public OpRewritePattern<TCGen5MMAOp> {
  ArefUseGraph &graph;

public:
  using OpRewritePattern::OpRewritePattern;

  ParallelizeTCGen5MMA(mlir::MLIRContext *context, ArefUseGraph &graph)
      : OpRewritePattern<TCGen5MMAOp>(context), graph(graph) {}

  LogicalResult matchAndRewrite(TCGen5MMAOp op,
                                PatternRewriter &rewriter) const override {
    auto ctx = op.getContext();
    auto loc = op.getLoc();
    auto getOp = dyn_cast<ArefGetOp>(op->getBlock()->getParentOp());
    if (!getOp)
      return failure();

    auto kLoop = dyn_cast<scf::ForOp>(getOp->getBlock()->getParentOp());
    if (!kLoop)
      return failure();

    auto putOp = dyn_cast<ArefPutOp>(kLoop->getBlock()->getParentOp());
    if (!putOp)
      return failure();

    auto putAref = putOp.getOperand();
    auto getAref = getOp.getOperand();
    if (!graph.arefs.contains(putAref) || !graph.arefs.contains(getAref))
      return failure();

    auto getArefValue = graph.arefs[getAref];
    auto putArefValue = graph.arefs[putAref];

    if (!op.getBarriers().empty())
      return failure(); // Can't paralleize an already parallel mma

    if (putAref.getType().getBaseType().size() != 1)
      return failure(
          "TODO: support putting into more than one tmem_buffer at a time");
    if (putOp.getRegion().getArguments()[0] != op.getD())
      return failure("This mma isn't accumulating to the put argument");

    rewriter.setInsertionPoint(op);

    // This assumes inputs to an mma are both from a single aref get. Need to
    // relax that to allow nested aref gets for attention
    Value getIdx = getArefValue.depth > 1
                       ? getOp.getIndexes()[0]
                       : rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    auto getEmptyBarrier = getBarrier(rewriter, loc, getArefValue.emptyMbars,
                                      getIdx, getArefValue.depth);
    op.addCompletionBarrier(getEmptyBarrier,
                            rewriter.create<arith::ConstantIntOp>(loc, 1, 1));

    // Lowering the Put op will tell the get op that reads the mma accumulator
    // to that the mma loop is done, but we need to wait on the last mma
    // operation inside the get.
    for (auto user : putArefValue.users) {
      auto mmaGetOp = dyn_cast<ArefGetOp>(user->op);
      if (!mmaGetOp)
        continue;
      Value idx;
      if (putArefValue.depth == 1) {
        rewriter.setInsertionPointAfter(kLoop);
        idx = getNumIterations(rewriter, kLoop);
      } else {
        // persistent
        // Get should be inside of a for loop
        auto forOp = dyn_cast<scf::ForOp>(mmaGetOp->getBlock()->getParentOp());
        if (!forOp)
          return failure();
        rewriter.setInsertionPoint(&mmaGetOp.getRegion().front().front());
        idx = getNumIterations(rewriter, forOp);
      }
      waitOnMbar(rewriter, loc, getArefValue.emptyMbars,
                 rewriter.create<arith::ConstantIntOp>(loc, 0, 32), idx,
                 getArefValue.depth);
    }

    // Since we are signalling the inputs to the mma are done with the mma
    // barrier, we don't need to add an arrival when lowering the get.
    graph.nodes[getOp].containsAsync = true;

    return success();
  }
};

class ParallelizeWarpGroupDot : public OpRewritePattern<WarpGroupDotOp> {
  ArefUseGraph &graph;

public:
  using OpRewritePattern::OpRewritePattern;

  ParallelizeWarpGroupDot(mlir::MLIRContext *context, ArefUseGraph &graph)
      : OpRewritePattern<WarpGroupDotOp>(context), graph(graph) {}

  LogicalResult matchAndRewrite(WarpGroupDotOp op,
                                PatternRewriter &rewriter) const override {
    auto ctx = op.getContext();
    auto loc = op.getLoc();
    auto getOp = dyn_cast<ArefGetOp>(op->getBlock()->getParentOp());
    if (!getOp)
      return failure();

    auto kLoop = dyn_cast<scf::ForOp>(getOp->getBlock()->getParentOp());
    if (!kLoop)
      return failure();

    auto getAref = getOp.getOperand();
    if (!graph.arefs.contains(getAref))
      return failure();

    auto getArefValue = graph.arefs[getAref];

    SmallVector<Operation *> users(op->user_begin(), op->user_end());
    if (users.size() != 1)
      return failure();

    auto wait = dyn_cast<WarpGroupDotWaitOp>(users[0]);
    if (!wait)
      return failure();

    if (wait.getPendings() == (getArefValue.depth - 1))
      return failure(); // we've already processed this

    wait.setPendings(getArefValue.depth - 1);
    rewriter.setInsertionPointAfter(wait);

    // This assumes inputs to an mma are both from a single aref get. Need to
    // relax that to allow nested aref gets for attention
    Value getIdx = getArefValue.depth > 1
                       ? getOp.getIndexes()[0]
                       : rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    auto depthVal =
        rewriter.create<arith::ConstantIntOp>(loc, getArefValue.depth - 1, 32);
    Value idx = rewriter.create<arith::SubIOp>(loc, getIdx, depthVal);
    auto zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    Value cond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                                idx, zero);
    auto ifOp = rewriter.create<scf::IfOp>(loc, SmallVector<Type>(), cond);
    rewriter.setInsertionPointToStart(&ifOp.getThenRegion().emplaceBlock());
    signalArrival(rewriter, loc, getArefValue.emptyMbars, idx,
                  getArefValue.depth);
    rewriter.create<scf::YieldOp>(loc);

    // Find the return and wait on it after the loop
    auto &retOp = getOp.getRegion().front().back();
    auto getRetIdx =
        llvm::find(retOp.getOperands(), wait.getResult(0)).getIndex();

    auto &yieldOp = kLoop.getRegion().front().back();
    auto yieldIdx =
        llvm::find(yieldOp.getOperands(), getOp.getResult(getRetIdx))
            .getIndex();
    assert(yieldIdx > 0 && yieldIdx < yieldOp.getNumResults());
    Value acc = kLoop.getResult(yieldIdx);

    rewriter.setInsertionPointAfter(kLoop);

    Value seqAcc =
        rewriter.create<WarpGroupDotWaitOp>(loc, acc, 0).getResult(0);
    rewriter.replaceAllUsesExcept(acc, seqAcc, seqAcc.getDefiningOp());
    // Since we manually signaled the that the barriers are empty,
    // we don't need to automatically do it
    graph.nodes[getOp].containsAsync = true;
    return success();
  }
};

static ArefUseGraph analyzeArefUseDef(ModuleOp m) {
  ArefUseGraph graph;
  DenseSet<Operation *> seen;

  ArefUseNode *parent = nullptr;
  std::function<void(Operation * op)> createGraph;
  // Psuedo recursion to get the dependency graph
  createGraph = [&](Operation *op) {
    if (seen.contains(op))
      return;
    ArefUseNode node;
    node.parent = parent;
    node.op = op;
    if (isa<ArefCreateOp>(op)) {
      graph.nodes[op] = node;
    } else if (auto get = dyn_cast<ArefGetOp>(op)) {
      graph.nodes[op] = node;
      auto old_parent = parent;
      parent = &node;
      get.getRegion().walk(createGraph);
      parent = old_parent;
    } else if (auto put = dyn_cast<ArefPutOp>(op)) {
      graph.nodes[op] = node;
      auto old_parent = parent;
      parent = &graph.nodes[op];
      put.getRegion().walk(createGraph);
      parent = old_parent;
    } else {
      return;
    }
    if (parent)
      graph.nodes[parent->op].subOps.push_back(&graph.nodes[op]);
    seen.insert(op);
  };

  m.walk(createGraph);

  return graph;
}

class NVWSLowerAref : public NVWSLowerArefBase<NVWSLowerAref> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::ModuleOp m = getOperation();

    ArefUseGraph graph = analyzeArefUseDef(m);

    mlir::RewritePatternSet patterns(context);
    patterns.add<LowerArefCreate, ParallelizeDescriptorLoad,
                 ParallelizeTCGen5MMA, ParallelizeWarpGroupDot>(context, graph);
    GreedyRewriteConfig config;

    if (applyPatternsGreedily(m, std::move(patterns), config).failed())
      signalPassFailure();

    OpPassManager pm;
    pm.addPass(mlir::createCSEPass());
    if (failed(runPipeline(pm, m)))
      return signalPassFailure();

    mlir::RewritePatternSet patterns2(context);
    patterns2.add<LowerArefPut, LowerArefGet>(context, graph);
    GreedyRewriteConfig config2;

    if (applyPatternsGreedily(m, std::move(patterns2), config2).failed())
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createNVWSLowerArefPass() {
  return std::make_unique<NVWSLowerAref>();
}
