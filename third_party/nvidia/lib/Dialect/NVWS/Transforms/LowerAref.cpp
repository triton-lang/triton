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

Value getBarrier(PatternRewriter &rewriter, Location loc, Value mBars,
                 Value idx) {
  return getBarrierAt(rewriter, loc, mBars, idx);
}

void waitOnMbar(PatternRewriter &rewriter, Location loc, Value mBars, Value idx,
                Value phase) {
  auto ctx = rewriter.getContext();
  auto barrier = getBarrier(rewriter, loc, mBars, idx);
  // wait on empty/full
  auto waitOp =
      rewriter.create<triton::nvidia_gpu::WaitBarrierOp>(loc, barrier, phase);
}

void signalArrival(PatternRewriter &rewriter, Location loc, Value mBars,
                   Value idx) {
  // And a barrier arrive to signal completion of the region if none of the
  // underlying ops are already async.
  auto barrier = getBarrierAt(rewriter, loc, mBars, idx);
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
LogicalResult LowerEnter(T op, PatternRewriter &rewriter, ArefUseGraph &graph) {
  auto ctx = op.getContext();
  auto loc = op.getLoc();
  auto aref = op.getAref();
  auto emptyMbars = graph.arefs[aref].emptyMbars;
  auto fullMbars = graph.arefs[aref].fullMbars;

  rewriter.setInsertionPoint(op);

  auto zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
  auto one = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);

  waitOnMbar(rewriter, loc, put ? emptyMbars : fullMbars, op.getIndex(),
             op.getPhase());

  SmallVector<Value> views;

  // slice aref values
  for (auto value : aref.template getDefiningOp<ArefCreateOp>().getOperands()) {
    if (auto mType = dyn_cast<MemDescType>(value.getType())) {
      auto shape = mType.getShape();
      SmallVector<int64_t> tensorShape(shape.begin() + 1, shape.end());
      auto memDescTypeNew =
          MemDescType::get(tensorShape, mType.getElementType(),
                           mType.getEncoding(), mType.getMemorySpace(), true);
      SmallVector<Value> offsets(mType.getShape().size(), zero);
      offsets[0] = op.getIndex();
      Value singleBuffer = rewriter.create<triton::gpu::MemDescSubviewOp>(
          loc, memDescTypeNew, value, offsets);
      views.push_back(singleBuffer);
    } else if (auto rType = dyn_cast<RankedTensorType>(value.getType())) {
      return op.emitError("FIXME: In-register Tensors not yet supported");
    } else {
      return op.emitError("Aref input type not supported for slicing");
    }
  }

  if (op.getResults().size() != views.size())
    return op.emitError("number of views and arguments mismatch.");
  for (auto [ret, view] : zip(op.getResults(), views))
    ret.replaceAllUsesWith(view);

  rewriter.eraseOp(op);
  return success();
}

template <bool put, typename T>
LogicalResult LowerExit(T op, PatternRewriter &rewriter, ArefUseGraph &graph) {
  auto ctx = op.getContext();
  auto loc = op.getLoc();
  auto aref = op.getAref();
  auto emptyMbars = graph.arefs[aref].emptyMbars;
  auto fullMbars = graph.arefs[aref].fullMbars;

  signalArrival(rewriter, loc, put ? fullMbars : emptyMbars, op.getIndex());
  rewriter.eraseOp(op);
  return success();
}

class LowerArefGetEnter : public OpRewritePattern<ArefGetEnterOp> {
  ArefUseGraph &graph;

public:
  using OpRewritePattern::OpRewritePattern;

  LowerArefGetEnter(mlir::MLIRContext *context, ArefUseGraph &graph)
      : OpRewritePattern<ArefGetEnterOp>(context), graph(graph) {}

  LogicalResult matchAndRewrite(ArefGetEnterOp op,
                                PatternRewriter &rewriter) const override {
    return LowerEnter<false>(op, rewriter, graph);
  }
};

class LowerArefGetExit : public OpRewritePattern<ArefGetExitOp> {
  ArefUseGraph &graph;

public:
  using OpRewritePattern::OpRewritePattern;

  LowerArefGetExit(mlir::MLIRContext *context, ArefUseGraph &graph)
      : OpRewritePattern<ArefGetExitOp>(context), graph(graph) {}

  LogicalResult matchAndRewrite(ArefGetExitOp op,
                                PatternRewriter &rewriter) const override {
    return LowerExit<false>(op, rewriter, graph);
  }
};

class LowerArefPutEnter : public OpRewritePattern<ArefPutEnterOp> {
  ArefUseGraph &graph;

public:
  using OpRewritePattern::OpRewritePattern;

  LowerArefPutEnter(mlir::MLIRContext *context, ArefUseGraph &graph)
      : OpRewritePattern<ArefPutEnterOp>(context), graph(graph) {}

  LogicalResult matchAndRewrite(ArefPutEnterOp op,
                                PatternRewriter &rewriter) const override {
    return LowerEnter<true>(op, rewriter, graph);
  }
};

class LowerArefPutExit : public OpRewritePattern<ArefPutExitOp> {
  ArefUseGraph &graph;

public:
  using OpRewritePattern::OpRewritePattern;

  LowerArefPutExit(mlir::MLIRContext *context, ArefUseGraph &graph)
      : OpRewritePattern<ArefPutExitOp>(context), graph(graph) {}

  LogicalResult matchAndRewrite(ArefPutExitOp op,
                                PatternRewriter &rewriter) const override {
    return LowerExit<true>(op, rewriter, graph);
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
    } else {
      // TODO: Fill this out
      return;
    }
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
    patterns.add<LowerArefCreate>(context, graph);
    GreedyRewriteConfig config;

    if (applyPatternsGreedily(m, std::move(patterns), config).failed())
      signalPassFailure();

    OpPassManager pm;
    pm.addPass(mlir::createCSEPass());
    if (failed(runPipeline(pm, m)))
      return signalPassFailure();

    mlir::RewritePatternSet patterns2(context);
    patterns2.add<LowerArefGetEnter, LowerArefGetExit, LowerArefPutEnter,
                  LowerArefPutExit>(context, graph);
    GreedyRewriteConfig config2;

    if (applyPatternsGreedily(m, std::move(patterns2), config2).failed())
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createNVWSLowerArefPass() {
  return std::make_unique<NVWSLowerAref>();
}
