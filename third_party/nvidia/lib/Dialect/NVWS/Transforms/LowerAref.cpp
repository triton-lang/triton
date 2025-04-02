
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
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"

#include <memory>
#include <optional>

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
  SmallVector<ArefUseNode *> inputs;
  ArefUseNode *parent;
  Operation *op;
  SmallVector<ArefUseNode *> subOps;
};

struct ArefValue {
  Value emptyMbars;
  Value fullMbars;
  int depth;
};

struct ArefUseGraph {
  llvm::MapVector<Operation *, ArefUseNode> nodes;
  llvm::MapVector<Value, ArefValue> arefs;
  SmallVector<ArefUseNode *> topLevelUses;
};

static MemDescType getBarrierMemDesc(MLIRContext *ctx,
                                     PatternRewriter &rewriter,
                                     llvm::ArrayRef<int64_t> shape) {
  Attribute sharedMemorySpace = triton::gpu::SharedMemorySpaceAttr::get(ctx);
  auto barrierCTALayout = CTALayoutAttr::get(
      /*context=*/ctx, /*CTAsPerCGA=*/{1},
      /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
  auto barrierEncoding =
      SwizzledSharedEncodingAttr::get(ctx, 1, 1, 1, {0}, barrierCTALayout);
  return MemDescType::get(shape, rewriter.getI64Type(), barrierEncoding,
                          sharedMemorySpace, /*mutableMemory=*/true);
}

static Value getBarrierAt(MLIRContext *ctx, Location loc,
                          PatternRewriter &rewriter, Value mbars, Value idx) {
  auto memDesc = getBarrierMemDesc(ctx, rewriter, {1});
  return rewriter.create<triton::gpu::MemDescSubviewOp>(
      loc, memDesc, mbars, SmallVector<Value>{idx});
}

static Value getStageIndex(PatternRewriter &rewriter, Location loc, Value idx,
                           int depth) {
  Value stageIdx = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
  if (depth > 1) {
    auto stage = rewriter.create<arith::RemSIOp>(
        loc, idx, rewriter.create<arith::ConstantIntOp>(loc, depth, 32));
    stageIdx = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIntegerType(32), stage);
  }
  return stageIdx;
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
    auto rank = op.getResult().getType().getNumBatchAxes();
    if (rank && *rank > 1)
      op.emitError("TODO: Implement multi-axis slicing");
    int depth = 1;
    if (rank) {
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
      auto memDesc = getBarrierMemDesc(ctx, rewriter, {1});
      auto singleBarrier = rewriter.create<triton::gpu::MemDescSubviewOp>(
          loc, memDesc, i == 0 ? emptyMbars.getResult() : fullMbars.getResult(),
          SmallVector<Value>{dLoop.getInductionVar()});

      int count = i == 0 ? 0 : 1;
      rewriter.create<InitBarrierOp>(loc, singleBarrier, count);
    }

    graph.arefs[op] =
        ArefValue{emptyMbars.getResult(), fullMbars.getResult(), depth};

    return success();
  }
};

template <int full = 0, typename T>
LogicalResult lowerRegion(T op, PatternRewriter &rewriter,
                          ArefUseGraph &graph) {
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

  auto zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
  auto one = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);

  Value idx = zero;
  if (depth > 1)
    idx = op.getIndexes()[0];

  rewriter.setInsertionPoint(op);

  auto stageIdx = getStageIndex(rewriter, loc, idx, depth);

  auto fullBarrier =
      getBarrierAt(ctx, loc, rewriter, full ? fullMbars : emptyMbars, stageIdx);
  // wait on full. Use a default phase for now, will rewrite later
  auto waitOp =
      rewriter.create<triton::nvidia_gpu::WaitBarrierOp>(loc, fullBarrier, one);

  SmallVector<Value> views;

  // slice aref values
  if (depth > 1) {
    for (auto value :
         aref.template getDefiningOp<ArefCreateOp>().getOperands()) {
      llvm::errs() << value << '\n';
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
    for (auto value : aref.template getDefiningOp<ArefCreateOp>().getOperands())
      views.push_back(value);
  }

  auto putOpBody = &op->getRegion(0).front();
  if (putOpBody->getArguments().size() != views.size()) {
    return op.emitError("number of views and arguments mismatch.");
  };
  for (auto [arg, view] : zip(putOpBody->getArguments(), views)) {
    arg.replaceAllUsesWith(view);
  }
  for (auto &bodyOp :
       llvm::make_early_inc_range(putOpBody->without_terminator()))
    bodyOp.moveBefore(op);

  // arrive on full mbar. To be rewritten later.
  auto emptyBarrier = getBarrierAt(ctx, loc, rewriter, emptyMbars, stageIdx);
  // wait on empty. Use a default phase for now, will rewrite later
  rewriter.create<triton::nvidia_gpu::ArriveBarrierOp>(
      loc, full ? emptyBarrier : fullBarrier, 1);

  rewriter.eraseOp(op);

  return success();
}

class LowerArefGet : public OpRewritePattern<ArefGetOp> {
  ArefUseGraph &graph;

public:
  using OpRewritePattern::OpRewritePattern;

  LowerArefGet(mlir::MLIRContext *context, ArefUseGraph &graph)
      : OpRewritePattern<ArefGetOp>(context), graph(graph) {}

  LogicalResult matchAndRewrite(ArefGetOp op,
                                PatternRewriter &rewriter) const override {
    return lowerRegion<1, ArefGetOp>(op, rewriter, graph);
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

    return lowerRegion<1, ArefPutOp>(op, rewriter, graph);
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
    patterns.add<LowerArefCreate, LowerArefGet, LowerArefPut>(context, graph);
    GreedyRewriteConfig config;

    if (applyPatternsGreedily(m, std::move(patterns), config).failed())
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createNVWSLowerArefPass() {
  return std::make_unique<NVWSLowerAref>();
}
