#include <algorithm>
#include <cassert>
#include <deque>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/ADT/DenseSet.h"

namespace ttg = mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUASSIGNCGALAYOUTSPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

// Default CGA layouts are assigned during Triton-to-TritonGPU conversion. This
// pass only gives Dot/Reduce ops their preferred CGA layout and materializes
// the boundary with ttg.convert_layout; RemoveLayoutConversions cleans up
// later.
struct DotCGASplit {
  unsigned m;
  unsigned n;
};

// Clone an encoding with `cgaLayout` while retaining its existing intra-CTA
// layout choices. The shape-aware blocked layout builder redistributes threads
// and warps when the shape per CTA changes.
Attribute cloneWithCGALayout(RankedTensorType tensorTy,
                             ttg::CGAEncodingAttr cgaLayout, Operation *scope) {
  Attribute layout = tensorTy.getEncoding();
  int numWarps = ttg::lookupNumWarps(scope);
  OpBuilder builder(scope);
  int threadsPerWarp = ttg::lookupThreadsPerWarp(builder);
  if (auto dot = dyn_cast<ttg::DotOperandEncodingAttr>(layout)) {
    auto parent = cast<ttg::BlockedEncodingAttr>(dot.getParent());
    auto newParent = ttg::BlockedEncodingAttr::get(
        tensorTy.getContext(), parent.getSizePerThread(),
        parent.getThreadsPerWarp(), parent.getWarpsPerCTA(), parent.getOrder(),
        cgaLayout);
    return ttg::DotOperandEncodingAttr::get(
        tensorTy.getContext(), dot.getOpIdx(), newParent, dot.getKWidth());
  }

  if (auto blocked = dyn_cast<ttg::BlockedEncodingAttr>(layout)) {
    return ttg::BlockedEncodingAttr::get(
        tensorTy.getContext(), tensorTy.getShape(), blocked.getSizePerThread(),
        blocked.getOrder(), numWarps, threadsPerWarp, cgaLayout);
  }

  if (auto slice = dyn_cast<ttg::SliceEncodingAttr>(layout)) {
    if (auto parent = dyn_cast<ttg::BlockedEncodingAttr>(slice.getParent())) {
      auto newParent = ttg::BlockedEncodingAttr::get(
          tensorTy.getContext(), parent.getSizePerThread(),
          parent.getThreadsPerWarp(), parent.getWarpsPerCTA(),
          parent.getOrder(), cgaLayout);
      return ttg::SliceEncodingAttr::get(tensorTy.getContext(), slice.getDim(),
                                         newParent);
    } else if (auto parent =
                   dyn_cast<ttg::SliceEncodingAttr>(slice.getParent())) {
      auto parentShape = slice.paddedShape(tensorTy.getShape());
      auto newParent = cloneWithCGALayout(
          RankedTensorType::get(parentShape, tensorTy.getElementType(), parent),
          cgaLayout, scope);
      return ttg::SliceEncodingAttr::get(
          tensorTy.getContext(), slice.getDim(),
          cast<ttg::DistributedEncodingTrait>(newParent));
    }
  }

  assert(false && "cloneWithCGALayout not implemented for encoding");
  return {};
}

// Rematerialize an exclusively-used producer slice in the requested CTA
// layout. Loads are replaced, never duplicated: if any value in the old slice
// has a user outside the slice and the specific consumer operand being
// rewritten, the rewrite is rejected.
class CGARematerialization {
public:
  CGARematerialization(OpOperand &root, Attribute desiredEncoding)
      : root(root), desiredEncoding(desiredEncoding) {}

  LogicalResult run() {
    if (failed(plan(root.get(), desiredEncoding)) ||
        !foundLoadWithDifferentCGALayout || !hasExclusiveUses())
      return failure();

    root.set(rewrite(root.get()));
    for (Operation *op : llvm::reverse(originalOps))
      op->erase();
    return success();
  }

private:
  LogicalResult plan(Value rootValue, Attribute rootEncoding) {
    std::deque<std::pair<Value, Attribute>> queue;
    queue.emplace_back(rootValue, rootEncoding);

    while (!queue.empty()) {
      auto [value, encoding] = queue.front();
      queue.pop_front();
      auto tensorTy = cast<RankedTensorType>(value.getType());

      auto [it, inserted] = layouts.try_emplace(value, encoding);
      if (!inserted) {
        if (it->second != encoding)
          return failure();
        continue;
      }

      Operation *op = value.getDefiningOp();
      // 1. !op: don't slice back through block arguments as it requires alias
      // 2. analysis op->getNumResults() != 1: Complexity consideration. If an
      // op has multiple results, we would have to slice all of them.
      // 3. !op->getRegions().empty(): Complexity consideration. Change region
      // operations requires rewriting yields, arguments and capture values
      if (!op || op->getNumResults() != 1 || !op->getRegions().empty())
        return failure();

      if (auto load = dyn_cast<triton::LoadOp>(op)) {
        if (load.getIsVolatile())
          return failure();
        foundLoadWithDifferentCGALayout |=
            ttg::getCGALayout(tensorTy.getEncoding()) !=
            ttg::getCGALayout(encoding);
        continue;
      }
      if (isa<triton::DescriptorLoadLikeOpInterface>(op)) {
        foundLoadWithDifferentCGALayout |=
            ttg::getCGALayout(tensorTy.getEncoding()) !=
            ttg::getCGALayout(encoding);
        continue;
      }

      if (!isMemoryEffectFree(op) || !canBeRematerialized(op))
        return failure();

      if (auto gather = dyn_cast<triton::GatherOp>(op)) {
        queue.emplace_back(gather.getIndices(), encoding);
        continue;
      }

      for (Value operand : op->getOperands()) {
        auto operandTy = dyn_cast<RankedTensorType>(operand.getType());
        if (!operandTy)
          continue;
        Attribute srcEncoding =
            cloneWithCGALayout(operandTy, ttg::getCGALayout(encoding), op);
        queue.emplace_back(operand, srcEncoding);
      }
    }
    return success();
  }

  bool hasExclusiveUses() {
    DenseSet<Operation *> sliceOps;
    for (auto &entry : layouts)
      sliceOps.insert(entry.first.getDefiningOp());

    for (auto &entry : layouts) {
      for (OpOperand &use : entry.first.getUses()) {
        Operation *user = use.getOwner();
        if (&use != &root && !sliceOps.contains(user))
          return false;
      }
    }
    return true;
  }

  Value rewrite(Value value) {
    auto layoutIt = layouts.find(value);
    if (layoutIt == layouts.end())
      return value;
    if (auto it = rewritten.find(value); it != rewritten.end())
      return it->second;

    Operation *op = value.getDefiningOp();
    OpBuilder builder(op);
    IRMapping mapping;

    if (isa<triton::LoadOp>(op)) {
      for (OpOperand &operand : op->getOpOperands()) {
        auto tensorTy = dyn_cast<RankedTensorType>(operand.get().getType());
        if (!tensorTy)
          continue;
        auto newType = tensorTy.cloneWithEncoding(layoutIt->second);
        Value converted = ttg::ConvertLayoutOp::create(builder, op->getLoc(),
                                                       newType, operand.get());
        mapping.map(operand.get(), converted);
      }
    } else if (!isa<triton::DescriptorLoadLikeOpInterface>(op)) {
      for (Value operand : op->getOperands()) {
        if (isa<RankedTensorType>(operand.getType()))
          mapping.map(operand, rewrite(operand));
      }
    }

    Operation *newOp = builder.clone(*op, mapping);
    for (auto [oldResult, newResult] :
         llvm::zip(op->getResults(), newOp->getResults())) {
      auto it = layouts.find(oldResult);
      if (it == layouts.end())
        continue;
      auto oldType = cast<RankedTensorType>(oldResult.getType());
      newResult.setType(oldType.cloneWithEncoding(it->second));
      rewritten[oldResult] = newResult;
    }
    originalOps.push_back(op);
    return rewritten.lookup(value);
  }

  OpOperand &root;
  Attribute desiredEncoding;
  DenseMap<Value, Attribute> layouts;
  DenseMap<Value, Value> rewritten;
  SmallVector<Operation *> originalOps;
  bool foundLoadWithDifferentCGALayout = false;
};

Value convertValueToLayout(OpBuilder &builder, Location loc, Value value,
                           Attribute layout) {
  auto tensorTy = cast<RankedTensorType>(value.getType());
  if (tensorTy.getEncoding() == layout)
    return value;

  auto newTy = tensorTy.cloneWithEncoding(layout);
  return ttg::ConvertLayoutOp::create(builder, loc, newTy, value);
}

void convertOpOperandsToLayouts(Operation *op,
                                llvm::ArrayRef<Attribute> operandLayouts) {
  OpBuilder builder(op);
  Location loc = op->getLoc();
  for (auto [operand, layout] :
       llvm::zip(op->getOpOperands(), operandLayouts)) {
    Value value = operand.get();
    auto tensorTy = dyn_cast<RankedTensorType>(value.getType());
    if (!tensorTy)
      continue;

    // Probe the original producer slice before materializing a conversion.
    // Planning is side-effect free, so a failed rematerialization can cleanly
    // fall back to the ordinary layout conversion.
    if (succeeded(CGARematerialization(operand, layout).run()))
      continue;
    if (tensorTy.getEncoding() == layout)
      continue;
    operand.set(convertValueToLayout(builder, loc, value, layout));
  }
}

void convertOpResultsFromLayouts(Operation *op,
                                 llvm::ArrayRef<Attribute> resultLayouts) {
  OpBuilder builder(op->getContext());
  builder.setInsertionPointAfter(op);
  Location loc = op->getLoc();
  for (auto [result, resultLayout] :
       llvm::zip(op->getResults(), resultLayouts)) {
    if (auto originalTy = dyn_cast<RankedTensorType>(result.getType())) {
      result.setType(originalTy.cloneWithEncoding(resultLayout));
      Value converted =
          convertValueToLayout(builder, loc, result, originalTy.getEncoding());
      result.replaceAllUsesExcept(converted, converted.getDefiningOp());
    }
  }
}

DotCGASplit getDotCGASplit(int64_t m, int64_t n, unsigned numCTAs) {
  constexpr unsigned kPreferredChunkSize = 128;
  constexpr unsigned kMinChunkSize = 64;
  auto isLegalChunkSize = [](unsigned chunk) { return chunk >= kMinChunkSize; };

  unsigned splitM = 1;
  unsigned splitN = numCTAs;
  // Prefer a larger M chunk, up to 128 elements, by assigning splitM first.
  // splitN gets the remaining CTAs as long as each N chunk has at least 64
  // elements.
  for (unsigned chunkM = kPreferredChunkSize; isLegalChunkSize(chunkM);
       chunkM /= 2) {
    splitM = std::clamp<unsigned>(m / chunkM, 1, numCTAs);
    splitN = numCTAs / splitM;
    if (isLegalChunkSize(n / splitN))
      break;
  }

  return {splitM, splitN};
}

void assignDotCGALayout(triton::DotOp dot) {
  MLIRContext *ctx = dot.getContext();

  auto aTy = cast<RankedTensorType>(dot.getA().getType());
  auto bTy = cast<RankedTensorType>(dot.getB().getType());
  auto dTy = cast<RankedTensorType>(dot.getD().getType());

  auto aLayout = cast<ttg::DotOperandEncodingAttr>(aTy.getEncoding());
  auto bLayout = cast<ttg::DotOperandEncodingAttr>(bTy.getEncoding());
  auto dLayout = cast<ttg::BlockedEncodingAttr>(dTy.getEncoding());

  DotCGASplit split = getDotCGASplit(dTy.getShape()[0], dTy.getShape()[1],
                                     ttg::getNumCTAs(dLayout));

  OpBuilder builder(dot);
  int threadsPerWarp = ttg::lookupThreadsPerWarp(builder);
  int numWarps = ttg::lookupNumWarps(dot);

  auto newCGALayout = ttg::CGAEncodingAttr::fromSplitParams(
      ctx, {split.m, split.n}, {split.m, split.n}, {1, 0});
  auto newDLayout = ttg::BlockedEncodingAttr::get(
      ctx, dTy.getShape(), dLayout.getSizePerThread(), dLayout.getOrder(),
      numWarps, threadsPerWarp, newCGALayout);
  auto newALayout = ttg::DotOperandEncodingAttr::get(
      ctx, aLayout.getOpIdx(), newDLayout, aLayout.getKWidth());
  auto newBLayout = ttg::DotOperandEncodingAttr::get(
      ctx, bLayout.getOpIdx(), newDLayout, bLayout.getKWidth());

  convertOpOperandsToLayouts(dot.getOperation(),
                             {newALayout, newBLayout, newDLayout});
  convertOpResultsFromLayouts(dot.getOperation(), {newDLayout});
}

ttg::CGAEncodingAttr getReduceCGALayout(triton::ReduceOp reduce,
                                        RankedTensorType srcTy) {
  unsigned rank = srcTy.getRank();
  auto order = ttg::getOrder(srcTy);
  auto sizePerThread = ttg::getContigPerThread(srcTy);
  auto srcLayout = cast<ttg::DistributedEncodingTrait>(srcTy.getEncoding());

  SmallVector<unsigned> ctasPerCGA(rank, 0);
  unsigned remainingCTAs = ttg::getNumCTAs(srcLayout);
  // Keep the reduced dimension within a single CTA so reductions do not cross
  // CTAs. Assign CTAs to the remaining dimensions in layout order, bounded by
  // how many elements each CTA can cover in that dimension.
  for (int i = rank - 1; i >= 0; --i) {
    unsigned dim = order[i];
    if (dim == reduce.getAxis()) {
      ctasPerCGA[dim] = 1;
      continue;
    }

    ctasPerCGA[dim] = std::min<unsigned>(
        srcTy.getShape()[dim] / sizePerThread[dim], remainingCTAs);
    ctasPerCGA[dim] = std::max(ctasPerCGA[dim], 1u);
    remainingCTAs /= ctasPerCGA[dim];
  }

  // Put any leftover CTAs on the fastest non-reduced dimension.
  bool assignedRemainingCTAs = false;
  for (int i = rank - 1; i >= 0; --i) {
    unsigned dim = order[i];
    if (dim == reduce.getAxis())
      continue;
    ctasPerCGA[dim] *= remainingCTAs;
    assignedRemainingCTAs = true;
    break;
  }

  SmallVector<unsigned> ctaSplitNum = ctasPerCGA;
  // If numCTAs > 1 and the only dimension is the reduced dimension, the loops
  // above leave all CTAs unassigned. Put the remaining CTAs on that dimension
  // so that they all collaborate in the reduction.
  if (!assignedRemainingCTAs && remainingCTAs > 0) {
    ctasPerCGA[order[rank - 1]] *= remainingCTAs;
    ctaSplitNum[order[rank - 1]] = ctasPerCGA[order[rank - 1]];
  }

  auto ctaOrder = ttg::getCTAOrder(srcLayout);
  return ttg::CGAEncodingAttr::fromSplitParams(reduce.getContext(), ctasPerCGA,
                                               ctaSplitNum, ctaOrder);
}

void assignReduceCGALayout(triton::ReduceOp reduce) {
  MLIRContext *ctx = reduce.getContext();
  Value src = reduce.getOperand(0);
  auto srcTy = cast<RankedTensorType>(src.getType());

  auto cgaLayout = getReduceCGALayout(reduce, srcTy);
  Attribute newSrcLayout = cloneWithCGALayout(srcTy, cgaLayout, reduce);

  SmallVector<Attribute> operandLayouts(reduce.getNumOperands(), newSrcLayout);
  Attribute resultLayout;
  if (srcTy.getRank() > 1)
    resultLayout = ttg::SliceEncodingAttr::get(
        ctx, reduce.getAxis(),
        cast<ttg::DistributedEncodingTrait>(newSrcLayout));
  SmallVector<Attribute> resultLayouts(reduce.getNumResults(), resultLayout);

  convertOpOperandsToLayouts(reduce.getOperation(), operandLayouts);
  convertOpResultsFromLayouts(reduce.getOperation(), resultLayouts);
}

struct AssignCGALayoutsPass
    : public impl::TritonNvidiaGPUAssignCGALayoutsPassBase<
          AssignCGALayoutsPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    unsigned numCTAs = ttg::TritonGPUDialect::getNumCTAs(mod);
    if (numCTAs == 1)
      return;

    mod.walk([&](Operation *op) {
      if (auto dot = dyn_cast<triton::DotOp>(op))
        assignDotCGALayout(dot);
      if (auto reduce = dyn_cast<triton::ReduceOp>(op))
        assignReduceCGALayout(reduce);
    });
  }
};

} // anonymous namespace

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
