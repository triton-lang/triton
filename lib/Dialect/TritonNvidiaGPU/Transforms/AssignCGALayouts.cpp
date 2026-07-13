#include <algorithm>

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
#include "llvm/Support/ErrorHandling.h"
#include <optional>

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

ttg::DistributedEncodingTrait
cloneWithCGALayout(ttg::DistributedEncodingTrait layout,
                   llvm::ArrayRef<int64_t> shape, int numWarps,
                   int threadsPerWarp, ttg::CGAEncodingAttr newCGALayout) {
  if (auto blockedLayout = dyn_cast<ttg::BlockedEncodingAttr>(layout)) {
    return ttg::BlockedEncodingAttr::get(
        layout.getContext(), shape, blockedLayout.getSizePerThread(),
        blockedLayout.getOrder(), numWarps, threadsPerWarp, newCGALayout);
  }

  if (auto sliceLayout = dyn_cast<ttg::SliceEncodingAttr>(layout)) {
    return ttg::SliceEncodingAttr::get(
        layout.getContext(), sliceLayout.getDim(),
        cloneWithCGALayout(sliceLayout.getParent(), shape, numWarps,
                           threadsPerWarp, newCGALayout));
  }

  assert(false && "cloneWithCGALayout not implemented for layout");
}

// Clone a blocked or sliced encoding with a new CGA layout while retaining the
// layout choices that affect accesses within a CTA. The shape-aware blocked
// layout builder redistributes threads and warps when the shape per CTA
// changes.
std::optional<Attribute> cloneWithCGALayout(RankedTensorType type,
                                            ttg::CGAEncodingAttr cgaLayout,
                                            Operation *scope) {
  auto layout = dyn_cast<ttg::DistributedEncodingTrait>(type.getEncoding());
  if (!layout)
    return std::nullopt;

  int numWarps = ttg::lookupNumWarps(scope);
  OpBuilder builder(scope);
  int threadsPerWarp = ttg::lookupThreadsPerWarp(builder);
  if (auto blocked = dyn_cast<ttg::BlockedEncodingAttr>(layout)) {
    return ttg::BlockedEncodingAttr::get(
        type.getContext(), type.getShape(), blocked.getSizePerThread(),
        blocked.getOrder(), numWarps, threadsPerWarp, cgaLayout);
  }

  if (auto slice = dyn_cast<ttg::SliceEncodingAttr>(layout)) {
    auto parent = dyn_cast<ttg::BlockedEncodingAttr>(slice.getParent());
    if (!parent)
      return std::nullopt;
    auto newParent = ttg::BlockedEncodingAttr::get(
        type.getContext(), parent.getSizePerThread(),
        parent.getThreadsPerWarp(), parent.getWarpsPerCTA(), parent.getOrder(),
        cgaLayout);
    return ttg::SliceEncodingAttr::get(type.getContext(), slice.getDim(),
                                       newParent);
  }

  return std::nullopt;
}

// Return the encoding that `value` should use to stay in the CTA group of
// `preferredEncoding`. Dot operand encodings carry additional parent-layout
// decisions, so reuse the preferred encoding when it describes the same
// operand. Other producers retain their existing intra-CTA layout and only
// adopt the preferred CGA layout.
std::optional<Attribute>
getEncodingInCGA(Value value, Attribute preferredEncoding, Operation *scope) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type)
    return std::nullopt;

  auto currentDot = dyn_cast<ttg::DotOperandEncodingAttr>(type.getEncoding());
  auto preferredDot = dyn_cast<ttg::DotOperandEncodingAttr>(preferredEncoding);
  if (currentDot && preferredDot &&
      currentDot.getOpIdx() == preferredDot.getOpIdx() &&
      currentDot.getKWidth() == preferredDot.getKWidth())
    return preferredEncoding;

  return cloneWithCGALayout(type, ttg::getCGALayout(preferredEncoding), scope);
}

// Rematerialize an exclusively-used producer slice in the requested CTA
// layout. Loads are replaced, never duplicated: if any value in the old slice
// has a user outside the slice and the root conversion, the rewrite is
// rejected.
class CGARematerialization {
public:
  explicit CGARematerialization(ttg::ConvertLayoutOp root) : root(root) {}

  LogicalResult run() {
    auto desiredEncoding =
        getEncodingInCGA(root.getSrc(), root.getType().getEncoding(), root);
    if (!desiredEncoding || failed(plan(root.getSrc(), *desiredEncoding)) ||
        !foundLoad || !hasExclusiveUses())
      return failure();

    SmallVector<Operation *> originalOps;
    originalOps.reserve(layouts.size());
    for (auto &entry : layouts)
      originalOps.push_back(entry.first.getDefiningOp());

    root.getSrcMutable().assign(rewrite(root.getSrc()));
    eraseOriginalSlice(originalOps);
    return success();
  }

private:
  LogicalResult plan(Value value, Attribute encoding) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type || type.getEncoding() == encoding)
      return success();

    auto [it, inserted] = layouts.try_emplace(value, encoding);
    if (!inserted)
      return success(it->second == encoding);

    Operation *op = value.getDefiningOp();
    if (!op || op->getNumResults() != 1 || !op->getRegions().empty())
      return failure();

    if (auto load = dyn_cast<triton::LoadOp>(op)) {
      if (load.getIsVolatile())
        return failure();
      foundLoad = true;
      return success();
    }
    if (isa<triton::DescriptorLoadLikeOpInterface>(op)) {
      foundLoad = true;
      return success();
    }

    if (!isMemoryEffectFree(op) || !canBeRematerialized(op))
      return failure();

    if (auto gather = dyn_cast<triton::GatherOp>(op))
      return plan(gather.getIndices(), encoding);

    if (auto convert = dyn_cast<ttg::ConvertLayoutOp>(op)) {
      auto srcEncoding = getEncodingInCGA(convert.getSrc(), encoding, op);
      if (!srcEncoding)
        return failure();
      return plan(convert.getSrc(), *srcEncoding);
    }

    for (auto [index, operand] : llvm::enumerate(op->getOpOperands())) {
      if (!isa<RankedTensorType>(operand.get().getType()))
        continue;
      Attribute srcEncoding;
      if (auto upcast = dyn_cast<ttg::UpcastFpOpInterface>(op))
        srcEncoding = upcast.inferSrcEncoding(index, encoding);
      else
        srcEncoding = inferSrcEncoding(op, encoding);
      if (!srcEncoding || failed(plan(operand.get(), srcEncoding)))
        return failure();
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
        if (user != root.getOperation() && !sliceOps.contains(user))
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
        auto type = dyn_cast<RankedTensorType>(operand.get().getType());
        if (!type)
          continue;
        auto newType = type.cloneWithEncoding(layoutIt->second);
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
    return rewritten.lookup(value);
  }

  static void eraseOriginalSlice(SmallVectorImpl<Operation *> &ops) {
    while (!ops.empty()) {
      auto it = llvm::find_if(ops, [](Operation *op) {
        return llvm::all_of(op->getResults(),
                            [](Value result) { return result.use_empty(); });
      });
      assert(it != ops.end() && "exclusive rematerialization slice has a use");
      (*it)->erase();
      ops.erase(it);
    }
  }

  ttg::ConvertLayoutOp root;
  DenseMap<Value, Attribute> layouts;
  DenseMap<Value, Value> rewritten;
  bool foundLoad = false;
};

Value convertValueToLayout(OpBuilder &builder, Location loc, Value value,
                           Attribute layout) {
  auto tensorTy = cast<RankedTensorType>(value.getType());
  if (tensorTy.getEncoding() == layout)
    return value;

  auto newTy = tensorTy.cloneWithEncoding(layout);
  return ttg::ConvertLayoutOp::create(builder, loc, newTy, value);
}

void convertOpOperandsToLayouts(
    Operation *op, llvm::ArrayRef<Attribute> operandLayouts,
    SmallVectorImpl<ttg::ConvertLayoutOp> &rematerializationRoots) {
  OpBuilder builder(op);
  Location loc = op->getLoc();
  for (auto [operand, layout] :
       llvm::zip(op->getOpOperands(), operandLayouts)) {
    Value value = convertValueToLayout(builder, loc, operand.get(), layout);
    if (auto convert = value.getDefiningOp<ttg::ConvertLayoutOp>())
      rematerializationRoots.push_back(convert);
    operand.set(value);
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

void assignDotCGALayout(
    triton::DotOp dot,
    SmallVectorImpl<ttg::ConvertLayoutOp> &rematerializationRoots) {
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
                             {newALayout, newBLayout, newDLayout},
                             rematerializationRoots);
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

void assignReduceCGALayout(
    triton::ReduceOp reduce,
    SmallVectorImpl<ttg::ConvertLayoutOp> &rematerializationRoots) {
  MLIRContext *ctx = reduce.getContext();
  Value src = reduce.getOperand(0);
  auto srcTy = cast<RankedTensorType>(src.getType());

  auto srcLayout = cast<ttg::DistributedEncodingTrait>(srcTy.getEncoding());

  OpBuilder builder(reduce);
  int threadsPerWarp = ttg::lookupThreadsPerWarp(builder);
  int numWarps = ttg::lookupNumWarps(reduce);

  auto cgaLayout = getReduceCGALayout(reduce, srcTy);
  auto newSrcLayout = cloneWithCGALayout(srcLayout, srcTy.getShape(), numWarps,
                                         threadsPerWarp, cgaLayout);

  SmallVector<Attribute> operandLayouts(reduce.getNumOperands(), newSrcLayout);
  Attribute resultLayout;
  if (srcTy.getRank() > 1)
    resultLayout =
        ttg::SliceEncodingAttr::get(ctx, reduce.getAxis(), newSrcLayout);
  SmallVector<Attribute> resultLayouts(reduce.getNumResults(), resultLayout);

  convertOpOperandsToLayouts(reduce.getOperation(), operandLayouts,
                             rematerializationRoots);
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

    SmallVector<ttg::ConvertLayoutOp> rematerializationRoots;
    mod.walk([&](Operation *op) {
      if (auto dot = dyn_cast<triton::DotOp>(op))
        assignDotCGALayout(dot, rematerializationRoots);
      if (auto reduce = dyn_cast<triton::ReduceOp>(op))
        assignReduceCGALayout(reduce, rematerializationRoots);
    });

    for (ttg::ConvertLayoutOp convert : rematerializationRoots)
      (void)CGARematerialization(convert).run();
  }
};

} // anonymous namespace

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
