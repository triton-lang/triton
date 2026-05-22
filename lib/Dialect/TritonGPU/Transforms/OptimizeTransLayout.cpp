#include "mlir/Analysis/TopologicalSortUtils.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritongpu-optimize-trans-layout"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton::gpu {

#define GEN_PASS_DEF_TRITONGPUOPTIMIZETRANSLAYOUT
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;

// Returns true if `op`'s result encoding can be changed in place without
// cloning. Unlike RemoveLayoutConversions::canBeRemat, loads are allowed since
// no extra memory traffic is generated. Intentionally a narrow whitelist.
bool isInPlaceRetypeable(Operation *op) {
  // Layout-changing ops: result encoding derived from operand via
  // inferSrcEncoding.
  if (isa<tt::ExpandDimsOp, tt::BroadcastOp, tt::ReshapeOp, tt::JoinOp,
          tt::SplitOp, tt::TransOp>(op))
    return true;
  // Roots that can take any layout.
  if (isa<tt::LoadOp, tt::SplatOp, tt::MakeRangeOp, arith::ConstantOp>(op))
    return true;
  // Pointer arithmetic and elementwise ops preserve layout.
  if (isa<tt::AddPtrOp>(op))
    return true;
  if (op->hasTrait<OpTrait::Elementwise>() ||
      op->hasTrait<OpTrait::SameOperandsAndResultEncoding>())
    return true;
  return false;
}

// Walks backward from `root` under encoding `targetEnc`, populating `slice`
// with values that need retyping and `layoutMap` with their new encodings.
// Returns failure if any value in the backward slice is not retypeable.
LogicalResult collectRetypeSlice(Value root, Attribute targetEnc,
                                 llvm::SetVector<Value> &slice,
                                 llvm::DenseMap<Value, Attribute> &layoutMap) {
  SmallVector<std::pair<Value, Attribute>> worklist;
  worklist.push_back({root, targetEnc});

  while (!worklist.empty()) {
    auto [v, enc] = worklist.pop_back_val();
    auto tensorTy = dyn_cast<RankedTensorType>(v.getType());
    if (!tensorTy)
      continue;

    // Conflicting encoding for a previously visited value; bail.
    auto it = layoutMap.find(v);
    if (it != layoutMap.end()) {
      if (it->second != enc) {
        LDBG("conflicting encodings for value " << v);
        return failure();
      }
      continue;
    }

    // Already correct encoding; do not recurse further along this path.
    if (tensorTy.getEncoding() == enc) {
      layoutMap[v] = enc;
      continue;
    }

    layoutMap[v] = enc;
    slice.insert(v);

    Operation *def = v.getDefiningOp();
    if (!def) {
      LDBG("hit block argument; bailing");
      return failure();
    }
    if (!isInPlaceRetypeable(def)) {
      LDBG("non-retypeable op in slice: " << *def);
      return failure();
    }

    // inferSrcEncoding maps the result encoding to each operand's encoding,
    // accounting for shape-permuting ops (trans, expand_dims, etc.).
    for (OpOperand &operandUse : def->getOpOperands()) {
      Value operand = operandUse.get();
      if (!isa<RankedTensorType>(operand.getType()))
        continue;
      Attribute srcEnc = inferSrcEncoding(def, enc);
      if (!srcEnc) {
        LDBG("inferSrcEncoding failed for " << *def);
        return failure();
      }
      worklist.push_back({operand, srcEnc});
    }
  }
  return success();
}

// Returns true if every value in `slice` is used only by other ops in the
// slice or by `terminalConvert`. An external user would require an additional
// convert-layout to restore the original encoding, defeating the rewrite.
bool isSliceClosed(const llvm::SetVector<Value> &slice,
                   ttg::ConvertLayoutOp terminalConvert) {
  for (Value v : slice) {
    for (Operation *user : v.getUsers()) {
      if (user == terminalConvert)
        continue;
      // User op must have at least one result in the slice.
      bool userInSlice = false;
      for (Value r : user->getResults()) {
        if (slice.count(r)) {
          userInSlice = true;
          break;
        }
      }
      if (!userInSlice) {
        LDBG("value " << v << " has out-of-slice user " << *user);
        return false;
      }
    }
  }
  return true;
}

// Retypes `arith.constant` in place, updating its `value` attribute to match
// `newEnc`. Supports splat and non-splat DenseElementsAttr only.
LogicalResult retypeConstant(arith::ConstantOp constOp, Attribute newEnc) {
  auto rtt = dyn_cast<RankedTensorType>(constOp.getType());
  if (!rtt)
    return failure();
  auto newType = rtt.cloneWithEncoding(newEnc);
  auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue());
  if (!denseAttr)
    return failure();
  DenseElementsAttr newAttr;
  if (denseAttr.isSplat()) {
    newAttr = DenseElementsAttr::get(cast<ShapedType>(newType),
                                     denseAttr.getSplatValue<Attribute>());
  } else {
    // Non-splat: raw data is encoding-independent; reshape in place.
    newAttr = denseAttr.reshape(cast<ShapedType>(newType));
  }
  constOp.setValueAttr(newAttr);
  constOp.getResult().setType(newType);
  return success();
}

// Applies the encoding changes in `layoutMap` to all ops in `slice`,
// in topological order.
LogicalResult applyRetyping(const llvm::SetVector<Value> &slice,
                            const llvm::DenseMap<Value, Attribute> &layoutMap) {
  llvm::SetVector<Operation *> opsToRetype;
  for (Value v : slice) {
    if (Operation *def = v.getDefiningOp())
      opsToRetype.insert(def);
  }
  auto sorted = mlir::topologicalSort(opsToRetype);
  for (Operation *op : sorted) {
    if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
      auto it = layoutMap.find(constOp.getResult());
      if (it == layoutMap.end())
        continue;
      if (failed(retypeConstant(constOp, it->second)))
        return failure();
      continue;
    }
    for (Value result : op->getResults()) {
      auto it = layoutMap.find(result);
      if (it == layoutMap.end())
        continue;
      auto rtt = dyn_cast<RankedTensorType>(result.getType());
      if (!rtt)
        continue;
      result.setType(rtt.cloneWithEncoding(it->second));
    }
  }
  return success();
}

// Returns the number of contiguous elements in the memory-contig dimension
// of `loadOp` under `enc`, as determined by axis-info. Returns 0 when the
// encoding is non-blocked or axis-info is unavailable.
int64_t coalescedRunLength(tt::LoadOp loadOp, Attribute enc,
                           ModuleAxisInfoAnalysis &axisInfo) {
  auto blocked = dyn_cast<ttg::BlockedEncodingAttr>(enc);
  if (!blocked)
    return 0;
  Value ptr = loadOp.getPtr();
  auto *info = axisInfo.getAxisInfo(ptr);
  if (!info)
    return 0;
  SmallVector<int64_t> contiguity(info->getContiguity().begin(),
                                  info->getContiguity().end());
  SmallVector<unsigned> order = getOrderFromContiguity(contiguity);
  if (order.empty())
    return 0;
  unsigned memContigDim = order[0];
  if (memContigDim >= blocked.getSizePerThread().size())
    return 0;
  return int64_t(blocked.getSizePerThread()[memContigDim]) *
         int64_t(blocked.getThreadsPerWarp()[memContigDim]);
}

// Returns false if any load in `slice` would have a shorter coalesced run
// under the proposed encoding in `layoutMap`.
bool retypingPreservesCoalescing(
    const llvm::SetVector<Value> &slice,
    const llvm::DenseMap<Value, Attribute> &layoutMap,
    ModuleAxisInfoAnalysis &axisInfo) {
  for (Value v : slice) {
    Operation *def = v.getDefiningOp();
    auto loadOp = dyn_cast_or_null<tt::LoadOp>(def);
    if (!loadOp)
      continue;
    auto it = layoutMap.find(v);
    if (it == layoutMap.end())
      continue;
    auto oldRtt = dyn_cast<RankedTensorType>(loadOp.getType());
    if (!oldRtt)
      continue;
    int64_t oldRun = coalescedRunLength(loadOp, oldRtt.getEncoding(), axisInfo);
    int64_t newRun = coalescedRunLength(loadOp, it->second, axisInfo);
    if (oldRun == 0 || newRun == 0) {
      LDBG("coalescing check inconclusive for " << *loadOp
                                                << "; refusing rewrite");
      return false;
    }
    if (newRun < oldRun) {
      LDBG("retyping would shorten coalesced run on "
           << *loadOp << " from " << oldRun << " to " << newRun);
      return false;
    }
  }
  return true;
}

// Propagates the target encoding of `convertOp` backward through the upstream
// slice and retypes ops in place, then erases the convert. Returns true on
// success. Only fires when the slice contains a shape-permuting op (tt.trans
// or tt.reshape with allow_reorder), the slice is closed, and retyping would
// not reduce memory coalescing on any load.
bool tryOptimizeConvert(ttg::ConvertLayoutOp convertOp,
                        ModuleAxisInfoAnalysis &axisInfo) {
  Value srcVal = convertOp.getSrc();
  Attribute targetEnc = convertOp.getType().getEncoding();

  llvm::SetVector<Value> slice;
  llvm::DenseMap<Value, Attribute> layoutMap;
  if (failed(collectRetypeSlice(srcVal, targetEnc, slice, layoutMap)))
    return false;
  if (slice.empty())
    return false;

  // Only fire when the slice contains a shape-permuting op; plain encoding
  // mismatches are handled by RemoveLayoutConversions.
  bool hasShapePermutingOp = false;
  for (Value v : slice) {
    Operation *def = v.getDefiningOp();
    if (!def)
      continue;
    if (isa<tt::TransOp>(def)) {
      hasShapePermutingOp = true;
      break;
    }
    if (auto reshape = dyn_cast<tt::ReshapeOp>(def)) {
      if (reshape.getAllowReorder()) {
        hasShapePermutingOp = true;
        break;
      }
    }
  }
  if (!hasShapePermutingOp)
    return false;

  // Require closed slice: no external users.
  if (!isSliceClosed(slice, convertOp))
    return false;

  // Don't trade away memory coalescing on any load in the slice.
  if (!retypingPreservesCoalescing(slice, layoutMap, axisInfo))
    return false;

  LDBG("rewriting convert " << convertOp << " with slice of size "
                            << slice.size());

  if (failed(applyRetyping(slice, layoutMap)))
    return false;

  // Source now has the target encoding; replace uses and erase.
  convertOp.getResult().replaceAllUsesWith(convertOp.getSrc());
  convertOp.erase();
  return true;
}

struct OptimizeTransLayoutPass
    : public impl::TritonGPUOptimizeTransLayoutBase<OptimizeTransLayoutPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    ModuleAxisInfoAnalysis axisInfo(mod);

    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<ttg::ConvertLayoutOp> candidates;
      mod.walk([&](ttg::ConvertLayoutOp cvt) { candidates.push_back(cvt); });
      for (ttg::ConvertLayoutOp cvt : candidates) {
        // Skip converts erased by an earlier iteration.
        if (!cvt || !cvt->getParentOp())
          continue;
        if (tryOptimizeConvert(cvt, axisInfo))
          changed = true;
      }
    }
  }
};

} // namespace

} // namespace mlir::triton::gpu
