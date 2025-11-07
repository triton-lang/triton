#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Visitors.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Gluon/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/CoalesceUtils.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/ADT/PriorityWorklist.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/xxhash.h"

#define DEBUG_TYPE "gluon-infer-coalesced-encodings"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace ttg = mlir::triton::gpu;

namespace mlir::triton::gluon {

#define GEN_PASS_DEF_GLUONINFERCOALESCEDENCODINGSPASS
#include "triton/Dialect/Gluon/Transforms/Passes.h.inc"

namespace {

ttg::CTALayoutAttr getDefaultCTALayout(RankedTensorType refTensorType,
                                       int numCTAs) {
  // TODO support numCTAs > 1
  assert(numCTAs == 1 && "only numCTAs == 1 is supported for now");
  return ttg::CTALayoutAttr::getDefault(refTensorType.getContext(),
                                        refTensorType.getShape().size());
}
///
/// Propagation
///
bool encodingsMayVary(Operation *op) {
  return isa<triton::JoinOp, triton::SplitOp, triton::ReshapeOp, triton::CatOp,
             triton::TransOp>(op);
}

bool isCoalescedEncodingTensorType(Type ty) {
  auto tensorTy = dyn_cast<RankedTensorType>(ty);
  return tensorTy && isa<gluon::CoalescedEncodingAttr>(tensorTy.getEncoding());
}

struct LayoutInfo {
  Attribute encoding;
  // Some operations can infer one of many encodings,
  // we model this by setting the mayVary flag on encodings
  // derived from these ops.
  // If "may vary" is set then we allow conflicts, and when
  // resolving conflicts we prefer encodings that are not allowed to vary.
  bool mayVary = false;

  operator bool() { return bool(encoding); }
};

uint64_t hashWithMemo(Attribute attr,
                      llvm::MapVector<Attribute, uint64_t> &hashMemo) {
  auto it = hashMemo.find(attr);
  if (it != hashMemo.end()) {
    return it->second;
  }

  // llvm::hash_value is not stable, so instead we hash the string repr of the
  // attribute
  std::string str;
  llvm::raw_string_ostream os(str);
  attr.print(os);
  auto hash = llvm::xxh3_64bits(str);
  hashMemo.try_emplace(attr, hash);
  return hash;
}

bool compare(Attribute a, Attribute b,
             llvm::MapVector<Attribute, uint64_t> &hashMemo) {
  if (a == b)
    return false;

  return hashWithMemo(a, hashMemo) > hashWithMemo(b, hashMemo);
}

LayoutInfo combineInfo(LayoutInfo lhs, LayoutInfo rhs, Operation *op,
                       llvm::MapVector<Attribute, uint64_t> &hashMemo) {
  // Sort inputs so this operation is commutative
  if (compare(lhs.encoding, rhs.encoding, hashMemo)) {
    std::swap(lhs, rhs);
  }
  if (lhs.mayVary)
    return rhs;
  if (rhs.mayVary)
    return lhs;
  if (lhs.encoding == rhs.encoding)
    return lhs;
  op->emitOpError("found conflicting encodings for value:\n  ")
      << lhs.encoding << "\nand\n  " << rhs.encoding;
  return {};
}

LogicalResult
inferCoalescedLayouts(FuncOp func,
                      llvm::MapVector<Operation *, Attribute> &layoutMap) {
  // Disallow coalesced encoding accross function call boundaries
  for (auto argTy : func.getArgumentTypes()) {
    if (isCoalescedEncodingTensorType(argTy)) {
      return func->emitError(
          "Functions taking coalesced encoding must be fully inlined");
    }
  }
  for (auto resultTy : func.getResultTypes()) {
    if (isCoalescedEncodingTensorType(resultTy))
      return func->emitError(
          "Functions returning coalesced encoding must be fully inlined");
  }

  llvm::MapVector<Value, LayoutInfo> valueToEncoding;
  llvm::PriorityWorklist<Value> worklist;
  llvm::MapVector<Attribute, uint64_t> hashMemo;

  auto updateEncoding = [&](ArrayRef<Value> values,
                            LayoutInfo info) -> LogicalResult {
    for (auto value : values) {
      auto [it, inserted] = valueToEncoding.insert({value, info});
      if (!inserted) {
        auto defOp = value.getDefiningOp();
        auto op = defOp ? defOp : func;
        auto combine = combineInfo(it->second, info, op, hashMemo);
        if (!combine)
          return failure();
        if (combine == it->second)
          continue;
        it->second = combine;
      }
      LLVM_DEBUG({
        DBGS() << "Setting value:\n\t" << value << "\nto encoding:\n\t"
               << it->second.encoding << "\n";
      });
      worklist.insert(value);
    }
    return success();
  };

  // 1. Set seed values from layout map
  auto res = func.walk([&](Operation *op) -> WalkResult {
    if (layoutMap.find(op) == layoutMap.end())
      return WalkResult::advance();
    Attribute layout = layoutMap[op];
    return updateEncoding(llvm::to_vector_of<Value>(op->getOperands()),
                          LayoutInfo{layout, false});
  });
  if (res.wasInterrupted())
    return failure();

  // 2. Propagate encodings through the graph until fixed point, or conflict
  while (!worklist.empty()) {
    auto val = worklist.pop_back_val();
    auto info = valueToEncoding[val];
    assert(info);

    // Propagate to users
    for (OpOperand &use : val.getUses()) {
      auto op = use.getOwner();
      if (isa<scf::ForOp, scf::WhileOp>(op)) {
        auto offset = 3 * isa<scf::ForOp>(op);
        auto tiedArgs = getTiedArgs(op, use.getOperandNumber() - offset);
        if (failed(updateEncoding(tiedArgs, info)))
          return failure();
      } else if (isa<scf::YieldOp>(op)) {
        auto tiedArgs = getTiedArgs(op, use.getOperandNumber());
        if (failed(updateEncoding(tiedArgs, info)))
          return failure();
      } else if (isa<gluon::SetAutoLayoutOp>(op)) {
        // here users set Coalesced layout back to some layout,
        // should not happen
        return failure();
      } else {
        auto dstEnc = inferDstEncoding(op, info.encoding);
        if (dstEnc) {
          bool mayVary = info.mayVary || encodingsMayVary(op);
          LayoutInfo dstInfo{dstEnc, mayVary};
          if (failed(updateEncoding(llvm::to_vector_of<Value>(op->getResults()),
                                    dstInfo)))
            return failure();
        }
      }
    }

    // Propagate to defining ops
    if (auto opResult = dyn_cast<OpResult>(val)) {
      auto definingOp = opResult.getOwner();
      if (isa<scf::ForOp, scf::WhileOp, scf::IfOp>(definingOp)) {
        auto tiedArgs = getTiedArgs(definingOp, opResult.getResultNumber());
        if (failed(updateEncoding(tiedArgs, info)))
          return failure();
      } else {
        auto srcEncoding = inferSrcEncoding(definingOp, info.encoding);
        if (srcEncoding) {
          bool mayVary = info.mayVary || encodingsMayVary(definingOp);
          LayoutInfo srcInfo{srcEncoding, mayVary};
          llvm::SmallVector<Value> tensorOperands;
          for (auto operand : definingOp->getOperands())
            if (isa<RankedTensorType>(operand.getType()))
              tensorOperands.push_back(operand);

          if (failed(updateEncoding(tensorOperands, srcInfo)))
            return failure();
        }
      }
    } else if (auto blockArg = dyn_cast<BlockArgument>(val)) {
      auto parentOp = blockArg.getOwner()->getParentOp();
      if (isa<scf::ForOp, scf::WhileOp>(parentOp)) {
        auto offset = isa<scf::ForOp>(parentOp);
        auto tiedArgs = getTiedArgs(parentOp, blockArg.getArgNumber() - offset);
        if (failed(updateEncoding(tiedArgs, info)))
          return failure();
      }
    }
  }

  // 3. Transfer propagated encodings into the graph
  auto ctx = func.getContext();
  for (auto &[val, info] : valueToEncoding) {
    auto existingTy = cast<RankedTensorType>(val.getType());
    assert(isa<gluon::CoalescedEncodingAttr>(existingTy.getEncoding()));
    auto ty = existingTy.cloneWithEncoding(info.encoding);
    val.setType(ty);

    if (auto opResult = dyn_cast<OpResult>(val)) {
      if (auto constantOp = dyn_cast<arith::ConstantOp>(opResult.getOwner())) {
        auto value = cast<SplatElementsAttr>(constantOp.getValueAttr());
        auto newValue =
            SplatElementsAttr::get(ty, value.getSplatValue<Attribute>());
        constantOp.setValueAttr(newValue);
      }
    }
  }
  return success();
}

LogicalResult
inferCoalescedLayouts(ModuleOp &mod,
                      llvm::MapVector<Operation *, Attribute> &layoutMap) {
  for (auto &op : *mod.getBody()) {
    auto func = dyn_cast<FuncOp>(&op);
    if (!func)
      continue;
    if (failed(inferCoalescedLayouts(func, layoutMap)))
      return failure();
  }
  return success();
}
} // anonymous namespace

class GluonInferCoalescedEncodingsPass
    : public impl::GluonInferCoalescedEncodingsPassBase<
          GluonInferCoalescedEncodingsPass> {
  //
  // triton coalesce results for reference:
  // ./build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt --tritongpu-coalesce
  // custom_bench/tt_coalesc.mlir -debug-only tritongpu-coalesce > tmp.mlir
  //
  void runOnOperation() override {
    // Run axis info analysis
    ModuleOp moduleOp = getOperation();
    ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

    // 1. for every load/store with coalesced encoding,
    // infer coalesced encoding for ptrs
    //
    // similar to Coalesce.cpp
    //
    llvm::MapVector<Operation *, Attribute> layoutMap;
    int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(moduleOp);
    moduleOp.walk([&](Operation *curr) {
      Value ptr = getMemAccessPtr(curr);
      if (!ptr)
        return;
      // We only convert `tensor<tt.ptr<>>` load/store
      bool isPtrTensor = false;
      if (auto tensorType = dyn_cast<RankedTensorType>(ptr.getType()))
        isPtrTensor = isa<PointerType>(tensorType.getElementType());
      if (!isPtrTensor)
        return;
      // we only consider those with coalesced encoding
      if (auto tensorType = dyn_cast<RankedTensorType>(ptr.getType())) {
        auto encoding = tensorType.getEncoding();
        if (!encoding || !isa<gluon::CoalescedEncodingAttr>(encoding))
          return;
      }

      int numWarps = ttg::lookupNumWarps(curr);
      int numCTAs = ttg::lookupNumCTAs(curr);

      auto tensorType = cast<RankedTensorType>(ptr.getType());
      auto ctaLayout = getDefaultCTALayout(tensorType, numCTAs);
      auto shapePerCTA = ttg::getShapePerCTA(ctaLayout.getCTASplitNum(),
                                             tensorType.getShape());
      ttg::setCoalescedEncoding(&getContext(), axisInfoAnalysis, curr, numWarps,
                                threadsPerWarp, ctaLayout, shapePerCTA,
                                layoutMap);
    });

    // 2. propagate forward/backward
    // similar to ResolveAutoLayoutPass.cpp
    //
    // for backward slice, it doesn't cross the set_auto_layout boundary
    // i.e. gl.set_auto_layout(val, gl.CoalescedLayout())
    // -> gl.set_auto_layout(val, concrete coalesced layout)
    // then ResolveAutoLayoutPass will handle the rest
    //
    if (failed(inferCoalescedLayouts(moduleOp, layoutMap)))
      return signalPassFailure();

    // Double check we didn't miss anything
    auto res = moduleOp.walk([](Operation *op) -> WalkResult {
      for (auto resTy : op->getResultTypes()) {
        if (isCoalescedEncodingTensorType(resTy)) {
          return op->emitOpError("Failed to infer return type");
        }
      }
      return success();
    });
    if (res.wasInterrupted())
      return signalPassFailure();

    res = moduleOp.walk([](Block *block) -> WalkResult {
      for (auto argTy : block->getArgumentTypes()) {
        if (isCoalescedEncodingTensorType(argTy)) {
          return block->getParentOp()->emitError(
              "Failed to infer block argument type");
        }
      }
      return success();
    });
    if (res.wasInterrupted())
      return signalPassFailure();
  }
};
} // namespace mlir::triton::gluon
