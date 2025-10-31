#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Visitors.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Gluon/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/ADT/PriorityWorklist.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/xxhash.h"

#define DEBUG_TYPE "gluon-infer-efficient-encodings"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace ttg = mlir::triton::gpu;

namespace mlir::triton::gluon {

#define GEN_PASS_DEF_GLUONINFEREFFICIENTENCODINGSPASS
#include "triton/Dialect/Gluon/Transforms/Passes.h.inc"

namespace {
///
/// Coalesce
///
unsigned getNumElementsPerThread(Operation *op, SmallVector<unsigned> order,
                                 ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                 SmallVector<int64_t> shapePerCTA) {
  Value val = getMemAccessPtr(op);
  auto ty = cast<RankedTensorType>(val.getType());
  AxisInfo &valInfo = *axisInfoAnalysis.getAxisInfo(val);
  unsigned elemNumBits = getElementBitWidth(ty);
  unsigned elemNumBytes = std::max(elemNumBits / 8, 1u);
  unsigned maxMultipleBytes = valInfo.getDivisibility(order[0]);
  unsigned maxMultiple = std::max(maxMultipleBytes / elemNumBytes, 1u);
  unsigned maxContig =
      std::min(valInfo.getContiguity(order[0]), shapePerCTA[order[0]]);
  unsigned alignment = std::min(maxMultiple, maxContig);
  unsigned currPerThread = std::min(alignment, 128 / elemNumBits);
  LDBG("elemNumBytes: " << elemNumBytes
                        << ", divisibility: " << maxMultipleBytes
                        << ", contig: " << valInfo.getContiguity(order[0])
                        << ", alignment: " << alignment);
  return currPerThread;
}

ttg::CTALayoutAttr
getCTALayoutForEfficientEncodings(RankedTensorType refTensorType,
                                  unsigned numCTAs) {
  // TODO support numCTAs > 1
  assert(numCTAs == 1 && "only numCTAs == 1 is supported for now");
  assert(isa<gluon::EfficientEncodingAttr>(refTensorType.getEncoding()) &&
         "expected CTALayoutAttr encoding");
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

bool isEfficientEncodingTensorType(Type ty) {
  auto tensorTy = dyn_cast<RankedTensorType>(ty);
  return tensorTy && isa<gluon::EfficientEncodingAttr>(tensorTy.getEncoding());
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
inferEfficientLayouts(FuncOp func,
                      llvm::MapVector<Operation *, Attribute> &layoutMap) {
  // Disallow efficient encoding accross function call boundaries
  for (auto argTy : func.getArgumentTypes()) {
    if (isEfficientEncodingTensorType(argTy)) {
      return func->emitError(
          "Functions taking auto encoding must be fully inlined");
    }
  }
  for (auto resultTy : func.getResultTypes()) {
    if (isEfficientEncodingTensorType(resultTy))
      return func->emitError(
          "Functions returning auto encoding must be fully inlined");
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
        // here users set efficient layout back to some layout,
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
    assert(isa<gluon::EfficientEncodingAttr>(existingTy.getEncoding()));
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
inferEfficientLayouts(ModuleOp &mod,
                      llvm::MapVector<Operation *, Attribute> &layoutMap) {
  for (auto &op : *mod.getBody()) {
    auto func = dyn_cast<FuncOp>(&op);
    if (!func)
      continue;
    if (failed(inferEfficientLayouts(func, layoutMap)))
      return failure();
  }
  return success();
}
} // anonymous namespace

class GluonInferEfficientEncodingsPass
    : public impl::GluonInferEfficientEncodingsPassBase<
          GluonInferEfficientEncodingsPass> {
  void
  setCoalescedEncoding(ModuleAxisInfoAnalysis &axisInfoAnalysis, Operation *op,
                       int numWarps, int threadsPerWarp, int numCTAs,
                       llvm::MapVector<Operation *, Attribute> &layoutMap) {

    Value ptr = getMemAccessPtr(op);
    auto refTensorType = cast<RankedTensorType>(ptr.getType());
    auto CTALayout = getCTALayoutForEfficientEncodings(refTensorType, numCTAs);

    LDBG("Considering op: " << *op);
    LLVM_DEBUG({
      DBGS() << "axis info of pointer: ";
      axisInfoAnalysis.getAxisInfo(ptr)->print(llvm::dbgs());
      llvm::dbgs() << "\n";
    });

    auto contiguity = axisInfoAnalysis.getAxisInfo(ptr)->getContiguity();
    SmallVector<unsigned> order = getOrderFromContiguity(contiguity);
    LDBG("order=[" << triton::join(order, ", ") << "]");

    auto matchesShape = [&refTensorType](const Value &val) {
      auto rttType = dyn_cast<RankedTensorType>(val.getType());
      return rttType && rttType.getShape() == refTensorType.getShape();
    };

    // The desired divisibility is the maximum divisibility among all dependent
    // pointers which have the same shape and order as `ptr`.
    llvm::SmallSetVector<Operation *, 32> memAccessesSameOrder;
    memAccessesSameOrder.insert(op);
    if (ptr.getDefiningOp()) {
      for (Operation *use : mlir::getSlice(op)) {
        Value val = getMemAccessPtr(use);
        if (!val || !matchesShape(val) || memAccessesSameOrder.contains(use))
          continue;
        auto currOrder = getOrderFromContiguity(
            axisInfoAnalysis.getAxisInfo(val)->getContiguity());
        if (order == currOrder) {
          LDBG("multi-root-slice: insert to memAccessesSameOrder " << *use);
          memAccessesSameOrder.insert(use);
        }
      }
    }

    auto shapePerCTA = ttg::getShapePerCTA(CTALayout.getCTASplitNum(),
                                           refTensorType.getShape());

    LDBG("shapePerCTA=[" << triton::join(shapePerCTA, ", ") << "]");

    int numElems = product<int64_t>(shapePerCTA);
    int numThreads = numWarps * threadsPerWarp;

    unsigned perThread =
        getNumElementsPerThread(op, order, axisInfoAnalysis, shapePerCTA);
    LDBG("perThread for op: " << perThread);

    for (Operation *opSameOrder : memAccessesSameOrder) {
      if (opSameOrder == op)
        continue;
      unsigned currPerThread = getNumElementsPerThread(
          opSameOrder, order, axisInfoAnalysis, shapePerCTA);
      LDBG("perThread for opSameOrder: " << currPerThread);
      perThread = std::max(perThread, currPerThread);
    }

    perThread = std::min<int>(perThread, std::max(numElems / numThreads, 1));
    LDBG("perThread: " << perThread);

    if (!dyn_cast<triton::LoadOp>(op)) {
      // For ops that can result in a global memory write, we should enforce
      // that each thread handles at most 128 bits, which is the widest
      // available vectorized store op; otherwise, the store will have "gaps"
      // in the memory write at the warp level, resulting in worse performance.
      // For loads, we can expect that the gaps won't matter due to the L1
      // cache.
      perThread = std::min<int>(
          perThread,
          getNumElementsPerThread(op, order, axisInfoAnalysis, shapePerCTA));
    }
    SmallVector<unsigned> sizePerThread(refTensorType.getRank(), 1);
    sizePerThread[order[0]] = perThread;

    layoutMap[op] = triton::gpu::BlockedEncodingAttr::get(
        &getContext(), refTensorType.getShape(), sizePerThread, order, numWarps,
        threadsPerWarp, CTALayout);
  }

  //
  // triton coalesce results for reference:
  // ./build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt --tritongpu-coalesce
  // custom_bench/tt_coalesc.mlir -debug-only tritongpu-coalesce > tmp.mlir
  //
  void runOnOperation() override {
    // Run axis info analysis
    ModuleOp moduleOp = getOperation();
    ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

    // 1. for every load/store with efficient encoding,
    // infer efficient encoding for ptrs
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
      // we only consider those with efficient encoding
      if (auto tensorType = dyn_cast<RankedTensorType>(ptr.getType())) {
        auto encoding = tensorType.getEncoding();
        if (!encoding || !isa<gluon::EfficientEncodingAttr>(encoding))
          return;
      }

      int numWarps = ttg::lookupNumWarps(curr);
      int numCTAs = ttg::lookupNumCTAs(curr);
      setCoalescedEncoding(axisInfoAnalysis, curr, numWarps, threadsPerWarp,
                           numCTAs, layoutMap);
    });

    // TODO: descriptor load/store??

    // 2. propagate forward/backward
    // similar to ResolveAutoLayoutPass.cpp
    //
    // for backward slice, it doesn't cross the set_auto_layout boundary
    // i.e. gl.set_auto_layout(val, gl.EfficientLayout())
    // -> gl.set_auto_layout(val, concrete coalesced layout)
    // then ResolveAutoLayoutPass will handle the rest
    //
    if (failed(inferEfficientLayouts(moduleOp, layoutMap)))
      return signalPassFailure();

    // Double check we didn't miss anything
    auto res = moduleOp.walk([](Operation *op) -> WalkResult {
      for (auto resTy : op->getResultTypes()) {
        if (isEfficientEncodingTensorType(resTy)) {
          return op->emitOpError("Failed to infer return type");
        }
      }
      return success();
    });
    if (res.wasInterrupted())
      return signalPassFailure();

    res = moduleOp.walk([](Block *block) -> WalkResult {
      for (auto argTy : block->getArgumentTypes()) {
        if (isEfficientEncodingTensorType(argTy)) {
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
