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

ttg::CTALayoutAttr getCTALayout(mlir::MLIRContext *ctx, unsigned rank) {
  return ttg::CTALayoutAttr::getDefault(ctx, rank);
}

RankedTensorType toInferred(RankedTensorType rt, Attribute inferredLayout) {
  auto *ctx = rt.getContext();
  return RankedTensorType::get(rt.getShape(), rt.getElementType(), inferredLayout);
}

Value convertOne(Value v, OpBuilder &b, Location loc, Attribute inferredLayout) {
  if (auto rt = dyn_cast<RankedTensorType>(v.getType())) {
    if (rt.getEncoding() && isa<gluon::AutoEncodingAttr>(rt.getEncoding())) {
      auto resType = toInferred(rt, inferredLayout);
      return b.create<gluon::SetAutoLayoutOp>(loc, resType, v);
    }
  }
  return v;
}

void rewriteInferredEncodings(Operation* op, Attribute inferredLayout) {
  // Insert all new ops immediately before the load.
  OpBuilder builder(op);
  Location loc = op->getLoc();

  bool changed = false;
  SmallVector<Value, 8> newOperands;
  newOperands.reserve(op->getNumOperands());

  // For *every* operand of the load, convert auto->inferred
  for (Value operand : op->getOperands()) {
    Value maybeNew = convertOne(operand, builder, loc, inferredLayout);
    newOperands.push_back(maybeNew);
    changed |= (maybeNew != operand);
  }

  if (!changed)
    return;

  // Wire the operands in-place 
  for (auto it : llvm::enumerate(newOperands))
    op->setOperand(it.index(), it.value());

  // Finally, convert the result types 
  for (auto result : op->getResults()) {
    if (auto rt = dyn_cast<RankedTensorType>(result.getType())) {
      if (rt.getEncoding()) {
        assert(isa<gluon::AutoEncodingAttr>(rt.getEncoding()) &&
               "Expected auto encoding");
        auto newType = toInferred(rt, inferredLayout);
        result.setType(newType);
      }
    }
  }
}


///
/// Propagation
///

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

 
LogicalResult inferEfficientLayouts(FuncOp func, llvm::MapVector<Operation *, Attribute> &layoutMap) {
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
        return failure(); // TODO: resolve conflict??
      }
      // LLVM_DEBUG({
      //   DBGS() << "Setting value:\n\t" << value << "\nto encoding:\n\t"
      //          << it->second.encoding << "\n";
      // });
      llvm::outs() << "Setting value:\n\t" << value << "\nto encoding:\n\t"
                   << it->second.encoding << "\n";
      worklist.insert(value);
    }
    return success();
  };

  // 1. Set seed values from layout map
  auto res = func.walk([&](Operation *op) -> WalkResult {
    if (layoutMap.find(op) == layoutMap.end())
      return WalkResult::advance();
    Attribute layout = layoutMap[op];
    if (failed(updateEncoding(llvm::to_vector_of<Value>(op->getOperands()), LayoutInfo{layout, false})))
      return WalkResult::interrupt();
  });
  if (res.wasInterrupted())
    return failure();
  
  llvm::outs() << "Starting propagation...\n" << worklist.size() << " items in worklist.\n";

  // 2. Propagate encodings through the graph until fixed point, or conflict
  while (!worklist.empty()) {
    auto val = worklist.pop_back_val();
    auto info = valueToEncoding[val];
    assert(info);
  }

  // 3. Transfer propagated encodings into the graph
  auto ctx = func.getContext();
  for (auto &[val, info] : valueToEncoding) {
    auto existingTy = cast<RankedTensorType>(val.getType());
    assert(isa<gluon::AutoEncodingAttr>(existingTy.getEncoding()) ||
           isa<gluon::EfficientEncodingAttr>(existingTy.getEncoding()));
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

  // 4. Cleanup set_auto_layout ops
  func.walk([&](gluon::SetAutoLayoutOp op) {
    assert(op.getSrc().getType() == op.getType());
    op.getResult().replaceAllUsesWith(op.getSrc());
    op->erase();
  });

  return success();
}

LogicalResult inferEfficientLayouts(ModuleOp &mod, llvm::MapVector<Operation *, Attribute> &layoutMap) {
    
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
                       int numWarps, int threadsPerWarp,
                       llvm::MapVector<Operation *, Attribute> &layoutMap) {

    Value ptr = getMemAccessPtr(op);
    auto refTensorType = cast<RankedTensorType>(ptr.getType());

    // LDBG("Considering op: " << *op);
    // LLVM_DEBUG({
    //     DBGS() << "axis info of pointer: ";
    //     axisInfoAnalysis.getAxisInfo(ptr)->print(llvm::dbgs());
    //     llvm::dbgs() << "\n";
    // });
    llvm::outs() << "\n";
    llvm::outs() << "Considering op: " << *op << "\n";
    llvm::outs() << "axis info of pointer: ";
    axisInfoAnalysis.getAxisInfo(ptr)->print(llvm::outs());
    llvm::outs() << "\n";

    auto contiguity = axisInfoAnalysis.getAxisInfo(ptr)->getContiguity();
    SmallVector<unsigned> order = getOrderFromContiguity(contiguity);
    // LDBG("order=[" << triton::join(order, ", ") << "]");
    llvm::outs() << "order=[" << triton::join(order, ", ") << "]\n";

    auto matchesShape = [&refTensorType](const Value &val) {
      auto rttType = dyn_cast<RankedTensorType>(val.getType());
      return rttType && rttType.getShape() == refTensorType.getShape();
    };

    // The desired divisibility is the maximum divisibility among all dependent
    // pointers which have the same shape and order as `ptr`.
    llvm::SmallSetVector<Operation *, 32> memAccessesSameOrder;
    memAccessesSameOrder.insert(op);
    if (ptr.getDefiningOp()) {
      for (Operation *use : mlir::multiRootGetSlice(op)) {
        Value val = getMemAccessPtr(use);
        if (!val || !matchesShape(val) || memAccessesSameOrder.contains(use))
          continue;
        auto currOrder = getOrderFromContiguity(
            axisInfoAnalysis.getAxisInfo(val)->getContiguity());
        if (order == currOrder) {
          // LDBG("multi-root-slice: insert to memAccessesSameOrder " << *use);
          llvm::outs() << "multi-root-slice: insert to memAccessesSameOrder "
                       << *use << "\n";
          memAccessesSameOrder.insert(use);
        }
      }
    }

    // TODO: hardcode ctaSplitNum for now. read ctaSplitNum from frontend
    // TODO: options?
    // TODO: Or we implement LayoutEncodingTrait for efficient encoding ??
    // TODO: then we can just reuse the utility
    unsigned rank = refTensorType.getShape().size();
    SmallVector<unsigned> ctaSplitNum(rank, 1);
    auto shapePerCTA =
        ttg::getShapePerCTA(ctaSplitNum, refTensorType.getShape());
    // auto shapePerCTA = ttg::getShapePerCTA(refTensorType);

    // LDBG("shapePerCTA=[" << triton::join(shapePerCTA, ", ") << "]");
    llvm::outs() << "shapePerCTA=[" << triton::join(shapePerCTA, ", ") << "]\n";

    int numElems = product<int64_t>(shapePerCTA);
    int numThreads = numWarps * threadsPerWarp;

    unsigned perThread =
        getNumElementsPerThread(op, order, axisInfoAnalysis, shapePerCTA);
    // LDBG("perThread for op: " << perThread);
    llvm::outs() << "perThread for op: " << perThread << "\n";

    for (Operation *opSameOrder : memAccessesSameOrder) {
      if (opSameOrder == op)
        continue;
      unsigned currPerThread = getNumElementsPerThread(
          opSameOrder, order, axisInfoAnalysis, shapePerCTA);
      // LDBG("perThread for opSameOrder: " << currPerThread);
      llvm::outs() << "perThread for opSameOrder: " << currPerThread << "\n";
      perThread = std::max(perThread, currPerThread);
    }

    perThread = std::min<int>(perThread, std::max(numElems / numThreads, 1));
    // LDBG("perThread: " << perThread);
    llvm::outs() << "perThread: " << perThread << "\n";

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

    // TODO: hardcode for now. read ctaSplitNum from frontend options?
    // auto CTALayout = triton::gpu::getCTALayout(refTensorType.getEncoding());
    auto CTALayout =
        getCTALayout(&getContext(), refTensorType.getShape().size());
    layoutMap[op] = triton::gpu::BlockedEncodingAttr::get(
        &getContext(), refTensorType.getShape(), sizePerThread, order, numWarps,
        threadsPerWarp, CTALayout);
  }

  //
  // triton coalesce results for reference:
  // ./build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt --tritongpu-coalesce custom_bench/tt_coalesc.mlir -debug-only tritongpu-coalesce > tmp.mlir
  //
  void runOnOperation() override {

    // Run axis info analysis
    ModuleOp moduleOp = getOperation();
    ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

    // llvm::outs() << "\n";
    // llvm::outs() << "\n";
    // llvm::outs() << "\n";
    // llvm::outs() << "[AXIS INFO ANALYSIS RESULT]:\n";
    // for (auto &funcEntry : axisInfoAnalysis.funcMap) {
    //   llvm::outs() << "Function: " << funcEntry.first << "\n";
    //   for (auto &axisEntry : funcEntry.second) {
    //     llvm::outs() << "  Value: " << axisEntry.first << "\n";
    //     axisEntry.second.print(llvm::outs());
    //     llvm::outs() << "\n";
    //   }
    // }
    // llvm::outs() << "[END] Axis Info Analysis Result\n";
    // llvm::outs() << "\n";

    // 0. for every load/store with efficient encoding,
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
      setCoalescedEncoding(axisInfoAnalysis, curr, numWarps, threadsPerWarp,
                           layoutMap);
    });

    llvm::outs() << "\n";
    llvm::outs() << "[INFERRED LAYOUTS]:\n";
    for (auto &pair : layoutMap) {
      Operation *op = pair.first;
      Attribute layout = pair.second;
      llvm::outs() << "inferred layout for op: " << *op << "\n";
      layout.print(llvm::outs());
      llvm::outs() << "\n";
    }
    llvm::outs() << "[END] Inferred layouts:\n";
    llvm::outs() << "\n";
    llvm::outs() << "\n";

    // TODO: descriptor load/store??




    // 2. propagate forward/backward
    //
    // Do layout inference
    // similar to ResolveAutoLayoutPass.cpp
    //
    //if (failed(inferEfficientLayouts(moduleOp, layoutMap)))
    //  return signalPassFailure();


    // 1. for gluon.set_auto_layout ops, if it set to efficient encoding, erase it
    //    and set the operand to the result
    moduleOp.walk([&](gluon::SetAutoLayoutOp op) {
      auto layout = op.getType().getEncoding();
      if (!layout || !isa<gluon::EfficientEncodingAttr>(layout))
        return;
      llvm::outs() << "Erasing set_auto_layout op: " << *op << "\n";
      op.getResult().replaceAllUsesWith(op.getSrc());
      op->erase();
    });
    llvm::outs() << "\n";
    llvm::outs() << "[ERASE set_auto_layout]:\n";
    moduleOp.print(llvm::outs());
    llvm::outs() << "\n";
    llvm::outs() << "[END] REWRITE\n";

    // 2. for operation with an efficient encoding, rewrite to auto encoding
    // so that ResolveLayoutPass can propogate
    //
    moduleOp.walk([&](Operation *op) {
      // we will handle the inferred ones in the next step
      if (layoutMap.find(op) != layoutMap.end())
        return;

      for (auto result : op->getResults()) {
        auto tensorType = dyn_cast<RankedTensorType>(result.getType());
        if (!tensorType)
          continue;
        auto encoding = tensorType.getEncoding();
        if (!encoding || !isa<gluon::EfficientEncodingAttr>(encoding))
          continue;
        // rewrite to auto encoding
        auto newType = tensorType.cloneWithEncoding(
            gluon::AutoEncodingAttr::get(&getContext()));
        result.setType(newType);
        llvm::outs() << "Rewrote result type to auto encoding for op: " << *op
                     << "\n";
      }
      for (auto operand : op->getOperands()) {
        auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
        if (!tensorType)
          continue;
        auto encoding = tensorType.getEncoding();
        if (!encoding || !isa<gluon::EfficientEncodingAttr>(encoding))
          continue;
        // rewrite to auto encoding
        auto newType = tensorType.cloneWithEncoding(
            gluon::AutoEncodingAttr::get(&getContext()));
        operand.setType(newType);
        llvm::outs() << "Rewrote operand type to auto encoding for op: " << *op
                     << "\n";
      }

      // for arith.constant ops, need to rewrite its attribute
      if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
        Attribute attr = cst.getValue();                        
        auto elems = dyn_cast<ElementsAttr>(attr);
        if (!elems) return;
        auto splat = dyn_cast<SplatElementsAttr>(elems);
        if (!splat) return; 
        auto rttType = dyn_cast<RankedTensorType>(splat.getType());
        if (!rttType) return;

        auto enc = rttType.getEncoding();
        if (!enc || !isa<gluon::EfficientEncodingAttr>(enc)) return;

        auto newType = RankedTensorType::get(
            rttType.getShape(), rttType.getElementType(),
            gluon::AutoEncodingAttr::get(&getContext()));
        auto newValue = SplatElementsAttr::get(newType,
                                              splat.getSplatValue<Attribute>());
        cst.setValueAttr(newValue);
      }
    });
    llvm::outs() << "\n";
    llvm::outs() << "[EFFICIENT to AUTOLAYOUT]:\n";
    moduleOp.print(llvm::outs());
    llvm::outs() << "\n";
    llvm::outs() << "[END] REWRITE\n";

    // at this point, all efficient encoding should be gone,
    // check that 
    moduleOp.walk([&](Operation *op) {
      for (auto result : op->getResults()) {
        auto tensorType = dyn_cast<RankedTensorType>(result.getType());
        if (!tensorType)
          continue;
        auto encoding = tensorType.getEncoding();
        if (!encoding)
          continue;
        if (isa<gluon::EfficientEncodingAttr>(encoding)) {
          llvm::outs() << "Error: found remaining efficient encoding in result type for op: " << *op << "\n";
          signalPassFailure();
        }
      }
      for (auto operand : op->getOperands()) {
        auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
        if (!tensorType)
          continue;
        auto encoding = tensorType.getEncoding();
        if (!encoding)
          continue;
        if (isa<gluon::EfficientEncodingAttr>(encoding)) {
          llvm::outs() << "Error: found remaining efficient encoding in operand type for op: " << *op << "\n";
          signalPassFailure();
        }
      }
    });

    // 3. for operations in layoutMap,
    // i. convert its efficient encoding to inferred one from the layoutMap
    // ii. insert gluon.set_auto_layout op before it to set the inferred layout
    // 
    // conceptually, we help users select a coalesced layout right before load/store
    // then let resolveAutoEncoding to propgate the layout
    // 
    for (auto &pair : layoutMap) {
      Operation *op = pair.first;
      Attribute layout = pair.second;
      Value ptr = getMemAccessPtr(op);
      assert(isa<gluon::AutoEncodingAttr>(cast<RankedTensorType>(ptr.getType()).getEncoding()));

      rewriteInferredEncodings(op, layout);
    }
    llvm::outs() << "\n";
    llvm::outs() << "[REWRITE ENCODINGS to INFERRED]:\n";
    moduleOp.print(llvm::outs());
    llvm::outs() << "\n";
    llvm::outs() << "[END] REWRITE\n";
  }
};
} // namespace mlir::triton::gluon
