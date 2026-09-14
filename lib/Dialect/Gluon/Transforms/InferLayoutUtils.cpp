#include "triton/Dialect/Gluon/Transforms/InferLayoutUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Gluon/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Traits.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PriorityWorklist.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"

#define DEBUG_TYPE "gluon-infer-layout-utils"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton::gluon {

namespace {
struct LayoutInfo {
  Attribute encoding;
  // Some operations can infer one of many encodings, we model this by
  // incrementing the "distance" each time a fuzzy inference occurs.
  // We preferrentially resolve to encodings with smallest distance, but ban
  // ambiguity when there is zero distance, since that implies set_auto_layout
  // was called on the op.
  unsigned distance = 0;

  operator bool() { return bool(encoding); }

  bool operator==(const LayoutInfo &other) const {
    return encoding == other.encoding && distance == other.distance;
  }
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
  if (lhs.distance != rhs.distance)
    return lhs.distance < rhs.distance ? lhs : rhs;

  // Sort inputs so this operation is commutative
  if (compare(lhs.encoding, rhs.encoding, hashMemo)) {
    std::swap(lhs, rhs);
  }
  if (lhs.distance != 0)
    return rhs;
  if (lhs.encoding == rhs.encoding)
    return lhs;
  op->emitOpError("found conflicting encodings for value:\n  ")
      << lhs.encoding << "\nand\n  " << rhs.encoding;
  return {};
}

bool encodingsMayVary(Operation *op) {
  return isa<triton::JoinOp, triton::SplitOp, triton::ReshapeOp,
             triton::TransOp>(op);
}

bool hasSameOperandsAndResultEncoding(Operation *op) {
  return op->hasTrait<OpTrait::SameOperandsAndResultType>() ||
         op->hasTrait<OpTrait::SameOperandsAndResultEncoding>() ||
         op->hasTrait<OpTrait::SameLoadStoreOperandsAndResultEncoding>();
}

bool hasSameOperandsEncoding(Operation *op) {
  return op->hasTrait<OpTrait::SameTypeOperands>() ||
         op->hasTrait<OpTrait::SameOperandsEncoding>() ||
         op->hasTrait<OpTrait::SameLoadStoreOperandsEncoding>() ||
         hasSameOperandsAndResultEncoding(op);
}

bool hasSameResultsEncoding(Operation *op) {
  return op->hasTrait<OpTrait::SameResultsType>() ||
         hasSameOperandsAndResultEncoding(op);
}

LogicalResult
updateEncoding(ArrayRef<Value> values, LayoutInfo info, FuncOp *func,
               llvm::MapVector<Value, LayoutInfo> &valueToEncoding,
               llvm::PriorityWorklist<Value> &worklist,
               llvm::MapVector<Attribute, uint64_t> &hashMemo) {
  for (auto value : values) {
    auto [it, inserted] = valueToEncoding.insert({value, info});
    if (!inserted) {
      auto defOp = value.getDefiningOp();
      auto op = defOp ? defOp : func->getOperation();
      auto combine = combineInfo(it->second, info, op, hashMemo);
      if (!combine)
        return failure();
      if (combine == it->second)
        continue;
      it->second = combine;
    }
    LLVM_DEBUG({
      DBGS() << "Setting value:\n\t" << value << "\nto encoding:\n\t"
             << it->second.encoding << " (distance " << it->second.distance
             << ")\n";
    });
    worklist.insert(value);
  }
  return success();
}
} // namespace

LogicalResult inferLayout(
    FuncOp func, llvm::function_ref<bool(Type)> typeCheck,
    const llvm::SmallVector<std::pair<Value, Attribute>> &seedEncodings) {
  // Disallow auto encoding accross function call boundaries
  for (auto argTy : func.getArgumentTypes()) {
    if (typeCheck(argTy)) {
      return func->emitError(
          "Functions taking auto encoding must be fully inlined");
    }
  }
  for (auto resultTy : func.getResultTypes()) {
    if (typeCheck(resultTy))
      return func->emitError(
          "Functions returning auto encoding must be fully inlined");
  }

  // set seed
  llvm::MapVector<Value, LayoutInfo> valueToEncoding;
  llvm::PriorityWorklist<Value> worklist;
  llvm::MapVector<Attribute, uint64_t> hashMemo;
  for (auto &[value, encoding] : seedEncodings) {
    if (failed(updateEncoding({value}, LayoutInfo{encoding, 0}, &func,
                              valueToEncoding, worklist, hashMemo)))
      return failure();
  }

  // Equality constraints propagate without introducing an encoding choice.
  auto propagateSameEncoding = [&](ValueRange values, LayoutInfo info) {
    SmallVector<Value> inferredValues;
    for (Value value : values)
      if (typeCheck(value.getType()))
        inferredValues.push_back(value);
    return updateEncoding(inferredValues, info, &func, valueToEncoding,
                          worklist, hashMemo);
  };

  // Propagate encodings through the graph until fixed point, or conflict
  while (!worklist.empty()) {
    auto val = worklist.pop_back_val();
    auto info = valueToEncoding[val];
    assert(info);

    // Propagate to users
    for (OpOperand &use : val.getUses()) {
      auto op = use.getOwner();
      if (hasSameOperandsEncoding(op) &&
          failed(propagateSameEncoding(op->getOperands(), info)))
        return failure();
      if (isa<scf::ForOp, scf::WhileOp>(op)) {
        auto offset = 3 * isa<scf::ForOp>(op);
        auto tiedArgs = getTiedArgs(op, use.getOperandNumber() - offset);
        if (failed(updateEncoding(tiedArgs, info, &func, valueToEncoding,
                                  worklist, hashMemo)))
          return failure();
      } else if (isa<scf::YieldOp>(op)) {
        auto parentOp = op->getParentOp();
        auto tiedArgs = getTiedArgs(parentOp, use.getOperandNumber());
        if (failed(updateEncoding(tiedArgs, info, &func, valueToEncoding,
                                  worklist, hashMemo)))
          return failure();
      } else {
        auto dstEnc = inferDstEncoding(op, info.encoding);
        if (dstEnc) {
          unsigned distance = info.distance + encodingsMayVary(op);
          LayoutInfo dstInfo{dstEnc, distance};
          if (failed(updateEncoding(llvm::to_vector_of<Value>(op->getResults()),
                                    dstInfo, &func, valueToEncoding, worklist,
                                    hashMemo)))
            return failure();
        }
      }
    }

    // Propagate to defining ops
    if (auto opResult = dyn_cast<OpResult>(val)) {
      auto definingOp = opResult.getOwner();
      if (hasSameResultsEncoding(definingOp) &&
          failed(propagateSameEncoding(definingOp->getResults(), info)))
        return failure();
      if (isa<scf::ForOp, scf::WhileOp, scf::IfOp>(definingOp)) {
        auto tiedArgs = getTiedArgs(definingOp, opResult.getResultNumber());
        if (failed(updateEncoding(tiedArgs, info, &func, valueToEncoding,
                                  worklist, hashMemo)))
          return failure();
      } else {
        auto srcEncoding = inferSrcEncoding(definingOp, info.encoding);
        if (srcEncoding) {
          unsigned distance = info.distance + encodingsMayVary(definingOp);
          LayoutInfo srcInfo{srcEncoding, distance};
          llvm::SmallVector<Value> tensorOperands;
          for (auto operand : definingOp->getOperands())
            if (isa<RankedTensorType>(operand.getType()))
              tensorOperands.push_back(operand);

          if (failed(updateEncoding(tensorOperands, srcInfo, &func,
                                    valueToEncoding, worklist, hashMemo)))
            return failure();
        }
      }
    } else if (auto blockArg = dyn_cast<BlockArgument>(val)) {
      auto parentOp = blockArg.getOwner()->getParentOp();
      if (isa<scf::ForOp, scf::WhileOp>(parentOp)) {
        auto offset = isa<scf::ForOp>(parentOp);
        auto tiedArgs = getTiedArgs(parentOp, blockArg.getArgNumber() - offset);
        if (failed(updateEncoding(tiedArgs, info, &func, valueToEncoding,
                                  worklist, hashMemo)))
          return failure();
      }
    }
  }

  // Transfer propagated encodings into the graph
  for (auto &[val, info] : valueToEncoding) {
    assert(typeCheck(val.getType()));
    auto existingTy = cast<RankedTensorType>(val.getType());
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

LogicalResult doubleCheckEncodings(ModuleOp &mod,
                                   llvm::function_ref<bool(Type)> typeCheck) {
  auto res = mod.walk([&](Operation *op) -> WalkResult {
    for (auto resTy : op->getResultTypes()) {
      if (typeCheck(resTy)) {
        return op->emitOpError("Failed to infer return type");
      }
    }
    return success();
  });
  if (res.wasInterrupted())
    return failure();

  res = mod.walk([&](Block *block) -> WalkResult {
    for (auto argTy : block->getArgumentTypes()) {
      if (typeCheck(argTy)) {
        return block->getParentOp()->emitError(
            "Failed to infer block argument type");
      }
    }
    return success();
  });
  if (res.wasInterrupted())
    return failure();
  return success();
}

} // namespace mlir::triton::gluon
