#include "triton/Dialect/Gluon/Transforms/InferLayoutUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Gluon/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PriorityWorklist.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"

#define DEBUG_TYPE "gluon-layout-propagation-utils"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton::gluon {

namespace {
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

bool encodingsMayVary(Operation *op) {
  return isa<triton::JoinOp, triton::SplitOp, triton::ReshapeOp, triton::CatOp,
             triton::TransOp>(op);
}
} // namespace

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
             << it->second.encoding << "\n";
    });
    worklist.insert(value);
  }
  return success();
}

LogicalResult inferLayout(FuncOp func, llvm::function_ref<bool(Type)> typeCheck,
                          llvm::MapVector<Value, LayoutInfo> &valueToEncoding,
                          llvm::PriorityWorklist<Value> &worklist,
                          llvm::MapVector<Attribute, uint64_t> &hashMemo) {
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

  // Propagate encodings through the graph until fixed point, or conflict
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
        if (failed(updateEncoding(tiedArgs, info, &func, valueToEncoding,
                                  worklist, hashMemo)))
          return failure();
      } else if (isa<scf::YieldOp>(op)) {
        auto tiedArgs = getTiedArgs(op, use.getOperandNumber());
        if (failed(updateEncoding(tiedArgs, info, &func, valueToEncoding,
                                  worklist, hashMemo)))
          return failure();
      } else {
        auto dstEnc = inferDstEncoding(op, info.encoding);
        if (dstEnc) {
          bool mayVary = info.mayVary || encodingsMayVary(op);
          LayoutInfo dstInfo{dstEnc, mayVary};
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
      if (isa<scf::ForOp, scf::WhileOp, scf::IfOp>(definingOp)) {
        auto tiedArgs = getTiedArgs(definingOp, opResult.getResultNumber());
        if (failed(updateEncoding(tiedArgs, info, &func, valueToEncoding,
                                  worklist, hashMemo)))
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
  auto ctx = func.getContext();
  for (auto &[val, info] : valueToEncoding) {
    auto existingTy = cast<RankedTensorType>(val.getType());
    assert(isa<gluon::AutoEncodingAttr>(existingTy.getEncoding()));
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

  // Cleanup set_auto_layout ops
  func.walk([&](gluon::SetAutoLayoutOp op) {
    assert(op.getSrc().getType() == op.getType());
    op.getResult().replaceAllUsesWith(op.getSrc());
    op->erase();
  });

  return success();
}

LogicalResult inferLayout(
    ModuleOp &mod, llvm::function_ref<bool(Type)> typeCheck,
    llvm::MapVector<FuncOp, llvm::MapVector<Value, LayoutInfo>> &funcValueEnc,
    llvm::MapVector<FuncOp, llvm::PriorityWorklist<Value>> &funcWorklist,
    llvm::MapVector<FuncOp, llvm::MapVector<Attribute, uint64_t>>
        &funcHashMemo) {
  for (auto &op : *mod.getBody()) {
    auto func = dyn_cast<FuncOp>(&op);
    if (!func)
      continue;
    if (failed(inferLayout(func, typeCheck, funcValueEnc[func],
                           funcWorklist[func], funcHashMemo[func])))
      return failure();
  }
  return success();
}

} // namespace mlir::triton::gluon
