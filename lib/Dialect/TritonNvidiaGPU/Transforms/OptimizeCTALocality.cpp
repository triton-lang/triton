/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates. All rights reserved.
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
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
 * OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Traits.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace ttg = mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUOPTIMIZECTALOCALITYPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

bool isCrossCTAConversion(ttg::ConvertLayoutOp convert) {
  auto srcTy = dyn_cast<RankedTensorType>(convert.getSrc().getType());
  auto dstTy = dyn_cast<RankedTensorType>(convert.getType());
  if (!srcTy || !dstTy || !srcTy.getEncoding() || !dstTy.getEncoding())
    return false;

  LinearLayout conversion = minimalCvtLayout(srcTy, dstTy);
  auto kBlock = StringAttr::get(convert.getContext(), "block");
  return conversion.hasInDim(kBlock);
}

bool hasLayout(Value value, Attribute layout, int64_t rank) {
  auto ty = dyn_cast<RankedTensorType>(value.getType());
  return ty && ty.getRank() == rank && ty.getEncoding() == layout;
}

bool allUsesAreInBlock(Value value, Block *block) {
  return llvm::all_of(value.getUses(), [&](OpOperand &use) {
    return use.getOwner()->getBlock() == block;
  });
}

struct PropagationPlan {
  Attribute oldLayout;
  Attribute newLayout;
  int64_t rank;
  Block *block;
  llvm::DenseMap<Value, Value> replacements;
  llvm::SetVector<Value> valuesToRetype;
  SmallVector<Operation *> worklist;

  PropagationPlan(Attribute oldLayout, Attribute newLayout, int64_t rank,
                  Block *block)
      : oldLayout(oldLayout), newLayout(newLayout), rank(rank), block(block) {}

  bool addValue(Value value) {
    if (!hasLayout(value, oldLayout, rank) || replacements.contains(value) ||
        valuesToRetype.contains(value))
      return true;

    if (isa<BlockArgument>(value))
      return false;

    Operation *defOp = value.getDefiningOp();
    if (!defOp || defOp->getBlock() != block)
      return false;

    if (auto convert = dyn_cast<ttg::ConvertLayoutOp>(defOp)) {
      if (!isCrossCTAConversion(convert) ||
          !hasLayout(convert.getSrc(), newLayout, rank))
        return false;
      replacements[value] = convert.getSrc();
      return true;
    }

    if (!allUsesAreInBlock(value, block))
      return false;

    valuesToRetype.insert(value);
    worklist.push_back(defOp);
    for (OpOperand &use : value.getUses())
      worklist.push_back(use.getOwner());
    return true;
  }

  bool addOp(Operation *op) {
    if (!op->hasTrait<OpTrait::SameOperandsAndResultEncoding>() &&
        !op->hasTrait<OpTrait::SameLoadStoreOperandsEncoding>() &&
        !op->hasTrait<OpTrait::SameLoadStoreOperandsAndResultEncoding>() &&
        !op->hasTrait<OpTrait::Elementwise>())
      return true;

    for (OpOperand &operand : op->getOpOperands())
      if (!addValue(operand.get()))
        return false;
    for (Value result : op->getResults())
      if (!addValue(result))
        return false;
    return true;
  }

  bool propagate() {
    llvm::SmallPtrSet<Operation *, 16> visited;
    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      if (op->getBlock() != block || isa<ttg::ConvertLayoutOp>(op) ||
          !visited.insert(op).second)
        continue;
      if (!addOp(op))
        return false;
    }
    return true;
  }

  void apply(llvm::SetVector<Operation *> &opsToErase) {
    for (Value value : valuesToRetype) {
      auto ty = cast<RankedTensorType>(value.getType());
      value.setType(ty.cloneWithEncoding(newLayout));
    }

    for (auto [oldValue, newValue] : replacements) {
      oldValue.replaceUsesWithIf(newValue, [&](OpOperand &use) {
        return use.getOwner()->getBlock() == block;
      });
      if (auto convert = oldValue.getDefiningOp<ttg::ConvertLayoutOp>())
        if (convert->use_empty())
          opsToErase.insert(convert);
    }
  }
};

void propagateConvertLayout(ttg::ConvertLayoutOp convert,
                            llvm::SetVector<Operation *> &opsToErase) {
  if (!isCrossCTAConversion(convert))
    return;

  auto srcTy = cast<RankedTensorType>(convert.getSrc().getType());
  auto dstTy = cast<RankedTensorType>(convert.getType());
  Block *block = convert->getBlock();
  PropagationPlan plan(dstTy.getEncoding(), srcTy.getEncoding(),
                       srcTy.getRank(), block);
  plan.replacements[convert.getResult()] = convert.getSrc();
  for (OpOperand &use : convert.getResult().getUses())
    if (use.getOwner()->getBlock() == block)
      plan.worklist.push_back(use.getOwner());

  if (plan.propagate())
    plan.apply(opsToErase);
}

struct OptimizeCTALocalityPass
    : public impl::TritonNvidiaGPUOptimizeCTALocalityPassBase<
          OptimizeCTALocalityPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (ttg::TritonGPUDialect::getNumCTAs(mod) == 1)
      return;

    SmallVector<ttg::ConvertLayoutOp> converts;
    mod.walk(
        [&](ttg::ConvertLayoutOp convert) { converts.push_back(convert); });

    llvm::SetVector<Operation *> opsToErase;
    for (ttg::ConvertLayoutOp convert : converts)
      propagateConvertLayout(convert, opsToErase);
    for (Operation *op : opsToErase)
      op->erase();
  }
};

} // namespace

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
