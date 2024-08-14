#include "TritonAMDGPUTransforms/PointerCanonicalizer.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/STLExtras.h"
#include <limits>

using namespace mlir;
using namespace mlir::triton::AMD;

namespace {

// Extract the element type of a (scalar or tensor) type
static Type getElementType(Value val) {
  Type elementType = val.getType();
  if (auto tensorType = dyn_cast<RankedTensorType>(elementType))
    elementType = tensorType.getElementType();
  return elementType;
}

// Extend a 32bit `offset` into 64bit using a arith.extsi operation
static Value extend32bitOffsetTo64Bits(IRRewriter &rewriter, Location loc,
                                       Value offset) {
  Type elementType = getElementType(offset);

  if (auto tensorType = dyn_cast<RankedTensorType>(offset.getType())) {
    auto shape = tensorType.getShape();
    auto newTensorType = RankedTensorType::get(shape, rewriter.getI64Type(),
                                               tensorType.getEncoding());
    return rewriter.create<arith::ExtSIOp>(loc, newTensorType, offset);
  }
  return rewriter.create<arith::ExtSIOp>(loc, rewriter.getI64Type(), offset);
}

// Narrow a 64bit `offset` into 32bit using a arith.trunci operation
static Value narrow64bitOffsetTo32bits(IRRewriter &rewriter, Location loc,
                                       Value offset) {
  Type elementType = getElementType(offset);
  if (elementType.isInteger(32))
    return offset;

  if (auto tensorType = dyn_cast<RankedTensorType>(offset.getType())) {
    auto shape = tensorType.getShape();
    auto newTensorType = RankedTensorType::get(shape, rewriter.getI32Type(),
                                               tensorType.getEncoding());
    return rewriter.create<arith::TruncIOp>(loc, newTensorType, offset);
  }
  return rewriter.create<arith::TruncIOp>(loc, rewriter.getI32Type(), offset);
}

// Try to extract a scalar `offset` by a (possibly) tensor offset
Value getScalarOffset(IRRewriter &rewriter, Location loc, Value offset) {
  if (!isa<RankedTensorType>(offset.getType()))
    return offset;

  if (auto splatOp = dyn_cast<triton::SplatOp>(offset.getDefiningOp()))
    return splatOp.getSrc();

  if (auto constOp = dyn_cast<arith::ConstantOp>(offset.getDefiningOp())) {
    auto denseAttr = dyn_cast<DenseIntElementsAttr>(constOp.getValueAttr());
    if (denseAttr && denseAttr.isSplat())
      return rewriter.create<arith::ConstantOp>(
          loc, denseAttr.getSplatValue<IntegerAttr>());
  }

  return Value();
}
} // namespace

// Create a tensor pointer from a fat pointer `fatPtr`. The tensor pointer is
// obtained by splatting the scalar pointer using the `fatPtr.offset` shape.
Value PointerCanonicalizer::createTensorPointer(FatPtr fatPtr, Location loc) {
  Value basePtr = fatPtr.basePtr;
  Value offset = fatPtr.offset;
  // Get the offset shape
  auto offsetType = dyn_cast<RankedTensorType>(offset.getType());
  ArrayRef<int64_t> offsetShape = offsetType.getShape();
  // Splat the scalar pointer
  auto tensorPtrType = RankedTensorType::get(offsetShape, basePtr.getType(),
                                             offsetType.getEncoding());
  Value tensorPtr =
      rewriter.create<triton::SplatOp>(loc, tensorPtrType, basePtr);
  return tensorPtr;
}

// Rewrite a memory operation
LogicalResult PointerCanonicalizer::materializeFatPointer(Operation *op,
                                                          Location loc,
                                                          Value ptr) {
  auto fatPtr = pointers[ptr];
  Value basePtr = fatPtr.basePtr;
  Value offset = fatPtr.offset;
  if (fatPtr.canNarrow)
    offset = narrow64bitOffsetTo32bits(rewriter, loc, offset);

  Value newPtr = basePtr;
  if (isa<RankedTensorType>(ptr.getType())) {
    // Splat the base pointer
    Value tensorPtr = createTensorPointer(fatPtr, loc);
    // This is creating `tt.addptr(%tensorBasePtr, %fatPtr.offset)
    newPtr = rewriter.create<triton::AddPtrOp>(loc, tensorPtr.getType(),
                                               tensorPtr, offset);
  }

  // Map and replace the load
  IRMapping mapper;
  mapper.map(ptr, newPtr);
  Operation *newOp = rewriter.clone(*op, mapper);
  rewriter.replaceAllOpUsesWith(op, newOp);
  return success();
}

// Start from an argument of a function and propagate its
// fat pointers
LogicalResult PointerCanonicalizer::rewritePointer(Value argPtr) {
  // List of IR to delete
  SetVector<Operation *> opToDelete;
  // This structure is used to point to the right operation during the traversal
  DenseMap<Operation *, Operation *> rewriteOpMap;
  // Queue of operations to visit
  SmallVector<OpOperand *> queue;

  // Start the visit
  for (OpOperand &use : argPtr.getUses())
    queue.push_back(&use);

  while (!queue.empty()) {
    OpOperand *curOperand = queue.back();
    queue.pop_back();

    Operation *curOp = curOperand->getOwner();
    Location curLoc = curOp->getLoc();
    Value nextPtr;

    rewriter.setInsertionPoint(curOp);
    if (auto splatOp = dyn_cast<triton::SplatOp>(curOp)) {
      nextPtr = splatOp.getResult();
      auto fatPtr = pointers[splatOp.getSrc()];
      auto outType = splatOp.getResult().getType();
      auto ptrShape = outType.getShape();
      auto newOffsetType = RankedTensorType::get(
          ptrShape, fatPtr.offset.getType(), outType.getEncoding());
      Value offset = rewriter.create<triton::SplatOp>(curLoc, newOffsetType,
                                                      fatPtr.offset);
      // The shape of the fat pointer is contained within the offset. We don't
      // need to keep the `splat` operation here.
      opToDelete.insert(splatOp);
      pointers[nextPtr] = FatPtr{splatOp.getSrc(), offset, fatPtr.canNarrow};
    } else if (auto broadcastOp = dyn_cast<triton::BroadcastOp>(curOp)) {
      nextPtr = broadcastOp.getResult();
      auto fatPtr = pointers[broadcastOp.getSrc()];
      auto outType =
          dyn_cast<RankedTensorType>(broadcastOp.getResult().getType());
      auto ptrShape = outType.getShape();
      auto offsetType = dyn_cast<RankedTensorType>(fatPtr.offset.getType());
      if (!offsetType)
        return failure();

      opToDelete.insert(broadcastOp);

      auto newOffsetType = RankedTensorType::get(
          ptrShape, offsetType.getElementType(), outType.getEncoding());
      Value offset = rewriter.create<triton::BroadcastOp>(curLoc, newOffsetType,
                                                          fatPtr.offset);
      pointers[nextPtr] = FatPtr{fatPtr.basePtr, offset, fatPtr.canNarrow};

    } else if (auto addPtr = dyn_cast<triton::AddPtrOp>(curOp)) {
      nextPtr = addPtr.getResult();
      auto fatPtr = pointers[addPtr.getPtr()];
      Value fatPtrOffset = fatPtr.offset;
      Value addPtrOffset = addPtr.getOffset();

      Type fatPtrOffsetType = getElementType(fatPtrOffset);
      Type addPtrOffsetType = getElementType(addPtrOffset);
      Value newOffset = fatPtrOffset;
      Value newPtr = fatPtr.basePtr;
      bool canNarrow = fatPtr.canNarrow;
      if (Value scalarOffset =
              getScalarOffset(rewriter, curLoc, addPtrOffset)) {
        // Scalar pointer update
        newPtr = rewriter.create<triton::AddPtrOp>(curLoc, newPtr.getType(),
                                                   newPtr, scalarOffset);
      } else {
        // If we the incoming offset is 32 bits, then we have to cast to 64
        if (addPtrOffsetType.isInteger(32))
          addPtrOffset =
              extend32bitOffsetTo64Bits(rewriter, curLoc, addPtrOffset);

        newOffset =
            rewriter.create<arith::AddIOp>(curLoc, addPtrOffset, fatPtrOffset);

        // If we do : %newOffset = %offset(32-bit) + %zero(64-bit) the user of
        // %newOffset should always be able to narrow it down to 32 bits
        bool isFatPtrZero = fatPtrOffset.getDefiningOp() &&
                            isa<triton::SplatOp>(fatPtrOffset.getDefiningOp());
        canNarrow = canNarrow && addPtrOffsetType.isInteger(32) && isFatPtrZero;
      }
      pointers[nextPtr] = FatPtr{newPtr, newOffset, canNarrow};
      opToDelete.insert(addPtr);
    } else if (auto loadOp = dyn_cast<triton::LoadOp>(curOp)) {
      if (failed(materializeFatPointer(curOp, curLoc, loadOp.getPtr())))
        return failure();
      // Delete the old operation
      opToDelete.insert(loadOp);
    } else if (auto storeOp = dyn_cast<triton::StoreOp>(curOp)) {
      if (failed(materializeFatPointer(curOp, curLoc, storeOp.getPtr())))
        return failure();
      // Delete the old operation
      opToDelete.insert(storeOp);
    } else if (auto atomicRmwOp = dyn_cast<triton::AtomicRMWOp>(curOp)) {
      if (failed(materializeFatPointer(curOp, curLoc, atomicRmwOp.getPtr())))
        return failure();
      // Delete the old operation
      opToDelete.insert(atomicRmwOp);
    } else if (auto atomicCasOp = dyn_cast<triton::AtomicCASOp>(curOp)) {
      if (failed(materializeFatPointer(curOp, curLoc, atomicCasOp.getPtr())))
        return failure();
      // Delete the old operation
      opToDelete.insert(atomicCasOp);
    } else if (auto forOp = resolveOp<scf::ForOp>(curOp, rewriteOpMap)) {
      size_t operandNum = curOperand->getOperandNumber();
      FatPtr fatPtr = pointers[curOperand->get()];
      Value offset = fatPtr.offset;
      Value basePtr = fatPtr.basePtr;

      // Replace the forOp with two additional argument (i.e., the curOperand's
      // scalar pointer and the offset)
      Value tensorPtr = createTensorPointer(fatPtr, curLoc);
      auto newForOp =
          replaceForOpWithNewSignature(rewriter, forOp, {basePtr, offset});
      rewriteOpMap[forOp] = newForOp;

      newForOp->setOperand(operandNum, tensorPtr);
      OpOperand *forOperand = &newForOp->getOpOperand(operandNum);
      // This is making sure we propagate the visit from the forOp result
      nextPtr = newForOp.getTiedLoopResult(forOperand);

      // This is making sure we visit the uses withint the forOp region
      Value arg = newForOp.getTiedLoopRegionIterArg(forOperand);
      size_t numIterArgs = newForOp.getNumRegionIterArgs();
      pointers[arg] =
          FatPtr{newForOp.getRegionIterArg(numIterArgs - 2),
                 newForOp.getRegionIterArg(numIterArgs - 1), fatPtr.canNarrow};
      for (OpOperand &use : arg.getUses())
        queue.push_back(&use);

      // This is setting the fat pointer for the users of the loop
      // and then propagatin the result
      size_t numResults = newForOp->getNumResults();
      pointers[nextPtr] =
          FatPtr{newForOp->getResult(numResults - 2),
                 newForOp.getResult(numResults - 1), fatPtr.canNarrow};

      opToDelete.insert(forOp);
    } else if (auto yieldOp = dyn_cast<scf::YieldOp>(curOp)) {
      // Rewriting the yield op is a bit more complicated, because a
      // yield op can be inside of a ForOp, WhileOp(in the AfterRegion) or
      // IfOp
      size_t operandNum = curOperand->getOperandNumber();
      FatPtr fatPtr = pointers[curOperand->get()];
      yieldOp.getResultsMutable().append(fatPtr.basePtr);
      yieldOp.getResultsMutable().append(fatPtr.offset);

      if (auto forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp())) {
        auto v = forOp.getRegionIterArg(operandNum);
        yieldOp->setOperand(operandNum, forOp.getRegionIterArg(operandNum));
      } else if (auto ifOp = dyn_cast<scf::IfOp>(yieldOp->getParentOp())) {
        // Case 1: the yieldOp is contained within an IfOp. One of the
        // two branches is responsible to rewrite the operation. The other
        // branch only update the yieldOp with the right parameters
        Value tensorPtr = createTensorPointer(fatPtr, curLoc);
        yieldOp->setOperand(operandNum, tensorPtr);

        if (yieldOp->getBlock() == &ifOp.getThenRegion().front()) {
          auto newIfOp = replaceIfOpWithNewSignature(
              rewriter, ifOp,
              {fatPtr.basePtr.getType(), fatPtr.offset.getType()});
          nextPtr = newIfOp.getResult(operandNum);
          size_t numResults = newIfOp->getNumResults();
          pointers[nextPtr] =
              FatPtr{newIfOp->getResult(numResults - 2),
                     newIfOp.getResult(numResults - 1), fatPtr.canNarrow};
          opToDelete.insert(ifOp);
        }

      } else if (auto whileOp = resolveOp<scf::WhileOp>(yieldOp->getParentOp(),
                                                        rewriteOpMap)) {
        // Case 2: the yieldOp is contained within the AfterRegion of a
        // WhileOp. In this case, we know that the before region should have
        // already been replaced (when we met the WhileOp), hence we can
        // simply replace the WhileOp with a new AfterRegion (and hance a new
        // set of return types)
        auto newWhileOp = replaceWhileOpWithNewSignature(
            rewriter, whileOp, {},
            {fatPtr.basePtr.getType(), fatPtr.offset.getType()});
        nextPtr = newWhileOp.getResult(operandNum);
        size_t numResults = newWhileOp->getNumResults();
        pointers[nextPtr] =
            FatPtr{newWhileOp->getResult(numResults - 2),
                   newWhileOp->getResult(numResults - 1), fatPtr.canNarrow};
        rewriteOpMap[whileOp] = newWhileOp;
        opToDelete.insert(whileOp.getOperation());
        yieldOp.setOperand(operandNum,
                           newWhileOp.getAfterArguments()[operandNum]);
      }
    } else if (auto whileOp = resolveOp<scf::WhileOp>(curOp, rewriteOpMap)) {
      // WhileOp rewrite happens in two phases: first rewrite the operand list
      // and then rewrite the types when we meet the yieldOp
      size_t operandNum = curOperand->getOperandNumber();
      FatPtr fatPtr = pointers[curOperand->get()];
      Value offset = fatPtr.offset;
      Value basePtr = fatPtr.basePtr;
      // Rewrite the while op with a new set of operands (but with the same
      // set of return types)
      Value tensorPtr = createTensorPointer(fatPtr, curLoc);
      auto newWhileOp = replaceWhileOpWithNewSignature(rewriter, whileOp,
                                                       {basePtr, offset}, {});
      newWhileOp->setOperand(operandNum, tensorPtr);
      Value arg = newWhileOp.getBeforeBody()->getArgument(operandNum);
      // Propagate inside the BeforeRegion
      size_t numArguments = newWhileOp.getBeforeBody()->getNumArguments();
      pointers[arg] =
          FatPtr{newWhileOp.getBeforeBody()->getArgument(numArguments - 2),
                 newWhileOp.getBeforeBody()->getArgument(numArguments - 1),
                 fatPtr.canNarrow};
      nextPtr = arg;
      rewriteOpMap[whileOp] = newWhileOp;
      opToDelete.insert(curOp);
    } else if (auto conditionOp = dyn_cast<scf::ConditionOp>(curOp)) {
      // ConditionOp can only be contained within the BeforeRegion of a
      // WhileOp. We already rewrote the WhileOp with the right operands, so
      // we need only to add the offset the current operand to be the base
      // pointer and continue the walk inside the AfterRegion
      size_t operandNum = curOperand->getOperandNumber();
      FatPtr fatPtr = pointers[curOperand->get()];
      Value offset = fatPtr.offset;
      Value basePtr = fatPtr.basePtr;
      auto whileOp = dyn_cast<scf::WhileOp>(conditionOp->getParentOp());
      assert(whileOp &&
             "ConditionOp inside an operation different from WhileOp");

      // Update the condition op
      auto afterBlock = whileOp.getAfterBody();
      conditionOp.getArgsMutable().append({basePtr, offset});

      // Propagate through the after region
      afterBlock->addArgument(basePtr.getType(), curLoc);
      afterBlock->addArgument(offset.getType(), curLoc);
      nextPtr = afterBlock->getArgument(operandNum - 1);
      size_t numArguments = afterBlock->getNumArguments();
      conditionOp.setOperand(operandNum,
                             whileOp.getRegionIterArgs()[operandNum - 1]);
      pointers[nextPtr] =
          FatPtr{afterBlock->getArgument(numArguments - 2),
                 afterBlock->getArgument(numArguments - 1), fatPtr.canNarrow};
    } else if (auto condBrOp = dyn_cast<cf::CondBranchOp>(curOp)) {
      // CondBranchOp is a bit tricky to handle. Because we might be inserting
      // the basePtr+offset as a TrueDestOperand(s), which is not the end of
      // `condBrOp.getOperands()`
      auto falseOperands = llvm::to_vector(condBrOp.getFalseDestOperands());
      auto trueOperands = llvm::to_vector(condBrOp.getTrueOperands());
      auto it = llvm::find(falseOperands, curOperand->get());
      bool isFalseOperand = (it != falseOperands.end());
      size_t operandNum = curOperand->getOperandNumber();

      if (rewriteOpMap.contains(condBrOp)) {
        // If we need to use a different condBrOp, we might also need to
        // update `operandNum`
        auto condBranchReplacement =
            dyn_cast<cf::CondBranchOp>(rewriteOpMap[condBrOp]);
        if (isFalseOperand) {
          // basePtr+offset need to be added if we are on the FalseOperands
          // side, but the true operands have been rewritten
          bool needOffset =
              (condBranchReplacement.getTrueDestOperands().size() !=
               condBrOp.getTrueDestOperands().size());
          int maybeOffset = (needOffset ? 2 : 0);
          operandNum += maybeOffset;
          curOperand = &condBranchReplacement->getOpOperand(operandNum);
        }
        // Now we need to recompute the currentOperation and its {true,false}
        // operands
        curOp = condBranchReplacement.getOperation();
        falseOperands =
            llvm::to_vector(condBranchReplacement.getFalseDestOperands());
        trueOperands =
            llvm::to_vector(condBranchReplacement.getTrueDestOperands());
        condBrOp = condBranchReplacement;
      }

      // Now we can proceed almost normally
      FatPtr fatPtr = pointers[curOperand->get()];
      Value offset = fatPtr.offset;
      Value basePtr = fatPtr.basePtr;

      Block *falseDest = condBrOp.getFalseDest();
      Block *trueDest = condBrOp.getTrueDest();
      // Walk the destination block only if you don't have visited it yet
      if (isFalseOperand) {
        falseOperands.push_back(basePtr);
        falseOperands.push_back(offset);
        Value falseDestArg = falseDest->getArgument(
            operandNum - condBrOp.getNumTrueOperands() - 1);
        if (!pointers.contains(falseDestArg)) {
          nextPtr = falseDestArg;
          Value basePtrArg = falseDest->addArgument(basePtr.getType(), curLoc);
          Value offsetArg = falseDest->addArgument(offset.getType(), curLoc);
          pointers[nextPtr] = FatPtr{basePtrArg, offsetArg, fatPtr.canNarrow};
        }
      } else {
        trueOperands.push_back(basePtr);
        trueOperands.push_back(offset);
        Value trueDestArg = trueDest->getArgument(operandNum - 1);
        if (!pointers.contains(trueDestArg)) {
          nextPtr = trueDestArg;
          Value basePtrArg = trueDest->addArgument(basePtr.getType(), curLoc);
          Value offsetArg = trueDest->addArgument(offset.getType(), curLoc);
          pointers[nextPtr] = FatPtr{basePtrArg, offsetArg, fatPtr.canNarrow};
        }
      }

      // Create a new condBranch. We cannot simply extend the operands,
      // because this would invalidate other operands pointing at the same
      // cond branch
      Value tensorPtr = createTensorPointer(fatPtr, curLoc);
      auto newCondBranch = rewriter.create<cf::CondBranchOp>(
          curLoc, condBrOp.getCondition(), trueDest, trueOperands, falseDest,
          falseOperands);

      newCondBranch.setOperand(operandNum, tensorPtr);
      rewriteOpMap[condBrOp] = newCondBranch;
      opToDelete.insert(condBrOp);
    } else if (auto branchOp = dyn_cast<cf::BranchOp>(curOp)) {
      size_t operandNum = curOperand->getOperandNumber();

      FatPtr fatPtr = pointers[curOperand->get()];
      Value offset = fatPtr.offset;
      Value basePtr = fatPtr.basePtr;

      branchOp.getDestOperandsMutable().append({basePtr, fatPtr.offset});
      Value tensorPtr = createTensorPointer(fatPtr, curLoc);
      branchOp->setOperand(operandNum, tensorPtr);
      Block *dest = branchOp.getDest();

      // Walk the destination block only if you don't have visited it yet
      if (!pointers.contains(dest->getArgument(operandNum))) {
        Value basePtrArg = dest->addArgument(basePtr.getType(), curLoc);
        Value offsetArg = dest->addArgument(offset.getType(), curLoc);
        nextPtr = dest->getArgument(operandNum);
        pointers[nextPtr] = {basePtrArg, offsetArg, fatPtr.canNarrow};
      }
    } else if (auto ptrCastOp = dyn_cast<triton::PtrToIntOp>(curOp)) {
      if (failed(materializeFatPointer(curOp, curLoc, ptrCastOp.getSrc())))
        return failure();
      // Delete the old operation
      opToDelete.insert(ptrCastOp);
    } else {
      // If we meet an unsupported operation, materialize the fat pointer and
      // continue. We flag that we met unknown usage of the pointer, hence we
      // cannot ensure that all the pointers have been canonicalized
      unknownOp = true;
      if (failed(materializeFatPointer(curOp, curLoc, curOperand->get())))
        return failure();
    }

    // Keep propagating the fat pointer down the IR
    if (nextPtr)
      for (OpOperand &use : nextPtr.getUses())
        queue.push_back(&use);
  }
  for (Operation *op : llvm::reverse(opToDelete)) {
    op->erase();
  }
  return success();
}

LogicalResult PointerCanonicalizer::rewriteRegion(Region &region) {
  for (Value arg : region.getArguments()) {
    // The pointer argument needs to be a scalar
    if (!isa<triton::PointerType>(arg.getType()))
      continue;

    rewriter.setInsertionPointToStart(&region.front());
    Value zeroOffset =
        rewriter.create<arith::ConstantIntOp>(region.getLoc(), 0, 64);

    // Take note of the scalar pointer to introduce more optimizations
    pointers[arg] = FatPtr{arg, zeroOffset, true};
    if (failed(rewritePointer(arg)))
      return failure();
  }
  return success();
}

LogicalResult PointerCanonicalizer::run() {
  llvm::SmallVector<triton::FuncOp> funcOps;

  mod.walk([&](triton::FuncOp funcOp) { funcOps.push_back(funcOp); });

  for (triton::FuncOp funcOp : funcOps) {
    if (failed(rewriteRegion(funcOp->getRegion(0))))
      return failure();
  }
  return success();
}
