#include "TritonAMDGPUTransforms/PointerCanonicalizer.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::triton::AMD;

// Rewrite a memory operation
LogicalResult PointerCanonicalizer::materializeFatPointer(Operation *op,
                                                          Location loc,
                                                          Value ptr) {
  auto fatPtr = pointers[ptr];
  Value basePtr = fatPtr.basePtr;
  Value offset = fatPtr.offset;
  // This is creating `tt.addptr(%fatPtr.bastPtr, %fatPtr.offset)
  Value newPtr = rewriter.create<triton::AddPtrOp>(loc, basePtr.getType(),
                                                   basePtr, offset);
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
      curOperand->set(fatPtr.basePtr);
      pointers[nextPtr] = FatPtr{nextPtr, offset};
    } else if (auto broadcastOp = dyn_cast<triton::BroadcastOp>(curOp)) {
      nextPtr = broadcastOp.getResult();
      auto fatPtr = pointers[broadcastOp.getSrc()];
      auto outType =
          dyn_cast<RankedTensorType>(broadcastOp.getResult().getType());
      auto ptrShape = outType.getShape();
      auto offsetType = dyn_cast<RankedTensorType>(fatPtr.offset.getType());
      curOperand->set(fatPtr.basePtr);
      if (!offsetType)
        return failure();

      auto newOffsetType = RankedTensorType::get(
          ptrShape, offsetType.getElementType(), outType.getEncoding());
      Value offset = rewriter.create<triton::BroadcastOp>(curLoc, newOffsetType,
                                                          fatPtr.offset);
      pointers[nextPtr] = FatPtr{nextPtr, offset};

    } else if (auto addPtr = dyn_cast<triton::AddPtrOp>(curOp)) {
      nextPtr = addPtr.getResult();
      auto fatPtr = pointers[addPtr.getPtr()];
      Type offsetType = addPtr.getOffset().getType();
      Value fatPtrOffset = fatPtr.offset;
      if (offsetType.isInteger(64)) {
        offset64 = true;
        fatPtrOffset = rewriter.create<arith::ExtSIOp>(
            curLoc, rewriter.getI64Type(), fatPtrOffset);
      } else if (auto offsetTensorType =
                     dyn_cast<RankedTensorType>(offsetType)) {
        if (offsetTensorType.isInteger(64)) {
          auto destType = RankedTensorType::get(offsetTensorType.getShape(),
                                                rewriter.getI64Type());
          fatPtrOffset =
              rewriter.create<arith::ExtSIOp>(curLoc, destType, fatPtrOffset);
        }
      }
      Value offset = rewriter.create<arith::AddIOp>(curLoc, fatPtrOffset,
                                                    addPtr.getOffset());
      pointers[nextPtr] = FatPtr{fatPtr.basePtr, offset};
      opToDelete.insert(addPtr);
    } else if (auto loadOp = dyn_cast<triton::LoadOp>(curOp)) {
      if (failed(materializeFatPointer(curOp, curLoc, loadOp.getPtr())))
        return failure();
      // Delete the old load operation
      opToDelete.insert(loadOp);
    } else if (auto storeOp = dyn_cast<triton::StoreOp>(curOp)) {
      if (failed(materializeFatPointer(curOp, curLoc, storeOp.getPtr())))
        return failure();
      // Delete the old load operation
      opToDelete.insert(storeOp);
    } else if (auto atomicOp = dyn_cast<triton::AtomicRMWOp>(curOp)) {
      if (failed(materializeFatPointer(curOp, curLoc, atomicOp.getPtr())))
        return failure();
      // Delete the old load operation
      opToDelete.insert(atomicOp);
    } else if (auto forOp = resolveOp<scf::ForOp>(curOp, rewriteOpMap)) {
      size_t operandNum = curOperand->getOperandNumber();
      FatPtr fatPtr = pointers[curOperand->get()];
      Value offset = fatPtr.offset;
      Value basePtr = fatPtr.basePtr;

      // Replace the forOp with an additional argument (i.e., the curOperand's
      // offset)
      auto newForOp = replaceForOpWithNewSignature(rewriter, forOp, offset);
      rewriteOpMap[forOp] = newForOp;

      newForOp->setOperand(operandNum, basePtr);
      OpOperand *forOperand = &newForOp->getOpOperand(operandNum);
      // This is making sure we propagate the visit from the forOp result
      nextPtr = newForOp.getTiedLoopResult(forOperand);

      // This is making sure we visit the uses withint the forOp region
      Value arg = newForOp.getTiedLoopRegionIterArg(forOperand);
      pointers[arg] = FatPtr{arg, newForOp.getRegionIterArgs().back()};
      for (OpOperand &use : arg.getUses())
        queue.push_back(&use);

      // This is setting the fat pointer for the users of the loop
      // and then propagatin the result
      pointers[nextPtr] = FatPtr{nextPtr, newForOp.getResults().back()};

      opToDelete.insert(forOp);
    } else if (auto yieldOp = dyn_cast<scf::YieldOp>(curOp)) {
      // Rewriting the yield op is a bit more complicated, because a
      // yield op can be inside of a ForOp, WhileOp(in the AfterRegion) or
      // IfOp
      size_t operandNum = curOperand->getOperandNumber();
      FatPtr fatPtr = pointers[curOperand->get()];
      yieldOp.getResultsMutable().append(fatPtr.offset);
      yieldOp->setOperand(operandNum, fatPtr.basePtr);

      if (auto ifOp = dyn_cast<scf::IfOp>(yieldOp->getParentOp())) {
        // Case 1: the yieldOp is contained within an IfOp. One of the
        // two branches is responsible to rewrite the operation. The other
        // branch only update the yieldOp with the right parameters
        if (yieldOp->getBlock() == &ifOp.getThenRegion().front()) {
          auto newIfOp = replaceIfOpWithNewSignature(rewriter, ifOp,
                                                     fatPtr.offset.getType());
          nextPtr = newIfOp.getResult(operandNum);
          pointers[nextPtr] = FatPtr{nextPtr, newIfOp.getResults().back()};
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
            rewriter, whileOp, {}, fatPtr.offset.getType());
        nextPtr = newWhileOp.getResult(operandNum);
        pointers[nextPtr] = FatPtr{nextPtr, newWhileOp.getResults().back()};
        rewriteOpMap[whileOp] = newWhileOp;
        opToDelete.insert(whileOp.getOperation());
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
      auto newWhileOp =
          replaceWhileOpWithNewSignature(rewriter, whileOp, offset, {});
      newWhileOp->setOperand(operandNum, fatPtr.basePtr);
      Value arg = newWhileOp.getBeforeBody()->getArgument(operandNum);
      // Propagate inside the BeforeRegion
      pointers[arg] =
          FatPtr{arg, newWhileOp.getBeforeBody()->getArguments().back()};
      nextPtr = arg;
      rewriteOpMap[whileOp] = newWhileOp;
      opToDelete.insert(curOp);
    } else if (auto conditionOp = dyn_cast<scf::ConditionOp>(curOp)) {
      // ConditionOp can only be contained within the BeforeRegion of a
      // WhileOp. We already rewrotw the WhileOp with the right operands, so
      // we need only to add the offset the the current operand to be the base
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
      conditionOp.getArgsMutable().append(offset);

      // Propagate through the after region
      afterBlock->addArgument(offset.getType(), curLoc);
      nextPtr = afterBlock->getArgument(operandNum - 1);
      pointers[nextPtr] = FatPtr{nextPtr, afterBlock->getArguments().back()};
    } else if (auto condBrOp = dyn_cast<cf::CondBranchOp>(curOp)) {
      // CondBranchOp is a bit tricky to handle. Because we might be inserting
      // the offset as a TrueDestOperand, which is not the end of
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
          // The offset needs to be added if we are on the FalseOperands side,
          // but the true operands have been rewritten
          int maybeOffset =
              (condBranchReplacement.getTrueDestOperands().size() !=
               condBrOp.getTrueDestOperands().size());
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
        falseOperands.push_back(offset);
        Value falseDestArg = falseDest->getArgument(
            operandNum - condBrOp.getNumTrueOperands() - 1);
        if (!pointers.contains(falseDestArg)) {
          nextPtr = falseDestArg;
          falseDest->addArgument(offset.getType(), curLoc);
          pointers[nextPtr] = FatPtr{nextPtr, falseDest->getArguments().back()};
        }
      } else {
        trueOperands.push_back(offset);
        Value trueDestArg = trueDest->getArgument(operandNum - 1);
        if (!pointers.contains(trueDestArg)) {
          nextPtr = trueDestArg;
          trueDest->addArgument(offset.getType(), curLoc);
          pointers[nextPtr] = FatPtr{nextPtr, trueDest->getArguments().back()};
        }
      }

      // Create a new condBranch. We cannot simply extend the operands,
      // because this would invalidate other operands pointing at the same
      // cond branch
      auto newCondBranch = rewriter.create<cf::CondBranchOp>(
          curLoc, condBrOp.getCondition(), trueDest, trueOperands, falseDest,
          falseOperands);

      newCondBranch.setOperand(operandNum, basePtr);
      rewriteOpMap[condBrOp] = newCondBranch;
      opToDelete.insert(condBrOp);
    } else if (auto branchOp = dyn_cast<cf::BranchOp>(curOp)) {
      size_t operandNum = curOperand->getOperandNumber();

      FatPtr fatPtr = pointers[curOperand->get()];
      Value offset = fatPtr.offset;
      Value basePtr = fatPtr.basePtr;

      branchOp.getDestOperandsMutable().append(fatPtr.offset);
      branchOp->setOperand(operandNum, basePtr);
      Block *dest = branchOp.getDest();

      // Walk the destination block only if you don't have visited it yet
      if (!pointers.contains(dest->getArgument(operandNum))) {
        dest->addArgument(offset.getType(), curLoc);
        nextPtr = dest->getArgument(operandNum);
        pointers[nextPtr] = {nextPtr, dest->getArguments().back()};
      }
    } else {
      unknownOp = true;
      if (failed(materializeFatPointer(curOp, curLoc, curOperand->get())))
        return failure();
    }

    // Keep propagating the fat pointer down the IR
    if (nextPtr)
      for (OpOperand &use : nextPtr.getUses())
        queue.push_back(&use);
  }
  for (Operation *op : llvm::reverse(opToDelete))
    op->erase();
  return success();
}

LogicalResult PointerCanonicalizer::rewriteRegion(Region &region) {
  for (Value arg : region.getArguments()) {

    auto tensorType = dyn_cast<RankedTensorType>(arg.getType());
    bool isTensorPointerType =
        tensorType && isa<triton::PointerType>(tensorType.getElementType());

    if (!isa<triton::PointerType>(arg.getType()) && !isTensorPointerType)
      continue;

    rewriter.setInsertionPointToStart(&region.front());
    Value zeroOffset =
        rewriter.create<arith::ConstantIntOp>(region.getLoc(), 0, 32);
    if (isTensorPointerType) {
      auto newOffsetType =
          RankedTensorType::get(tensorType.getShape(), zeroOffset.getType(),
                                tensorType.getEncoding());
      zeroOffset = rewriter.create<triton::SplatOp>(region.getLoc(),
                                                    newOffsetType, zeroOffset);
    }
    pointers[arg] = FatPtr{arg, zeroOffset};
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
