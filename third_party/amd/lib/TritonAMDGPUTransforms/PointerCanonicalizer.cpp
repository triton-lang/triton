#include "TritonAMDGPUTransforms/PointerCanonicalizer.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TypeSize.h"
#include <limits>

#define DEBUG_TYPE "tritonamdgpu-canonicalize-pointers"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::triton::AMD;

namespace {

// Extend a 32bit `offset` into 64bit using a arith.extsi operation
static Value extend32bitOffsetTo64Bits(IRRewriter &rewriter, Location loc,
                                       Value offset) {
  Type elementType = getElementTypeOrSelf(offset);

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
  Type elementType = getElementTypeOrSelf(offset);
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

// Helper function to determine if the given `op` is a constant tensor and in
// that case return the scalar value.
Value getScalarConstant(IRRewriter &rewriter, Location loc, Operation *op) {

  // Check for splatness
  if (auto splatOp = dyn_cast<triton::SplatOp>(op))
    return splatOp.getSrc();

  // Check for constant
  DenseIntElementsAttr constVal;
  if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
    Value val = constOp.getResult();
    if (matchPattern(val, m_Constant(&constVal)) && constVal.isSplat())
      return rewriter.create<arith::ConstantOp>(
          loc, constVal.getSplatValue<IntegerAttr>());
  }
  return Value();
}

// Narrowing logic
// For now we allow to narrow down to 32 bits only in the following case:
// - `baseOffset` is 32-bits and `addOffset`(64-bits) is zero
bool canNarrowOffset(Value baseOffset, Value addOffset) {
  Type addOffsetType = getElementTypeOrSelf(addOffset);
  Operation *baseOp = baseOffset.getDefiningOp();
  bool isBaseOffsetZero = baseOp && isa<triton::SplatOp>(baseOp);
  return (isBaseOffsetZero && addOffsetType.isInteger(32));
}

} // namespace

// Offset extraction logic for a binary op A*B or A+B
Value PointerCanonicalizer::extractScalarOffsetFromAddOrMul(
    Location loc, Operation *binOp, OpOperand &curOperand, int64_t bitness,
    bool &hasNonUniformComponent) {

  assert((isa<arith::MulIOp, arith::AddIOp>(binOp)) &&
         "Only arith.addi or arith.muli ops are supported");

  // Common useful variables
  OpOperand &lhsOperand = binOp->getOpOperand(0);
  OpOperand &rhsOperand = binOp->getOpOperand(1);
  Operation *lhsOp = lhsOperand.get().getDefiningOp();
  Operation *rhsOp = rhsOperand.get().getDefiningOp();

  // Useful creator function
  auto createBinOp = [&](Value lhs, Value rhs) -> Value {
    if (isa<arith::AddIOp>(binOp))
      return rewriter.create<arith::AddIOp>(loc, lhs, rhs);
    else
      return rewriter.create<arith::MulIOp>(loc, lhs, rhs);
  };

  // Propagate addition. If the binary operation is a arith.add, we
  // can simply replace its use with the uniform operand. I.e., if we
  // have:
  // ```
  //   %a = arith.addi %u, %nu
  //   %ptr0 = addptr %ptr, %a
  // ```
  // We can replace this with:
  // ```
  //   %scalar_offset = %u
  //   %ptr0 = addptr %ptr, %nu
  // ```
  auto propagateNonUniformOffset = [&](Value nonUniformOffset) {
    if (isa<arith::AddIOp>(binOp))
      curOperand.set(nonUniformOffset);
  };

  // If one out lhs or rhs is a scalar, set the current operand to the
  // non-scalar value and keep extracting
  if (Value lhsScalar = getScalarConstant(rewriter, loc, lhsOp)) {
    propagateNonUniformOffset(rhsOperand.get());
    Value scalarOffset = extractScalarOffsetFromExpr(
        loc, rhsOp, rhsOperand, bitness, hasNonUniformComponent);
    return createBinOp(scalarOffset, lhsScalar);
  } else if (Value rhsScalar = getScalarConstant(rewriter, loc, rhsOp)) {
    propagateNonUniformOffset(lhsOperand.get());
    Value scalarOffset = extractScalarOffsetFromExpr(
        loc, lhsOp, lhsOperand, bitness, hasNonUniformComponent);
    return createBinOp(scalarOffset, rhsScalar);
  } else if (isa<arith::MulIOp>(binOp)) {
    // Stop if this is a non-scalar * non-scalar
    hasNonUniformComponent = true;
    return rewriter.create<arith::ConstantIntOp>(loc, 0, bitness);
  } else {
    // Keep extracting otherwise
    Value scalarOffsetLhs = extractScalarOffsetFromExpr(
        loc, lhsOp, lhsOperand, bitness, hasNonUniformComponent);
    Value scalarOffsetRhs = extractScalarOffsetFromExpr(
        loc, rhsOp, rhsOperand, bitness, hasNonUniformComponent);
    return createBinOp(scalarOffsetLhs, scalarOffsetRhs);
  }
}

Value PointerCanonicalizer::extractScalarOffsetFromExpr(
    Location loc, Operation *curOp, OpOperand &curOperand, int64_t bitness,
    bool &hasNonUniformComponent) {
  if (Value scalarConst = getScalarConstant(rewriter, loc, curOp))
    return scalarConst;

  Value scalarOffset =
      llvm::TypeSwitch<Operation *, Value>(curOp)
          .Case<triton::BroadcastOp>([&](auto broadcastOp) {
            return extractScalarOffsetFromExpr(
                loc, broadcastOp.getSrc().getDefiningOp(),
                broadcastOp->getOpOperand(0), bitness, hasNonUniformComponent);
          })
          .Case<triton::ExpandDimsOp>([&](auto expandOp) {
            return extractScalarOffsetFromExpr(
                loc, expandOp.getSrc().getDefiningOp(),
                expandOp->getOpOperand(0), bitness, hasNonUniformComponent);
          })
          .Case<arith::AddIOp, arith::MulIOp>([&](Operation *op) {
            return extractScalarOffsetFromAddOrMul(loc, op, curOperand, bitness,
                                                   hasNonUniformComponent);
          })
          .Default([&](Operation *op) {
            hasNonUniformComponent = true;
            return rewriter.create<arith::ConstantIntOp>(loc, 0, bitness);
          });

  return scalarOffset;
}

// Try to extract a scalar `offset` by a (possibly) tensor offset. This will
// potentially modify the `offset` field of the `addPtrOp` (which will be
// potentially stripped off the non-uniform component)
Value PointerCanonicalizer::getScalarOffset(Location loc,
                                            triton::AddPtrOp addPtrOp,
                                            bool &hasNonUniformComponent) {
  hasNonUniformComponent = false;
  Value offset = addPtrOp.getOffset();
  if (!isa<RankedTensorType>(offset.getType()))
    return offset;

  // Early exist for the case of a constant tensor
  if (Value scalarConst = getScalarConstant(rewriter, loc, addPtrOp))
    return scalarConst;

  // Additional optimization to catch cases where `offset = f(non-splat, splat)`
  // and `f=={AddIOp,MulIOp}`. In this case we can get the splat contribution
  // out of the add and use that to update the scalar pointer. We use a queue of
  // <Operation, Operand> to traverse the offset contribution, in this way we
  // can propagate the non-splat component to the right place
  int64_t bitness =
      cast<RankedTensorType>(offset.getType()).getElementTypeBitWidth();
  return extractScalarOffsetFromExpr(loc, addPtrOp.getOffset().getDefiningOp(),
                                     addPtrOp->getOpOperand(1), bitness,
                                     hasNonUniformComponent);
}

// Create a tensor pointer from a fat pointer `fatPtr`. The tensor pointer is
// obtained by splatting the scalar pointer using the `fatPtr.offset` shape.
Value PointerCanonicalizer::createTensorPointer(FatPtr fatPtr, Location loc) {
  Value basePtr = fatPtr.basePtr;
  Value offset = fatPtr.offset;
  // Get the offset shape
  auto offsetType = cast<RankedTensorType>(offset.getType());
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
    // Add the tensor offset to the base pointer
    newPtr = rewriter.create<triton::AddPtrOp>(loc, tensorPtr.getType(),
                                               tensorPtr, offset);
  }

  // Map and replace the load
  IRMapping mapper;
  mapper.map(ptr, newPtr);
  Operation *newOp = rewriter.clone(*op, mapper);
  rewriter.replaceAllOpUsesWith(op, newOp);
  opToDelete.insert(op);
  return success();
}

LogicalResult PointerCanonicalizer::rewriteSplatOp(triton::SplatOp splatOp,
                                                   Location curLoc,
                                                   Value &nextPtr) {
  nextPtr = splatOp.getResult();
  auto fatPtr = pointers[splatOp.getSrc()];
  auto outType = splatOp.getResult().getType();
  auto ptrShape = outType.getShape();
  auto newOffsetType = RankedTensorType::get(ptrShape, fatPtr.offset.getType(),
                                             outType.getEncoding());
  Value offset =
      rewriter.create<triton::SplatOp>(curLoc, newOffsetType, fatPtr.offset);
  // The shape of the fat pointer is contained within the offset. We don't
  // need to keep the `splat` operation here.
  opToDelete.insert(splatOp);
  pointers[nextPtr] = fatPtr.copy(splatOp.getSrc(), offset);
  return success();
}

LogicalResult
PointerCanonicalizer::rewriteBroadcastOp(triton::BroadcastOp broadcastOp,
                                         Location curLoc, Value &nextPtr) {
  nextPtr = broadcastOp.getResult();
  auto fatPtr = pointers[broadcastOp.getSrc()];
  auto outType = dyn_cast<RankedTensorType>(broadcastOp.getResult().getType());
  auto ptrShape = outType.getShape();
  auto offsetType = dyn_cast<RankedTensorType>(fatPtr.offset.getType());
  if (!offsetType)
    return failure();

  opToDelete.insert(broadcastOp);

  auto newOffsetType = RankedTensorType::get(
      ptrShape, offsetType.getElementType(), outType.getEncoding());
  Value offset = rewriter.create<triton::BroadcastOp>(curLoc, newOffsetType,
                                                      fatPtr.offset);
  pointers[nextPtr] = fatPtr.copyWithBase(offset);
  return success();
}

LogicalResult PointerCanonicalizer::rewriteAddPtrOp(triton::AddPtrOp addPtrOp,
                                                    Location curLoc,
                                                    Value &nextPtr) {
  nextPtr = addPtrOp.getResult();
  auto fatPtr = pointers[addPtrOp.getPtr()];
  Value newPtr = fatPtr.basePtr;
  bool hasNonUniformComponent = false;
  Value scalarOffset =
      getScalarOffset(curLoc, addPtrOp, hasNonUniformComponent);
  if (!isZeroConst(scalarOffset)) {
    // Scalar pointer update
    newPtr = rewriter.create<triton::AddPtrOp>(curLoc, newPtr.getType(), newPtr,
                                               scalarOffset);
  }

  Value fatPtrOffset = fatPtr.offset;
  bool canNarrow = fatPtr.canNarrow;
  Value newOffset = fatPtrOffset;
  // Vector offset update
  if (hasNonUniformComponent) {
    Value addPtrOffset = addPtrOp.getOffset();
    Type addPtrOffsetType = getElementTypeOrSelf(addPtrOffset);
    canNarrow = canNarrow && canNarrowOffset(fatPtrOffset, addPtrOffset);

    // If we the incoming offset is 32 bits, then we have to cast to 64
    if (addPtrOffsetType.isInteger(32))
      addPtrOffset = extend32bitOffsetTo64Bits(rewriter, curLoc, addPtrOffset);

    newOffset =
        rewriter.create<arith::AddIOp>(curLoc, addPtrOffset, fatPtrOffset);
  }
  pointers[nextPtr] = FatPtr{newPtr, newOffset, canNarrow};
  opToDelete.insert(addPtrOp);
  return success();
}

LogicalResult PointerCanonicalizer::rewriteForOp(scf::ForOp forOp,
                                                 Location curLoc,
                                                 OpOperand *curOperand,
                                                 Value &nextPtr) {
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

  // This is making sure we visit the uses within the forOp region
  Value arg = newForOp.getTiedLoopRegionIterArg(forOperand);
  size_t numIterArgs = newForOp.getNumRegionIterArgs();
  pointers[arg] =
      FatPtr{newForOp.getRegionIterArg(numIterArgs - 2),
             newForOp.getRegionIterArg(numIterArgs - 1), fatPtr.canNarrow};
  for (OpOperand &use : arg.getUses())
    queue.push_back(&use);

  // This is setting the fat pointer for the users of the loop
  // and then propagate the result
  size_t numResults = newForOp->getNumResults();
  pointers[nextPtr] = fatPtr.copy(newForOp->getResult(numResults - 2),
                                  newForOp.getResult(numResults - 1));

  opToDelete.insert(forOp);
  return success();
}

LogicalResult PointerCanonicalizer::rewriteYieldOp(scf::YieldOp yieldOp,
                                                   Location curLoc,
                                                   OpOperand *curOperand,
                                                   Value &nextPtr) {

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
          rewriter, ifOp, {fatPtr.basePtr.getType(), fatPtr.offset.getType()});
      nextPtr = newIfOp.getResult(operandNum);
      size_t numResults = newIfOp->getNumResults();
      pointers[nextPtr] = fatPtr.copy(newIfOp->getResult(numResults - 2),
                                      newIfOp.getResult(numResults - 1));
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
    pointers[nextPtr] = fatPtr.copy(newWhileOp->getResult(numResults - 2),
                                    newWhileOp->getResult(numResults - 1));
    rewriteOpMap[whileOp] = newWhileOp;
    opToDelete.insert(whileOp.getOperation());
    yieldOp.setOperand(operandNum, newWhileOp.getAfterArguments()[operandNum]);
  }
  return success();
}

LogicalResult PointerCanonicalizer::rewriteWhileOp(scf::WhileOp whileOp,
                                                   Location curLoc,
                                                   OpOperand *curOperand,
                                                   Value &nextPtr) {
  // WhileOp rewrite happens in two phases: first rewrite the operand list
  // and then rewrite the types when we meet the yieldOp
  size_t operandNum = curOperand->getOperandNumber();
  FatPtr fatPtr = pointers[curOperand->get()];
  Value offset = fatPtr.offset;
  Value basePtr = fatPtr.basePtr;
  // Rewrite the while op with a new set of operands (but with the same
  // set of return types)
  Value tensorPtr = createTensorPointer(fatPtr, curLoc);
  auto newWhileOp =
      replaceWhileOpWithNewSignature(rewriter, whileOp, {basePtr, offset}, {});
  newWhileOp->setOperand(operandNum, tensorPtr);
  Value arg = newWhileOp.getBeforeBody()->getArgument(operandNum);
  // Propagate inside the BeforeRegion
  size_t numArguments = newWhileOp.getBeforeBody()->getNumArguments();
  pointers[arg] =
      fatPtr.copy(newWhileOp.getBeforeBody()->getArgument(numArguments - 2),
                  newWhileOp.getBeforeBody()->getArgument(numArguments - 1));
  nextPtr = arg;
  rewriteOpMap[whileOp] = newWhileOp;
  opToDelete.insert(whileOp);
  return success();
}

// ConditionOp can only be contained within the BeforeRegion of a
// WhileOp. We already rewrote the WhileOp with the right operands, so
// we need only to add the offset the current operand to be the base
// pointer and continue the walk inside the AfterRegion
LogicalResult
PointerCanonicalizer::rewriteConditionOp(scf::ConditionOp conditionOp,
                                         Location curLoc, OpOperand *curOperand,
                                         Value &nextPtr) {

  size_t operandNum = curOperand->getOperandNumber();
  FatPtr fatPtr = pointers[curOperand->get()];
  Value offset = fatPtr.offset;
  Value basePtr = fatPtr.basePtr;
  auto whileOp = cast<scf::WhileOp>(conditionOp->getParentOp());

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
  pointers[nextPtr] = fatPtr.copy(afterBlock->getArgument(numArguments - 2),
                                  afterBlock->getArgument(numArguments - 1));
  return success();
}

LogicalResult PointerCanonicalizer::rewriteCondBranchOp(
    cf::CondBranchOp condBrOp, Location curLoc, OpOperand *curOperand,
    Value &nextPtr) {
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
      bool needOffset = (condBranchReplacement.getTrueDestOperands().size() !=
                         condBrOp.getTrueDestOperands().size());
      int maybeOffset = (needOffset ? 2 : 0);
      operandNum += maybeOffset;
      curOperand = &condBranchReplacement->getOpOperand(operandNum);
    }
    // Now we need to recompute the currentOperation and its {true,false}
    // operands
    falseOperands =
        llvm::to_vector(condBranchReplacement.getFalseDestOperands());
    trueOperands = llvm::to_vector(condBranchReplacement.getTrueDestOperands());
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
    Value falseDestArg =
        falseDest->getArgument(operandNum - condBrOp.getNumTrueOperands() - 1);
    if (!pointers.contains(falseDestArg)) {
      nextPtr = falseDestArg;
      Value basePtrArg = falseDest->addArgument(basePtr.getType(), curLoc);
      Value offsetArg = falseDest->addArgument(offset.getType(), curLoc);
      pointers[nextPtr] = fatPtr.copy(basePtrArg, offsetArg);
    }
  } else {
    trueOperands.push_back(basePtr);
    trueOperands.push_back(offset);
    Value trueDestArg = trueDest->getArgument(operandNum - 1);
    if (!pointers.contains(trueDestArg)) {
      nextPtr = trueDestArg;
      Value basePtrArg = trueDest->addArgument(basePtr.getType(), curLoc);
      Value offsetArg = trueDest->addArgument(offset.getType(), curLoc);
      pointers[nextPtr] = fatPtr.copy(basePtrArg, offsetArg);
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
  return success();
}

LogicalResult PointerCanonicalizer::rewriteBranchOp(cf::BranchOp branchOp,
                                                    Location curLoc,
                                                    OpOperand *curOperand,
                                                    Value &nextPtr) {
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
  return success();
}

// Start from an argument of a function and propagate its
// fat pointers
LogicalResult PointerCanonicalizer::rewritePointer(Value argPtr) {
  // Start the visit
  for (OpOperand &use : argPtr.getUses())
    queue.push_back(&use);

  while (!queue.empty()) {
    OpOperand *curOperand = queue.pop_back_val();
    Operation *curOp = curOperand->getOwner();
    Location curLoc = curOp->getLoc();

    rewriter.setInsertionPoint(curOp);
    LogicalResult res = success();
    Value nextPtr;
    // We need to propagate the fat pointer throughout the IR
    llvm::TypeSwitch<Operation *>(curOp)
        .Case<triton::SplatOp>([&](auto splatOp) {
          res = rewriteSplatOp(splatOp, curLoc, nextPtr);
        })
        .Case<triton::BroadcastOp>([&](auto broadcastOp) {
          res = rewriteBroadcastOp(broadcastOp, curLoc, nextPtr);
        })
        .Case<triton::AddPtrOp>([&](auto addPtrOp) {
          res = rewriteAddPtrOp(addPtrOp, curLoc, nextPtr);
        })
        .Case<scf::ForOp>([&](auto forOp) {
          res = rewriteForOp(resolveOp<scf::ForOp>(forOp, rewriteOpMap), curLoc,
                             curOperand, nextPtr);
        })
        .Case<scf::YieldOp>([&](auto yieldOp) {
          res = rewriteYieldOp(yieldOp, curLoc, curOperand, nextPtr);
        })
        .Case<scf::WhileOp>([&](auto whileOp) {
          res = rewriteWhileOp(resolveOp<scf::WhileOp>(whileOp, rewriteOpMap),
                               curLoc, curOperand, nextPtr);
        })
        .Case<scf::ConditionOp>([&](auto conditionOp) {
          res = rewriteConditionOp(conditionOp, curLoc, curOperand, nextPtr);
        })
        .Case<cf::CondBranchOp>([&](auto condBrOp) {
          res = rewriteCondBranchOp(condBrOp, curLoc, curOperand, nextPtr);
        })
        .Case<cf::BranchOp>([&](auto branchOp) {
          res = rewriteBranchOp(branchOp, curLoc, curOperand, nextPtr);
        })
        .Case<triton::LoadOp, triton::StoreOp, triton::AtomicCASOp,
              triton::AtomicRMWOp, triton::PtrToIntOp>([&](Operation *op) {
          res = materializeFatPointer(curOp, curLoc, op->getOperand(0));
        })
        .Default([&](Operation *op) {
          // If we meet an unsupported operation, materialize the fat pointer
          // and continue.
          LDBG("Unknown op during pointer canonicalization: " << *curOp);
          res = materializeFatPointer(op, curLoc, curOperand->get());
        });

    // Keep propagating the fat pointer down the IR
    if (nextPtr)
      for (OpOperand &use : nextPtr.getUses())
        queue.push_back(&use);
  }
  return success();
}

LogicalResult PointerCanonicalizer::rewriteFunction(triton::FuncOp funcOp) {
  Region &region = funcOp.getRegion();
  for (Value arg : region.getArguments()) {
    // The pointer argument needs to be a scalar
    if (!isa<triton::PointerType>(arg.getType()))
      continue;

    rewriter.setInsertionPointToStart(&region.front());
    Value zeroOffset =
        rewriter.create<arith::ConstantIntOp>(region.getLoc(), 0, 64);

    // Start the rewrite
    clearFunctionState();
    pointers[arg] = FatPtr{arg, zeroOffset, true};
    if (failed(rewritePointer(arg)))
      return failure();

    // Clean-up
    for (Operation *op : llvm::reverse(opToDelete))
      op->erase();
  }
  return success();
}

LogicalResult PointerCanonicalizer::run() {
  llvm::SmallVector<triton::FuncOp> funcOps;

  // For now we don't cross function boundaries, but we should do that whenever
  // is possible
  mod.walk([&](triton::FuncOp funcOp) { funcOps.push_back(funcOp); });

  for (triton::FuncOp funcOp : funcOps) {
    if (failed(rewriteFunction(funcOp)))
      return failure();
  }
  return success();
}
