#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_POINTER_CANONICALIZER_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_POINTER_CANONICALIZER_H_

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;

namespace mlir::triton::AMD {

// -----------------------------------------------------------------------------
//
// -----------------------------------------------------------------------------
// This class iterates through the argument of the `funcOp`, if the argument is
// a pointer, it starts a walk to replace the pointer through all the IR with an
// offset. Only when the pointer is really needed the offset is added to the
// base pointer and passed to the operation that needs the pointer (usually a
// load or a store)
//
// Let's suppose that `arg0` is a pointer. The algorithm works like that:
// a) At the beginning the offset is zero, and we associate with `arg0` a
//    `FatPtr{arg0, offset}`
// b) Follow the pointer through the IR, and replace any
//    `tt.addptr(%ptr, %offset)` with
//    `add(%fatPoniters[ptr].basePtr, %fatPointers[ptr].offset)`.
// c) When you meet the `tt.load(%ptr)` or `tt.store(%ptr)` instructions,
//    replace that instruction with:
//    `%fat_ptr = tt.addptr(%fatPointers[ptr].basePtr,
//    %fatPointers[ptr].offset)`
//    `%data = tt.load(%fat_ptr)`
//
class PointerCanonicalizer {
private:
  // This is the internal representation of a fat pointer: `fatPtr = basePtr +
  // offset`.
  struct FatPtr {
    Value basePtr;
    Value offset;
    bool canNarrow = false;
  };

  // Rewrite any operation that needs a pointer
  LogicalResult materializeFatPointer(Operation *op, Location loc, Value ptr);

  // Start from an argument of a function and propagate its fat pointers
  LogicalResult rewritePointer(Value argPtr);

  Value createTensorPointer(FatPtr fatPtr, Location loc);

  // Rewrite a given function, canonicalizing the different pointer arguments of
  // the region
  LogicalResult rewriteFunction(triton::FuncOp funcOp);

  // Rewriters for different operation a pointer can walk into
  LogicalResult rewriteSplatOp(triton::SplatOp splatOp, Location curLoc,
                               Value &nextPtr);
  LogicalResult rewriteBroadcastOp(triton::BroadcastOp broadcastOp,
                                   Location curLoc, Value &nextPtr);
  LogicalResult rewriteAddPtrOp(triton::AddPtrOp addPtrOp, Location curLoc,
                                Value &nextPtr);
  LogicalResult rewriteForOp(scf::ForOp forOp, Location curLoc,
                             OpOperand *operand, Value &nextPtr);
  LogicalResult rewriteYieldOp(scf::YieldOp yieldOp, Location curLoc,
                               OpOperand *operand, Value &nextPtr);
  LogicalResult rewriteWhileOp(scf::WhileOp whileOp, Location curLoc,
                               OpOperand *operand, Value &nextPtr);
  LogicalResult rewriteConditionOp(scf::ConditionOp conditionOp,
                                   Location curLoc, OpOperand *operand,
                                   Value &nextPtr);
  LogicalResult rewriteCondBranchOp(cf::CondBranchOp condBrOp, Location curLoc,
                                    OpOperand *operand, Value &nextPtr);
  LogicalResult rewriteBranchOp(cf::BranchOp branchOp, Location curLoc,
                                OpOperand *operand, Value &nextPtr);

  // Return either the operation or its rewritten op
  template <typename OpTy>
  OpTy resolveOp(Operation *op,
                 const DenseMap<Operation *, Operation *> &rewriteOpMap) {
    OpTy resolvedOp = dyn_cast<OpTy>(op);
    if (rewriteOpMap.contains(op))
      resolvedOp = dyn_cast<OpTy>(rewriteOpMap.at(op));
    return resolvedOp;
  }

public:
  PointerCanonicalizer(ModuleOp moduleOp)
      : mod(moduleOp), rewriter(moduleOp.getContext()) {}

  // Propagate fat pointers in all the functions of the module
  LogicalResult run();

private:
  // IR Rewriter
  mlir::IRRewriter rewriter;

  // Actual moduleOp
  ModuleOp mod;

  // Symbol table: association between pointers and fatPointers
  llvm::MapVector<Value, FatPtr> pointers;

  void clearFunctionState() {
    rewriteOpMap.clear();
    queue.clear();
    opToDelete.clear();
  }

  // This structure is used to point to the right operation during the traversal
  // of a function
  DenseMap<Operation *, Operation *> rewriteOpMap;

  // Queue of operations to visit in the current function
  SmallVector<OpOperand *> queue;

  // List of IR to delete in the current function
  SetVector<Operation *> opToDelete;
};

} // namespace mlir::triton::AMD

#endif
