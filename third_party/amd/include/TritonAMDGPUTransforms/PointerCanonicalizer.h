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
// Pointer canonicalizer utility class
// -----------------------------------------------------------------------------
// This class iterates through the argument of the `funcOp`, if the argument is
// a pointer, starts a walk through its transitive uses to build a in-memory
// data structure to record the current offset to that pointer. Only when the
// pointer is really loaded/stored we materialize the base pointer with the
// offset.
//
// Let's suppose that `arg0` is a pointer. The algorithm works like that:
//
// a) At the beginning the offset is a tensor initialized to zero, and we
//    associate with `%arg0` a `FatPtr{basePtr=%arg0, offset=0}`. Through the
//    algorithm `FatPtr.basePtr` represents the scalar base pointer (all the
//    uniform updates will go into that) and `FatPtr.offset` represents the
//    tensor offset (all the non-uniform updates will go into that)
//
//
// b) Follow the pointer through the IR. When we meet:
//    `%ptr = tt.addptr(%arg0, %offset)`
//
//    Isolate the uniform and the non-uniform contributions of %offset =
//    (%u_offset, %nu_offset) and update the scalar pointer and the tensor
//    offset
//    ```
//    %s_ptr = addi(%fatPoniters[ptr].basePtr, %u_offset)
//    %t_offset = addi(%fatPoniters[ptr].offset, %nu_offset)
//    %fatPointers[%ptr0] = FatPtr{base=%s_ptr, offset=%t_offset}
//    ```
// c) When we meet the `tt.load(%ptr)` or `tt.store(%ptr)` instructions,
//    replace that instruction with:
//    `%t_ptr = tt.splat(%fatPointers[%ptr].basePtr)
//    `%fat_ptr = tt.addptr(%t_ptr, %fatPointers[ptr].offset)`
//    `%data = tt.load(%fat_ptr)`
//
class PointerCanonicalizer {
public:
  explicit PointerCanonicalizer(ModuleOp moduleOp)
      : mod(moduleOp), rewriter(moduleOp.getContext()) {}

  // Propagate fat pointers in all the functions of the module
  LogicalResult run();

private:
  // A fat pointer is represented as `basePtr + offset` internally.
  struct FatPtr {
    // Scalar base pointer. Needs to be `tt.splat`ed before used
    Value basePtr;
    // Tensor offset
    Value offset;
    // Flag to express if we can narrow the uses of the offset down to 32 bits
    bool canNarrow = false;

    // Utility copy functions
    FatPtr copy(Value newBasePtr, Value newOffset) {
      return FatPtr{newBasePtr, newOffset, canNarrow};
    };
    FatPtr copyWithBase(Value newOffset) {
      return FatPtr{basePtr, newOffset, canNarrow};
    }
    FatPtr copyWithOffset(Value newBase) {
      return FatPtr{newBase, offset, canNarrow};
    }
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

  // Extract the scalar/uniform offset from a `tt.addptr` operation
  Value getScalarOffset(Location loc, triton::AddPtrOp addPtrOp,
                        bool &hasNonUniformComponent);

  // Perform simplified scalar extraction. An offset can be composed by Unifrom
  // (U) and non-uniform(N) components. A uniform component is basically a
  // tensor constant (or a splat). A NonUniform value is a `make_range` or
  // whatever we multiply with a `make_range` operation. We consider the
  // following expressions:
  //   offset = (U+U) -> ScalarOffset = (U+U)
  //   offset = (U+N) -> ScalarOffset = (U)
  //   offset = (N+N) -> ScalarOffset = 0
  //   offset = (U*(N+U)) -> ScalarOffset = U*U
  //   offset = (N*U) -> ScalarOffset = 0
  // We do not consider the more generic expression:
  //   offset = (N+U)*(N+U)
  // Or any other expression not involving * and +.
  //
  // The function accepts the `rewriter`, the `location` and start recursing at
  // the `curOperand` offset used by the `curOp` operation.
  //
  // E.g., in  the following IR:
  // ```
  //   %offset = arith.add %uniform, %non_uniform
  //   %ptr0 = addptr %ptr, %offset`
  // ```
  // `arith.add` is the current op and `%offset` is the operand.
  //
  // Note that if we have the above use-def chain, we can rewrite this as :
  // ```
  // %ptr0 = addptr %ptr, %uniform : !ptr<f32>
  // %ptr1 = addptr %ptr0, %non_uniform : tensor<..x !ptr<f32>>
  // ```
  //
  //  We also pass the bitness of the offset and a boolean
  //  `hasNonUniformComponent` initialized to `false` to flag if the remaining
  //  tree of operations with root in `curOp` has still some non-uniform
  //  component.
  Value extractScalarOffsetFromExpr(Location loc, Operation *curOp,
                                    OpOperand &curOperand, int64_t bitness,
                                    bool &hasNonUniformComponent);

  // Extract the offset from a binary operator
  Value extractScalarOffsetFromAddOrMul(Location loc, Operation *binOp,
                                        OpOperand &curOperand, int64_t bitness,
                                        bool &hasNonUniformComponent);

  // Return either the operation or its rewritten op
  template <typename OpTy>
  OpTy resolveOp(Operation *op,
                 const DenseMap<Operation *, Operation *> &rewriteOpMap) {
    OpTy resolvedOp = dyn_cast<OpTy>(op);
    if (rewriteOpMap.contains(op))
      resolvedOp = dyn_cast<OpTy>(rewriteOpMap.at(op));
    return resolvedOp;
  }

  mlir::IRRewriter rewriter;
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
