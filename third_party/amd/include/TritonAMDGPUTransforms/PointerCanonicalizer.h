#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_POINTER_CANONICALIZER_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_POINTER_CANONICALIZER_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;

namespace mlir::triton::AMD {

// -----------------------------------------------------------------------------
//
// -----------------------------------------------------------------------------
// This class iterates throught the argument of the `funcOp`, if the argument is
// a pointer, it starts a walk to replace the pointer through all the IR with an
// offset. Only when the pointer is really needed the offset is added to the
// base pointer and passed to the operation that needs the pointer (usually a
// load or a store)
//
// Let's suppose that `arg0` is a pointer. The algorithm works like that:
// a) At the beginning the offset is zero, and we associate with `arg0` a
// `FatPtr{arg0, offset}` b) Follow the pointer through the IR, and replace any
// `tt.addptr(%ptr, %offset)` with
//    `add(%fatPoniters[ptr].basePtr, %fatPointers[ptr].offset)`.
// c) When you meet the `tt.load(%ptr)` or `tt.store(%ptr)` instructions,
// replace that instruction with:
//    %fat_ptr = tt.addptr(%fatPointers[ptr].basePtr, %fatPointers[ptr].offset)
//    %data = tt.load(%fat_ptr)
//
class PointerCanonicalizer {
private:
  template <typename OpTy>
  OpTy resolveOp(Operation *op,
                 const DenseMap<Operation *, Operation *> &rewriteOpMap) {
    OpTy resolvedOp = dyn_cast<OpTy>(op);
    if (rewriteOpMap.contains(op))
      resolvedOp = dyn_cast<OpTy>(rewriteOpMap.at(op));
    return resolvedOp;
  }

  // Rewrite any operation that needs a pointer
  LogicalResult materializeFatPointer(Operation *op, Location loc, Value ptr);

  // Start from an argument of a function and propagate its fat pointers
  LogicalResult rewritePointer(Value argPtr);

  // Rewrite a given region, canonicalizing the different pointer arguments of
  // the region
  LogicalResult rewriteRegion(Region &region);

public:
  PointerCanonicalizer(ModuleOp moduleOp)
      : mod(moduleOp), rewriter(moduleOp.getContext()), offset64(false),
        unknownOp(false) {}

  // Propagate fat pointers in all the functions of the module
  LogicalResult run();

  // Returns if, while propagating any fat pointer,
  // we met any unkwnown operation that needed a pointer
  bool hasUnknownOps() const { return unknownOp; }

  // Returns if, while propagating any fat pointer, we met any 64
  // bit offset
  bool has64BitOffset() const { return offset64; }

private:
  // This is the internal representation of a fat pointer: `fatPtr = basePtr +
  // offset`
  struct FatPtr {
    Value basePtr;
    Value offset;
  };

  // Rewriter
  mlir::IRRewriter rewriter;

  // Actual moduleOp
  ModuleOp mod;

  // Symbol table: association between pointers and fatPointers
  llvm::MapVector<Value, FatPtr> pointers;

  bool unknownOp;

  bool offset64;
};

} // namespace mlir::triton::AMD

#endif
