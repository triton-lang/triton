#ifndef TRITON_DIALECT_GLUON_TRANSFORMS_INFERLAYOUTUTILS_H_
#define TRITON_DIALECT_GLUON_TRANSFORMS_INFERLAYOUTUTILS_H_

#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Gluon/Transforms/Passes.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PriorityWorklist.h"

namespace mlir::triton::gluon {

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
LogicalResult
updateEncoding(ArrayRef<Value> values, LayoutInfo info, FuncOp *func,
               llvm::MapVector<Value, LayoutInfo> &valueToEncoding,
               llvm::PriorityWorklist<Value> &worklist,
               llvm::MapVector<Attribute, uint64_t> &hashMemo);

LogicalResult inferLayout(FuncOp func, llvm::function_ref<bool(Type)> typeCheck,
                          llvm::MapVector<Value, LayoutInfo> &valueToEncoding,
                          llvm::PriorityWorklist<Value> &worklist,
                          llvm::MapVector<Attribute, uint64_t> &hashMemo);

LogicalResult inferLayout(
    ModuleOp &mod, llvm::function_ref<bool(Type)> typeCheck,
    llvm::MapVector<FuncOp, llvm::MapVector<Value, LayoutInfo>> &funcValueEnc,
    llvm::MapVector<FuncOp, llvm::PriorityWorklist<Value>> &funcWorklist,
    llvm::MapVector<FuncOp, llvm::MapVector<Attribute, uint64_t>>
        &funcHashMemo);

} // namespace mlir::triton::gluon

#endif // TRITON_DIALECT_GLUON_TRANSFORMS_INFERLAYOUTUTILS_H_
