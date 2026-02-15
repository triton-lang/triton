#ifndef TRITON_DIALECT_GLUON_TRANSFORMS_INFERLAYOUTUTILS_H_
#define TRITON_DIALECT_GLUON_TRANSFORMS_INFERLAYOUTUTILS_H_

#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Gluon/Transforms/Passes.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PriorityWorklist.h"

namespace mlir::triton::gluon {

LogicalResult
inferLayout(FuncOp func, llvm::function_ref<bool(Type)> typeCheck,
            const SmallVector<std::pair<Value, Attribute>> &seedEncodings);

LogicalResult doubleCheckEncodings(ModuleOp &mod,
                                   llvm::function_ref<bool(Type)> typeCheck);

} // namespace mlir::triton::gluon

#endif // TRITON_DIALECT_GLUON_TRANSFORMS_INFERLAYOUTUTILS_H_
