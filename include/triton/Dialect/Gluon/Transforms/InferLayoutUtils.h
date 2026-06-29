#ifndef TRITON_DIALECT_GLUON_TRANSFORMS_INFERLAYOUTUTILS_H_
#define TRITON_DIALECT_GLUON_TRANSFORMS_INFERLAYOUTUTILS_H_

#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Gluon/Transforms/Passes.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PriorityWorklist.h"

namespace mlir::triton::gluon {

enum class EncodingMaterializationPhase {
  Probe,
  Materialize,
};

// Return true during Probe to defer an operation result until all of its
// ranked tensor operands have concrete encodings. During Materialize, the
// callback must consume the deferred result.
using DeferredEncodingMaterialization = llvm::function_ref<FailureOr<bool>(
    OpResult, RankedTensorType, EncodingMaterializationPhase)>;

FailureOr<bool>
materializeRequireSlicedReshape(OpResult result, RankedTensorType resultType,
                                EncodingMaterializationPhase phase);

LogicalResult
inferLayout(FuncOp func, llvm::function_ref<bool(Type)> typeCheck,
            const SmallVector<std::pair<Value, Attribute>> &seedEncodings,
            DeferredEncodingMaterialization materializeEncoding = {});

LogicalResult doubleCheckEncodings(ModuleOp &mod,
                                   llvm::function_ref<bool(Type)> typeCheck);

} // namespace mlir::triton::gluon

#endif // TRITON_DIALECT_GLUON_TRANSFORMS_INFERLAYOUTUTILS_H_
