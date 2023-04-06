#ifndef TRITON_LIB_DIALECT_TRITONGPU_TRANSFORMS_UTILITY_H_
#define TRITON_LIB_DIALECT_TRITONGPU_TRANSFORMS_UTILITY_H_
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/MapVector.h"

namespace mlir {

LogicalResult fixupLoops(ModuleOp mod);

// TODO: Interface
LogicalResult invertEncoding(Attribute targetEncoding, Operation *op,
                             Attribute &ret);

bool expensiveLoadOrStore(Operation *op, Attribute &targetEncoding);

bool expensiveToRemat(Operation *op, Attribute &targetEncoding);

int simulateBackwardRematerialization(
    Operation *initOp, SetVector<Operation *> &processed,
    SetVector<Attribute> &layout, llvm::MapVector<Value, Attribute> &toConvert,
    const Attribute &targetEncoding);

Operation *cloneWithInferType(mlir::PatternRewriter &rewriter, Operation *op,
                              IRMapping &mapping);

void rematerializeConversionChain(
    const llvm::MapVector<Value, Attribute> &toConvert,
    mlir::PatternRewriter &rewriter, SetVector<Operation *> &processed,
    IRMapping &mapping);
} // namespace mlir

#endif // TRITON_LIB_DIALECT_TRITONGPU_TRANSFORMS_UTILITY_H_
