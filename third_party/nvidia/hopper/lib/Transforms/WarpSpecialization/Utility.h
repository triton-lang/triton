
#ifndef NV_DIALECT_HOPPER_TRANSFORMS_UTILITY_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {

typedef int AsyncTaskId;
SmallVector<AsyncTaskId> getAsyncTaskIds(Operation *op);
bool hasAsyncTaskId(Operation *op, AsyncTaskId asyncTaskId);
void setAsyncTaskIds(Operation *op, ArrayRef<AsyncTaskId> asyncTaskIds);
SmallVector<AsyncTaskId> getNestedAsyncTaskIds(Operation *op);
void addAsyncTaskIds(Operation *op, ArrayRef<int> asyncTasks);
void removeAsyncTaskId(Operation *op, AsyncTaskId asyncTaskId);
void removeAsyncTaskIds(Operation *op);

} // namespace mlir
#endif // NV_DIALECT_HOPPER_TRANSFORMS_UTILITY_H_
