
#ifndef NV_DIALECT_HOPPER_TRANSFORMS_UTILITY_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {

typedef int AsyncTaskId;

// Retrieves the async task ids of the given operation.
SmallVector<AsyncTaskId> getAsyncTaskIds(Operation *op);

// Checks if the given operation has the given async task id.
bool hasAsyncTaskId(Operation *op, AsyncTaskId asyncTaskId);

// Sets the async task ids of the given operation.
void setAsyncTaskIds(Operation *op, ArrayRef<AsyncTaskId> asyncTaskIds);

// Retrieves the async task IDs of all operations nested within the given
// operation, including the operation itself.
SmallVector<AsyncTaskId> getNestedAsyncTaskIds(Operation *op);

// Adds the given async task ids to the given operation.
void addAsyncTaskIds(Operation *op, ArrayRef<AsyncTaskId> asyncTasks);

// Removes the given async task id from the given operation.
void removeAsyncTaskId(Operation *op, AsyncTaskId asyncTaskId);

// Removes all async task ids from the given operation.
void removeAsyncTaskIds(Operation *op);

} // namespace mlir
#endif // NV_DIALECT_HOPPER_TRANSFORMS_UTILITY_H_
