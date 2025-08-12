#ifndef NV_DIALECT_HOPPER_TRANSFORMS_UTILITY_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {

typedef int AsyncTaskId;

// Retrieves the async task ids of the given operation.
SmallVector<AsyncTaskId> getAsyncTaskIds(Operation *op);

// Checks if the given operation has the given async task id.
bool hasAsyncTaskId(Operation *op, AsyncTaskId asyncTaskId);

// Sets the async task ids of the given operation.
void setAsyncTaskIds(Operation *op, ArrayRef<AsyncTaskId> asyncTaskIds);

// Propagate the async task ids of the given operation to its parent ops.
void labelParentOps(Operation *op);

// Retrieves the async task IDs of all operations nested within the given
// operation, including the operation itself.
SmallVector<AsyncTaskId> getNestedAsyncTaskIds(Operation *op);

// Adds the given async task ids to the given operation.
void addAsyncTaskIds(Operation *op, ArrayRef<AsyncTaskId> asyncTasks);

// Removes the given async task id from the given operation.
void removeAsyncTaskId(Operation *op, AsyncTaskId asyncTaskId);

// Removes all async task ids from the given operation.
void removeAsyncTaskIds(Operation *op);

class OpBuilderWithAsyncTaskIds : public OpBuilder {
public:
  OpBuilderWithAsyncTaskIds(MLIRContext *context) : OpBuilder(context) {}

  explicit OpBuilderWithAsyncTaskIds(Operation *op) : OpBuilder(op) {
    setAsyncTaskIdsFromOp(op);
  }

  void setAsynTaskIdsFromArray(ArrayRef<AsyncTaskId> newAsyncTaskIds) {
    asyncTaskIds = SmallVector<AsyncTaskId>(newAsyncTaskIds.begin(),
                                            newAsyncTaskIds.end());
  }

  void setAsyncTaskIdsFromOp(Operation *op) {
    setAsynTaskIdsFromArray(getAsyncTaskIds(op));
  }

  void setAsyncTaskIdsFromValueUsers(Value value) {
    SetVector<AsyncTaskId> asyncTaskIdSet;
    for (Operation *user : value.getUsers())
      for (AsyncTaskId asyncTaskId : getAsyncTaskIds(user))
        asyncTaskIdSet.insert(asyncTaskId);
    setAsynTaskIdsFromArray(asyncTaskIdSet.getArrayRef());
  }

  template <typename OpTy, typename... Args>
  OpTy createWithAsyncTaskIds(Args &&...args) {
    OpTy op = OpBuilder::create<OpTy>(std::forward<Args>(args)...);
    if (!asyncTaskIds.empty())
      setAsyncTaskIds(op, asyncTaskIds);
    return op;
  }

  template <typename OpTy, typename... Args> OpTy create(Args &&...args) {
    OpTy op = createWithAsyncTaskIds<OpTy>(std::forward<Args>(args)...);
    return op;
  }

private:
  SmallVector<AsyncTaskId> asyncTaskIds;
};
} // namespace mlir
#endif // NV_DIALECT_HOPPER_TRANSFORMS_UTILITY_H_
