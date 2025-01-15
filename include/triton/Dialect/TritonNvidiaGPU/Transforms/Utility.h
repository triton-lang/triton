
#ifndef TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_UTILITY_H_
#define TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_UTILITY_H_

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "llvm/ADT/MapVector.h"

namespace mlir {

// 0 is reserved for default sync.
// TODO: comprehensive mechanism to globally manage namedbarrier.
static int const nameBarrierIdBegin = 1;
static int nameBarrierIdEnd = 16;

/// Helper functions for async task
typedef int AsyncTaskId;
SmallVector<AsyncTaskId> getAsyncTaskIds(Operation *op);
bool hasAsyncTaskId(Operation *op, AsyncTaskId asyncTaskId);
void setAsyncTaskIds(Operation *op, ArrayRef<AsyncTaskId> asyncTaskIds);
SmallVector<AsyncTaskId> getNestedAsyncTaskIds(Operation *op);
void addAsyncTaskIds(Operation *op, ArrayRef<int> asyncTasks);
void removeAsyncTaskId(Operation *op, AsyncTaskId asyncTaskId);
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
    OpTy op = create<OpTy>(std::forward<Args>(args)...);
    if (!asyncTaskIds.empty())
      setAsyncTaskIds(op, asyncTaskIds);
    return op;
  }

private:
  SmallVector<AsyncTaskId> asyncTaskIds;
};

class PatternRewriterWithAsyncTaskIds {
public:
  PatternRewriterWithAsyncTaskIds(PatternRewriter &rewriter, Operation *op)
      : rewriter(&rewriter) {
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
  OpTy create(Location location, Args &&...args) {
    OpTy op = rewriter->create<OpTy>(location, std::forward<Args>(args)...);
    if (!asyncTaskIds.empty())
      setAsyncTaskIds(op, asyncTaskIds);
    return op;
  }

  template <typename OpTy, typename... Args>
  OpTy replaceOpWithNewOp(Operation *op, Args &&...args) {
    auto newOp =
        rewriter->replaceOpWithNewOp<OpTy>(op, std::forward<Args>(args)...);
    return newOp;
  }

private:
  PatternRewriter *rewriter;
  SmallVector<AsyncTaskId> asyncTaskIds;
};

} // namespace mlir

#endif // TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_UTILITY_H_
