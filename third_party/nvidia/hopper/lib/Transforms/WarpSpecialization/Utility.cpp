#include "Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// Helper functions for async task
//===----------------------------------------------------------------------===//

SmallVector<AsyncTaskId> getAsyncTaskIds(Operation *op) {
  SmallVector<AsyncTaskId> asyncTaskIds;
  if (auto attr = op->getAttrOfType<DenseI32ArrayAttr>("async_task_id")) {
    for (AsyncTaskId asyncTaskId : attr.asArrayRef()) {
      // TODO(Arda): Remove this check once we figure out why we have duplicate
      // async task ids
      if (asyncTaskIds.empty() ||
          asyncTaskIds[asyncTaskIds.size() - 1] != asyncTaskId)
        asyncTaskIds.push_back(asyncTaskId);
    }
  }
  return asyncTaskIds;
}

bool hasAsyncTaskId(Operation *op, AsyncTaskId asyncTaskId) {
  return llvm::is_contained(getAsyncTaskIds(op), asyncTaskId);
}

void setAsyncTaskIds(Operation *op, ArrayRef<AsyncTaskId> asyncTaskIds) {
  SmallVector<AsyncTaskId> sortedAsyncTaskIds(asyncTaskIds.begin(),
                                              asyncTaskIds.end());
  sort(sortedAsyncTaskIds);
  auto i32Ty = IntegerType::get(op->getContext(), 32);
  auto size = static_cast<int64_t>(sortedAsyncTaskIds.size());
  auto vecTy = VectorType::get(size, i32Ty);
  op->setAttr("async_task_id",
              DenseI32ArrayAttr::get(op->getContext(), sortedAsyncTaskIds));
}

void labelParentOps(Operation *op) {
  auto asyncTaskIds = mlir::getAsyncTaskIds(op);
  auto parent = op->getParentOp();
  while (parent && !isa<triton::FuncOp>(parent)) {
    addAsyncTaskIds(parent, asyncTaskIds);
    parent = parent->getParentOp();
  }
}

SmallVector<AsyncTaskId> getNestedAsyncTaskIds(Operation *op) {
  SetVector<AsyncTaskId> asyncTaskIds;
  op->walk([&](Operation *curOp) {
    asyncTaskIds.insert_range(getAsyncTaskIds(curOp));
  });
  SmallVector<AsyncTaskId> res = asyncTaskIds.takeVector();
  llvm::sort(res);
  return res;
}

void addAsyncTaskIds(Operation *op, ArrayRef<AsyncTaskId> asyncTasks) {
  auto asyncTasksVec = getAsyncTaskIds(op);
  DenseSet<AsyncTaskId> asyncTasksSet(asyncTasksVec.begin(),
                                      asyncTasksVec.end());
  for (auto a : asyncTasks) {
    if (!asyncTasksSet.contains(a)) {
      asyncTasksVec.push_back(a);
    }
  }
  if (asyncTasksVec.size() > 0) {
    setAsyncTaskIds(op, asyncTasksVec);
  }
}

void removeAsyncTaskId(Operation *op, AsyncTaskId asyncTaskId) {
  auto origAsyncTaskIds = getAsyncTaskIds(op);
  llvm::erase(origAsyncTaskIds, asyncTaskId);
  if (origAsyncTaskIds.empty())
    op->removeAttr("async_task_id");
  else
    setAsyncTaskIds(op, origAsyncTaskIds);
}

void removeAsyncTaskIds(Operation *op) { op->removeAttr("async_task_id"); }

} // namespace mlir
