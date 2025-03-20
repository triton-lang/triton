
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"
#include <fstream>

namespace mlir {

namespace ttg = triton::gpu;

//===----------------------------------------------------------------------===//
// Helper functions for async task
//===----------------------------------------------------------------------===//

SmallVector<AsyncTaskId> getAsyncTaskIds(Operation *op) {
  SmallVector<AsyncTaskId> asyncTaskIds;
  if (auto attr = op->getAttrOfType<DenseIntElementsAttr>("async_task_id"))
    for (AsyncTaskId asyncTaskId : attr.getValues<AsyncTaskId>())
      asyncTaskIds.push_back(asyncTaskId);
  return asyncTaskIds;
}

bool hasAsyncTaskId(Operation *op, AsyncTaskId asyncTaskId) {
  for (AsyncTaskId candidate : getAsyncTaskIds(op))
    if (candidate == asyncTaskId)
      return true;
  return false;
}

void setAsyncTaskIds(Operation *op, ArrayRef<AsyncTaskId> asyncTaskIds) {
  SmallVector<AsyncTaskId> sortedAsyncTaskIds(asyncTaskIds.begin(),
                                              asyncTaskIds.end());
  sort(sortedAsyncTaskIds);
  auto i32Ty = IntegerType::get(op->getContext(), 32);
  auto size = static_cast<int64_t>(sortedAsyncTaskIds.size());
  auto vecTy = VectorType::get(size, i32Ty);
  op->setAttr("async_task_id",
              DenseIntElementsAttr::get(vecTy, sortedAsyncTaskIds));
}

SmallVector<AsyncTaskId> getNestedAsyncTaskIds(Operation *op) {
  SetVector<AsyncTaskId> asyncTaskIds;
  op->walk([&](Operation *curOp) {
    for (AsyncTaskId asyncTaskId : getAsyncTaskIds(curOp))
      asyncTaskIds.insert(asyncTaskId);
  });
  SmallVector<AsyncTaskId> res(asyncTaskIds.begin(), asyncTaskIds.end());
  llvm::sort(res);
  return res;
}

void addAsyncTaskIds(Operation *op, ArrayRef<int> asyncTasks) {
  auto asyncTasksVec = getAsyncTaskIds(op);
  DenseSet<int> asyncTasksSet(asyncTasksVec.begin(), asyncTasksVec.end());
  for (int a : asyncTasks) {
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
  auto end = std::remove(origAsyncTaskIds.begin(), origAsyncTaskIds.end(),
                         asyncTaskId);
  origAsyncTaskIds.erase(end, origAsyncTaskIds.end());
  if (origAsyncTaskIds.empty())
    op->removeAttr("async_task_id");
  else
    setAsyncTaskIds(op, origAsyncTaskIds);
}

void removeAsyncTaskIds(Operation *op) { op->removeAttr("async_task_id"); }
//===----------------------------------------------------------------------===//
// Implementations for general auto WS
//===----------------------------------------------------------------------===//

} // namespace mlir
