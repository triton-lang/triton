#pragma once
#include "mlir/IR/Builders.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/ADT/ArrayRef.h"
#include <memory>

typedef int AsyncTaskId;
void setAsyncTaskIds(mlir::Operation *op,
                     llvm::ArrayRef<AsyncTaskId> asyncTaskIds);

// A custom op builder that keeps track of the last location
class TritonOpBuilder {
public:
  TritonOpBuilder(mlir::MLIRContext *context) {
    builder = std::make_unique<mlir::OpBuilder>(context);
    lastLoc = std::make_unique<mlir::Location>(builder->getUnknownLoc());
  }

  mlir::OpBuilder &getBuilder() { return *builder; }
  mlir::MLIRContext *getContext() { return builder->getContext(); }

  bool isLineInfoEnabled() { return lineInfoEnabled; }

  void setLastLoc(mlir::Location loc) {
    if (lineInfoEnabled)
      lastLoc = std::make_unique<mlir::Location>(loc);
  }

  void setLastLoc(const std::string &fileName, int line, int column) {
    auto context = builder->getContext();
    setLastLoc(mlir::FileLineColLoc::get(context, fileName, line, column));
  }

  mlir::Location getLastLoc() {
    assert(lastLoc);
    return *lastLoc;
  }

  void setInsertionPointToStart(mlir::Block &block) {
    if (!block.empty())
      setLastLoc(block.begin()->getLoc());
    else
      setLastLoc(builder->getUnknownLoc());
    builder->setInsertionPointToStart(&block);
  }

  void setInsertionPointToEnd(mlir::Block &block) {
    if (!block.empty())
      setLastLoc(block.back().getLoc());
    else
      setLastLoc(builder->getUnknownLoc());
    builder->setInsertionPointToEnd(&block);
  }

  void setInsertionPointAfter(mlir::Operation &op) {
    setLastLoc(op.getLoc());
    builder->setInsertionPointAfter(&op);
  }

  void restoreInsertionPoint(mlir::OpBuilder::InsertPoint pt) {
    if (pt.isSet() && pt.getPoint() != pt.getBlock()->end())
      setLastLoc(pt.getPoint()->getLoc());
    else
      setLastLoc(builder->getUnknownLoc());
    builder->restoreInsertionPoint(pt);
  }

  template <typename OpTy, typename... Args> OpTy create(Args &&...args) {
    auto loc = getLastLoc();
    auto ret = builder->create<OpTy>(loc, std::forward<Args>(args)...);
    if (asyncTaskIds)
      ::setAsyncTaskIds(ret, *asyncTaskIds);
    return ret;
  }

  // Overload to create or fold a single result operation.
  template <typename OpTy, typename... Args>
  std::enable_if_t<OpTy::template hasTrait<mlir::OpTrait::OneResult>(),
                   mlir::Value>
  createOrFold(Args &&...args) {
    auto loc = getLastLoc();
    return builder->createOrFold<OpTy>(loc, std::forward<Args>(args)...);
  }

  // Overload to create or fold a zero result operation.
  template <typename OpTy, typename... Args>
  std::enable_if_t<OpTy::template hasTrait<mlir::OpTrait::ZeroResults>(), OpTy>
  createOrFold(Args &&...args) {
    auto loc = getLastLoc();
    return builder->createOrFold<OpTy>(loc, std::forward<Args>(args)...);
  }

  void setAsyncTaskIds(std::vector<int> taskIds) {
    this->asyncTaskIds = taskIds;
  }

  void unsetAsyncTaskIds() { this->asyncTaskIds = std::nullopt; }

private:
  std::unique_ptr<mlir::OpBuilder> builder;
  std::unique_ptr<mlir::Location> lastLoc;
  std::optional<std::vector<int>> asyncTaskIds;
  bool lineInfoEnabled =
      !mlir::triton::tools::getBoolEnv("TRITON_DISABLE_LINE_INFO");
};
