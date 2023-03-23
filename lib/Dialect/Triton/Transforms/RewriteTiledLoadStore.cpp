#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

#include <memory>
#include <stack>

using namespace mlir;

#define GEN_PASS_CLASSES
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

/// An additional struct to record the meta information of tiled operations
struct RewritedInfo {
private:
  Value base;
  SmallVector<Value> shape;
  SmallVector<Value> strides;
  SmallVector<Value> offsets;
  ArrayRef<int64_t> tileShape;

  // A cache to avoid generating the same offset with range
  DenseMap<unsigned, Value> cachedOffsetWithRange;

public:
  RewritedInfo() = default;

  RewritedInfo(const RewritedInfo &other) = default;

  RewritedInfo(Value base, const SmallVector<Value> &shape,
               const SmallVector<Value> &strides,
               const SmallVector<Value> &offsets,
               const ArrayRef<int64_t> &tileShape)
      : base(base), shape(shape), strides(strides), offsets(offsets),
        tileShape(tileShape) {
    assert(shape.size() == strides.size() && shape.size() == offsets.size() &&
           shape.size() == tileShape.size());
  }

  unsigned int length() const { return shape.size(); }

  Value getOffset(unsigned i) { return offsets[i]; }

  SmallVector<Value> getOffsets() { return offsets; }

  void setOffset(unsigned i, Value newOffset) {
    offsets[i] = newOffset;
    cachedOffsetWithRange.clear();
  }

  void setOffsets(const SmallVector<Value> &newOffsets) {
    offsets = newOffsets;
    cachedOffsetWithRange.clear();
  }

  Value getExpandedI64OffsetWithRange(OpBuilder &builder, const Location &loc,
                                      unsigned i) {
    if (cachedOffsetWithRange.count(i))
      return cachedOffsetWithRange[i];

    auto indexI32RowType =
        RankedTensorType::get({tileShape[i]}, builder.getI32Type());
    Value splatOffset =
        builder.create<triton::SplatOp>(loc, indexI32RowType, offsets[i]);
    Value range = builder.create<triton::MakeRangeOp>(loc, indexI32RowType, 0,
                                                      tileShape[i]);
    Value expandedResult =
        builder.create<arith::AddIOp>(loc, splatOffset, range);
    for (int j = 0; j < tileShape.size(); ++j) {
      if (j == i)
        continue;
      expandedResult =
          builder.create<triton::ExpandDimsOp>(loc, expandedResult, j);
    }

    // `expandedShape` should be like {1, 1, ..., tileShape[i], 1}
    auto expandedShape =
        expandedResult.getType().cast<RankedTensorType>().getShape();
    auto expandedI64Type =
        RankedTensorType::get(expandedShape, builder.getI64Type());
    // Current `tt.make_range` and `offsets` only support I32,
    // so we need a cast here
    Value i64Result =
        builder.create<arith::ExtSIOp>(loc, expandedI64Type, expandedResult);
    return cachedOffsetWithRange[i] = i64Result;
  }

  Value generatePtr(OpBuilder &builder, const Location &loc) {
    assert(tileShape.size() == offsets.size() &&
           tileShape.size() == strides.size());
    auto indexTensorType =
        RankedTensorType::get(tileShape, builder.getI64Type());
    auto ptrType = base.getType().cast<triton::PointerType>();
    auto ptrTensorType = RankedTensorType::get(tileShape, ptrType);

    // Generate offsets per dimension
    Value ptr = builder.create<triton::SplatOp>(loc, ptrTensorType, base);
    for (unsigned i = 0; i < tileShape.size(); ++i) {
      auto offsetWithRange = getExpandedI64OffsetWithRange(builder, loc, i);

      // We must splat strides into the expanded shape not a row for retaining
      // the divisibility information given by strides
      Value splatStride = builder.create<triton::SplatOp>(
          loc, offsetWithRange.getType(), strides[i]);
      Value offsetWithStride =
          builder.create<arith::MulIOp>(loc, offsetWithRange, splatStride);
      Value broadcasted = builder.create<triton::BroadcastOp>(
          loc, indexTensorType, offsetWithStride);

      // Add to the pointer
      ptr = builder.create<triton::AddPtrOp>(loc, ptrTensorType, ptr,
                                             broadcasted);
    }

    return ptr;
  }

  Value generateMask(OpBuilder &builder, const Location &loc,
                     const std::optional<ArrayRef<int32_t>> &boundaryCheck) {
    if (!boundaryCheck.has_value())
      return {};

    // Generate mask per dimension
    auto maskTensorType = RankedTensorType::get(tileShape, builder.getI1Type());
    Value mask;
    for (auto i : boundaryCheck.value()) {
      auto offsetWithRange = getExpandedI64OffsetWithRange(builder, loc, i);

      // Compare with lower bound
      Value lowerBound = builder.create<mlir::arith::ConstantIntOp>(
          loc, 0, builder.getI64Type());
      Value splatLowerBound = builder.create<triton::SplatOp>(
          loc, offsetWithRange.getType(), lowerBound);
      Value cmpLower = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sge, offsetWithRange, splatLowerBound);

      // Compare with upper bound
      Value splatUpperBound = builder.create<triton::SplatOp>(
          loc, offsetWithRange.getType(), shape[i]);
      Value cmpUpper = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, offsetWithRange, splatUpperBound);

      // And and broadcast
      Value andResult = builder.create<arith::AndIOp>(loc, cmpLower, cmpUpper);
      Value broadcasted =
          builder.create<triton::BroadcastOp>(loc, maskTensorType, andResult);

      // And up all results
      if (!mask) {
        mask = broadcasted;
      } else {
        mask = builder.create<arith::AndIOp>(loc, mask, broadcasted);
      }
    }

    return mask;
  }

  Value generateOther(OpBuilder &builder, const Location &loc,
                      const std::optional<triton::PaddingOption> &padding) {
    if (!padding.has_value())
      return Value();

    // Create element attribute
    auto elementType =
        base.getType().cast<triton::PointerType>().getPointeeType();
    auto otherTensorType = RankedTensorType::get(tileShape, elementType);
    auto floatAttr = builder.getFloatAttr(elementType, 0);
    // TODO(Chenggang): add tests to check alignment with TMA
    if (padding.value() == triton::PaddingOption::PAD_NAN) {
      auto apNaN = llvm::APFloat::getNaN(floatAttr.getValue().getSemantics());
      floatAttr = builder.getFloatAttr(elementType, apNaN);
    }

    // Create tensor
    Value constant = builder.create<arith::ConstantOp>(loc, floatAttr);
    return builder.create<triton::SplatOp>(loc, otherTensorType, constant);
  }
};

class RewriteTiledLoadStorePass
    : public TritonRewriteTiledLoadStoreBase<RewriteTiledLoadStorePass> {
private:
  DenseMap<Value, RewritedInfo> rewritedInfo;

public:
  static bool needRewrite(Operation *op) {
    return std::any_of(
        op->getOperands().begin(), op->getOperands().end(),
        [](Value operand) { return isTilePointerType(operand.getType()); });
  }

  static SmallVector<Value>
  generateNewOperands(const SmallVector<Value> &oldOperands, unsigned index,
                      const SmallVector<Value> &newValues) {
    assert(index < oldOperands.size());
    SmallVector<Value> newOperands;
    for (int i = 0; i < index; ++i)
      newOperands.push_back(oldOperands[i]);
    for (auto value : newValues)
      newOperands.push_back(value);
    for (auto i = index + 1; i < oldOperands.size(); ++i)
      newOperands.push_back(oldOperands[i]);
    return newOperands;
  }

  Operation *rewriteMakeTilePtrOp(OpBuilder &builder, triton::MakeTilePtrOp op,
                                  std::stack<Operation *> &eraser) {
    // Save info for later use
    auto ptrType = op.getResult().getType().cast<triton::PointerType>();
    auto tensorType = ptrType.getPointeeType().cast<RankedTensorType>();
    rewritedInfo[op.getResult()] =
        RewritedInfo(op.getBase(), op.getShape(), op.getStrides(),
                     op.getOffsets(), tensorType.getShape());

    // Erase the original operation
    eraser.push(op);
    return nullptr;
  }

  Operation *rewriteAdvanceOp(OpBuilder &builder, triton::AdvanceOp op,
                              std::stack<Operation *> &eraser) {
    // Get info from previous results
    assert(rewritedInfo.count(op.getPtr()));
    auto info = rewritedInfo[op.getPtr()];

    // Calculate new offsets
    assert(info.length() == op.getOffsets().size());
    SmallVector<Value> newOffsets;
    for (int i = 0; i < info.length(); ++i) {
      Value newOffset = builder.create<arith::AddIOp>(
          op.getLoc(), info.getOffset(i), op.getOffsets()[i]);
      newOffsets.push_back(newOffset);
    }

    // Save info for later use
    info.setOffsets(newOffsets);
    rewritedInfo[op.getResult()] = info;

    // Erase the original operation
    eraser.push(op);
    return nullptr;
  }

  Operation *rewriteLoadStoreOp(OpBuilder &builder, Operation *op,
                                std::stack<Operation *> &eraser) {
    assert(isa<triton::LoadOp>(op) || isa<triton::StoreOp>(op));

    // We only have to rewrite tiled load/stores
    auto ptr = op->getOperand(0);
    if (!isTilePointerType(ptr.getType()))
      return nullptr;

    // Get info from previous results
    assert(rewritedInfo.count(ptr));
    auto info = rewritedInfo[ptr];

    // Tiled load/store implicitly will check the bound while accessing memory,
    // so we should set `mask` and `other` (according to the padding)
    // Also note that tiled load do not have `mask` and `other` while building
    // IR from Python AST
    std::optional<ArrayRef<int>> boundaryCheck;
    if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
      assert(!loadOp.getMask() && !loadOp.getOther());
      boundaryCheck = loadOp.getBoundaryCheck();
    } else if (auto storeOp = dyn_cast<triton::StoreOp>(op)) {
      assert(!storeOp.getMask());
      llvm_unreachable("working in progress");
    }

    // Generate new `ptr`, `mask` and `other`
    auto newPtr = info.generatePtr(builder, op->getLoc());
    auto newMask = info.generateMask(builder, op->getLoc(), boundaryCheck);
    Value newOther;
    if (auto loadOp = dyn_cast<triton::LoadOp>(op))
      newOther = info.generateOther(builder, op->getLoc(), loadOp.getPadding());

    // Create a new operation
    if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
      auto newResult = builder.create<triton::LoadOp>(
          loadOp.getLoc(), newPtr, newMask, newOther, loadOp.getCache(),
          loadOp.getEvict(), loadOp.getIsVolatile());
      op->getResult(0).replaceAllUsesWith(newResult);
    } else if (auto storeOp = dyn_cast<triton::StoreOp>(op)) {
      builder.create<triton::StoreOp>(storeOp.getLoc(), newPtr,
                                      storeOp.getValue(), newMask,
                                      storeOp.getCache(), storeOp.getEvict());
    }

    // Erase the original operation
    eraser.push(op);
    return nullptr;
  }

  Operation *rewriteForOp(OpBuilder &builder, scf::ForOp op,
                          std::stack<Operation *> &eraser) {
    // Generate new iteration operands and set rewrited information
    SmallVector<Value> oldIterOperands = op.getIterOperands();
    SmallVector<Value> newIterOperands = op.getIterOperands();
    for (unsigned i = 0, oldI = 0, size = op.getNumIterOperands(); i < size;
         ++i, ++oldI) {
      if (!isTilePointerType(newIterOperands[i].getType()))
        continue;

      // Expand the tile pointer into offsets
      assert(rewritedInfo.count(newIterOperands[i]));
      auto info = rewritedInfo[newIterOperands[i]];
      newIterOperands =
          generateNewOperands(newIterOperands, i, info.getOffsets());
      i += info.length() - 1;
      size += info.length() - 1;
    }

    // Rebuild the loop type
    auto newForOp = builder.create<scf::ForOp>(op.getLoc(), op.getLowerBound(),
                                               op.getUpperBound(), op.getStep(),
                                               newIterOperands);

    // Create value mapping. Note that for tile pointers, we use identity
    // mapping. It may refer to a value in the old loop, but we will rewrite it
    // later.
    IRMapping mapping;
    for (unsigned i = 0, oldI = 0; oldI < op.getNumIterOperands();
         ++i, ++oldI) {
      auto oldRegionIterArg = op.getRegionIterArg(oldI);
      if (isTilePointerType(oldRegionIterArg.getType())) {
        // Pass rewrited info inside
        assert(rewritedInfo.count(oldIterOperands[oldI]));
        auto info = rewritedInfo[oldIterOperands[oldI]];
        mapping.map(oldRegionIterArg, oldRegionIterArg);
        for (unsigned j = 0; j < info.length(); ++j)
          info.setOffset(j, newForOp.getRegionIterArg(i + j));
        rewritedInfo[oldRegionIterArg] = info;
        i += info.length() - 1;
      } else {
        mapping.map(oldRegionIterArg, newForOp.getRegionIterArg(i));
      }
    }
    mapping.map(op.getInductionVar(), newForOp.getInductionVar());

    // Clone body
    builder.setInsertionPointToStart(newForOp.getBody());
    for (auto &opInFor : *op.getBody()) {
      auto *newOp = builder.clone(opInFor, mapping);
      for (unsigned i = 0; i < opInFor.getNumResults(); ++i)
        mapping.map(op->getResult(i), newOp->getResult(i));
    }

    // Replace later usages
    assert(op.getNumResults() == op.getNumIterOperands());
    for (unsigned i = 0, oldI = 0; oldI < op.getNumResults(); ++i, ++oldI) {
      auto oldResult = op.getResult(oldI);
      if (isTilePointerType(oldResult.getType())) {
        // Pack new offsets into rewrited info
        assert(rewritedInfo.count(oldIterOperands[oldI]));
        auto info = rewritedInfo[oldIterOperands[oldI]];
        for (unsigned j = 0; j < info.length(); ++j)
          info.setOffset(j, newForOp.getResult(i + j));
        i += info.length() - 1;
        rewritedInfo[oldResult] = info;
      } else {
        oldResult.replaceAllUsesWith(newForOp.getResult(i));
      }
    }

    // Erase later
    eraser.push(op);
    return newForOp;
  }

  Operation *rewriteYieldOp(OpBuilder &builder, scf::YieldOp op,
                            std::stack<Operation *> &eraser) {
    // Replace tile pointers with offsets
    SmallVector<Value> newOperands = op->getOperands();
    for (unsigned i = 0, size = op.getNumOperands(); i < size; ++i) {
      if (!isTilePointerType(newOperands[i].getType()))
        continue;

      assert(rewritedInfo.count(newOperands[i]));
      auto info = rewritedInfo[newOperands[i]];
      newOperands = generateNewOperands(newOperands, i, info.getOffsets());
      i += info.length() - 1;
      size += info.length() - 1;
    }
    op->setOperands(newOperands);

    // No need to erase
    return nullptr;
  }

  Operation *rewriteOp(Operation *op, std::stack<Operation *> &eraser) {
    OpBuilder builder(op);

    // Rewrite `make_tile_ptr` and `advance` and make a tensor of pointers
    // Rewriting functions return the next operation to visit, if there is no
    // next one, simply return `nullptr`
    std::pair<Value, RewritedInfo> rewrited;
    if (auto makeTilePtrOp = dyn_cast<triton::MakeTilePtrOp>(op)) {
      return rewriteMakeTilePtrOp(builder, makeTilePtrOp, eraser);
    } else if (auto advanceOp = dyn_cast<triton::AdvanceOp>(op)) {
      return rewriteAdvanceOp(builder, advanceOp, eraser);
    } else if (isa<triton::LoadOp>(op) || isa<triton::StoreOp>(op)) {
      return rewriteLoadStoreOp(builder, op, eraser);
    } else if (auto storeOp = dyn_cast<triton::StoreOp>(op)) {
      llvm_unreachable("Working in progress");
    } else if (op->getDialect()->getNamespace() == "scf" ||
               op->getDialect()->getNamespace() == "cf") {
      if (!needRewrite(op))
        return op;

      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        return rewriteForOp(builder, forOp, eraser);
      } else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
        return rewriteYieldOp(builder, yieldOp, eraser);
      } else {
        llvm_unreachable("Currently we only support tile pointer usages inside"
                         "a `scf::ForOp`, others such as `scf::IfOp`,"
                         "`scf::WhileOp`, `cf::BranchOp` or `cf::CondBranchOp` "
                         "are not supported yet");
      }
    }

    // Otherwise return the original one
    return op;
  }

  void visitOperation(Operation *op, std::stack<Operation *> &eraser) {
    for (auto &region : op->getRegions()) {
      for (auto &block : region) {
        // We need an extra copy because erasing operations may break the
        // iterator behavior
        SmallVector<Operation *> blockCopy;
        for (auto &nestedOp : block)
          blockCopy.push_back(&nestedOp);

        // Rewrite and recursively visit
        for (auto &nestedOp : blockCopy) {
          if (auto newOp = rewriteOp(nestedOp, eraser))
            visitOperation(newOp, eraser);
        }
      }
    }
  }

  void runOnOperation() override {
    // NOTES(Chenggang): we don't use `ConversionPatternRewriter`, because
    // MLIR does not support one-multiple value mapping. If we convert the
    // result of `tt.make_tile_ptr` into `tuple<base, strides, offsets>`,
    // we have to define pack and unpack operations for this tuple type,
    // which is much more effort.
    // So here we recursively build the IR, to be specific, we have to rewrite
    // `tt.make_tile_ptr`, `tt.advance`, `tt.load`, `tt.store`,
    // `scf.for` (tile-based semantics may be in a loop fashion)
    std::stack<Operation *> eraser;
    visitOperation(getOperation(), eraser);

    // The operation could not be erased during visit, because they may have
    // later usages, so we erase after visit
    rewritedInfo.clear();
    while (!eraser.empty()) {
      auto op = eraser.top();
      eraser.pop();
      op->erase();
    }
  }
};

std::unique_ptr<Pass> triton::createRewriteTiledLoadStorePass() {
  return std::make_unique<RewriteTiledLoadStorePass>();
}
