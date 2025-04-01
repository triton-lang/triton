/*
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/LogicalResult.h"

#include <memory>
#include <optional>

#define GEN_PASS_CLASSES
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

#define DEBUG_TYPE "nvws-lower-warp-group"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

using namespace mlir;
using namespace triton;
using namespace triton::nvws;
using namespace triton::gpu;

class LowerWarpGroup : public OpRewritePattern<WarpGroupOp> {
  void visitNestedOperands(Operation *op,
                           function_ref<void(OpOperand &)> visitor) const {
    op->walk([&](Operation *nestedOp) {
      for (OpOperand &operand : nestedOp->getOpOperands()) {
        if (operand.get().getParentBlock()->getParentOp()->isProperAncestor(op))
          visitor(operand);
      }
    });
  }

  SmallVector<Value> iterateInputs(mlir::Block *block) const {
    SmallVector<Value> captures;
    for (Operation &op : *block) {
      visitNestedOperands(&op, [&](OpOperand &operand) {
        Value value = operand.get();
        if (value.getParentBlock() != block)
          captures.push_back(value);
      });
    }
    return captures;
  }

  void populateBlock(PatternRewriter &rewriter, mlir::Block *input_block,
                     mlir::Block *output_block, SmallVector<Value> &inputs,
                     SmallVector<Value> &constants) const {

    DenseMap<Value, Value> valueMap;
    rewriter.setInsertionPointToEnd(output_block);
    for (auto &value : inputs) {
      auto new_value =
          output_block->addArgument(value.getType(), value.getLoc());
      valueMap[value] = new_value;
    }
    for (auto &value : constants) {
      auto cOp = cast<arith::ConstantOp>(*value.getDefiningOp());
      auto newConst =
          rewriter.create<arith::ConstantOp>(value.getLoc(), cOp.getValue());
      valueMap[value] = newConst->getResult(0);
    }
    auto retOp = rewriter.create<triton::gpu::WarpReturnOp>(
        input_block->getParentOp()->getLoc(), ArrayRef<Type>(),
        ArrayRef<Value>());

    for (auto &op : llvm::make_early_inc_range(
             llvm::make_range(input_block->getOperations().begin(),
                              input_block->getTerminator()->getIterator()))) {
      op.moveBefore(retOp);
    }

    auto traverseParentBlocks = [&](Operation *op) -> bool {
      Block *currentBlock = op->getBlock();
      while (currentBlock) {
        // Check the current block
        if (currentBlock == output_block)
          return true;
        // Get parent operation of this block
        Operation *parentOp = currentBlock->getParentOp();
        if (!parentOp)
          break;
        // Move up to parent block
        currentBlock = parentOp->getBlock();
      }
      return false;
    };

    for (auto pair : valueMap) {
      auto oldValue = pair.first;
      auto newValue = pair.second;
      rewriter.replaceUsesWithIf(oldValue, newValue, [&](OpOperand &operand) {
        return traverseParentBlocks(operand.getOwner());
      });
    }
  }

  LogicalResult createTtgWSOp(Location loc, PatternRewriter &rewriter,
                              Block *defaultPartition,
                              ArrayRef<Block *> partitions,
                              ArrayRef<int> numWarps,
                              ArrayRef<int> warpGroupStartIds) const {
    if (partitions.size() != numWarps.size())
      return failure("mismatched number of warp groups and number of warps per "
                     "warp group");
    if (partitions.size() != warpGroupStartIds.size())
      return failure(
          "mismatched number of warp groups and number of warp start ids");
    DenseSet<Value> uniqueInputs;
    DenseSet<Value> uniqueConstants;
    for (auto *block : partitions) {
      for (Value &input : iterateInputs(block)) {
        if (!isa<BlockArgument>(input) &&
            isa<arith::ConstantOp>(input.getDefiningOp())) {
          uniqueConstants.insert(input);
        } else {
          uniqueInputs.insert(input);
        }
      }
    }
    SmallVector<Value> inputs;
    SmallVector<Value> constants;
    for (auto value : uniqueInputs)
      inputs.push_back(value);
    for (auto value : uniqueConstants)
      constants.push_back(value);

    auto wsOp = rewriter.create<WarpSpecializeOp>(loc, TypeRange(), inputs);

    wsOp.setPartitionNumWarps(numWarps);
    wsOp.setWarpGroupStartIds(warpGroupStartIds);

    auto &defaultBlock = wsOp.getDefaultRegion().emplaceBlock();
    rewriter.setInsertionPointToEnd(&defaultBlock);
    auto yieldOp =
        rewriter.create<WarpYieldOp>(loc, TypeRange(), ArrayRef<Value>());
    if (defaultPartition) {
      for (auto &op : llvm::make_early_inc_range(llvm::make_range(
               defaultPartition->getOperations().begin(),
               defaultPartition->getTerminator()->getIterator()))) {
        op.moveBefore(yieldOp);
      }
    }

    auto &block = wsOp.getPartitionOpHolder().emplaceBlock();
    rewriter.setInsertionPointToStart(&block);
    auto wspOp =
        rewriter.create<WarpSpecializePartitionsOp>(loc, partitions.size());
    auto regions = wspOp.getPartitionRegions();
    for (size_t i = 0, e = regions.size(); i < e; ++i) {
      auto &block = regions[i].emplaceBlock();
      populateBlock(rewriter, partitions[i], &block, inputs, constants);
    }
    return success();
  }

public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(WarpGroupOp warpGroupOp,
                                PatternRewriter &rewriter) const override {
    auto loc = warpGroupOp.getLoc();
    rewriter.setInsertionPointAfter(warpGroupOp);
    auto mod = warpGroupOp->getParentOfType<ModuleOp>();
    int32_t globalNumWarps =
        mlir::cast<mlir::IntegerAttr>(mod->getAttr("ttg.num-warps")).getInt();

    auto regions = warpGroupOp.getRegions();
    Block *defaultBlock = nullptr;
    int startWarp = 0;
    auto numWarps = warpGroupOp.getNumWarps();

    if (numWarps[0] == globalNumWarps) {
      defaultBlock = &regions.front()->getBlocks().front();
      regions = regions.drop_front();
      startWarp = globalNumWarps;
      numWarps = numWarps.drop_front();
    }

    auto result = createTtgWSOp(
        loc, rewriter, defaultBlock,
        llvm::map_to_vector(
            regions,
            [](Region *region) { return &region->getBlocks().front(); }),
        numWarps, llvm::map_to_vector(numWarps, [&](int numWarps) {
          int result = startWarp;
          startWarp += numWarps;
          return result;
        }));
    if (result.succeeded())
      rewriter.eraseOp(warpGroupOp);
    else
      assert(false);

    return result;
  }
};

class NVWSLowerWarpGroup : public NVWSLowerWarpGroupBase<NVWSLowerWarpGroup> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns.add<LowerWarpGroup>(context);
    GreedyRewriteConfig config;

    if (applyPatternsGreedily(m, std::move(patterns), config).failed())
      signalPassFailure();

    if (failed(m.verify()))
      assert(false);
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createNVWSLowerWarpGroupPass() {
  return std::make_unique<NVWSLowerWarpGroup>();
}
