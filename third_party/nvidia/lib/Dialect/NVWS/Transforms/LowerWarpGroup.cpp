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

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/LogicalResult.h"
#include <optional>
#include <queue>

using namespace mlir::triton;
using namespace mlir::triton::nvws;
using namespace mlir::triton::gpu;

#define DEBUG_TYPE "nvws-lower-warp-group"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_NVWSLOWERWARPGROUP
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

namespace {

class LowerWarpGroup : public OpRewritePattern<WarpGroupOp> {

  void populateRegion(PatternRewriter &rewriter, Region *inputRegion,
                      Region *outputRegion, SmallVector<Value> &inputs,
                      IRMapping &mapping) const {
    Block *output_block = &outputRegion->emplaceBlock();
    DenseMap<Value, Value> valueMap;
    rewriter.setInsertionPointToEnd(output_block);

    for (auto &value : inputs) {
      auto new_value =
          output_block->addArgument(value.getType(), value.getLoc());
      valueMap[value] = new_value;
    }
    auto retOp = rewriter.create<triton::gpu::WarpReturnOp>(
        inputRegion->getLoc(), ArrayRef<Type>(), ArrayRef<Value>());

    for (auto &op : llvm::make_early_inc_range(
             inputRegion->getBlocks().front().without_terminator())) {
      op.moveBefore(retOp);
    }

    for (auto pair : valueMap)
      replaceAllUsesInRegionWith(pair.first, pair.second, *outputRegion);
  }

  LogicalResult createWarpSpecializeOp(Location loc, WarpGroupOp warpGroupOp,
                                       PatternRewriter &rewriter,
                                       Region *defaultPartition,
                                       RegionRange partitions,
                                       ArrayRef<int> numWarps,
                                       ArrayRef<int> warpGroupStartIds) const {
    if (partitions.size() != numWarps.size())
      return failure("mismatched number of warp groups and number of warps per "
                     "warp group");
    if (partitions.size() != warpGroupStartIds.size())
      return failure(
          "mismatched number of warp groups and number of warp start ids");

    SetVector<Value> captures;
    for (auto partition : partitions)
      mlir::getUsedValuesDefinedAbove(*partition, captures);

    SmallVector<Value> inputs;
    SmallVector<IRMapping> mappings(partitions.size());
    SmallVector<OpBuilder> builders;
    for (auto region : partitions) {
      builders.push_back(OpBuilder::atBlockBegin(&region->front()));
    }

    SetVector<Operation *> opsToClone;
    std::queue<Value> que;
    for (auto capture : captures) {
      que.push(capture);
    }

    while (!que.empty()) {
      Value capture = que.front();
      // Rematerialize constants and also pure tensor ops to get around the
      // restriction below on capturing tensors.
      Operation *defOp = capture.getDefiningOp();
      if (!isa<BlockArgument>(capture) && defOp && isPure(defOp) &&
          (defOp->hasTrait<OpTrait::ConstantLike>() ||
           isa<RankedTensorType>(capture.getType()))) {
        for (auto operand : defOp->getOperands()) {
          que.push(operand);
        }
        opsToClone.insert(defOp);
      } else if (auto tensorTy =
                     dyn_cast<RankedTensorType>(capture.getType())) {
        SharedEncodingTrait sharedEnc = getSharedEncoding(tensorTy);
        auto memdescTy = MemDescType::get(
            tensorTy.getShape(), tensorTy.getElementType(), sharedEnc,
            SharedMemorySpaceAttr::get(tensorTy.getContext()));
        auto alloc = rewriter.create<LocalAllocOp>(loc, memdescTy, capture);
        for (auto [i, region] : llvm::enumerate(partitions)) {
          Value value = builders[i].create<LocalLoadOp>(capture.getLoc(),
                                                        tensorTy, alloc);
          replaceAllUsesInRegionWith(capture, value, *region);
          mappings[i].map(capture, value);
        }
        inputs.push_back(alloc);
      } else {
        inputs.push_back(capture);
      }
      que.pop();
    }

    opsToClone = topologicalSort(opsToClone);

    for (auto [region, b, mapping] :
         llvm::zip(partitions, builders, mappings)) {
      for (Operation *op : opsToClone) {
        auto copy = b.clone(*op, mapping)->getResult(0);
        mapping.map(op->getResult(0), copy);
        replaceAllUsesInRegionWith(op->getResult(0), copy, *region);
      }
    }

    auto wsOp = rewriter.create<WarpSpecializeOp>(
        loc, warpGroupOp.getResultTypes(), inputs);

    wsOp.setPartitionNumWarps(numWarps);

    auto &defaultBlock = wsOp.getDefaultRegion().emplaceBlock();
    rewriter.setInsertionPointToEnd(&defaultBlock);

    if (defaultPartition) {
      auto yieldOp = defaultPartition->front().getTerminator();
      auto newYieldOp = rewriter.create<WarpYieldOp>(
          loc, yieldOp->getResultTypes(), yieldOp->getOperands());

      for (auto &op : llvm::make_early_inc_range(
               defaultPartition->getBlocks().front().without_terminator())) {
        op.moveBefore(newYieldOp);
      }
    } else {
      rewriter.create<WarpYieldOp>(loc, TypeRange(), ArrayRef<Value>());
    }

    auto &block = wsOp.getPartitionOpHolder().emplaceBlock();
    rewriter.setInsertionPointToStart(&block);
    auto wspOp =
        rewriter.create<WarpSpecializePartitionsOp>(loc, partitions.size());
    auto regions = wspOp.getPartitionRegions();

    for (auto [in, out, mapping] : zip(partitions, regions, mappings))
      populateRegion(rewriter, in, &out, inputs, mapping);

    warpGroupOp.replaceAllUsesWith(wsOp);

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
    Region *defaultRegion = nullptr;
    int startWarp = 0;
    auto numWarps = warpGroupOp.getNumWarps();

    if (numWarps[0] == globalNumWarps) {
      defaultRegion = regions.front();
      regions = regions.drop_front();
      startWarp = globalNumWarps;
      numWarps = numWarps.drop_front();
    } else if (warpGroupOp.getNumResults() > 0) {
      return failure("The first warp group does not use the default number of "
                     "warps. The default partition cannot be created. When "
                     "nvws.warp_group op returns results, there must be a "
                     "default region.");
    }

    auto result = createWarpSpecializeOp(
        loc, warpGroupOp, rewriter, defaultRegion, regions, numWarps,
        llvm::map_to_vector(numWarps, [&](int numWarps) {
          int result = startWarp;
          startWarp += numWarps;
          return result;
        }));

    if (result.succeeded())
      rewriter.eraseOp(warpGroupOp);

    return result;
  }
};

} // namespace

class NVWSLowerWarpGroup
    : public impl::NVWSLowerWarpGroupBase<NVWSLowerWarpGroup> {
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

} // namespace triton
} // namespace mlir
