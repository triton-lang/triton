#include "../../TritonGPU/Transforms/WSUtility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/LogicalResult.h"

#include <iostream>
#include <memory>
#include <optional>

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tritonnvidiagpu-aref-if-to-ttg-ws"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
namespace ttng = triton::nvidia_gpu;

void visitNestedOperands(Operation *op,
                         function_ref<void(OpOperand &)> visitor) {
  op->walk([&](Operation *nestedOp) {
    for (OpOperand &operand : nestedOp->getOpOperands()) {
      if (operand.get().getParentBlock()->getParentOp()->isProperAncestor(op))
        visitor(operand);
    }
  });
}

SmallVector<Value> iterateInputs(mlir::Block *block) {
  SmallVector<Value> captures;
  for (Operation &op : *block) {
    visitNestedOperands(&op, [&](OpOperand &operand) {
      // Ignore implicit captures.
      Value value = operand.get();
      if (value.getParentBlock() != block)
        captures.push_back(value);
    });
  }
  return captures;
}

void populateBlock(OpBuilder &builder, mlir::Block *input_block,
                   mlir::Block *output_block, SmallVector<Value> &inputs,
                   SmallVector<Value> &constants) {

  DenseMap<Value, Value> valueMap;
  builder.setInsertionPointToEnd(output_block);
  for (auto &value : inputs) {
    auto new_value = output_block->addArgument(value.getType(), value.getLoc());
    valueMap[value] = new_value;
  }
  for (auto &value : constants) {
    auto cOp = cast<arith::ConstantOp>(*value.getDefiningOp());
    auto newConst =
        builder.create<arith::ConstantOp>(value.getLoc(), cOp.getValue());
    valueMap[value] = newConst->getResult(0);
  }
  auto retOp =
      builder.create<WarpReturnOp>(input_block->getParentOp()->getLoc(),
                                   ArrayRef<Type>(), ArrayRef<Value>());

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
    oldValue.replaceUsesWithIf(newValue, [&](OpOperand &operand) {
      return traverseParentBlocks(operand.getOwner());
    });
  }
}

LogicalResult createTtgWSOp(Location loc, OpBuilder &builder,
                            SmallVector<Block *> partitions,
                            SmallVector<int> numWarps,
                            SmallVector<int> warpGroupStartIds,
                            SmallVector<int> barIds) {

  // This methods assumes warp specialized partitions have already been added
  // to a block such as the then block of an if statement
  if (partitions.size() != numWarps.size())
    return failure("mismatched number of warp groups and number of warps per "
                   "warp group");
  if (partitions.size() != barIds.size())
    return failure("mismatched number of warp groups and number of warps per "
                   "warp group");
  if (partitions.size() != warpGroupStartIds.size())
    return failure(
        "mismatched number of warp groups and number of warp start ids");

  auto defaultPartition = partitions.front();
  partitions.erase(partitions.begin());

  if (warpGroupStartIds.front() != 0)
    return failure("warp group start ids must start at 0");
  warpGroupStartIds.erase(warpGroupStartIds.begin());

  // get ttg.num-warps attribute from module
  auto op = builder.getInsertionBlock()->getParentOp();
  auto modNumWarps = builder.getInsertionBlock()
                         ->getParentOp()
                         ->getParentOfType<ModuleOp>()
                         ->getAttrOfType<IntegerAttr>("ttg.num-warps")
                         .getInt();
  if (numWarps.front() != modNumWarps)
    return failure("ttg.num-warps attribute does not match the number of warps "
                   "in the first warp group");
  numWarps.erase(numWarps.begin());

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

  auto wsOp = builder.create<WarpSpecializeOp>(loc, TypeRange(), inputs);
  // convert barIds to attribute list and set on wsOp

  auto barIdType = RankedTensorType::get({static_cast<int64_t>(barIds.size())},
                                         builder.getIntegerType(32));
  auto barIdAttr = DenseIntElementsAttr::get(barIdType, barIds);
  wsOp->setAttr("barIds", barIdAttr);

  wsOp.setPartitionNumWarps(numWarps);
  wsOp.setWarpGroupStartIds(warpGroupStartIds);

  // TODO: Handle returns from the default block
  auto &defaultBlock = wsOp.getDefaultRegion().emplaceBlock();
  builder.setInsertionPointToEnd(&defaultBlock);
  auto yieldOp =
      builder.create<WarpYieldOp>(loc, TypeRange(), ArrayRef<Value>());
  for (auto &op : llvm::make_early_inc_range(llvm::make_range(
           defaultPartition->getOperations().begin(),
           defaultPartition->getTerminator()->getIterator()))) {
    op.moveBefore(yieldOp);
  }

  auto &block = wsOp.getPartitionOpHolder().emplaceBlock();
  builder.setInsertionPointToStart(&block);
  auto wspOp =
      builder.create<WarpSpecializePartitionsOp>(loc, partitions.size());
  auto regions = wspOp.getPartitionRegions();
  for (size_t i = 0, e = regions.size(); i < e; ++i) {
    auto &block = regions[i].emplaceBlock();
    populateBlock(builder, partitions[i], &block, inputs, constants);
  }
  return success();
}

class TritonNvidiaGPUTtngWgToTtgWs
    : public TritonNvidiaGPUTtngWgToTtgWsBase<TritonNvidiaGPUTtngWgToTtgWs> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::ModuleOp m = getOperation();

    SmallVector<ttng::WarpGroupOp> wgOps = findWarpGroupOps(m);

    DenseMap<int32_t, int32_t> warpGroups;
    DenseMap<int32_t, ttng::WarpGroupOp> wgOpsMap;
    uint32_t totalNumWarps = 0;
    for (auto wgOp : wgOps) {
      auto startWarp = wgOp.getStartWarp();
      auto numWarps = wgOp.getNumWarps();
      assert(warpGroups.count(startWarp) == 0 &&
             "multiple ttng.warp_group with same startWarp");
      warpGroups[startWarp] = numWarps;
      wgOpsMap[startWarp] = wgOp;
      totalNumWarps = std::max(totalNumWarps, startWarp + numWarps);
    }

    // we can only convert to ttg.warp_specialized if there is a
    // ttng.warp_group with startWarp = 0 and numWarps = ttg.num-warps.
    assert(warpGroups.find(0) != warpGroups.end());
    assert(warpGroups[0] ==
               m->getAttrOfType<IntegerAttr>("ttg.num-warps").getInt() &&
           "ttng.warp_group with startWarp = 0 and numWarps != ttg.num-warps");

    SmallVector<int> warpGroupStartIds;
    SmallVector<int> numWarps;
    for (auto [startWarp, _numWarps] : warpGroups) {
      warpGroupStartIds.push_back(startWarp);
      numWarps.push_back(_numWarps);
      // ttg.warp_specialize does not allow gaps between warp groups, so
      // we would need to fill any gaps with dummy partitions. For now, we
      // assert that there are no gaps.
      if (startWarp + _numWarps < totalNumWarps)
        assert(warpGroups.find(startWarp + _numWarps) != warpGroups.end());
    }

    SmallVector<Block *> partitions;
    SmallVector<int> barIds;
    for (auto [startWarp, wgOp] : wgOpsMap) {
      auto wgRegion = &wgOp.getPartitionRegions()[0].front();
      partitions.push_back(wgRegion);
      auto barId = triton::gpu::getBarrierID(wgOp);
      barIds.push_back(barId);
    }
    OpBuilder builder(wgOps.back());
    builder.setInsertionPointAfter(wgOps.back());
    auto result = createTtgWSOp(wgOps.back().getLoc(), builder, partitions,
                                numWarps, warpGroupStartIds, barIds);

    if (failed(result)) {
      signalPassFailure();
      return;
    }

    for (auto wgOp : wgOps) {
      wgOp.erase();
    }

    constexpr static char AttrWarpSpecializedName[] = "ttg.warp-specialized";
    Attribute boolAttribute = BoolAttr::get(context, false);
    m->setAttr(AttrWarpSpecializedName, boolAttribute);
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createTritonNvidiaGPUTtngWgToTtgWsPass() {
  return std::make_unique<TritonNvidiaGPUTtngWgToTtgWs>();
}
