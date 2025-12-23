#include "triton/Analysis/Membar.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include <deque>

namespace mlir {

AllocationSlice::AllocationSlice(Value value,
                                 Interval<size_t> allocationInterval)
    : allocationInterval(allocationInterval) {
  auto accessTy = cast<triton::gpu::MemDescType>(value.getType());
  this->accessTy = accessTy;

  // Get the memdesc_subslice information if present. If no subslice is
  // present the whole interval is accessed
  if (auto subslice = value.getDefiningOp<triton::gpu::MemDescSubsliceOp>()) {
    // We know there aren't subslices before the one because of subslice::fold
    // Still need to check this for where a fold isn't possible (control flow)
    // and when a subslice is carried in a loop
    if (accessTy.getAllocShape() == subslice.getSrc().getType().getShape()) {
      subsliceOffsets = SmallVector<int64_t>(subslice.getOffsets());
    }
  }
}

bool AllocationSlice::intersects(const AllocationSlice &other) const {
  // Disjoint intervals don't overlap
  if (!allocationInterval.intersects(other.allocationInterval))
    return false;

  // If access types are unknown, assume intersection
  if (!accessTy || !other.accessTy)
    return true;

  // If offsets are unknown, conservatively assume overlap
  if (subsliceOffsets.empty() || other.subsliceOffsets.empty())
    return true;

  // If layouts differ, we assume intersection as we currently only work on
  // logical elements
  if (accessTy.getEncoding() != other.accessTy.getEncoding())
    return true;

  auto shapeA = SmallVector<int64_t>(accessTy.getShape());
  auto shapeB = SmallVector<int64_t>(other.accessTy.getShape());
  // Chek if all subslice region dimensions have some intersection
  // [offsetA, offsetA + shape) and [offsetB, offsetB + other.shape)
  // If any dimension doesn't intersect, we are looking at disjoint subslices
  for (size_t i = 0; i < subsliceOffsets.size(); ++i) {
    int64_t startA = subsliceOffsets[i];
    int64_t endA = startA + shapeA[i];
    int64_t startB = other.subsliceOffsets[i];
    int64_t endB = startB + shapeB[i];

    // Is A completely before B? Is B completely before A? If so, disjoint
    if (endA <= startB || endB <= startA)
      return false;
  }

  // All dimensions of subslices have some intersection
  return true;
}

void AllocationSlice::print(raw_ostream &os) const {
  os << "interval=[" << allocationInterval.start() << ","
     << allocationInterval.end() << ")";

  os << " offsets=[";
  if (!subsliceOffsets.empty()) {
    llvm::interleaveComma(subsliceOffsets, os);
  } else {
    os << "unknown";
  }
  os << "]";

  os << " shape=";
  if (accessTy) {
    llvm::interleave(accessTy.getShape(), os, "x");
    os << " layout=" << accessTy.getEncoding();
  } else {
    os << "? layout=unknown";
  }
}

void MembarOrFenceAnalysis::run(FuncBlockInfoMapT &funcBlockInfoMap) {
  FunctionOpInterface funcOp =
      dyn_cast<FunctionOpInterface>(allocation->getOperation());
  OpBuilder builder(funcOp.getContext());
  resolve(funcOp, &funcBlockInfoMap, &builder);
}

void MembarOrFenceAnalysis::resolve(FunctionOpInterface funcOp,
                                    FuncBlockInfoMapT *funcBlockInfoMap,
                                    OpBuilder *builder) {
  // Initialize the blockList. Operations are organized into "virtual blocks",
  // which represent segments of straight-line code analyzed by each iteration
  // of the dataflow analysis. Virtual blocks abstract over both control flow
  // represented by basic blocks and block successors (i.e. `BranchOpInterface`)
  // and control flow represented by regions (i.e. `RegionBranchOpInterface`).
  //
  // A virtual block consists of a parent block and a starting iterator, where
  // the virtual block starts on the operation *after* the starting iterator. A
  // null iterator is used to represent the beginning of the block. The virtual
  // block ends at any region branch operation or the basic block terminator.
  // Thus, basic blocks are broken up into multiple virtual blocks at each
  // region operation.
  //
  // Entry virtual blocks are represented by a null iterator. Populate the
  // blockList with the entry virtual blocks in the function. Then, each
  // iteration scans until a terminator or region branch operation is found.
  DenseMap<VirtualBlock, BlockInfo> inputBlockInfoMap;
  DenseMap<VirtualBlock, BlockInfo> outputBlockInfoMap;
  std::deque<VirtualBlock> blockList;
  // Start the analysis from the entry block of the function.
  blockList.emplace_back(&funcOp.getBlocks().front(), Block::iterator());

  // A fixed point algorithm
  while (!blockList.empty()) {
    VirtualBlock block = blockList.front();
    blockList.pop_front();
    // Make a copy of the inputblockInfo but not update
    auto inputBlockInfo = inputBlockInfoMap[block];
    SmallVector<VirtualBlock> successors;
    Block::iterator startIt =
        block.second.isValid() ? std::next(block.second) : block.first->begin();
    for (Operation &op : llvm::make_range(startIt, block.first->end())) {
      // Update inputBlockInfo based on the current operation. Note that we do
      // this before we process terminators and branch-like ops, because some of
      // them (e.g. WarpSpecializePartitionsOp) may have synchronizing effects.
      update(&op, &inputBlockInfo, funcBlockInfoMap, builder);
      if (op.hasTrait<OpTrait::IsTerminator>() ||
          isa<RegionBranchOpInterface>(op)) {
        visitTerminator(&op, successors);
        break;
      }
    }
    // Get the reference because we want to update if it changed
    if (outputBlockInfoMap.count(block) &&
        inputBlockInfo == outputBlockInfoMap[block]) {
      // If we have seen the block before and the inputBlockInfo is the same as
      // the outputBlockInfo, we skip the successors
      continue;
    }
    // Update the current block. The block transfer function is not monotonic,
    // so overwrite the output state entirely.
    outputBlockInfoMap[block] = inputBlockInfo;
    // Update the successors
    for (VirtualBlock successor : successors) {
      inputBlockInfoMap[successor].join(outputBlockInfoMap[block]);
      blockList.emplace_back(successor);
    }
  }

  // Update the final dangling buffers that haven't been synced
  BlockInfo &funcBlockInfo = (*funcBlockInfoMap)[funcOp];
  funcOp.walk<WalkOrder::PreOrder>([&](triton::ReturnOp returnOp) {
    // A basic block can be broken into several virtual blocks. Find all virtual
    // blocks that belong to the basic block containing the return.
    SmallVector<std::pair<VirtualBlock, BlockInfo>> virtualBlocks;
    for (auto &[block, blockInfo] : outputBlockInfoMap) {
      if (block.first == returnOp->getBlock())
        virtualBlocks.emplace_back(block, blockInfo);
    }
    // The return is a terminator, so the virtual block that contains this
    // return starts after all other ones. Find it by comparing the start
    // iterators of the virtual blocks.
    auto maxIt = llvm::max_element(virtualBlocks, [&](auto &lhs, auto &rhs) {
      assert(lhs.first.first == rhs.first.first);
      Block::iterator lhsIt = lhs.first.second, rhsIt = rhs.first.second;
      return !lhsIt.isValid() ||
             (rhsIt.isValid() && lhsIt->isBeforeInBlock(&*rhsIt));
    });

    funcBlockInfo.join(maxIt->second);
  });
}

void MembarOrFenceAnalysis::visitTerminator(
    Operation *op, SmallVector<VirtualBlock> &successors) {
  if (isa<BranchOpInterface>(op)) {
    // Collect the block successors of the branch.
    for (Block *successor : op->getSuccessors())
      successors.emplace_back(successor, Block::iterator());
    return;
  }

  if (auto br = dyn_cast<RegionBranchOpInterface>(op)) {
    // The successors of an operation with regions can be queried via an
    // interface. The operation branches to the entry blocks of its region
    // successors. It can also branch to after itself.
    SmallVector<RegionSuccessor> regions;
    br.getSuccessorRegions(RegionBranchPoint::parent(), regions);
    for (RegionSuccessor &region : regions) {
      if (region.isParent()) {
        successors.emplace_back(br->getBlock(), br->getIterator());
      } else {
        Block &block = region.getSuccessor()->front();
        successors.emplace_back(&block, Block::iterator());
      }
    }
    return;
  }

  // FIXME: `ReturnLike` adds `RegionBranchTerminatorOpInterface` for some
  // reason. Check that the parent is actually a `RegionBranchOpInterface`.
  auto br = dyn_cast<RegionBranchTerminatorOpInterface>(op);
  if (br && isa<RegionBranchOpInterface>(br->getParentOp())) {
    // Check the successors of a region branch terminator. It can branch to
    // another region of its parent operation or to after the parent op.
    SmallVector<Attribute> operands(br->getNumOperands());
    SmallVector<RegionSuccessor> regions;
    br.getSuccessorRegions(operands, regions);
    for (RegionSuccessor &region : regions) {
      if (region.isParent()) {
        Operation *parent = br->getParentOp();
        successors.emplace_back(parent->getBlock(), parent->getIterator());
      } else {
        Block &block = region.getSuccessor()->front();
        successors.emplace_back(&block, Block::iterator());
      }
    }
    return;
  }

  // Otherwise, it could be a return op
  if (op->hasTrait<OpTrait::ReturnLike>())
    return;
  llvm_unreachable("Unknown terminator encountered in membar analysis");
}

void MembarAnalysis::insertBarrier(Operation *op, OpBuilder *builder) {
  OpBuilder::InsertionGuard g(*builder);
  auto barrierOp = triton::gpu::LocalBarrierOp::create(*builder, op->getLoc());
}

void MembarAnalysis::update(Operation *op, BlockInfo *blockInfo,
                            FuncBlockInfoMapT *funcBlockInfoMap,
                            OpBuilder *builder) {
  if (isa<gpu::BarrierOp, triton::gpu::LocalBarrierOp,
          triton::gpu::WarpSpecializePartitionsOp>(op)) {
    // If the current op is a barrier, we sync previous reads and writes
    blockInfo->sync();
    return;
  }

  if (op->hasTrait<mlir::OpTrait::MemWaitOpTrait>() &&
      !isa<gpu::BarrierOp, triton::gpu::LocalBarrierOp>(op->getNextNode())) {
    // If the current op is an async wait and the next op is not a barrier we
    // insert a barrier op and sync
    builder->setInsertionPointAfter(op);
    insertBarrier(op, builder);
    blockInfo->sync();
    return;
  }

  BlockInfo curBlockInfo;
  auto scratchBufferId = Allocation::InvalidBufferId;
  if (isa<triton::CallOp>(op)) {
    // Inter-function dependencies
    auto callOpInterface = dyn_cast<CallOpInterface>(op);
    if (auto callee =
            dyn_cast<FunctionOpInterface>(callOpInterface.resolveCallable()))
      curBlockInfo = funcBlockInfoMap->lookup(callee);
  } else {
    // Intra-function dependencies
    if (auto memoryEffectOpInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
      // Explicit buffer
      SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>>
          effectInstances;
      memoryEffectOpInterface.getEffects(effectInstances);
      for (auto effectInstance : effectInstances) {
        if (auto value = effectInstance.getValue()) {
          for (auto bufferId : allocation->getBufferIds(value)) {
            if (bufferId != Allocation::InvalidBufferId) {
              auto interval = allocation->getAllocatedInterval(bufferId);
              auto slice = AllocationSlice(value, interval);

              if (isa<MemoryEffects::Write>(effectInstance.getEffect()))
                curBlockInfo.syncWriteSlices[slice].insert(op);
              else if (isa<MemoryEffects::Read>(effectInstance.getEffect()))
                curBlockInfo.syncReadSlices[slice].insert(op);
            }
          }
        }
      }
    }
    // If this op is may be signalling other threads asynchronously, make sure
    // all shared memory transactions are complete beforehand.
    if (isa<triton::nvidia_gpu::ArriveBarrierOp>(op)) {
      Interval<size_t> allIntervals(0, std::numeric_limits<size_t>::max());
      auto allMemorySlice = AllocationSlice(allIntervals);
      curBlockInfo.syncWriteSlices[allMemorySlice].insert(op);
      curBlockInfo.syncReadSlices[allMemorySlice].insert(op);
    }
    scratchBufferId = allocation->getBufferId(op);
  }

  // Scratch buffer operations consist of a series of shared memory operations
  // starting from a shared memory write, followed by a series of shared memory
  // read/write operations, and ending with a shared memory read, i.e., shared
  // memory write -> ... -> shared memory read.
  if (scratchBufferId != Allocation::InvalidBufferId) {
    // Detect warp-synchronous convert-layout operations. These emit a
    // warp-level barrier (warp.sync) rather than a CTA-wide barrier between
    // the internal shared-memory write and read phases. For these ops, we must
    // not globally clear pending dependencies.
    bool isWarpSync = false;
    if (auto cvt = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
      auto srcTy = cast<RankedTensorType>(cvt.getSrc().getType());
      auto dstTy = cast<RankedTensorType>(cvt.getType());
      auto srcLayout = triton::gpu::toLinearLayout(srcTy);
      auto dstLayout = triton::gpu::toLinearLayout(dstTy);
      isWarpSync = mlir::isCvtWarpSync(srcLayout, dstLayout);
    }

    if (!curBlockInfo.syncReadSlices.empty() ||
        !curBlockInfo.syncWriteSlices.empty()) {
      llvm::report_fatal_error(
          "scratch buffer operations should not have any shared memory "
          "dependencies");
    }
    auto interval = allocation->getAllocatedInterval(scratchBufferId);
    auto scratchSlice = AllocationSlice(interval);
    curBlockInfo.syncWriteSlices[scratchSlice].insert(op);
    auto insertCTABarrier = blockInfo->isIntersected(curBlockInfo, filter);
    if (insertCTABarrier) {
      builder->setInsertionPoint(op);
      insertBarrier(op, builder);
    }
    // Ops with a scratch buffer that don't use warp.sync internally sync
    // read/write on shared memory
    if (insertCTABarrier || !isWarpSync)
      blockInfo->sync();
    curBlockInfo.syncReadSlices[scratchSlice].insert(op);
  } else if (blockInfo->isIntersected(curBlockInfo, filter)) {
    builder->setInsertionPoint(op);
    insertBarrier(op, builder);
    blockInfo->sync();
  }
  // Update the region info, even if barrier is inserted, we have to maintain
  // the current op's read/write buffers.
  blockInfo->join(curBlockInfo);
}
} // namespace mlir
