#include "triton/Analysis/Membar.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <deque>

#define DEBUG_TYPE "tritongpu-membar-analysis"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace ttg = mlir::triton::gpu;

namespace mlir {

AllocationSlice::AllocationSlice(ttg::MemDescType allocTy,
                                 Interval<size_t> allocationInterval)
    : allocationInterval(allocationInterval), allocTy(allocTy),
      subsliceOffsets(allocTy.getShape().size(), 0),
      subsliceShape(allocTy.getShape()) {}

AllocationSlice::AllocationSlice(ttg::MemDescType allocTy,
                                 Interval<size_t> allocationInterval,
                                 ArrayRef<int64_t> curShape)
    : AllocationSlice(allocTy, allocationInterval) {
  const auto shapeDiff = subsliceShape.size() - curShape.size();
  for (auto i : llvm::seq(shapeDiff)) {
    subsliceOffsets[i] = OffsetValue{};
    subsliceShape[i] = 1;
  }
  for (auto i : llvm::seq(curShape.size())) {
    if (subsliceShape[i + shapeDiff] != curShape[i]) {
      // If the value is already a subslice, there is some unknown offset
      subsliceOffsets[i] = OffsetValue{};
    }
    subsliceShape[i + shapeDiff] = curShape[i];
  }
}

AllocationSlice AllocationSlice::subslice(ArrayRef<int32_t> offsets,
                                          ArrayRef<int64_t> resShape) const {
  auto newSlice = *this;
  if (subsliceShape.empty() || subsliceOffsets.empty())
    return newSlice;

  assert(resShape.size() <= subsliceShape.size());
  const auto shapeDiff = subsliceShape.size() - resShape.size();
  for (auto i : llvm::seq(shapeDiff))
    newSlice.subsliceShape[i] = 1;
  for (auto i : llvm::seq(resShape.size())) {
    newSlice.subsliceShape[i + shapeDiff] = resShape[i];
  }

  assert(offsets.size() <= subsliceShape.size());
  auto offsetDiff = subsliceShape.size() - offsets.size();
  for (auto i : llvm::seq(offsets.size())) {
    newSlice.subsliceOffsets[i + offsetDiff] += offsets[i];
  }
  return newSlice;
}

AllocationSlice AllocationSlice::index(Value indexVal,
                                       ArrayRef<int64_t> resShape) const {
  auto newSlice = *this;
  if (subsliceShape.empty() || subsliceOffsets.empty())
    return newSlice;

  assert(resShape.size() < subsliceShape.size());
  auto indexDim = subsliceShape.size() - resShape.size() - 1;
  for (auto i : llvm::seq(indexDim))
    assert(newSlice.subsliceShape[i] == 1);
  newSlice.subsliceShape[indexDim] = 1;
  for (auto i : llvm::seq(resShape.size()))
    assert(newSlice.subsliceShape[indexDim + 1 + i] == resShape[i]);

  APInt index;
  if (matchPattern(indexVal, m_ConstantInt(&index))) {
    newSlice.subsliceOffsets[indexDim] += index.getSExtValue();
  } else {
    // Reset to unknown
    newSlice.subsliceOffsets[indexDim] = OffsetValue{};
  }
  return newSlice;
}

bool memDescAllocTypeMatches(Type allocType, Type otherType) {
  auto descA = dyn_cast<ttg::MemDescType>(allocType);
  auto descB = dyn_cast<ttg::MemDescType>(otherType);
  if (!descA || !descB)
    return false;

  auto otherNdim = descB.getAllocShape().size();
  auto allocShape = descA.getShape().take_back(otherNdim);
  return allocShape == descB.getAllocShape() &&
         descA.getElementType() == descB.getElementType() &&
         descA.getEncoding() == descB.getEncoding();
}

bool AllocationSlice::intersects(const AllocationSlice &other) const {
  // Disjoint intervals don't overlap
  if (!allocationInterval.intersects(other.allocationInterval))
    return false;

  // If access types are unknown, assume intersection
  if (!allocTy || !other.allocTy)
    return true;

  // If offsets are unknown, conservatively assume overlap
  if (subsliceOffsets.empty() || other.subsliceOffsets.empty())
    return true;
  if (subsliceShape.empty() || other.subsliceShape.empty())
    return true;

  // If types differ, we assume intersection as we currently only work on
  // logical elements
  if (!memDescAllocTypeMatches(allocTy, other.allocTy))
    return true;

  const auto &shapeA = subsliceShape;
  const auto &shapeB = other.subsliceShape;
  // Check if all subslice region dimensions have some intersection
  // [offsetA, offsetA + shape) and [offsetB, offsetB + other.shape)
  // If any dimension doesn't intersect, we are looking at disjoint subslices
  for (size_t i = 0; i < subsliceOffsets.size(); ++i) {
    auto startA = subsliceOffsets[i];
    auto endA = startA + shapeA[i];
    auto startB = other.subsliceOffsets[i];
    auto endB = startB + shapeB[i];

    // Is A completely before B? Is B completely before A? If so, disjoint
    if (endA.known_leq(startB) || endB.known_leq(startA))
      return false;
  }

  // All dimensions of subslices have some intersection
  return true;
}

void AllocationSlice::OffsetValue::print(raw_ostream &os) const {
  if (isKnown())
    os << offset_;
  else
    os << "unknown";
}

void AllocationSlice::print(raw_ostream &os) const {
  os << "interval=[" << allocationInterval.start() << ","
     << allocationInterval.end() << ")";

  if (subsliceOffsets.empty()) {
    os << " offsets=unknown";
  } else {
    os << " offsets=[";
    llvm::interleaveComma(subsliceOffsets, os);
    os << "]";
  }
  if (subsliceShape.empty()) {
    os << " subsliceShape=unknown";
  } else {
    os << " subsliceShape=[";
    llvm::interleaveComma(subsliceShape, os);
    os << "]";
  }
  if (!allocTy) {
    os << " allocTy=unknown";
  } else {
    os << " allocTy=" << allocTy;
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
  triton::gpu::BarrierOp::create(*builder, op->getLoc(),
                                 triton::gpu::AddrSpace::Local);
}

void AllocationSliceAnalysis::update(Operation *op) {
  if (isa<ttg::LocalAllocOp>(op)) {
    getAllocationSlices(op->getResult(0));
    return;
  }

  if (!isa<ttg::MemDescSubsliceOp, ttg::MemDescIndexOp>(op))
    return;
  auto src = op->getOperand(0);
  std::vector<AllocationSlice> slices;
  for (const auto &parentSlice : getAllocationSlices(src)) {
    if (auto sliceOp = dyn_cast<ttg::MemDescSubsliceOp>(op)) {
      auto slice = parentSlice.subslice(sliceOp.getOffsets(),
                                        sliceOp.getType().getShape());
      slices.push_back(std::move(slice));
    } else {
      auto indexOp = cast<ttg::MemDescIndexOp>(op);
      auto slice =
          parentSlice.index(indexOp.getIndex(), indexOp.getType().getShape());
      slices.push_back(std::move(slice));
    }
  }
  sliceMap.emplace(op->getResult(0), std::move(slices));
}

const std::vector<AllocationSlice> &
AllocationSliceAnalysis::getAllocationSlices(Value value) {
  auto it = sliceMap.find(value);
  if (it != sliceMap.end()) {
    return it->second;
  }

  std::vector<AllocationSlice> slices;
  if (auto alloc = value.getDefiningOp<ttg::LocalAllocOp>()) {
    // new allocation
    auto bufferId = allocation_->getBufferId(value);
    allocTypeMap[bufferId] = alloc.getType();
    assert(bufferId != Allocation::InvalidBufferId);
    auto interval = allocation_->getAllocatedInterval(bufferId);
    slices.emplace_back(alloc.getType(), interval);
  } else {
    // unknown buffer, potentially aliased
    for (auto bufferId : allocation_->getBufferIds(value)) {
      if (bufferId == Allocation::InvalidBufferId)
        continue;
      auto interval = allocation_->getAllocatedInterval(bufferId);
      auto typeIt = allocTypeMap.find(bufferId);
      if (typeIt != allocTypeMap.end()) {
        auto bufferType = typeIt->second;
        if (memDescAllocTypeMatches(bufferType, value.getType())) {
          slices.emplace_back(
              bufferType, interval,
              cast<ttg::MemDescType>(value.getType()).getShape());
          continue;
        }
      }
      slices.emplace_back(interval);
    }
  }
  return sliceMap.emplace(value, std::move(slices)).first->second;
}

void MembarAnalysis::update(Operation *op, BlockInfo *blockInfo,
                            FuncBlockInfoMapT *funcBlockInfoMap,
                            OpBuilder *builder) {
  sliceAnalysis.update(op);

  auto containsLocalBarrier = [](Operation *op) {
    if (isa<gpu::BarrierOp>(op))
      return true;
    if (isa<triton::gpu::WarpSpecializePartitionsOp>(op))
      return true;
    if (auto barrier = dyn_cast<triton::gpu::BarrierOp>(op))
      return barrier.hasLocal();
    return false;
  };

  if (containsLocalBarrier(op)) {
    // If the current op is a local barrier, we sync previous reads and writes
    blockInfo->sync();
    return;
  }

  if (op->hasTrait<mlir::OpTrait::MemWaitOpTrait>() &&
      !containsLocalBarrier(op->getNextNode())) {
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
          for (auto slice : sliceAnalysis.getAllocationSlices(value)) {
            if (isa<MemoryEffects::Write>(effectInstance.getEffect()))
              curBlockInfo.syncWriteSlices[slice].insert(op);
            else if (isa<MemoryEffects::Read>(effectInstance.getEffect()))
              curBlockInfo.syncReadSlices[slice].insert(op);
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
    LLVM_DEBUG({
      DBGS()
          << "Inserting barrier due to memory region overlap in operation:\n";
      op->dump();
      llvm::dbgs() << "Previous Block info =\n";
      blockInfo->dump();
      llvm::dbgs() << "Operation Block info =\n";
      curBlockInfo.dump();
    });

    builder->setInsertionPoint(op);
    insertBarrier(op, builder);
    blockInfo->sync();
  }
  // Update the region info, even if barrier is inserted, we have to maintain
  // the current op's read/write buffers.
  blockInfo->join(curBlockInfo);
}
} // namespace mlir
