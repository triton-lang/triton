#include "triton/Analysis/Membar.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include <deque>

namespace mlir {

namespace {

using IntervalVector = SmallVector<Interval<size_t>>;

// Computes strides for `ty`, using allocShape if available.
// Returns a vector with strides[i] is the stride in elements for dimension i.
// Example: 4x32x64 will have strides [2048, 64, 1]
SmallVector<int64_t> computeStrides(triton::gpu::MemDescType ty) {
  auto shape = ty.getShape();
  auto allocShape = ty.getAllocShape();
  auto strideShape =
      allocShape.empty() ? shape : allocShape.take_back(shape.size());

  SmallVector<int64_t> strides(strideShape.size());
  int64_t stride = 1;
  for (int64_t i = strideShape.size() - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= strideShape[i];
  }
  return strides;
}

// Generates intervals for a given buffer type.
// For a tensor of shape AxBxCxD, generates A*B*C intervals of size D
// All interval offsets are generated assuming unswizzled row major layout
IntervalVector generateIntervalsForType(triton::gpu::MemDescType srcTy,
                                        int64_t baseElementOffset) {
  IntervalVector intervals;
  auto shape = srcTy.getShape();
  auto bytesPerElement = srcTy.getElementTypeBitWidth() / 8;

  // The total number of intervals is the product of all dimensions except
  // the last one. This is because the last dimension is the size of each
  // interval.
  int64_t numIntervals = 1;
  for (int64_t i = 0; i < shape.size() - 1; ++i) {
    numIntervals *= shape[i];
  }
  int64_t intervalLength = shape[shape.size() - 1];

  auto strides = computeStrides(srcTy);

  // Generate intervals by converting interval indices to memory offsets.
  // Each intervalIdx is mapped into multi-dimensional coordinates, then
  // converted to a linear memory offset using the tensor's strides.
  // Example:
  //   With intervalIdx=171 for shape [2, 16, 8, 32] -> coords [1, 5, 3]:
  //   - flatIdx = intervalIdx = 171
  //   - coord[2] = 171 % 8 = 3   (4th element in dim 2)
  //     flatIdx = 171 / 8 = 21
  //   - coord[1] = 21 % 16 = 5   (6th element in dim 1)
  //     flatIdx = 21 / 16 = 1
  //   - coord[0] = 1 % 2 = 1     (2nd element in dim 0)
  //     flatIdx = 1 / 2 = 0
  //   Moving to offset (in elements) calculation, we do the following:
  //   - offset = 1*stride[0] + 5*stride[1] + 3*stride[2]
  //            = 1*4096 + 5*256 + 3*32 = 5472
  //   So intervalIdx=171 will be interval [5472, 5504)
  for (int64_t intervalIdx = 0; intervalIdx < numIntervals; ++intervalIdx) {
    // Extract coordinates from the interval index
    SmallVector<int64_t> coords(shape.size() - 1);
    int64_t flatIdx = intervalIdx;
    for (int64_t dim = shape.size() - 2; dim >= 0; --dim) {
      coords[dim] = flatIdx % shape[dim];
      flatIdx /= shape[dim];
    }

    // Calculate the interval element offset using coordinates and strides
    int64_t intervalElementOffset = baseElementOffset;
    for (int64_t dim = 0; dim < shape.size() - 1; ++dim) {
      intervalElementOffset += coords[dim] * strides[dim];
    }

    // Convert element offset to byte offset and create the interval
    size_t intervalStart = intervalElementOffset * bytesPerElement;
    size_t intervalEnd = intervalStart + intervalLength * bytesPerElement;
    intervals.push_back(Interval<size_t>(intervalStart, intervalEnd));
  }
  return intervals;
}

// Processes all view ops top down to compute the memory intervals.
// Starts with element based offsets and returns byte intervals
// The intervals are generated using the shape and data type of `leafTy`
IntervalVector applyViewOperations(int64_t baseElementOffset,
                                   triton::gpu::MemDescType leafTy,
                                   ArrayRef<Operation *> viewOps) {
  if (viewOps.empty()) {
    return generateIntervalsForType(leafTy, baseElementOffset);
  }

  Operation *op = viewOps[viewOps.size() - 1];
  ArrayRef<Operation *> remainingOps = viewOps.drop_back();

  // Each view operation just changes the starting offset of the buffer that
  // we will be read/write to. The intervals generated only depend on the
  // `leafTy`. This still holds when dealing with dynamic index, except in
  // that case we recurse multiple times one for each of the possible indices.
  // The processing for the various cases is:
  // - MemdescSubslice: adjust starting element offset based on all strides
  //                    and process next view operation
  // - MemdescIndex (constant): adjust the starting element offset based on
  //                            the dim0 stride and process next view op
  // - MemdescIndex (dynamic): for each possible index of dim0, adjust the
  //                           starting element offset based on dim0 stride
  //                           and process the next view op
  if (auto subsliceOp = dyn_cast<triton::gpu::MemDescSubsliceOp>(op)) {
    auto srcTy = subsliceOp.getSrc().getType();
    auto strides = computeStrides(srcTy);
    auto offsets = subsliceOp.getOffsets();
    assert(offsets.size() == strides.size());
    int64_t elementOffset = 0;
    for (size_t i = 0; i < offsets.size(); ++i) {
      elementOffset += offsets[i] * strides[i];
    }
    return applyViewOperations(baseElementOffset + elementOffset, leafTy,
                               remainingOps);
  } else if (auto indexOp = dyn_cast<triton::gpu::MemDescIndexOp>(op)) {
    auto srcTy = indexOp.getSrc().getType();
    int64_t elementStride = computeStrides(srcTy)[0];

    if (auto constantIndex =
            indexOp.getIndex().getDefiningOp<arith::ConstantIntOp>()) {
      // Apply constant index offset
      int64_t elementOffset = constantIndex.value() * elementStride;
      return applyViewOperations(baseElementOffset + elementOffset, leafTy,
                                 remainingOps);
    } else {
      // This can be made smarter in the future by looking at loop iterations
      // and if we alternate index then we can reduce the analysis to just
      // one of the slices
      IntervalVector intervals;
      int64_t dim0Size = srcTy.getDimSize(0);
      // Dynamic index: recursively handle all possible values of dim0
      for (int64_t i = 0; i < dim0Size; ++i) {
        int64_t elementOffset = i * elementStride;
        llvm::append_range(
            intervals, applyViewOperations(baseElementOffset + elementOffset,
                                           leafTy, remainingOps));
      }
      return intervals;
    }
  }

  llvm_unreachable("Unexpected operation type in view chain");
}

// Computes all memory intervals touched by a memory operation.
//
// Computes the portions of shared memory that a memory operation accesses.
// Returns a vector of intervals with byte offsets.
IntervalVector getFineGrainedIntervals(Interval<size_t> baseInterval,
                                       const Value value) {
  // Get all view ops between the memory op and the allocation
  SmallVector<Operation *> viewOps;

  Value currentValue = value;
  while (auto op = currentValue.getDefiningOp()) {
    if (auto subsliceOp = dyn_cast<triton::gpu::MemDescSubsliceOp>(op)) {
      viewOps.push_back(subsliceOp);
      currentValue = subsliceOp.getSrc();
    } else if (auto indexOp = dyn_cast<triton::gpu::MemDescIndexOp>(op)) {
      viewOps.push_back(indexOp);
      currentValue = indexOp.getSrc();
    } else {
      // Start of the chain of ops (local_alloc/function parameter/...)
      break;
    }
  }

  auto ty = cast<triton::gpu::MemDescType>(value.getType());
  auto bytesPerElement = ty.getElementTypeBitWidth() / 8;
  auto baseElementOffset = baseInterval.start() / bytesPerElement;
  auto intervals = applyViewOperations(baseElementOffset, ty, viewOps);

  // Check maths, all intervals need to be within the original allocation
  for (auto interval : intervals) {
    assert(interval.start() >= baseInterval.start() &&
           interval.end() <= baseInterval.end());
  }

  return intervals;
}

} // namespace

template <bool PrintIntervals>
void MembarOrFenceAnalysisImpl<PrintIntervals>::run(
    FuncBlockInfoMapT &funcBlockInfoMap) {
  FunctionOpInterface funcOp =
      dyn_cast<FunctionOpInterface>(this->allocation->getOperation());
  if constexpr (PrintIntervals) {
    llvm::errs() << "Function: " << funcOp.getName() << "\n";
  }
  OpBuilder builder(funcOp.getContext());
  resolve(funcOp, &funcBlockInfoMap, &builder);
}

template <bool PrintIntervals>
void MembarOrFenceAnalysisImpl<PrintIntervals>::resolve(
    FunctionOpInterface funcOp, FuncBlockInfoMapT *funcBlockInfoMap,
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
  funcOp.walk<WalkOrder::PreOrder>([&](Block *block) {
    // Start the analysis from the entry blocks of any nested isolated from
    // above regions.
    if (block->isEntryBlock() &&
        !isa<RegionBranchOpInterface>(block->getParentOp()))
      blockList.emplace_back(block, Block::iterator());
  });

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
      if (op.hasTrait<OpTrait::IsTerminator>() ||
          isa<RegionBranchOpInterface>(op)) {
        visitTerminator(&op, successors);
        break;
      }
      update(&op, &inputBlockInfo, funcBlockInfoMap, builder);
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

template <bool PrintIntervals>
void MembarOrFenceAnalysisImpl<PrintIntervals>::visitTerminator(
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

template <bool PrintIntervals>
void MembarAnalysisImpl<PrintIntervals>::insertBarrier(Operation *op,
                                                       OpBuilder *builder) {
  OpBuilder::InsertionGuard g(*builder);
  auto barrierOp = triton::gpu::LocalBarrierOp::create(*builder, op->getLoc());
}

template <bool PrintIntervals>
void MembarAnalysisImpl<PrintIntervals>::update(
    Operation *op, BlockInfo *blockInfo, FuncBlockInfoMapT *funcBlockInfoMap,
    OpBuilder *builder) {
  if (isa<gpu::BarrierOp, triton::gpu::LocalBarrierOp>(op)) {
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
          for (auto bufferId : this->allocation->getBufferIds(value)) {
            if (bufferId != Allocation::InvalidBufferId) {
              auto baseInterval =
                  this->allocation->getAllocatedInterval(bufferId);
              IntervalVector intervals;

              // Check that all operations on a buffer use the same layout.
              // If not, we need to conservatively assume it touches the
              // whole buffer.
              auto valTy = cast<triton::gpu::MemDescType>(value.getType());
              auto currentLayout = valTy.getEncoding();
              // Track the layout for this operation. If we've already seen
              // this buffer in this operation, encodings must match
              auto [it, inserted] =
                  curBlockInfo.bufferLayouts.emplace(bufferId, currentLayout);
              assert(inserted ||
                     it->second == currentLayout &&
                         "Same buffer used with multiple layouts in same op");
              auto previousLayout = blockInfo->bufferLayouts.find(bufferId);
              bool layoutChanged =
                  (previousLayout != blockInfo->bufferLayouts.end() &&
                   previousLayout->second != currentLayout);

              if (layoutChanged) {
                // Layout changed, use the full interval as we tracking ranges
                // across layouts is complex and not commonly needed
                intervals.push_back(baseInterval);
              } else {
                // Get fine grained intervals for the operation
                intervals = getFineGrainedIntervals(baseInterval, value);
              }

              if (isa<MemoryEffects::Write>(effectInstance.getEffect())) {
                // Insert operation for each interval touched
                for (auto interval : intervals) {
                  curBlockInfo.syncWriteIntervals[interval].insert(op);
                }
              } else if (isa<MemoryEffects::Read>(effectInstance.getEffect())) {
                // Insert operation for each interval touched
                for (auto interval : intervals) {
                  curBlockInfo.syncReadIntervals[interval].insert(op);
                }
              }
            }
          }
        }
      }
    }
    // If this op is may be signalling other threads asynchronously, make sure
    // all shared memory transactions are complete beforehand.
    if (isa<triton::nvidia_gpu::ArriveBarrierOp>(op)) {
      Interval<size_t> allIntervals(0, std::numeric_limits<size_t>::max());
      curBlockInfo.syncWriteIntervals[allIntervals].insert(op);
      curBlockInfo.syncReadIntervals[allIntervals].insert(op);
    }
    scratchBufferId = this->allocation->getBufferId(op);
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

    if (!curBlockInfo.syncReadIntervals.empty() ||
        !curBlockInfo.syncWriteIntervals.empty()) {
      llvm::report_fatal_error(
          "scratch buffer operations should not have any shared memory "
          "dependencies");
    }
    auto interval = this->allocation->getAllocatedInterval(scratchBufferId);
    curBlockInfo.syncWriteIntervals[interval].insert(op);
    auto insertCTABarrier =
        blockInfo->isIntersected(curBlockInfo, this->filter);
    if (insertCTABarrier) {
      builder->setInsertionPoint(op);
      insertBarrier(op, builder);
    }
    // Ops with a scratch buffer that don't use warp.sync internally sync
    // read/write on shared memory
    if (insertCTABarrier || !isWarpSync)
      blockInfo->sync();
    curBlockInfo.syncReadIntervals[interval].insert(op);
  } else if (blockInfo->isIntersected(curBlockInfo, this->filter)) {
    builder->setInsertionPoint(op);
    insertBarrier(op, builder);
    blockInfo->sync();
  }
  if constexpr (PrintIntervals) {
    llvm::errs() << "Op: ";
    op->print(llvm::errs());
    llvm::errs() << "\n";
    curBlockInfo.dump();
  }

  // Update the region info, even if barrier is inserted, we have to maintain
  // the current op's read/write buffers and layout information.
  blockInfo->join(curBlockInfo);
}

// Explicit template instantiations
template class MembarOrFenceAnalysisImpl<>;
template class MembarOrFenceAnalysisImpl<true>;
template class MembarAnalysisImpl<>;
template class MembarAnalysisImpl<true>;

} // namespace mlir
