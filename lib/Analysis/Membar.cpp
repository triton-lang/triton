#include "triton/Analysis/Membar.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <deque>

namespace mlir {

static SmallPtrSet<Operation *, 2> parentAllocs(Operation *op) {
  SmallPtrSet<Operation *, 2> owners;
  auto opEffects = dyn_cast<MemoryEffectOpInterface>(op);
  if (!opEffects)
    return owners;

  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
  opEffects.getEffects(effects);
  for (auto &effect : effects) {
    if (effect.getResource() != triton::gpu::SharedMemory::get())
      continue;

    Value value = effect.getValue();
    // Hacky way to skip barriers...
    if (cast<triton::gpu::MemDescType>(value.getType()).getNumElements() == 1)
      continue;

    // Get a slice of all the operations that touch this shared memory
    // (subslice/index/memdesc views) all the way up to the local_alloc.
    BackwardSliceOptions options;
    options.omitUsesFromAbove = false;
    options.inclusive = true;
    auto isSharedMemDesc = [](Type ty) {
      auto memDescTy = dyn_cast<triton::gpu::MemDescType>(ty);
      if (!memDescTy)
        return false;
      return isa<triton::gpu::SharedMemorySpaceAttr>(
          memDescTy.getMemorySpace());
    };
    options.filter = [&](Operation *depOp) -> bool {
      return llvm::any_of(depOp->getOperandTypes(), isSharedMemDesc) ||
             // Add ops that have Memdesc in the result types to pick
             // local_alloc ops as well
             llvm::any_of(depOp->getResultTypes(), isSharedMemDesc);
    };
    llvm::SetVector<Operation *> slice;
    LogicalResult result = getBackwardSlice(value, &slice, options);
    assert(succeeded(result) && "backward slice must succeed");

    for (Operation *depOp : slice) {
      if (auto alloc = dyn_cast<triton::gpu::LocalAllocOp>(depOp))
        owners.insert(alloc.getOperation());
    }
  }
  return owners;
}

static std::pair<BlockInfo::CTA_UFDS, BlockInfo::CTA_UFDS>
getCTAEquivalenceSets(Operation *op) {
  auto numCTAs = triton::gpu::lookupNumCTAs(op);
  if (numCTAs == 1) {
    return {BlockInfo::CTA_UFDS(1), BlockInfo::CTA_UFDS(1)};
  }
  auto *ctx = op->getContext();
  auto kBlock = StringAttr::get(ctx, "block");
  if (auto cvt = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
    auto srcTy = cast<RankedTensorType>(cvt.getSrc().getType());
    auto dstTy = cast<RankedTensorType>(cvt.getType());
    auto cvtLayout = minimalCvtLayout(srcTy, dstTy);
    if (llvm::is_contained(cvtLayout.getInDimNames(), kBlock)) {
      auto readsUFDS = BlockInfo::CTA_UFDS(numCTAs);
      auto blockLayout =
          cvtLayout.sublayout({kBlock}, to_vector(cvtLayout.getOutDimNames()));
      for (int i = 0; i < numCTAs; i++) {
        auto res = blockLayout.apply({{kBlock, i}});
        assert(res.size() == 4);
        assert(res.back().first == kBlock);
        readsUFDS.unite(i, res.back().second);
      }
      // The writes are just each writing to their own shmem
      auto writesUFDS = BlockInfo::CTA_UFDS(numCTAs);
      return {readsUFDS, writesUFDS};
    }
  } else if (auto reduce = dyn_cast<triton::ReduceOp>(op)) {
    auto srcTy = cast<RankedTensorType>(reduce.getInputTypes()[0]);
    auto inCTALayout = triton::gpu::getCTALayout(srcTy.getEncoding());
    auto axis = reduce.getAxis();
    auto ll = inCTALayout.getLinearLayout();
    auto outdims = to_vector(ll.getOutDimNames());
    if (ll.getOutDimSize(outdims[axis]) != 1) {
      auto outCTALayout = triton::gpu::getCTALayout(
          cast<RankedTensorType>(reduce.getType(0)).getEncoding());
      // Maps the reads necessary in the reduction
      auto ctaLl = outCTALayout.getLinearLayout().invertAndCompose(ll);
      auto readsUFDS = BlockInfo::CTA_UFDS(numCTAs);
      for (int i = 0; i < numCTAs; i++) {
        auto res = ctaLl.apply({{kBlock, i}});
        assert(res.size() == 1);
        assert(res.front().first == kBlock);
        readsUFDS.unite(i, res.front().second);
      }
      // The writes are just each writing to their own shmem
      auto writesUFDS = BlockInfo::CTA_UFDS(numCTAs);
      return {readsUFDS, writesUFDS};
    }
  } else if (auto tma =
                 dyn_cast<triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp>(
                     op)) {
    if (tma.getMulticast()) {
      auto ctaLl =
          triton::gpu::getCTALayout(tma.getResult().getType().getEncoding())
              .getLinearLayout()
              .flattenOuts();
      // We build a map that's the identity on the non broadcasted blocks and
      // zero in the broadcasted
      auto ll = triton::LinearLayout::identity1D(numCTAs, kBlock, kBlock);
      auto bases = ll.getBases();
      auto &basesBlock = bases[kBlock];
      auto outDim = *ctaLl.getOutDimNames().begin();
      for (int i = 0; i < llvm::Log2_32(numCTAs); i++) {
        if (ctaLl.getBasis(kBlock, i, outDim) == 0) {
          basesBlock[i] = {0};
        }
      }
      ll = triton::LinearLayout(bases, {{kBlock, numCTAs}}, false);
      auto writesUFDS = BlockInfo::CTA_UFDS(numCTAs);
      for (int i = 0; i < numCTAs; i++) {
        auto res = ll.apply({{kBlock, i}});
        assert(res.size() == 1);
        assert(res.front().first == kBlock);
        writesUFDS.unite(i, res.front().second);
      }
      // It's not going to be used so it's fine
      auto defaultUFDS = BlockInfo::CTA_UFDS(numCTAs);
      return {defaultUFDS, writesUFDS};
    }
  }
  return {BlockInfo::CTA_UFDS(numCTAs), BlockInfo::CTA_UFDS(numCTAs)};
}

bool BlockInfo::haveSameAlloc(Operation *lhs, Operation *rhs) {
  auto lhsAllocs = parentAllocs(lhs);
  auto rhsAllocs = parentAllocs(rhs);
  // They can be empty when the buffer is internal, e.g. a convert_layout.
  if (lhsAllocs.empty() || rhsAllocs.empty())
    return false;

  return llvm::any_of(
      lhsAllocs, [&](Operation *alloc) { return rhsAllocs.contains(alloc); });
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

void MembarAnalysis::insertBarrier(Operation *op, OpBuilder *builder,
                                   const BlockInfo::CTA_UFDS &ctaClasses) {
  OpBuilder::InsertionGuard g(*builder);
  if (ctaClasses.isDistributed()) {
    // TODO Insert an membar when there is more than one CTA class to avoid
    // synchronising the whole cluster
    triton::nvidia_gpu::ClusterArriveOp::create(*builder, op->getLoc(),
                                                /*relaxed=*/false);
    triton::nvidia_gpu::ClusterWaitOp::create(*builder, op->getLoc());
  } else {
    triton::gpu::LocalBarrierOp::create(*builder, op->getLoc());
  }
}

void MembarAnalysis::update(Operation *op, BlockInfo *blockInfo,
                            FuncBlockInfoMapT *funcBlockInfoMap,
                            OpBuilder *builder) {
  if (isa<gpu::BarrierOp, triton::gpu::LocalBarrierOp,
          triton::nvidia_gpu::ClusterWaitOp>(op)) {
    // If the current op is a barrier, we sync previous reads and writes
    blockInfo->sync(isa<triton::nvidia_gpu::ClusterWaitOp>(op));
    return;
  }

  if (op->hasTrait<mlir::OpTrait::MemWaitOpTrait>() &&
      !isa<gpu::BarrierOp, triton::gpu::LocalBarrierOp>(op->getNextNode())) {
    // If the current op is an async wait and the next op is not a barrier we
    // insert a barrier op and sync
    builder->setInsertionPointAfter(op);
    auto nCTAs = triton::gpu::lookupNumCTAs(op);
    insertBarrier(op, builder, BlockInfo::CTA_UFDS(nCTAs));
    blockInfo->sync(false);
    return;
  }

  auto [readCTAs, writeCTAs] = getCTAEquivalenceSets(op);

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
              if (isa<MemoryEffects::Write>(effectInstance.getEffect()))
                curBlockInfo.syncWriteIntervals[{interval, writeCTAs}].insert(
                    op);
              else if (isa<MemoryEffects::Read>(effectInstance.getEffect()))
                curBlockInfo.syncReadIntervals[{interval, readCTAs}].insert(op);
            }
          }
        }
      }
    }
    // If this op may be signalling other threads asynchronously, make sure
    // all shared memory transactions are complete beforehand.
    if (isa<triton::nvidia_gpu::ArriveBarrierOp>(op)) {
      Interval<size_t> allIntervals(0, std::numeric_limits<size_t>::max());
      curBlockInfo.syncWriteIntervals[{allIntervals, writeCTAs}].insert(op);
      curBlockInfo.syncReadIntervals[{allIntervals, readCTAs}].insert(op);
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

    if (!curBlockInfo.syncReadIntervals.empty() ||
        !curBlockInfo.syncWriteIntervals.empty()) {
      llvm::report_fatal_error(
          "scratch buffer operations should not have any shared memory "
          "dependencies");
    }
    auto interval = allocation->getAllocatedInterval(scratchBufferId);
    curBlockInfo.syncWriteIntervals[{interval, writeCTAs}].insert(op);
    auto insertCTABarrier = blockInfo->isIntersected(curBlockInfo, filter);
    if (insertCTABarrier.has_value()) {
      builder->setInsertionPoint(op);
      insertBarrier(op, builder, *insertCTABarrier);
      blockInfo->sync(insertCTABarrier->isDistributed());
    } else if (!isWarpSync) {
      // Ops with a scratch buffer that don't use warp.sync internally sync
      // read/write on shared memory at the CTA level.
      blockInfo->sync(false);
    }
    curBlockInfo.syncReadIntervals[{interval, readCTAs}].insert(op);
  } else if (auto ctas = blockInfo->isIntersected(curBlockInfo, filter)) {
    builder->setInsertionPoint(op);
    insertBarrier(op, builder, *ctas);
    blockInfo->sync(ctas->isDistributed());
  }
  // Update the region info, even if barrier is inserted, we have to maintain
  // the current op's read/write buffers.
  blockInfo->join(curBlockInfo);
}
} // namespace mlir
