#include "Utilities.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/OpInterfaces.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PartitionBuilder.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_NVWSINSERTAREF
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

namespace {

using namespace mlir;
using namespace triton::gpu;
using namespace triton::nvidia_gpu;
using namespace triton::nvws;

struct ProducedValueInfo {
  SetVector<int> partitions;
  Value result;
};

SmallVector<ProducedValueInfo> getProducedValues(Operation *op,
                                                 Block *loopBody) {
  SmallVector<ProducedValueInfo> producedValues;

  if (!hasPartition(op))
    return {};

  // For ops without regions, all results share the same partition IDs
  auto partitionOutputs = op->getNumRegions() == 0
                              ? SmallVector<SetVector<int>, 4>(
                                    op->getNumResults(), getPartitionIds(op))
                              : getPartitionOutputs(op);

  for (auto result : op->getResults()) {
    if (isa<AsyncTokenType>(result.getType()))
      continue;
    producedValues.push_back(
        {partitionOutputs[result.getResultNumber()], result});
  }

  return producedValues;
};

template <typename AllocOp, typename LoadOp>
std::optional<std::pair<AllocOp, LoadOp>> isLoadAndAlloc(Value result) {
  auto alloc = result.getDefiningOp<AllocOp>();
  if (!alloc || !alloc.getSrc())
    return std::nullopt;
  if (auto load = alloc.getSrc().template getDefiningOp<LoadOp>();
      load && getPartitionIds(alloc) == getPartitionIds(load)) {
    // if alloc and load are in different partitions, they are treated as two
    // different producer operations.
    return std::make_pair(alloc, load);
  }
  return std::nullopt;
}

// if result is defined by descriptor_load followed by alloc, return the alloc
// and the load ops as a pair.
template <typename AllocOp> auto isDescLoadAndAlloc(Value result) {
  return isLoadAndAlloc<AllocOp, triton::DescriptorOpInterface>(result);
}

template <typename AllocOp> auto isGlobalLoadAndAlloc(Value result) {
  return isLoadAndAlloc<AllocOp, triton::LoadOp>(result);
}

RankedTensorType getTensorTypeFromScalar(OpBuilder &builder, Value scalar) {
  auto mod = scalar.getParentRegion()->getParentOfType<ModuleOp>();
  auto nWarps = lookupNumWarps(mod);
  auto threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
  int CTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);
  Attribute encoding = getDefaultBlockedEncoding(builder.getContext(), {1},
                                                 nWarps, threadsPerWarp, CTAs);
  return RankedTensorType::get({1}, scalar.getType(), encoding);
}

ArefCreateOp createAref(OpBuilder &builder, ProducedValueInfo &producedValue) {
  auto result = producedValue.result;

  auto getSmemDescType = [](RankedTensorType tensorType, Value tensorResult) {
    Attribute SharedMemorySpace =
        SharedMemorySpaceAttr::get(tensorType.getContext());
    Attribute encoding = tensorResult && tensorResult.getDefiningOp()
                             ? getSharedEncoding(tensorResult.getDefiningOp())
                             : getSharedEncoding(tensorType);
    auto memDescType =
        MemDescType::get(tensorType.getShape(), tensorType.getElementType(),
                         encoding, SharedMemorySpace);
    return memDescType;
  };

  MemDescType memDescType;
  if (result.getDefiningOp<LocalAllocOp>()) {
    memDescType = dyn_cast<MemDescType>(result.getType());
  } else if (auto tensorType = dyn_cast<RankedTensorType>(result.getType())) {
    memDescType = getSmemDescType(tensorType, result);
  } else if (isa<FloatType, IntegerType>(result.getType())) {
    auto tensorType = getTensorTypeFromScalar(builder, result);
    memDescType = getSmemDescType(tensorType, Value());
  } else {
    std::string msg = "createAref: unsupported produced value type: " +
                      mlir::debugString(result.getType());
    llvm::report_fatal_error(msg.c_str());
  }

  MemDescType arefBufType = getMultiBufferedType(memDescType, 1);
  assert(isa<SharedMemorySpaceAttr>(arefBufType.getMemorySpace()));
  auto loc = result.getLoc();
  auto alloc = triton::nvws::createAlloc(builder, loc, arefBufType, Value());
  return createArefCreateOp(builder, {arefBufType}, {alloc->getResult(0)}, loc);
}

int getTxCount(Operation *descOp) {
  auto getTensorTypeAndDesc =
      [](Operation *op) -> std::pair<RankedTensorType, Value> {
    if (auto loadOp = dyn_cast<triton::DescriptorLoadOp>(op)) {
      return {loadOp.getType(), loadOp.getDesc()};
    } else if (auto gatherOp = dyn_cast<triton::DescriptorGatherOp>(op)) {
      return {gatherOp.getType(), gatherOp.getDesc()};
    } else {
      llvm_unreachable("Unsupported operation type");
    }
  };
  auto [tensorType, desc] = getTensorTypeAndDesc(descOp);
  auto encoding = getEncodingFromDescriptor(descOp, tensorType, desc);
  auto shapePerCTA = getShapePerCTA(encoding, tensorType.getShape());
  return product(shapePerCTA) *
         getIntOrFloatOrPtrBitWidth(tensorType.getElementType()) / 8;
}

void createNVWSDescriptorLoadOp(OpBuilder &builder, Operation *ttDescLoadOp,
                                Value dataBuf,
                                SetVector<int> const &producerPartitions,
                                Location loc) {
  auto txCount = getTxCount(ttDescLoadOp);
  if (auto descLoad = dyn_cast<triton::DescriptorLoadOp>(ttDescLoadOp)) {
    auto newDescLoad = triton::nvws::DescriptorLoadOp::create(
        builder, loc, descLoad.getDesc(), descLoad.getIndices(), txCount,
        dataBuf, descLoad.getCache(), descLoad.getEvict());
    newDescLoad->setAttrs(descLoad->getAttrs());
    setPartition(newDescLoad, producerPartitions);
  } else if (auto descGather =
                 dyn_cast<triton::DescriptorGatherOp>(ttDescLoadOp)) {
    auto newDescGather = triton::nvws::DescriptorGatherOp::create(
        builder, loc, descGather.getDesc(), descGather.getXOffsets(),
        descGather.getYOffset(), txCount, dataBuf);
    newDescGather->setAttrs(descGather->getAttrs());
    setPartition(newDescGather, producerPartitions);
  } else {
    llvm_unreachable("unknown descriptor op.");
  }
}

StageCluster getStageClusterForProducer(Value producedValue) {
  if (auto opt = isDescLoadAndAlloc<LocalAllocOp>(producedValue)) {
    return getStageCluster(opt->second);
  } else if (auto opt = isGlobalLoadAndAlloc<LocalAllocOp>(producedValue)) {
    return getStageCluster(opt->second);
  } else if (auto op = producedValue.getDefiningOp()) {
    return getStageCluster(op);
  } else {
    return {};
  }
}

SmallVector<Operation *> createArefPut(OpBuilder &builder, ArefCreateOp aref,
                                       ProducedValueInfo producedValue) {
  auto loc = producedValue.result.getLoc();
  auto arefBufType = cast<MemDescType>(aref.getBuffers()[0].getType());
  Value result = producedValue.result;
  Type dataBufType = getBufferViewType(arefBufType, /*mutable*/ true);
  StageCluster stageCluster = getStageClusterForProducer(result);

  // elect a partition to put result into aref-buffer
  SetVector<int> producerPartitions;
  producerPartitions.insert(producedValue.partitions.front());

  Type token{builder.getType<AsyncTokenType>()};
  auto putEnterOp = triton::gpu::createInto<ArefPutEnterOp>(
      builder, loc, producerPartitions, stageCluster, aref,
      TypeRange{dataBufType}, token);
  auto dataBuf = putEnterOp.getBuffers()[0];

  auto producerKind = AsyncOp::NONE;
  SmallVector<Operation *> staleOps;
  if (auto opt = isDescLoadAndAlloc<LocalAllocOp>(result)) {
    auto [alloc, descOp] = *opt;
    createNVWSDescriptorLoadOp(builder, descOp, dataBuf, producerPartitions,
                               loc);
    producerKind = AsyncOp::TMALoad;
    staleOps.push_back(alloc);
    staleOps.push_back(descOp);
  } else if (isGlobalLoadAndAlloc<LocalAllocOp>(result)) {
    llvm_unreachable("cpasync not supported yet");
  } else if (auto alloc = result.getDefiningOp<LocalAllocOp>()) {
    triton::gpu::createInto<LocalStoreOp>(builder, loc, producerPartitions,
                                          stageCluster, alloc.getSrc(),
                                          dataBuf);
    staleOps.push_back(alloc);
  } else if (auto tensorType = dyn_cast<RankedTensorType>(result.getType())) {
    if (auto descOp = result.getDefiningOp<triton::DescriptorOpInterface>()) {
      createNVWSDescriptorLoadOp(builder, descOp, dataBuf, producerPartitions,
                                 loc);
      producerKind = AsyncOp::TMALoad;
      staleOps.push_back(descOp);
    } else if (auto loadOp = result.getDefiningOp<triton::LoadOp>()) {
      llvm_unreachable("cpasync not supported yet");
    } else {
      triton::gpu::createInto<LocalStoreOp>(builder, loc, producerPartitions,
                                            stageCluster, result, dataBuf);
      producerKind = AsyncOp::NONE;
    }
  } else if (isa<FloatType, IntegerType>(result.getType())) {
    auto tensorType = getTensorTypeFromScalar(builder, result);
    auto splatOp = triton::gpu::createInto<triton::SplatOp>(
        builder, loc, producerPartitions, stageCluster, tensorType, result);
    triton::gpu::createInto<LocalStoreOp>(builder, loc, producerPartitions,
                                          stageCluster, splatOp, dataBuf);
    producerKind = AsyncOp::NONE;
  } else {
    std::string msg = "createArefPut: unsupported produced value type: " +
                      mlir::debugString(result.getType());
    llvm::report_fatal_error(msg.c_str());
  }

  triton::gpu::createInto<ArefPutExitOp>(
      builder, loc, producerPartitions, stageCluster, aref,
      putEnterOp.getToken(),
      builder.getArrayAttr(SmallVector<Attribute>{
          AsyncOpAttr::get(aref.getContext(), producerKind)}));

  return staleOps;
};

SetVector<Operation *>
getTransitiveConsumers(Operation *op,
                       SetVector<int> const &consumerPartitions) {
  SetVector<Operation *> opConsumers;
  auto isMemDesc = [](auto res) { return isa<MemDescType>(res.getType()); };
  for (auto &use : op->getUses()) {
    if (llvm::count_if(use.getOwner()->getResults(), isMemDesc) > 0) {
      // Recurse into consumers of memdesc ops, since the liveness of the
      // produced value extends beyond such ops.
      auto consumers =
          getTransitiveConsumers(use.getOwner(), consumerPartitions);
      opConsumers.insert(consumers.begin(), consumers.end());
    } else {
      if (getPartitionIds(&use) == consumerPartitions) {
        opConsumers.insert(use.getOwner());
        // If an op is defined before an inner loop and used inside, the loop
        // itself should be considered as an additional consumer. This is
        // necessary for persistent attention, where the load of Q is done
        // before the inner loop.
        opConsumers.insert(
            op->getBlock()->findAncestorOpInBlock(*use.getOwner()));
      }
    }
  }
  return opConsumers;
}

SmallVector<Operation *>
getTransitiveConsumers(const SetVector<Value> &results,
                       SetVector<int> const &consumerPartitions) {
  SetVector<Operation *> opSet;
  for (auto result : results) {
    if (isa<BlockArgument>(result)) {
      for (auto &use : result.getUses()) {
        if (getPartitionIds(&use) == consumerPartitions) {
          opSet.insert(use.getOwner());
        }
      }
    } else {
      auto consumers =
          getTransitiveConsumers(result.getDefiningOp(), consumerPartitions);
      opSet.insert(consumers.begin(), consumers.end());
    }
  }
  return SmallVector<Operation *>{opSet.begin(), opSet.end()};
}

SmallVector<Attribute> getConsumerAsyncOpKinds(ArrayRef<Operation *> consumers,
                                               MLIRContext *ctx) {
  SetVector<AsyncOp> kindSet;
  for (auto consumer : consumers) {
    if (isa<scf::ForOp>(consumer) && consumers.size() > 1) {
      // In this case, a getExit is placed after the consumer loop. The
      // corresponding async kind attributes should be determined from other
      // consumer ops in the loop.
      continue;
    }
    if (isa<WarpGroupDotOp>(consumer)) {
      kindSet.insert(AsyncOp::WGMMA);
    } else if (isa<MMAv5OpInterface>(consumer)) {
      kindSet.insert(AsyncOp::TC5MMA);
    } else {
      kindSet.insert(AsyncOp::NONE);
    }
  }

  SmallVector<Attribute> kindAttrs;
  for (auto kind : kindSet) {
    kindAttrs.push_back(AsyncOpAttr::get(ctx, kind));
  }

  return kindAttrs;
}

std::pair<StageCluster, StageCluster>
getEnterAndExitStageClustersOfUses(const SetVector<Value> &producedResults,
                                   std::function<bool(Operation *)> filterUse,
                                   scf::ForOp forOp) {
  CoarseSchedule coarseSchedule;
  if (!forOp || failed(coarseSchedule.deSerialize(forOp))) {
    return std::make_pair(std::nullopt, std::nullopt);
  }

  SmallVector<Operation *> ops;
  for (auto res : producedResults) {
    if (auto blockArg = dyn_cast<BlockArgument>(res)) {
      // If the producer is a block argument, this means we need to communicate
      // iteration arguments from the producer partition in the previous
      // iteration to the consumer partition in the current iteration. There
      // must be only one produced result in this case.
      assert(producedResults.size() == 1);
      auto block = blockArg.getOwner();
      auto forOp = cast<scf::ForOp>(block->getParentOp());
      auto opnd = forOp.getYieldedValues()[blockArg.getArgNumber() - 1];
      auto op = opnd.getDefiningOp();
      auto stageCluster = getStageCluster(op);
      return std::make_pair(stageCluster, stageCluster);
    }
    auto op = res.getDefiningOp();
    ops.push_back(op);
  }

  auto firstOp =
      triton::getFirstUseOfPipelinedOp(ops, forOp, coarseSchedule, filterUse);
  auto lastOp =
      triton::getLastUseOfPipelinedOp(ops, forOp, coarseSchedule, filterUse);
  assert(firstOp && lastOp);

  return std::make_pair(getStageCluster(firstOp), getStageCluster(lastOp));
}

void createArefGet(OpBuilder &builder, scf::ForOp loop, ArefCreateOp aref,
                   const SetVector<Value> &results, int consumerPartition,
                   SmallVector<OpOperand *> &uses) {
  OpBuilder::InsertionGuard g(builder);
  // The vector "results" contains either
  // 1. One of local_load(desc_load()) or desc_load()
  // 2. Both of them
  // In the second case, we only need to emit one enter / exit since we know
  // that the two results are used by consumers in the same partition.
  assert(results.size() == 1 || results.size() == 2);
  auto loc = results[0].getLoc();

  scf::ForOp scheduledLoop;
  loop->walk([&](scf::ForOp op) {
    if (op->hasAttr(mlir::triton::kScheduledMaxStageAttrName)) {
      scheduledLoop = op;
    }
  });

  auto filterUse = [&](Operation *user) {
    if (hasPartition(user)) {
      return llvm::is_contained(getPartitionIds(user), consumerPartition);
    } else {
      return false;
    }
  };
  auto [stageClusterEnter, stageClusterExit] =
      getEnterAndExitStageClustersOfUses(results, filterUse, scheduledLoop);

  SetVector<int> consumerPartitions;
  consumerPartitions.insert(consumerPartition);
  auto arefBufType = cast<MemDescType>(aref.getOperand(0).getType());
  Type bufferType = getBufferViewType(arefBufType, /*mutable*/ false);
  Type tokenType = builder.getType<AsyncTokenType>();
  auto getEnterOp = triton::gpu::createInto<ArefGetEnterOp>(
      builder, loc, consumerPartitions, stageClusterEnter, aref,
      TypeRange{bufferType}, tokenType);

  auto consumers = getTransitiveConsumers(results, consumerPartitions);
  assert(consumers.size() > 0);
  auto asyncKinds = getConsumerAsyncOpKinds(consumers, aref.getContext());
  Value dataBuf = getEnterOp.getBuffers()[0];
  Value token = getEnterOp.getToken();

  Operation *exitInsertPointAfter = nullptr;

  auto replaceUsesWithLocalLoad = [&](Value result, StageCluster stageCluster) {
    auto localLoadOp = triton::gpu::createInto<LocalLoadOp>(
        builder, loc, consumerPartitions, stageCluster, result.getType(),
        dataBuf);

    for (auto use : uses) {
      if (use->get() == result) {
        use->set(localLoadOp.getResult());
      }
    }
    if (dataBuf.hasOneUse()) {
      // If there is only one consumer for dataBuf, it is localLoadOp created
      // above, and we hit this code path, the empty barrier can be released
      // after local load.
      exitInsertPointAfter = localLoadOp;
    }
  };

  for (auto result : results) {
    if (auto localAlloc = result.getDefiningOp<LocalAllocOp>()) {
      auto callback = [&](Operation *oldOp, Operation *newOp) {
        assert(llvm::is_contained(getPartitionIds(oldOp), consumerPartition));
        setPartition(newOp, consumerPartitions);
      };
      replaceUsesAndPropagateType(builder, localAlloc, dataBuf, callback);
    } else if (isa<RankedTensorType>(result.getType())) {
      replaceUsesWithLocalLoad(result, stageClusterEnter);
    } else if (isa<FloatType, IntegerType>(result.getType())) {
      auto tensorType = getTensorTypeFromScalar(builder, result);
      auto localLoadOp = triton::gpu::createInto<LocalLoadOp>(
          builder, loc, consumerPartitions, stageClusterEnter, tensorType,
          dataBuf);
      auto scalar = triton::gpu::createInto<triton::UnsplatOp>(
          builder, loc, consumerPartitions, stageClusterEnter, localLoadOp);
      for (auto use : uses) {
        use->set(scalar);
      }
      exitInsertPointAfter = localLoadOp;
    } else {
      std::string msg = "createArefGet: unsupported produced value type: " +
                        mlir::debugString(result.getType());
      llvm::report_fatal_error(msg.c_str());
    }
  }

  if (exitInsertPointAfter == nullptr) {
    PostDominanceInfo dom(loop);
    exitInsertPointAfter = findNearestCommonPostDominator(consumers, dom);
  }

  builder.setInsertionPointAfter(exitInsertPointAfter);

  triton::gpu::createInto<ArefGetExitOp>(builder, loc, consumerPartitions,
                                         stageClusterExit, aref, token,
                                         builder.getArrayAttr(asyncKinds));
};

Operation *getEarliestUserInBlock(Block *block, ArrayRef<OpOperand *> uses) {
  OpOperand *use =
      *llvm::min_element(uses, [block](OpOperand *lhs, OpOperand *rhs) {
        auto lhsOwner = block->findAncestorOpInBlock(*lhs->getOwner());
        auto rhsOwner = block->findAncestorOpInBlock(*rhs->getOwner());
        return lhsOwner->isBeforeInBlock(rhsOwner);
      });
  return block->findAncestorOpInBlock(*use->getOwner());
}

bool insertArefs(OpBuilder &builder, scf::ForOp loop, Block *block,
                 ProducedValueInfo producedValue) {
  // Collect uses of local_alloc(desc_load()) or desc_load() results by each
  // partition
  DenseMap<int, SetVector<Value>> resultsPerPartition;
  DenseMap<int, SmallVector<OpOperand *>> usesPerPartition;
  auto processResultUses = [&](Value result) {
    for (auto &use : result.getUses()) {
      auto user = use.getOwner();
      // if use is outside ttg.ws, it may not have partition ids, skip it
      if (!hasPartition(user))
        continue;
      auto userPartitions = getPartitionIds(&use);
      for (auto id : producedValue.partitions) {
        userPartitions.remove(id);
      }
      for (auto id : userPartitions) {
        resultsPerPartition[id].insert(result);
        usesPerPartition[id].push_back(&use);
      }
    }
  };

  processResultUses(producedValue.result);

  if (auto opt = isDescLoadAndAlloc<LocalAllocOp>(producedValue.result)) {
    // Process the register use as well
    auto alloc = opt->first;
    processResultUses(alloc.getSrc());
  }

  if (resultsPerPartition.empty()) {
    return false;
  }

  ArefCreateOp aref;
  {
    OpBuilder::InsertionGuard g(builder);
    auto wsLoop = getOuterWSLoop(loop);
    builder.setInsertionPoint(wsLoop);
    aref = createAref(builder, producedValue);
  }

  auto staleOps = createArefPut(builder, aref, producedValue);

  for (auto [consumerPartition, results] : resultsPerPartition) {
    OpBuilder::InsertionGuard g(builder);
    auto earliestUser =
        getEarliestUserInBlock(block, usesPerPartition[consumerPartition]);
    builder.setInsertionPoint(earliestUser);
    createArefGet(builder, loop, aref, results, consumerPartition,
                  usesPerPartition[consumerPartition]);
  }

  for (auto op : staleOps) {
    op->erase();
  }

  return true;
}

} // namespace

class NVWSArefInsertion
    : public triton::impl::NVWSInsertArefBase<NVWSArefInsertion> {
public:
  void runOnFunction(triton::FuncOp func) {
    SmallVector<scf::ForOp> loops;
    func.walk([&](scf::ForOp loop) {
      auto func = loop->getParentOfType<triton::FuncOp>();
      if (loop->hasAttr(triton::kWarpSpecializeAttrName) && hasPartition(loop))
        loops.push_back(loop);
    });

    for (scf::ForOp loop : loops) {
      loop.walk([&](scf::ForOp forOp) {
        // Communicate tensor arguments in iter_args from producer partition in
        // current iteration to consumer partition in previous iteration or
        // initial value
        for (auto arg : forOp.getRegionIterArgs()) {
          if (isa<RankedTensorType, FloatType, IntegerType>(arg.getType())) {
            auto producerPartition =
                getPartitionOutputs(forOp)[arg.getArgNumber() - 1];
            ProducedValueInfo producedValue{producerPartition, arg};
            OpBuilder builder(forOp);
            builder.setInsertionPointToStart(forOp.getBody());
            insertArefs(builder, loop, forOp.getBody(), producedValue);
          }
        }
      });

      // To handle cases where desc_load result in registers is used as is in
      // addition to being consumed by local_alloc op, we process
      // local_alloc(desc_load()) first, followed by remaining register uses of
      // desc_load results.
      SmallVector<Operation *> memoryOps;
      loop.walk([&](Operation *op) {
        if (op->getNumResults() > 0 &&
            (isDescLoadAndAlloc<LocalAllocOp>(op->getResult(0)) ||
             isa<LocalAllocOp>(op))) {
          memoryOps.push_back(op);
        }
      });

      for (auto op : memoryOps) {
        auto producedValues = getProducedValues(op, loop.getBody());
        for (auto producedValue : producedValues) {
          OpBuilder builder(op);
          insertArefs(builder, loop, op->getBlock(), producedValue);
        }
      }

      // handle non-tmem ops in the loop, including uses of desc_load results.
      loop.walk([&](Operation *op) {
        if (op == loop || isa<MMAv5OpInterface, TMEMAllocOp, TMEMStoreOp>(op)) {
          return WalkResult::advance();
        }
        auto producedValues = getProducedValues(op, loop.getBody());
        for (auto producedValue : producedValues) {
          OpBuilder builder(op);
          builder.setInsertionPointAfter(op);
          insertArefs(builder, loop, op->getBlock(), producedValue);
        }
        return WalkResult::advance();
      });
    }
  }

  void runOnOperation() override {
    getOperation().walk([&](triton::FuncOp func) { runOnFunction(func); });
  }
};

} // namespace triton
} // namespace mlir
