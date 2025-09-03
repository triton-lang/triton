#include "Utilities.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PartitionBuilder.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"

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
  Partition *partition;
  Value result;
};

SmallVector<ProducedValueInfo> getProducedValues(Operation *op, Block *loopBody,
                                                 WarpSchedule &schedule) {
  SmallVector<ProducedValueInfo> producedValues;
  auto partition = schedule.getPartition(loopBody->findAncestorOpInBlock(*op));

  if (partition != schedule.getRootPartition()) {
    for (auto result : op->getResults()) {
      producedValues.push_back({partition, result});
    }
  }

  return producedValues;
};

template <typename AllocOp, typename LoadOp>
std::optional<std::pair<AllocOp, LoadOp>> isLoadAndAlloc(Value result) {
  auto alloc = result.getDefiningOp<AllocOp>();
  if (!alloc)
    return std::nullopt;
  if (auto load = alloc.getSrc().template getDefiningOp<LoadOp>()) {
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

ArefCreateOp createAref(OpBuilder &builder, ProducedValueInfo &producedValue) {
  auto result = producedValue.result;

  auto getSmemDescType = [](Value tensorResult) {
    auto tensorType = cast<RankedTensorType>(tensorResult.getType());
    MemDescType memDescType;
    Attribute SharedMemorySpace =
        SharedMemorySpaceAttr::get(tensorType.getContext());
    if (auto load =
            tensorResult.getDefiningOp<triton::DescriptorOpInterface>()) {
      // A use of TMA which is not immediately consumed by LocalAlloc
      // This case applies, for example, when TMA is followed by SIMT ops
      // or MMAv2 is used.
      auto encoding =
          getEncodingFromDescriptor(load, tensorType, load.getDesc());
      memDescType =
          MemDescType::get(tensorType.getShape(), tensorType.getElementType(),
                           encoding, SharedMemorySpace);
    } else {
      llvm_unreachable("Only TMA is expected for now.");
    }
    return memDescType;
  };

  MemDescType memDescType;
  if (result.getDefiningOp<LocalAllocOp>()) {
    memDescType = dyn_cast<MemDescType>(result.getType());
  } else if (auto opt = isDescLoadAndAlloc<TMEMAllocOp>(result)) {
    auto descLoadResult = opt->first.getSrc();
    memDescType = getSmemDescType(descLoadResult);
  } else if (isa<RankedTensorType>(result.getType())) {
    memDescType = getSmemDescType(result);
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
         tensorType.getElementType().getIntOrFloatBitWidth() / 8;
}

void createNVWSDescriptorLoadOp(OpBuilder &builder, Operation *ttDescLoadOp,
                                Value dataBuf, Partition *producerPartition,
                                WarpSchedule &schedule, Location loc) {
  auto txCount = getTxCount(ttDescLoadOp);
  if (auto descLoad = dyn_cast<triton::DescriptorLoadOp>(ttDescLoadOp)) {
    auto newDescLoad = builder.create<triton::nvws::DescriptorLoadOp>(
        loc, descLoad.getDesc(), descLoad.getIndices(), txCount, dataBuf,
        descLoad.getCache(), descLoad.getEvict());
    newDescLoad->setAttrs(descLoad->getAttrs());
    schedule.insert(producerPartition, newDescLoad);
  } else if (auto descGather =
                 dyn_cast<triton::DescriptorGatherOp>(ttDescLoadOp)) {
    auto newDescGather = builder.create<triton::nvws::DescriptorGatherOp>(
        loc, descGather.getDesc(), descGather.getXOffsets(),
        descGather.getYOffset(), txCount, dataBuf);
    newDescGather->setAttrs(descGather->getAttrs());
    schedule.insert(producerPartition, newDescGather);
  } else {
    llvm_unreachable("unknown descriptor op.");
  }
}

StageCluster getStageClusterForProducer(Value producedValue) {
  if (auto opt = isDescLoadAndAlloc<LocalAllocOp>(producedValue)) {
    return getStageCluster(opt->second);
  } else if (auto opt = isDescLoadAndAlloc<TMEMAllocOp>(producedValue)) {
    return getStageCluster(opt->second);
  } else if (auto opt = isGlobalLoadAndAlloc<LocalAllocOp>(producedValue)) {
    return getStageCluster(opt->second);
  } else if (auto opt = isGlobalLoadAndAlloc<TMEMAllocOp>(producedValue)) {
    return getStageCluster(opt->second);
  }
  return getStageCluster(producedValue.getDefiningOp());
}

SmallVector<Operation *> createArefPut(PartitionBuilder &builder,
                                       ArefCreateOp aref, std::string arefTag,
                                       ProducedValueInfo producedValue,
                                       WarpSchedule &schedule) {
  auto loc = producedValue.result.getLoc();
  auto arefBufType = cast<MemDescType>(aref.getBuffers()[0].getType());
  Value result = producedValue.result;
  Type dataBufType = getBufferViewType(arefBufType, /*mutable*/ true);
  StageCluster stageCluster = getStageClusterForProducer(result);
  Partition *producerPartition = producedValue.partition;

  Type token{builder.getType<AsyncTokenType>()};
  auto c0Enter = builder.intCst(0);
  auto putEnterOp = builder.createInto<ArefPutEnterOp>(
      *producerPartition, stageCluster, SmallVector{dataBufType}, token, aref,
      c0Enter, c0Enter);
  schedule.insert(producerPartition, putEnterOp);
  schedule.insert(producerPartition, c0Enter.getDefiningOp());
  auto dataBuf = putEnterOp.getBuffers()[0];

  auto producerKind = AsyncOp::NONE;
  SmallVector<Operation *> staleOps;
  if (auto opt = isDescLoadAndAlloc<LocalAllocOp>(result)) {
    auto [alloc, descOp] = *opt;
    createNVWSDescriptorLoadOp(builder, descOp, dataBuf, producerPartition,
                               schedule, loc);
    producerKind = AsyncOp::TMALoad;
    staleOps.push_back(alloc);
    staleOps.push_back(descOp);
  } else if (auto opt = isDescLoadAndAlloc<TMEMAllocOp>(result)) {
    auto descOp = opt->second;
    createNVWSDescriptorLoadOp(builder, descOp, dataBuf, producerPartition,
                               schedule, loc);
    producerKind = AsyncOp::TMALoad;
    staleOps.push_back(descOp);
  } else if (isGlobalLoadAndAlloc<LocalAllocOp>(result) ||
             isGlobalLoadAndAlloc<TMEMAllocOp>(result)) {
    llvm_unreachable("cpasync not supported yet");
  } else if (auto alloc = result.getDefiningOp<LocalAllocOp>()) {
    builder.createInto<LocalStoreOp>(*producerPartition, stageCluster,
                                     alloc.getSrc(), dataBuf);
    staleOps.push_back(alloc);
  } else if (auto tensorType = dyn_cast<RankedTensorType>(result.getType())) {
    if (auto descOp = result.getDefiningOp<triton::DescriptorOpInterface>()) {
      createNVWSDescriptorLoadOp(builder, descOp, dataBuf, producerPartition,
                                 schedule, loc);
      producerKind = AsyncOp::TMALoad;
      staleOps.push_back(descOp);
    } else if (auto loadOp = result.getDefiningOp<triton::LoadOp>()) {
      llvm_unreachable("cpasync not supported yet");
    } else {
      // Create LocalStore of result into dataBuf. This is a value aref, not
      // supported for now.
      llvm_unreachable("Aref for values not supported yet");
    }
  } else {
    std::string msg = "createArefPut: unsupported produced value type: " +
                      mlir::debugString(result.getType());
    llvm::report_fatal_error(msg.c_str());
  }

  auto c0Exit = builder.intCst(0);
  auto putExitOp = builder.createInto<ArefPutExitOp>(
      *producerPartition, stageCluster, aref, putEnterOp.getToken(), c0Exit,
      builder.getArrayAttr(SmallVector<Attribute>{
          AsyncOpAttr::get(aref.getContext(), producerKind)}));
  schedule.insert(producerPartition, putExitOp);
  schedule.insert(producerPartition, c0Exit.getDefiningOp());

  return staleOps;
};

SetVector<Operation *> getTransitiveConsumers(Operation *op,
                                              Partition *consumerPartition,
                                              const WarpSchedule &schedule) {
  SetVector<Operation *> opConsumers;
  auto isMemDesc = [](auto res) { return isa<MemDescType>(res.getType()); };
  for (auto user : op->getUsers()) {
    if (llvm::count_if(user->getResults(), isMemDesc) > 0) {
      // Recurse into consumers of memdesc ops, since the liveness of the
      // produced value extends beyond such ops.
      auto consumers =
          getTransitiveConsumers(user, consumerPartition, schedule);
      opConsumers.insert(consumers.begin(), consumers.end());
    } else {
      if (schedule.getPartition(user) == consumerPartition) {
        opConsumers.insert(user);
      }
    }
  }
  return opConsumers;
}

SmallVector<Operation *> getTransitiveConsumers(const SetVector<Value> &results,
                                                Partition *consumerPartition,
                                                const WarpSchedule &schedule) {
  SetVector<Operation *> opSet;
  for (auto result : results) {
    auto consumers = getTransitiveConsumers(result.getDefiningOp(),
                                            consumerPartition, schedule);
    opSet.insert(consumers.begin(), consumers.end());
  }
  return SmallVector<Operation *>{opSet.begin(), opSet.end()};
}

SmallVector<Attribute> getConsumerAsyncOpKinds(ArrayRef<Operation *> consumers,
                                               MLIRContext *ctx) {
  SetVector<AsyncOp> kindSet;
  for (auto consumer : consumers) {
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
  if (failed(coarseSchedule.deSerialize(forOp))) {
    return std::make_pair(std::nullopt, std::nullopt);
  }

  SmallVector<Operation *> ops;
  for (auto res : producedResults) {
    ops.push_back(res.getDefiningOp());
  }

  auto firstOp =
      triton::getFirstUseOfPipelinedOp(ops, forOp, coarseSchedule, filterUse);
  auto lastOp =
      triton::getLastUseOfPipelinedOp(ops, forOp, coarseSchedule, filterUse);
  assert(firstOp && lastOp);

  return std::make_pair(getStageCluster(firstOp), getStageCluster(lastOp));
}

void createArefGet(PartitionBuilder &builder, scf::ForOp loop,
                   ArefCreateOp aref, std::string arefTag,
                   const SetVector<Value> &results,
                   Partition *consumerPartition, WarpSchedule &schedule) {
  OpBuilder::InsertionGuard g(builder);
  // The vector "results" contains either
  // 1. One of local_load(desc_load()) or desc_load()
  // 2. Both of them
  // In the second case, we only need to emit one enter / exit since we know
  // that the two results are used by consumers in the same partition.
  assert(results.size() == 1 || results.size() == 2);
  auto loc = results[0].getLoc();

  auto filterUse = [&](Operation *use) {
    return schedule.getPartition(use) == consumerPartition;
  };
  auto [stageClusterEnter, stageClusterExit] =
      getEnterAndExitStageClustersOfUses(results, filterUse, loop);

  auto arefBufType = cast<MemDescType>(aref.getOperand(0).getType());
  Type bufferType = getBufferViewType(arefBufType, /*mutable*/ false);
  Type tokenType = builder.getType<AsyncTokenType>();
  auto c0Enter = builder.intCst(0);
  auto getEnterOp = builder.createInto<ArefGetEnterOp>(
      *consumerPartition, stageClusterEnter, SmallVector{bufferType}, tokenType,
      aref, c0Enter, c0Enter);
  schedule.insert(consumerPartition, getEnterOp);
  schedule.insert(consumerPartition, c0Enter.getDefiningOp());

  auto consumers = getTransitiveConsumers(results, consumerPartition, schedule);
  assert(consumers.size() > 0);
  auto asyncKinds = getConsumerAsyncOpKinds(consumers, aref.getContext());
  Value dataBuf = getEnterOp.getBuffers()[0];
  Value token = getEnterOp.getToken();

  Operation *exitInsertPointAfter = nullptr;

  auto replaceUsesWithLocalLoad = [&](Value result, StageCluster stageCluster) {
    auto localLoadOp = builder.createInto<LocalLoadOp>(
        *consumerPartition, stageCluster, result.getType(), dataBuf);
    result.replaceAllUsesWith(localLoadOp.getResult());
    schedule.insert(consumerPartition, localLoadOp);
    if (consumers.size() == 1) {
      // If there is only one consumer and we hit this code path, the empty
      // barrier can be released after local load.
      exitInsertPointAfter = localLoadOp;
    }
  };

  for (auto result : results) {
    if (auto localAlloc = result.getDefiningOp<LocalAllocOp>()) {
      auto memDescType = cast<MemDescType>(result.getType());
      auto callback = [&](Operation *oldOp, Operation *newOp) {
        assert(schedule.getPartition(oldOp) == consumerPartition);
        schedule.insert(consumerPartition, newOp);
      };
      replaceUsesAndPropagateType(builder, localAlloc, dataBuf, callback);
    } else if (auto tmemAlloc = result.getDefiningOp<TMEMAllocOp>()) {
      builder.setInsertionPoint(tmemAlloc);
      replaceUsesWithLocalLoad(tmemAlloc.getSrc(), stageClusterEnter);
    } else if (isa<RankedTensorType>(result.getType())) {
      replaceUsesWithLocalLoad(result, stageClusterEnter);
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

  auto c0Exit = builder.intCst(0);
  auto getExitOp = builder.createInto<ArefGetExitOp>(
      *consumerPartition, stageClusterExit, aref, token, c0Exit,
      builder.getArrayAttr(asyncKinds));
  schedule.insert(consumerPartition, getExitOp);
  schedule.insert(consumerPartition, c0Exit.getDefiningOp());
};

bool insertArefs(PartitionBuilder &builder, scf::ForOp loop,
                 WarpSchedule &schedule, ProducedValueInfo producedValue,
                 int arefTag) {
  // Collect uses of local_alloc(desc_load()) or desc_load() results by each
  // partition
  DenseMap<Partition *, SetVector<Value>> resultsPerPartition;
  auto processResultUses = [&](Value result) {
    for (auto user : result.getUsers()) {
      Partition *userPartition = schedule.getPartition(user);
      if (producedValue.partition != userPartition) {
        resultsPerPartition[userPartition].insert(result);
      }
    }
  };

  processResultUses(producedValue.result);

  if (auto opt = isDescLoadAndAlloc<LocalAllocOp>(producedValue.result)) {
    // Process the register use as well
    auto alloc = opt->first;
    processResultUses(alloc.getSrc());
  } else if (auto opt = isDescLoadAndAlloc<TMEMAllocOp>(producedValue.result)) {
    auto alloc = opt->first;
    processResultUses(alloc.getSrc());
  }

  if (resultsPerPartition.empty()) {
    return false;
  }

  ArefCreateOp aref;
  {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPoint(loop);
    aref = createAref(builder, producedValue);
  }

  auto tag = "aref_" + std::to_string(arefTag);
  auto staleOps = createArefPut(builder, aref, tag, producedValue, schedule);

  for (auto [consumerPartition, results] : resultsPerPartition) {
    createArefGet(builder, loop, aref, tag, results, consumerPartition,
                  schedule);
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
  void runOnOperation() override {
    SmallVector<scf::ForOp> loops;
    getOperation().walk([&](scf::ForOp loop) {
      if (loop->hasAttr(triton::kWarpSpecializeAttrName))
        loops.push_back(loop);
    });

    for (scf::ForOp loop : loops) {
      FailureOr<WarpSchedule> schedule = WarpSchedule::deserialize(loop);
      if (failed(schedule))
        continue;

      int arefTag = 0;

      // To handle cases where desc_load result in registers is used as is in
      // addition to being consumed by local_alloc op, we process
      // local_alloc(desc_load()) first, followed by remaining register uses of
      // desc_load results.
      for (auto allowDescLoadRegUse : {false, true}) {
        SmallVector<Operation *> ops;
        loop.walk([&](Operation *op) {
          if (op->getNumResults() == 0) {
            return WalkResult::advance();
          }
          // Only handles load ops for now.
          if (isDescLoadAndAlloc<LocalAllocOp>(op->getResult(0)) ||
              isDescLoadAndAlloc<TMEMAllocOp>(op->getResult(0)) ||
              (allowDescLoadRegUse &&
               (isa<triton::DescriptorOpInterface>(op)))) {
            ops.push_back(op);
          } else if (isa<LocalAllocOp>(op)) {
            ops.push_back(op);
          }
          return WalkResult::advance();
        });

        for (auto op : ops) {
          auto producedValues =
              getProducedValues(op, loop.getBody(), *schedule);
          for (auto producedValue : producedValues) {
            PartitionBuilder builder(op->getLoc(), op);
            builder.setInsertionPoint(op);
            if (insertArefs(builder, loop, *schedule, producedValue, arefTag))
              arefTag++;
          }
        }
      }

      schedule->serialize(loop);
    }
  }
};

} // namespace triton
} // namespace mlir
