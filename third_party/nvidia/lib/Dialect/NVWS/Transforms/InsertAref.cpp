#include "Utilities.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PartitionBuilder.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
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

bool isDescLoadAndAlloc(Value result) {
  auto alloc = result.getDefiningOp<LocalAllocOp>();
  if (!alloc)
    return false;
  return alloc.getSrc().getDefiningOp<triton::DescriptorOpInterface>() !=
         nullptr;
}

bool isGlobalLoadAndAlloc(Value result) {
  auto alloc = result.getDefiningOp<LocalAllocOp>();
  if (!alloc)
    return false;
  return alloc.getSrc().getDefiningOp<triton::LoadOp>() != nullptr;
}

SmallVector<ProducedValueInfo> getProducedValues(Operation *op, Block *loopBody,
                                                 WarpSchedule &schedule) {
  SmallVector<ProducedValueInfo> producedValues;
  auto partition = schedule.getPartition(loopBody->findAncestorOpInBlock(*op));

  if (partition != schedule.getRootPartition()) {
    for (auto result : op->getResults()) {
      // Only handles load ops for now.
      if (isa<triton::DescriptorOpInterface>(result.getDefiningOp()) ||
          isDescLoadAndAlloc(result)) {
        producedValues.push_back({partition, result});
      }
    }
  }

  return producedValues;
};

ArefCreateOp createAref(OpBuilder &builder, ProducedValueInfo &producedValue) {
  auto result = producedValue.result;
  MemDescType arefBufType;

  if (auto memDescType = dyn_cast<MemDescType>(result.getType())) {
    arefBufType = getMultiBufferedType(memDescType, 1);
  } else if (auto tensorType = dyn_cast<RankedTensorType>(result.getType())) {
    // if result is a value, create memdesctype for location where value will
    // be stored
    MemDescType memDescType;
    Attribute SharedMemorySpace =
        SharedMemorySpaceAttr::get(tensorType.getContext());
    if (auto load = result.getDefiningOp<triton::DescriptorOpInterface>()) {
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
    arefBufType = getMultiBufferedType(memDescType, 1);
  } else {
    llvm_unreachable("unsupported type");
  }

  assert(arefBufType &&
         (isa<SharedMemorySpaceAttr>(arefBufType.getMemorySpace())));
  auto loc = result.getLoc();
  auto alloc = triton::nvws::createAlloc(builder, loc, arefBufType, Value());
  alloc->setAttr("aref_buffer", builder.getUnitAttr());
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

Value mkConstant(OpBuilder &builder, Location loc, int value, int width,
                 Partition *partition, WarpSchedule &schedule) {
  auto constValue = builder.create<arith::ConstantIntOp>(loc, value, width);
  schedule.insert(partition, constValue);
  return constValue;
}

StageCluster getStageClusterForProducer(Value producedValue) {
  if (isDescLoadAndAlloc(producedValue) ||
      isGlobalLoadAndAlloc(producedValue)) {
    auto alloc = producedValue.getDefiningOp<LocalAllocOp>();
    auto loadOp = alloc.getSrc().getDefiningOp();
    return getStageCluster(loadOp);
  }
  return getStageCluster(producedValue.getDefiningOp());
}

SmallVector<Operation *> createArefPut(PartitionBuilder &builder,
                                       ArefCreateOp aref, std::string arefTag,
                                       ProducedValueInfo producedValue,
                                       Partition *producerPartition,
                                       WarpSchedule &schedule) {
  auto loc = producedValue.result.getLoc();
  auto arefBufType = cast<MemDescType>(aref.getBuffers()[0].getType());
  Value result = producedValue.result;
  auto dataBufType = getBufferViewType(arefBufType, true);
  StageCluster stageCluster = getStageClusterForProducer(result);

  SmallVector<Type> buffers{dataBufType};

  auto putEnterOp = builder.createInto<ArefPutEnterOp>(
      *producerPartition, stageCluster, buffers, aref,
      mkConstant(builder, loc, 0, 32, producerPartition, schedule));
  schedule.insert(producerPartition, putEnterOp);
  // Attach a "tag" to each put enter / exit pair, to easily identify them
  // as a matching pair in later analysis.
  putEnterOp->setAttr(kArefTagAttrName, builder.getStringAttr(arefTag));
  auto dataBuf = putEnterOp.getResults()[0];

  auto producerKind = AsyncOp::NONE;
  SmallVector<Operation *> staleOps;
  if (isDescLoadAndAlloc(result)) {
    auto alloc = result.getDefiningOp<LocalAllocOp>();
    auto descOp = alloc.getSrc().getDefiningOp();
    createNVWSDescriptorLoadOp(builder, descOp, dataBuf, producerPartition,
                               schedule, loc);
    producerKind = AsyncOp::TMALoad;
    staleOps.push_back(alloc);
    staleOps.push_back(descOp);
  } else if (isGlobalLoadAndAlloc(result)) {
    llvm_unreachable("cpasync not supported yet");
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
    llvm_unreachable("unsupported type");
  }

  auto putExitOp = builder.createInto<ArefPutExitOp>(
      *producerPartition, stageCluster, aref,
      mkConstant(builder, loc, 0, 32, producerPartition, schedule),
      builder.getArrayAttr(SmallVector<Attribute>{
          AsyncOpAttr::get(aref.getContext(), producerKind)}));
  putExitOp->setAttr(kArefTagAttrName, builder.getStringAttr(arefTag));
  schedule.insert(producerPartition, putExitOp);

  return staleOps;
};

SetVector<Operation *> getTransitiveConsumers(Operation *op) {
  SetVector<Operation *> opConsumers;
  auto isMemDesc = [](auto res) { return isa<MemDescType>(res.getType()); };
  for (auto user : op->getUsers()) {
    if (llvm::count_if(user->getResults(), isMemDesc) > 0) {
      // Recurse into consumers of memdesc ops, since the liveness of the
      // produced value extends beyond such ops.
      auto consumers = getTransitiveConsumers(user);
      opConsumers.insert(consumers.begin(), consumers.end());
    } else {
      opConsumers.insert(user);
    }
  }
  return opConsumers;
}

SmallVector<Attribute>
getConsumerAsyncOpKinds(const SetVector<Operation *> &consumers,
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

void propagateAllocShape(Value value, int64_t arefAllocDepth) {
  for (Operation *user : value.getUsers()) {
    if (user->hasTrait<OpTrait::MemDescViewTrait>()) {
      auto type = cast<MemDescType>(user->getResult(0).getType());
      SmallVector<int64_t> allocShape{type.getShape()};
      allocShape.insert(allocShape.begin(), arefAllocDepth);
      auto newType = MemDescType::get(type.getShape(), type.getElementType(),
                                      type.getEncoding(), type.getMemorySpace(),
                                      type.getMutableMemory(), allocShape);
      user->getResult(0).setType(newType);
      propagateAllocShape(user->getResult(0), arefAllocDepth);
    }
  }
}

void createArefGet(PartitionBuilder &builder, ArefCreateOp aref,
                   std::string arefTag, ProducedValueInfo producedValue,
                   Partition *consumerPartition, WarpSchedule &schedule,
                   PostDominanceInfo &postDomInfo) {
  auto loc = producedValue.result.getLoc();
  auto arefBufType = cast<MemDescType>(aref.getOperand(0).getType());

  Value result = producedValue.result;
  StageCluster stageCluster = getStageCluster(result.getDefiningOp());

  Type bufferType = getBufferViewType(arefBufType, false);
  auto getEnterOp = builder.createInto<ArefGetEnterOp>(
      *consumerPartition, stageCluster, SmallVector{bufferType}, aref,
      mkConstant(builder, loc, 0, 32, consumerPartition, schedule));
  Value dataBuf = getEnterOp.getResults()[0];
  schedule.insert(consumerPartition, getEnterOp);
  getEnterOp->setAttr(kArefTagAttrName, builder.getStringAttr(arefTag));

  auto consumers = getTransitiveConsumers(result.getDefiningOp());
  auto asyncKinds = getConsumerAsyncOpKinds(consumers, aref.getContext());

  if (auto memDescType = dyn_cast<MemDescType>(result.getType())) {
    auto insertPoint = *llvm::min_element(consumers, [&](auto &lhs, auto &rhs) {
      return postDomInfo.postDominates(lhs, rhs);
    });
    builder.setInsertionPointAfter(insertPoint);
    result.replaceAllUsesWith(dataBuf);
    propagateAllocShape(dataBuf, arefBufType.getAllocShape()[0]);
  } else if (auto tensorType = dyn_cast<RankedTensorType>(result.getType())) {
    auto localLoadOp = builder.create<LocalLoadOp>(loc, tensorType, dataBuf);
    result.replaceAllUsesWith(localLoadOp.getResult());
    schedule.insert(consumerPartition, localLoadOp);
  } else {
    llvm_unreachable("unsupported type");
  }

  auto getExitOp = builder.createInto<ArefGetExitOp>(
      *consumerPartition, stageCluster, aref,
      mkConstant(builder, loc, 0, 32, consumerPartition, schedule),
      builder.getArrayAttr(asyncKinds));

  getExitOp->setAttr(kArefTagAttrName, builder.getStringAttr(arefTag));
  schedule.insert(consumerPartition, getExitOp);

  for (auto consumer : consumers) {
    if (auto mmav5 = dyn_cast<MMAv5OpInterface>(consumer)) {
      // MMAv5 now works with SMEM buffers obtained locally from ArefGeterOp. It
      // can be now considered as asynchronous - it starts executing when the
      // SMEM buffers are ready, and when it finishes, ArefGetExitOp will
      // release the empty barrier, which unblocks the load partition.
      mmav5.setIsAsync(true);
    }
  }
};

bool insertArefs(PartitionBuilder &builder, scf::ForOp loop,
                 WarpSchedule &schedule, ProducedValueInfo producedValue,
                 int arefTag) {
  Partition *consumerPartition = nullptr;
  auto [producerPartition, result] = producedValue;
  for (auto &use : result.getUses()) {
    Partition *userPartition = schedule.getPartition((use.getOwner()));
    if (producerPartition != userPartition) {
      consumerPartition = userPartition;
      break;
    }
  }

  if (!consumerPartition) {
    return false;
  }

  ArefCreateOp aref;
  {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPoint(loop);
    aref = createAref(builder, producedValue);
  }

  auto tag = "aref_" + std::to_string(arefTag);
  auto staleOps = createArefPut(builder, aref, tag, producedValue,
                                producerPartition, schedule);

  PostDominanceInfo postDomInfo(loop);
  createArefGet(builder, aref, tag, producedValue, consumerPartition, schedule,
                postDomInfo);

  for (auto op : staleOps) {
    op->erase();
  }

  return true;
}

} // namespace

class NVWSArefInsertion : public impl::NVWSInsertArefBase<NVWSArefInsertion> {
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

      SmallVector<Operation *> ops;
      loop.walk([&](Operation *op) {
        if (!isa<TMEMAllocOp, TMEMLoadOp, TMEMStoreOp, scf::YieldOp,
                 triton::FuncOp, triton::ReturnOp, scf::ForOp, scf::IfOp>(op)) {
          ops.push_back(op);
        }
      });

      int arefTag = 0;

      for (auto op : ops) {
        auto producedValues = getProducedValues(op, loop.getBody(), *schedule);
        for (auto producedValue : producedValues) {
          PartitionBuilder builder(op->getLoc(), op);
          builder.setInsertionPointAfter(op);
          if (insertArefs(builder, loop, *schedule, producedValue, arefTag))
            arefTag++;
        }
      }

      schedule->serialize(loop);
    }
  }
};

} // namespace triton
} // namespace mlir
