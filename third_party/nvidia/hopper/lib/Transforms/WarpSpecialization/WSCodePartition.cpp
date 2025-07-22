#include "CodePartitionUtility.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include <unordered_set>

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace ttnvws = ::mlir::triton::nvws;
namespace mlir {

#define DEBUG_TYPE "nvgpu-ws-code-partition"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

static unsigned getNumBuffersOrDefault(scf::ForOp forOp, unsigned numBuffers) {
  // Use the attribute attached to the loop if it exists otherwise use the
  // global control.
  if (!forOp->hasAttr(mlir::triton::kNumStagesAttrName))
    return numBuffers;
  return mlir::cast<IntegerAttr>(
             forOp->getAttr(mlir::triton::kNumStagesAttrName))
      .getInt();
}

// Get the bufferIdx and phase for the last iteration of the immediate scope.
std::pair<Value, Value> getOutOfScopeBufferIdxAndPhase(
    OpBuilderWithAsyncTaskIds &builder, Operation *op, unsigned numBuffers,
    const DenseSet<Operation *> &regionsWithChannels) {
  // Get the current in-scope accumulation count for op.
  Value accumCnt = getAccumCount(builder, op, regionsWithChannels);

  // Get the out-of-scope accumulation count.
  assert(isa<BlockArgument>(accumCnt) &&
         "Expected accumCnt to be a block argument");
  auto bbArg = dyn_cast<BlockArgument>(accumCnt);
  Operation *bbAargOwner = bbArg.getOwner()->getParentOp();
  if (auto forOp = dyn_cast<scf::ForOp>(bbAargOwner)) {
    accumCnt = forOp.getResult(bbArg.getArgNumber() - 1);
  } else {
    llvm_unreachable("Unexpected block argument owner");
  }

  // The accumulation count is one past the last iteration. Subtract one to get
  // the last valid iteration index.
  auto loc = bbAargOwner->getLoc();
  Value one = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 64);
  accumCnt = builder.createWithAsyncTaskIds<arith::SubIOp>(loc, accumCnt, one);

  return getBufferIdxAndPhase(builder, op->getLoc(), accumCnt, numBuffers);
}

// Find transitive users of the root op. Track through control flow ops (such as
// yield) to get to the real users.
void getTransitiveUsers(Value root,
                        SetVector<std::pair<Operation *, unsigned>> &users) {
  for (Operation *userOp : root.getUsers()) {
    if (auto yieldOp = dyn_cast<scf::YieldOp>(userOp)) {
      for (OpOperand &operand : yieldOp->getOpOperands()) {
        if (operand.get() == root) {
          auto result =
              yieldOp->getParentOp()->getResult(operand.getOperandNumber());
          getTransitiveUsers(result, users);
        }
      }
    } else {
      // find operand index of root
      unsigned operandIndex = 0;
      for (OpOperand &operand : userOp->getOpOperands()) {
        if (operand.get() == root) {
          break;
        }
        operandIndex++;
      }
      assert(operandIndex < userOp->getNumOperands() &&
             "root is not an operand of userOp");
      users.insert({userOp, operandIndex});
    }
  }
}

// When traversing gen5, producerOp can be either the defining op of operand
// A or the accumulator.
static void createChannel(Operation *producerOp, Operation *op,
                          mlir::DominanceInfo &dom,
                          SmallVector<std::unique_ptr<Channel>> &channels,
                          bool opndAOfGen5, unsigned producerNumBuffers) {
  // For TMEM channels, op is Gen5 op, producerOp can be either A operand
  // or accumulator.
  auto producerTaskIds = getAsyncTaskIds(opndAOfGen5 ? producerOp : op);
  auto producerTaskId = producerTaskIds.front();
  for (auto result : producerOp->getResults()) {
    if (result.use_empty()) {
      continue;
    }

    SetVector<std::pair<Operation *, unsigned>> users;
    getTransitiveUsers(result, users);
    for (auto user : users) {
      auto userOp = user.first;
      if (op == userOp && !opndAOfGen5)
        continue;
      // rule out users that are not dominated by op
      if (op->getBlock() != userOp->getBlock()) {
        if (!dom.properlyDominates(op->getParentOp(), userOp)) {
          continue;
        }
      } else {
        if (!dom.properlyDominates(op, userOp) && op != userOp)
          continue;
      }

      auto consumerTaskIds = getAsyncTaskIds(userOp);
      if (consumerTaskIds.empty())
        continue;
      // Remove producer task id from consumerTaskIds.
      auto iter = std::remove(consumerTaskIds.begin(), consumerTaskIds.end(),
                              producerTaskId);
      consumerTaskIds.erase(iter, consumerTaskIds.end());

      const unsigned NUM_TMEM_BUFFERS = 2;
      // Add a channel from the single producer task to consumerTaskIds.
      if (consumerTaskIds.size() > 0) {
        if (auto dotOp = dyn_cast<ttng::TCGen5MMAOp>(op)) {
          // When traversing Gen5MMA, we create channel for the accumulator.
          if (auto tmemAllocOp = dyn_cast<ttng::TMEMAllocOp>(producerOp)) {
            // Always use two buffers for TMEM channels.
            channels.push_back(std::make_unique<ttng::TmemDataChannel>(
                producerTaskId, consumerTaskIds, tmemAllocOp, dotOp, userOp,
                user.second, NUM_TMEM_BUFFERS));
          }
        } else {
          channels.push_back(
              std::make_unique<Channel>(producerTaskId, consumerTaskIds, userOp,
                                        user.second, producerNumBuffers));
        }
      }
    }
  }
}

// Loads will be in producer warp groups. For now, we only allow a single
// warp group/task for a producer. For each LoadOp, create a channel from it
// to any direct user which belongs to a different taskId.
void collectAsyncChannels(SmallVector<std::unique_ptr<Channel>> &channels,
                          triton::FuncOp &funcOp, unsigned numBuffers) {
  mlir::DominanceInfo dom(funcOp);
  funcOp.walk([&](Operation *op) {
    // FIXME: It is possible that a local_alloc can start a channel, when a
    // gemm's operand is in smem and comes from local_alloc.
    if (isa<tt::LoadOp, tt::DescriptorLoadOp>(op) ||
        isa<mlir::triton::DotOpInterface>(op)) {
      auto producerTaskIds = getAsyncTaskIds(op);
      if (producerTaskIds.empty() || producerTaskIds.size() > 1) {
        LLVM_DEBUG({
          LDBG(" ignoring load ops without async task id or with multiple task "
               "ids: ");
          op->dump();
        });
        return;
      }
      auto producerTaskId = producerTaskIds.front();
      unsigned producerNumBuffers = numBuffers;
      if (auto forOp = op->getParentOfType<scf::ForOp>()) {
        producerNumBuffers = getNumBuffersOrDefault(forOp, numBuffers);
      }

      auto producerOp = op;
      if (auto dotOp = dyn_cast<ttng::TCGen5MMAOp>(op)) {
        auto accumulator = dotOp.getD();
        producerOp = accumulator.getDefiningOp();
        createChannel(producerOp, op, dom, channels, false, producerNumBuffers);
        // We may need to create a TMEM channel for A operand.
        auto opndA = dotOp.getA();
        producerOp = opndA.getDefiningOp();
        if (isa<ttng::TMEMAllocOp>(producerOp))
          createChannel(producerOp, op, dom, channels, true /*opndA*/,
                        producerNumBuffers);
      } else {
        createChannel(producerOp, op, dom, channels, false, producerNumBuffers);
      }
    }
  });

  LLVM_DEBUG({
    LDBG("Async channels:");
    for (auto &channel : channels) {
      LDBG("producer op: " << channel->relation.first);
      channel->getSrcOp()->dump();
      for (auto &asyncTaskId : channel->relation.second)
        LDBG("consumer: " << asyncTaskId);
      channel->getDstOp()->dump();
      LDBG("numBuffers: " << channel->numBuffers);
    }
  });
}

// When the consumer is a local_alloc loading from shared memory to registers,
// look ahead for the actual consumers, usually dot ops, that can directly
// use shared memory. The local_alloc will be removed later.
static SmallVector<Operation *> getActualConsumers(Operation *consumerOp) {
  if (isa<ttg::LocalAllocOp>(consumerOp)) {
    DenseSet<Operation *> users;
    for (auto user : consumerOp->getUsers()) {
      if (isa<tt::TransOp, ttg::MemDescTransOp>(user)) {
        // TransOp is not a real consumer. It caculates the shared memory
        // address for the real consumer. Continue to find its transitive users
        // recursively.
        DenseSet<Operation *> visited;
        SmallVector<Operation *> transUsers;
        transUsers.push_back(user);
        while (!transUsers.empty()) {
          auto transUser = transUsers.pop_back_val();
          visited.insert(transUser);
          if (isa<tt::TransOp, ttg::MemDescTransOp>(transUser)) {
            for (auto transitiveUser : transUser->getUsers()) {
              if (!visited.count(transitiveUser))
                transUsers.push_back(transitiveUser);
            }
          } else {
            users.insert(transUser);
          }
        }
      } else {
        users.insert(user);
      }
    }

    return SmallVector<Operation *>(users.begin(), users.end());
  }
  return {consumerOp};
}

static Operation *getUniqueActualConsumer(Operation *consumerOp) {
  auto consumers = getActualConsumers(consumerOp);
  return consumers.size() == 1 ? consumers[0] : consumerOp;
}

static Operation *getUniqueActualConsumer(Operation *consumerOp,
                                          AsyncTaskId taskId) {
  auto consumers = getActualConsumers(consumerOp);
  if (consumers.size() == 1)
    return consumers[0];
  // Check to see if there is only one consumer with the specific taskId.
  Operation *uniqOp = nullptr;
  for (auto *op : consumers) {
    SmallVector<AsyncTaskId> asyncTasks = getAsyncTaskIds(op);
    assert(asyncTasks.size() > 0);
    if (asyncTasks.size() > 1)
      return consumerOp;
    if (asyncTasks[0] == taskId) {
      if (uniqOp)
        return consumerOp;
      uniqOp = op;
    }
  }
  return uniqOp ? uniqOp : consumerOp;
}

// Group channels in two ways:
//  - by producer ops. One producer corresponds to multiple channels. This
//    grouping will be used to create buffers per shared producer.
//  - by consumer ops. One consumer corresponds to multiple channels. This
//  grouping will be used to create barriers per shared consumer.
// Also compute orderedChannels, which will be keyed by getDstOp() of channels,
// to enforce deterministic order for map.
void groupChannels(
    SmallVector<Channel *> &channels,
    DenseMap<Channel *, SmallVector<Channel *>> &channelsGroupedByProducers,
    DenseMap<Channel *, SmallVector<Channel *>> &channelsGroupedByConsumers,
    SmallVector<Channel *> &orderedChannels) {

  // Group channels by producer op.
  DenseMap<Operation *, SmallVector<Channel *>> producerChannels;
  for (auto channel : channels) {
    producerChannels[channel->getSrcOp()].push_back(channel);
  }

#ifndef NDEBUG
  // Some sanity checks.
  for (auto &item : producerChannels) {
    auto &channels = item.second;
    unsigned numBuffers = channels.front()->numBuffers;
    for (auto c : channels) {
      assert(c->numBuffers == numBuffers && "Unmatched number of buffers");
    }
  }
#endif

  // Group channels by consumer op.
  DenseMap<Operation *, SmallVector<Channel *>> consumerChannels;

  // Two channels can be combined if
  //   src1 and src2 are in the same block and
  //   (dst1 == dst2 or
  //    (dst1 and dst2 are in the same block, both have a single user, and
  //     dst1User == dst2User and dst1User is in the same block as dst1))
  auto channelCanBeMerged = [](Channel *c1, Channel *c2) -> bool {
    if (c1->getSrcOp()->getBlock() != c2->getSrcOp()->getBlock())
      return false;
    Operation *dst1 = c1->getDstOp(), *dst2 = c2->getDstOp();
    if (dst1 == dst2)
      return true;
    // We only have one CommChannel for channels in channelsGroupedByConsumers.
    // A CommChannel can have multiple tokens, one for each consumer taskId.
    // Consider the case where channel v is between producer
    // task 0 and consumer task 1, while channel p is between producer task 2
    // and consumer task 1, but in createToken, we only consider the first
    // channel in the group.
    if (getAsyncTaskIds(c1->getSrcOp()) != getAsyncTaskIds(c2->getSrcOp()))
      return false;
    // Check taskIds on dstOps.
    if (getAsyncTaskIds(dst1) != getAsyncTaskIds(dst2))
      return false;
    auto dst1User = getUniqueActualConsumer(dst1);
    auto dst2User = getUniqueActualConsumer(dst2);
    if (!dst1User || !dst2User)
      return false;
    return dst1User == dst2User && dst1User->getBlock() == dst1->getBlock();
  };
  assert(channels.size() > 0 && "channel size is zero");
  // Compare with existing channels in the consumerChannels to see if
  // it can be combined.
  for (auto *c0 : channels) {
    bool merged = false;
    for (auto &kv : consumerChannels) {
      if (kv.second.size() > 0 && channelCanBeMerged(c0, kv.second.front())) {
        kv.second.push_back(c0);
        merged = true;
        break;
      }
    }
    if (!merged) { // Create a new entry.
      auto *keyOp = c0->getDstOp();
      if (!consumerChannels.count(keyOp))
        orderedChannels.push_back(c0);
      consumerChannels[keyOp].push_back(c0);
    }
  }

  // Reorder channels associated with one entry based on program order of the
  // producers.
  for (auto &kv : consumerChannels) {
    if (kv.second.size() > 1) {
      auto &allOps = kv.second.front()->getSrcOp()->getBlock()->getOperations();
      std::sort(
          kv.second.begin(), kv.second.end(), [&](Channel *a, Channel *b) {
            auto itrA =
                std::find_if(allOps.begin(), allOps.end(), [&](Operation &op) {
                  Operation *opPointer = &op;
                  return opPointer == a->getSrcOp();
                });
            auto itrB =
                std::find_if(allOps.begin(), allOps.end(), [&](Operation &op) {
                  Operation *opPointer = &op;
                  return opPointer == b->getSrcOp();
                });
            assert(itrA != allOps.end() && itrB != allOps.end());
            return std::distance(itrA, itrB) < 0;
          });
    }
  }

  // Switch to using channel as the key instead of ops as ops can be volatile.
  for (auto &kv : producerChannels) {
    channelsGroupedByProducers[kv.second.front()] = kv.second;
  }
  for (auto &kv : consumerChannels) {
    channelsGroupedByConsumers[kv.second.front()] = kv.second;
  }

  LLVM_DEBUG({
    DBGS() << "\n\n";
    LDBG("Grouped channels by producer:");
    unsigned i = 0;
    for (auto &kv : channelsGroupedByProducers) {
      DBGS() << "Channel  " << ++i << ":\n";
      DBGS() << "producer:  ";
      kv.getFirst()->getSrcOp()->dump();
      for (auto &channel : kv.second) {
        DBGS() << "consumer: ";
        channel->getDstOp()->dump();
        DBGS() << "] ";
        LDBG("numBuffers: " << channel->numBuffers);
        DBGS() << "\n";
      }
    }

    DBGS() << "\n\n";
    LDBG("Grouped channels by consumer:");
    i = 0;
    for (auto &kv : channelsGroupedByConsumers) {
      DBGS() << "Channel  " << ++i << ":\n";
      DBGS() << "consumer:  ";
      kv.getFirst()->getDstOp()->dump();
      for (auto &channel : kv.second) {
        DBGS() << "producer: ";
        channel->getSrcOp()->dump();
        for (auto &asyncTaskId : channel->relation.second)
          DBGS() << asyncTaskId << ", ";
        DBGS() << "] ";
        LDBG("numBuffers: " << channel->numBuffers);
        DBGS() << "\n";
      }
      DBGS() << "\n";
    }
  });
}

// Reorder producer ops to unblock consumers interleavingly.
void reorderProducerOps(SmallVector<Channel *> &channels) {
  if (channels.size() <= 1)
    return;

  // Bail out if channels are not in the same block
  auto block = channels.front()->getSrcOp()->getBlock();
  for (auto &channel : channels) {
    if (channel->getSrcOp()->getBlock() != block) {
      return;
    }
  }

  // Group channels by the first consumer taskId of each channel. Smaller taskId
  // has higher priority.
  // TODO: consider consumer priority
  std::map<AsyncTaskId, SmallVector<Channel *>> groupedProducerOps;
  for (auto &channel : channels) {
    auto asyncTaskId = channel->relation.second.front();
    groupedProducerOps[asyncTaskId].push_back(channel);
  }

  // No need to reorder if all channels are in the same group.
  if (groupedProducerOps.size() <= 1)
    return;

  // Sort each group by number of consumers.
  for (auto &group : groupedProducerOps) {
    std::sort(group.second.begin(), group.second.end(),
              [&](Channel *a, Channel *b) {
                return a->relation.second.size() < b->relation.second.size();
              });
  }

  // Start from the first producer in channels. Iterate through the groups
  // which are ordered by the first consumer taskId. Within each group, channels
  // are ordered by number of consumers.
  Operation *currOp = channels.front()->getSrcOp();
  for (auto &group : groupedProducerOps) {
    for (auto &channel : group.second) {
      channel->getSrcOp()->moveAfter(currOp);
      currOp = channel->getSrcOp();
    }
  }

  // Move backward dependency slice close to producer ops.
  // Start from the last producer op backwards and move backward slice to
  // before each op. This guarantees that the backward slice of each op is
  // scheduled as late as possible.
  for (auto &group : reverse(groupedProducerOps)) {
    for (auto &channel : reverse(group.second)) {
      BackwardSliceOptions opt;
      opt.omitBlockArguments = true;
      SetVector<Operation *> backwardSlice;
      (void)getBackwardSlice(channel->getSrcOp(), &backwardSlice, opt);
      for (auto &op : backwardSlice) {
        if (op->getBlock() == block)
          op->moveBefore(channel->getSrcOp());
      }
    }
  }

  LLVM_DEBUG({
    LDBG("\n");
    LDBG("after reordering producer ops");
    currOp->getParentOfType<triton::FuncOp>().dump();
    LDBG("\n");
  });
}

// Find top-level ops which contain at least one channel. If a channel's
// getSrcOp() and getDstOp() belong to the inner loop, the outer loop will be
// part of asyncTaskOps.
SmallVector<Operation *>
getTaskTopRegion(triton::FuncOp funcOp,
                 const SmallVector<Channel *> &channels) {
  SmallVector<Operation *> asyncTaskOps;
  auto isAsyncTaskTopOp = [&](Operation *taskTopOp) -> bool {
    for (auto c : channels) {
      Operation *producer = c->getSrcOp(), *consumer = c->getDstOp();
      while (producer && !isa<triton::FuncOp>(producer->getParentOp())) {
        producer = producer->getParentOp();
      }
      while (consumer && !isa<triton::FuncOp>(consumer->getParentOp())) {
        consumer = consumer->getParentOp();
      }
      if (producer == taskTopOp && consumer == taskTopOp)
        return true;
    }
    return false;
  };
  for (auto &block : funcOp.getBody().getBlocks()) {
    for (Operation &bodyOp : block.getOperations()) {
      Operation *op = &bodyOp;
      if (op->getNumRegions() <= 0)
        continue;
      // If this op does not contain both a producer taskId and a consumer
      // taskId, continue.
      if (getAsyncTaskIds(op).size() == 1)
        continue;
      if (isAsyncTaskTopOp(op))
        asyncTaskOps.push_back(op);
    }
  }

  LLVM_DEBUG({
    LDBG("\nTop Task Bodies");
    for (auto op : asyncTaskOps) {
      LDBG("\nTask Body:");
      op->dump();
    }
  });
  return asyncTaskOps;
}

// Create an allocation to hold the mbarriers.
static Value createBarrierAlloc(triton::FuncOp funcOp, unsigned distance) {
  OpBuilder builder(funcOp);
  builder.setInsertionPointToStart(&(funcOp.getBody().front()));
  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(funcOp.getContext());
  Location loc = funcOp.getLoc();
  auto context = funcOp.getContext();
  auto barrierCTALayout =
      ttg::CTALayoutAttr::get(context, /*CTAsPerCGA=*/{1},
                              /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
  auto barrierEncoding = ttg::SwizzledSharedEncodingAttr::get(
      context, 1, 1, 1, {0}, barrierCTALayout);
  Type barrierMemDescType = ttg::MemDescType::get(
      {distance}, builder.getI64Type(), barrierEncoding, sharedMemorySpace,
      /*mutableMemory=*/true);
  Type singleBarrierMemDescType =
      ttg::MemDescType::get({1}, builder.getI64Type(), barrierEncoding,
                            sharedMemorySpace, /*mutableMemory=*/true);
  Value barrierAlloc = builder.create<mlir::triton::gpu::LocalAllocOp>(
      loc, barrierMemDescType, Value());
  for (unsigned i = 0; i < distance; i++) {
    Value idx = builder.create<arith::ConstantIntOp>(loc, i, 32);
    Value barrierView = builder.create<ttg::MemDescSubviewOp>(
        loc, singleBarrierMemDescType, barrierAlloc, idx);
    builder.create<ttng::InitBarrierOp>(funcOp->getLoc(), barrierView, 1);
  }
  return barrierAlloc;
}

// channelsGroupedByConsumers: channels are grouped together.
// Go through each group, check the first channel in the group, create a token
// for each consumer taskId. Return a map that maps each channel + consumer
// taskId to a token. Also update barrierAllocMap that maps each channel +
// consumer taskId to a BarrierAlloc.
void createToken(
    const DenseMap<Channel *, SmallVector<Channel *>>
        &channelsGroupedByConsumers,
    const SmallVector<Channel *> &orderedChannels, triton::FuncOp funcOp,
    const DenseMap<Channel *, std::pair<Operation *, Operation *>> &copyOpMap,
    DenseMap<Channel *, CommChannel> &tokenMap) {
  OpBuilder builder(funcOp);
  builder.setInsertionPointToStart(&(funcOp.getBody().front()));
  DenseMap<ttng::TCGen5MMAOp, Channel *> gen5Barriers;
  for (auto *key : orderedChannels) {
    auto it = channelsGroupedByConsumers.find(key);
    Channel *channel = it->second.front();

    CommChannel commChannel;
    auto producerOp = it->second.front()->getSrcOp();
    auto dstOp = it->second.front()->getDstOp();

    if (isa<tt::DescriptorLoadOp>(producerOp)) {
      commChannel.producerBarrier =
          createBarrierAlloc(funcOp, channel->numBuffers);
    }

    for (auto consumerAsyncTaskId : channel->relation.second) {
      // It is possible that this channel has two consumer taskIds.
      Operation *consumerOp =
          getUniqueActualConsumer(dstOp, consumerAsyncTaskId);

      // For channels associated with acc of gen5, consumerOp is not the gen5,
      // it is usually tmem_load.
      bool useGen5Barrier = isa<ttng::TCGen5MMAOp>(consumerOp) &&
                            producerOp->getBlock() == consumerOp->getBlock();
      LLVM_DEBUG({
        LDBG("-- creatToken: useGen5Barrier = " << useGen5Barrier);
        consumerOp->dump();
      });
      if (useGen5Barrier) {
        auto mmaOp = cast<ttng::TCGen5MMAOp>(consumerOp);
        // If the gen5 barrier for this mmaOp is already used for another
        // channel, do not use it for this channel.
        if (gen5Barriers.count(mmaOp) && gen5Barriers[mmaOp] != channel) {
          useGen5Barrier = false;
          LDBG("-- mmaOp already has a channel associated");
        }
      }

      // No token is needed for a TMA <-> TCGen5MMAOp channel
      if (!isa<tt::DescriptorLoadOp>(producerOp) ||
          !useGen5Barrier) { // isa<nvidia_gpu::TCGen5MMAOp>(consumerOp)) {
        ttnvws::TokenLoadType tokenLoadType;
        auto copyOp = copyOpMap.find(channel)->second.first;
        if (isa<ttg::AsyncCopyGlobalToLocalOp>(copyOp)) {
          tokenLoadType = ttnvws::TokenLoadType::AsyncLoadOp;
        } else if (isa<tt::DescriptorLoadOp>(copyOp)) {
          tokenLoadType = ttnvws::TokenLoadType::TMALoadOp;
        } else if (isa<ttg::LocalStoreOp>(copyOp)) {
          tokenLoadType = ttnvws::TokenLoadType::LocalStoreOp;
        } else if (isa<ttng::TMEMLoadOp>(consumerOp)) {
          tokenLoadType = ttnvws::TokenLoadType::TmemLoadOp;
        } else if (isa<ttng::TCGen5MMAOp>(consumerOp)) {
          // For operand A of gen5, we have tmem_store + gen5.
          tokenLoadType = ttnvws::TokenLoadType::TmemLoadOp;
        } else {
          llvm_unreachable("Unexpected load type");
        }
        Value v;
        if (it->second.front()->getSrcOp()->getParentOfType<scf::ForOp>())
          v = builder.create<ttnvws::CreateTokenOp>(
              funcOp.getLoc(), channel->numBuffers, tokenLoadType);
        else
          v = builder.create<ttnvws::CreateTokenOp>(funcOp.getLoc(), 1,
                                                    tokenLoadType);
        commChannel.tokens[consumerAsyncTaskId] = v;
      }

      if (useGen5Barrier) {
        Value v = createBarrierAlloc(funcOp, channel->numBuffers);
        commChannel.consumerBarriers[consumerAsyncTaskId] = v;
        gen5Barriers[cast<ttng::TCGen5MMAOp>(consumerOp)] = channel;
      }
    }

    // Channels in the group share the same set of tokens.
    for (auto &c : it->second) {
      tokenMap[c] = commChannel;
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Communication Channels: \n";
    for (auto &item : tokenMap) {
      llvm::dbgs() << "\ndata channel: \n";
      llvm::dbgs() << *item.first->getSrcOp() << "\n";
      llvm::dbgs() << *item.first->getDstOp() << "\n";
      llvm::dbgs() << "communication channel: \n";
      for (auto &kv : item.second.tokens) {
        llvm::dbgs() << "token: " << kv.first << " " << kv.second << "\n";
      }
      if (item.second.producerBarrier)
        llvm::dbgs() << "producer barrier: " << *item.second.producerBarrier
                     << "\n";
      for (auto &kv : item.second.consumerBarriers)
        llvm::dbgs() << "consumer barrier: " << kv.first << " " << kv.second
                     << "\n";
    }
  });
}

static ttng::TMEMAllocOp createTMemAlloc(OpBuilder &builder,
                                         ttng::TMEMAllocOp oldTMemAllocOp,
                                         int numBuffers) {
  Location loc = oldTMemAllocOp.getLoc();
  auto oldRetType = oldTMemAllocOp.getType();
  SmallVector<int64_t> shape = {oldRetType.getShape().begin(),
                                oldRetType.getShape().end()};
  // We can still use subView in createTMEMCopy even if numBuffers is 1.
  if (numBuffers >= 1) {
    shape.insert(shape.begin(), numBuffers);
  }
  Type accMemDescType = triton::gpu::MemDescType::get(
      shape, oldRetType.getElementType(), oldRetType.getEncoding(),
      oldRetType.getMemorySpace(), /*mutableMemory=*/true);
  return builder.create<ttng::TMEMAllocOp>(oldTMemAllocOp.getLoc(),
                                           accMemDescType, nullptr);
}

// Create a buffer array for each producer op, if the producer is in a ForOp,
// the buffer array will contain numBuffers.
DenseMap<Channel *, Value> createBuffer(
    DenseMap<Channel *, SmallVector<Channel *>> &channelsGroupedByProducers,
    const SmallVector<Channel *> &orderedChannels, triton::FuncOp funcOp) {

  DenseMap<Channel *, Value> bufferMap;
  MLIRContext *context = funcOp.getContext();
  OpBuilder builder(funcOp);
  builder.setInsertionPointToStart(&(funcOp.getBody().front()));
  DenseSet<Channel *> visited;
  for (auto &item : channelsGroupedByProducers) {
    auto &channels = item.second;
    for (auto c : channels) {
      assert(!visited.count(c));
      visited.insert(c);
    }
  }
  for (auto *channelInOrder : orderedChannels) {
    if (channelsGroupedByProducers.find(channelInOrder) ==
        channelsGroupedByProducers.end())
      continue;
    auto &channels = channelsGroupedByProducers[channelInOrder];
    auto srcValue = channelInOrder->getSrcOperand();
    auto srcOp = channelInOrder->getSrcOp();
    auto dstOp = channelInOrder->getDstOp();
    auto *channel = channels.front();
    unsigned numBuffers = channel->numBuffers;
    Value buffer;

    LLVM_DEBUG({
      LDBG("Creating buffers for channel:");
      LDBG("Producer:");
      DBGS() << *srcOp << "\n";
      LDBG("Consumer:");
      DBGS() << *dstOp << "\n";
    });

    // For TMEM channel, multi-buffer TMEM alloc
    if (channel->channelKind == DataChannelKind::TMEM) {
      // Move TMEM alloc to the beginning of the function.
      ttng::TmemDataChannel *tmemChannel =
          static_cast<ttng::TmemDataChannel *>(channel);
      auto oldTMemAllocOp = tmemChannel->getAllocOp();
      buffer = createTMemAlloc(builder, oldTMemAllocOp, numBuffers);
    } else if (auto tensorType =
                   dyn_cast<RankedTensorType>(srcValue.getType())) {
      // Get basic information from tensorType
      auto order = ttg::getOrderForMemory(tensorType);
      auto CTALayout = ttg::getCTALayout(tensorType.getEncoding());
      auto elemType = tensorType.getElementType();

      // Get shape, layout and type of a slice
      auto sliceShape = tensorType.getShape();
      // Check the consumer type
      auto actualConsumers = getActualConsumers(dstOp);
      LLVM_DEBUG({
        DBGS() << "actual consumers: \n";
        for (auto consumerOp : actualConsumers) {
          DBGS() << *consumerOp << "\n";
        }
      });

      bool requireMMASharedEncoding =
          llvm::any_of(actualConsumers, [](Operation *op) {
            return isa<mlir::triton::DotOpInterface>(op);
          });

      Attribute sharedLayout;
      if (requireMMASharedEncoding) {
        sharedLayout = ttg::NVMMASharedEncodingAttr::get(
            context, sliceShape, order, CTALayout, elemType,
            /*fp4Padded*/ false);
      } else if (auto tmaLoad = dyn_cast<tt::DescriptorLoadOp>(srcOp)) {
        sharedLayout = ttng::getEncodingFromDescriptor(
            tmaLoad, tmaLoad.getType(), tmaLoad.getDesc());
      } else {
        // Create an unswizzled layout for now.
        // TODO: optimize it based on the consumer.
        sharedLayout = ttg::SwizzledSharedEncodingAttr::get(context, 1, 1, 1,
                                                            order, CTALayout);
      }

      // Get shape, layout and type of the complete buffer
      SmallVector<int64_t> bufferShape(sliceShape.begin(), sliceShape.end());
      if (srcOp->getParentOfType<scf::ForOp>())
        bufferShape.insert(bufferShape.begin(), numBuffers);
      else
        bufferShape.insert(bufferShape.begin(), 1);
      Attribute sharedMemorySpace =
          triton::gpu::SharedMemorySpaceAttr::get(context);
      Type memdescType =
          ttg::MemDescType::get(bufferShape, elemType, sharedLayout,
                                sharedMemorySpace, /*mutableMemory*/ true);
      buffer = builder.create<ttg::LocalAllocOp>(funcOp.getLoc(), memdescType);
    } else {
      llvm_unreachable("Unexpected result type");
    }

    // Channels in the group share the same buffer.
    for (auto c : channels)
      bufferMap[c] = buffer;
  }
  return bufferMap;
}

// Make TCGen5MMAOp fully asynchronous by de-synchronizing it. This leverages
// its inline barrier to synchronize with both the producer (TMA load) and the
// consumer (TMEM load). Return the WaitBarrierOp inserted before the consumer
// (TMEM load).
ttng::WaitBarrierOp
desyncTCGen5MMAOp(OpBuilderWithAsyncTaskIds &builder, ttng::TCGen5MMAOp mmaOp,
                  Value barrierAlloc, Value bufferIdx, Value inPhase,
                  unsigned numBuffers, Operation *headProducer,
                  DenseSet<Operation *> &regionsWithChannels,
                  mlir::DominanceInfo &dom) {
  // Attach the barrier as an operand of the mma op.
  builder.setInsertionPoint(mmaOp);
  builder.setAsyncTaskIdsFromOp(mmaOp);
  auto consumerBarrier =
      getBarrierForPipelineStage(builder, barrierAlloc, bufferIdx);
  assert(mmaOp.getBarriers().empty() && "mmaOp should not have barriers");
  auto pred = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
      mmaOp->getLoc(), true, 1);
  mmaOp.addCompletionBarrier(consumerBarrier, pred);
  mmaOp.setIsAsync(true);

  // Create a wait_barrier before the producer.
  builder.setInsertionPoint(headProducer);
  builder.setAsyncTaskIdsFromOp(headProducer);
  auto producerBarrier =
      getBarrierForPipelineStage(builder, barrierAlloc, bufferIdx);
  // curPhase = curPhase xor True for emptyBarrier.
  auto loc = headProducer->getLoc();
  Value _1_1b = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 1);
  // Creating phase for headProducer.
  Value phase =
      builder.createWithAsyncTaskIds<mlir::arith::XOrIOp>(loc, inPhase, _1_1b);
  phase = builder.createWithAsyncTaskIds<arith::ExtSIOp>(
      loc, builder.getI32Type(), phase);
  builder.createWithAsyncTaskIds<ttng::WaitBarrierOp>(loc, producerBarrier,
                                                      phase);

  LLVM_DEBUG({
    LDBG("desync: create wait_barrier for producer ");
    producerBarrier.dump();
  });
  // Create a wait_barrier before the tmem load.
  SetVector<std::pair<Operation *, unsigned>> users;
  getTransitiveUsers(mmaOp.getD(), users);
  for (auto item : users) {
    auto user = item.first;
    if (user == mmaOp)
      continue;
    // TODO: identify the real consumer of the mma op.
    // rule out users that are not dominated by op
    if (mmaOp->getBlock() != user->getBlock()) {
      if (!dom.properlyDominates(mmaOp->getParentOp(), user))
        continue;
    } else {
      if (!dom.properlyDominates(mmaOp, user))
        continue;
    }
    builder.setInsertionPoint(user);
    builder.setAsyncTaskIdsFromOp(mmaOp);
    // If user and mmaOp are in the same block, we can use the same barrier.
    if (user->getBlock() != mmaOp->getBlock()) {
      // Compute the barrier from the last consumer instance
      // Extract the accum count from the consumer block.
      std::tie(bufferIdx, phase) = getOutOfScopeBufferIdxAndPhase(
          builder, mmaOp, numBuffers, regionsWithChannels);
      phase = builder.createWithAsyncTaskIds<arith::ExtSIOp>(
          user->getLoc(), builder.getI32Type(), phase);
      consumerBarrier =
          getBarrierForPipelineStage(builder, barrierAlloc, bufferIdx);
    } else {
      // mmaOp can be in a different task from headProducer. Even if user and
      // mma are in the same block and they share the same barrier, but the
      // phases should be offset by 1.
      auto loc = user->getLoc();
      Value _1_1b =
          builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 1);
      phase = builder.createWithAsyncTaskIds<mlir::arith::XOrIOp>(loc, inPhase,
                                                                  _1_1b);
      phase = builder.createWithAsyncTaskIds<arith::ExtSIOp>(
          loc, builder.getI32Type(), phase);
    }

    // TODO: if there are multiple users of the mma op, we need to barrier
    // before the first user.
    return builder.createWithAsyncTaskIds<ttng::WaitBarrierOp>(
        user->getLoc(), consumerBarrier, phase);
  }

  llvm_unreachable("Failed to find the consumer of the mma op");
}

// Lower producers for channels. Here channels are grouped in
// "channelsGroupedByConsumers". tokenMap tracks the set of tokens for each
// channel.
void insertAsyncComm(
    triton::FuncOp funcOp,
    const DenseMap<Channel *, SmallVector<Channel *>>
        &channelsGroupedByConsumers,
    const DenseMap<Channel *, CommChannel> &tokenMap,
    const DenseMap<Channel *, DenseMap<int, Value>> &barrierAllocMap,
    const DenseMap<Channel *, Value> &bufferMap,
    const DenseMap<Channel *, std::pair<Operation *, Operation *>> &copyOpMap,
    DenseSet<Operation *> &regionsWithChannels) {

  // Find the operation that is along producer's parent chain, and its parent
  // is the same op as producer's parent. Here p is producer, and c is consumer.
  auto getSameLevelOp = [](Operation *p, Operation *c) -> Operation * {
    Operation *op = c;
    while (!isa<triton::FuncOp>(op)) {
      if (op->getParentOp() == p->getParentOp()) {
        return op;
      }
      op = op->getParentOp();
    }
    op = p;
    while (!isa<triton::FuncOp>(op)) {
      if (c->getParentOp() == op->getParentOp()) {
        return c;
      }
      op = op->getParentOp();
    }
    llvm_unreachable("Failed to find consumer's same level Op with producer");
  };

  mlir::DominanceInfo dom(funcOp);
  mlir::PostDominanceInfo pdom(funcOp);
  auto consumerReleaseHeuristic = [&](Operation *p, Operation *c,
                                      int consumerAsyncTaskId) -> Operation * {
    if (c->getBlock() != p->getBlock())
      return getSameLevelOp(p, c);

    // Find a common place for all users of the consumer, which would be the
    // common post dominator.
    auto actualConsumers = getActualConsumers(c);
    std::unordered_set<Operation *> mutuallyNonDominatingUsers;
    for (auto user : actualConsumers) {
      auto it = mutuallyNonDominatingUsers.begin();
      while (it != mutuallyNonDominatingUsers.end()) {
        if (pdom.properlyPostDominates(user, *it)) {
          it = mutuallyNonDominatingUsers.erase(it);
        } else if (pdom.properlyPostDominates(*it, user)) {
          break;
        } else {
          ++it;
        }
      }
      if (it == mutuallyNonDominatingUsers.end())
        mutuallyNonDominatingUsers.insert(user);
    }

    if (mutuallyNonDominatingUsers.size() == 1) {
      // Find the common parent of this user and c
      auto user = *mutuallyNonDominatingUsers.begin();
      while (user && user->getParentOp() != c->getParentOp())
        user = user->getParentOp();
      assert(user && "Failed to find common parent of this user and c");
      return user;
    }

    for (auto &op : reverse(c->getBlock()->getOperations())) {
      auto asyncTasks = getAsyncTaskIds(&op);
      if (asyncTasks.size() == 1 && asyncTasks[0] == consumerAsyncTaskId)
        return &op;
    }

    return nullptr;
  };

  DenseMap<ttng::TCGen5MMAOp, ttng::WaitBarrierOp> tmemWaitBarriers;

  // Postpont TMEM channels until all SMEM channels are processed.
  // TODO: Reorder the channels in channelsGroupedByConsumers in dependency
  // order. This is to ensure that we insert the synchronization primitives for
  // dependent before using it.
  SmallVector<std::pair<Channel *, SmallVector<Channel *>>>
      orderedChannelsGroupedByConsumers;
  for (auto kv : channelsGroupedByConsumers) {
    if (kv.first->channelKind == DataChannelKind::SMEM) {
      orderedChannelsGroupedByConsumers.push_back({kv.first, kv.second});
    }
  }
  for (auto kv : channelsGroupedByConsumers) {
    if (kv.first->channelKind == DataChannelKind::TMEM) {
      orderedChannelsGroupedByConsumers.push_back({kv.first, kv.second});
    }
  }

  // Go through each channel group.
  for (auto kv : orderedChannelsGroupedByConsumers) {
    // Find head and tail ops.
    DenseSet<Operation *> producerOps;
    DenseSet<Operation *> consumerOps;
    for (auto &c : kv.second) {
      auto pcOp = copyOpMap.find(c)->second;
      producerOps.insert(pcOp.first);
      consumerOps.insert(pcOp.second);
      consumerOps.insert(c->getDstOp());
      consumerOps.insert(getUniqueActualConsumer(c->getDstOp()));
    }

    // Find head producer
    auto producerBlock = kv.second.front()->getSrcOp()->getBlock();
    Operation *headProducer = nullptr;
    for (auto &op : producerBlock->getOperations()) {
      if (producerOps.count(&op)) {
        headProducer = &op;
        break;
      }
    }
    // Find tail producer
    Operation *tailProducer = nullptr;
    for (auto &op : reverse(producerBlock->getOperations())) {
      if (producerOps.count(&op)) {
        tailProducer = &op;
        break;
      }
    }

    // Find head consumer and tail consumer
    auto consumerBlock = kv.second.front()->getDstOp()->getBlock();
    Operation *headConsumer = nullptr;
    for (auto &op : consumerBlock->getOperations()) {
      if (consumerOps.count(&op)) {
        headConsumer = &op;
        break;
      }
    }
    Operation *tailConsumer = nullptr;
    for (auto &op : reverse(consumerBlock->getOperations())) {
      if (consumerOps.count(&op)) {
        tailConsumer = &op;
        break;
      }
    }

    // We have one set of tokens for each channel group.
    auto &commChannel = tokenMap.find(kv.second.front())->second;
    auto masterChannel = kv.first;

    SmallVector<AsyncTaskId> asyncTaskP;
    asyncTaskP.push_back(masterChannel->relation.first);
    SmallVector<AsyncTaskId> &asyncTaskC = masterChannel->relation.second;
    SmallVector<AsyncTaskId> asyncTasksPC = asyncTaskP;
    asyncTasksPC.insert(asyncTasksPC.end(), asyncTaskC.begin(),
                        asyncTaskC.end());

    OpBuilderWithAsyncTaskIds builder(headProducer->getContext());
    if (auto funcOp = dyn_cast<triton::FuncOp>(headProducer->getParentOp())) {
      builder.setInsertionPointToStart(&(funcOp.getBody().front()));
    } else {
      builder.setInsertionPoint(headProducer->getParentOp());
    }
    builder.setAsynTaskIdsFromArray(asyncTasksPC);

    Value bufferIdx;
    Value phase = Value();
    if (auto forOp = headProducer->getParentOfType<scf::ForOp>()) {
      builder.setInsertionPoint(headProducer);
      LLVM_DEBUG({
        LDBG("call getBufferIdxAndPhase2 ");
        headProducer->dump();
      });
      getBufferIdxAndPhase(builder, headProducer, kv.second.front()->numBuffers,
                           regionsWithChannels, bufferIdx, phase);
    } else {
      // Producer is not in a ForOp, create phase and bufferIdx here.
      bufferIdx = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
          headProducer->getLoc(), 0, 32);
      phase = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
          headProducer->getLoc(), 0, 1);
    }

    // Lower TMA loads and TCGen5MMAOp first before inserting synchronization
    // primitives to avoid displacement.
    SmallVector<tt::DescriptorLoadOp> tmaLoads;
    SmallVector<Value> buffers;
    // Go through all channels in this channel group.
    for (auto &c : kv.second) {
      if (auto tmaLoad = dyn_cast<tt::DescriptorLoadOp>(c->getSrcOp())) {
        tmaLoads.push_back(tmaLoad);
        buffers.push_back(bufferMap.find(c)->second);
      }
    }

    LLVM_DEBUG({
      LDBG("SrcOp of master Channel ");
      masterChannel->getSrcOp()->dump();
      LDBG("DstOp of master Channel ");
      masterChannel->getDstOp()->dump();
      LDBG("headProducer ");
      headProducer->dump();
      LDBG("tailProducer ");
      tailProducer->dump();
      LDBG("headConsumer ");
      headConsumer->dump();
      LDBG("tailConsumer ");
      tailConsumer->dump();
    });

    builder.setAsynTaskIdsFromArray(masterChannel->relation.first);

    // Channel can have multiple consumers.
    for (auto &consumerTaskId : masterChannel->relation.second) {
      // Desynchronize TCGen5MMAOp. Set up consumer release and producer
      // acquire.
      auto mmaOp = dyn_cast<ttng::TCGen5MMAOp>(
          getUniqueActualConsumer(masterChannel->getDstOp(), consumerTaskId));
      // Assume a single task for mmaOp.
      if (mmaOp) {
        SmallVector<AsyncTaskId> asyncTasksMma = getAsyncTaskIds(mmaOp);
        assert(asyncTasksMma.size() == 1 && asyncTasksMma[0] == consumerTaskId);
      }
      if (mmaOp && commChannel.consumerBarriers.count(consumerTaskId)) {
        LLVM_DEBUG({
          LDBG("unique actual consumer is gen5 mma ");
          mmaOp->dump();
        });
        auto iter = commChannel.consumerBarriers.find(consumerTaskId);
        Value consumerBarrier = iter->second;
        // Use consumerBarrier as gen5 inline barrier.
        auto tmemWaitBarrier = desyncTCGen5MMAOp(
            builder, mmaOp, consumerBarrier, bufferIdx, phase,
            masterChannel->numBuffers, headProducer, regionsWithChannels, dom);
        tmemWaitBarriers[mmaOp] = tmemWaitBarrier;
      }
    }

    for (const auto &token : commChannel.tokens) {
      if (commChannel.consumerBarriers.empty()) {
        // Insert ProducerAcquireOp before the producer.
        auto producerAcquirePoint = getSameLevelOp(headConsumer, headProducer);
        builder.setInsertionPoint(producerAcquirePoint);
        builder.createWithAsyncTaskIds<ttnvws::ProducerAcquireOp>(
            headProducer->getLoc(), token.second, bufferIdx, phase);
      }

      // Insert ProducerCommitOp if producer is not TMA. For TMA, TMA lowering
      // will handle the ProducerCommit.
      if (!commChannel.producerBarrier) {
        Operation *producerCommitPoint;
        if (masterChannel->channelKind == DataChannelKind::TMEM) {
          ttng::TmemDataChannel *tmemChannel =
              static_cast<ttng::TmemDataChannel *>(masterChannel);
          assert(tmemWaitBarriers.count(tmemChannel->tmemMmaOp) &&
                 "Failed to find tmemWaitBarriers");
          producerCommitPoint = tmemWaitBarriers[tmemChannel->tmemMmaOp];
        } else {
          producerCommitPoint = getSameLevelOp(headConsumer, tailProducer);
        }
        builder.setInsertionPointAfter(producerCommitPoint);
        builder.createWithAsyncTaskIds<ttnvws::ProducerCommitOp>(
            tailProducer->getLoc(), token.second, bufferIdx);
      }
    }

    for (const auto &token : commChannel.tokens) {
      builder.setAsynTaskIdsFromArray(token.first);
      // Insert ConsumerWaitOp
      if (!commChannel.producerBarrier) {
        auto consumerWaitPoint = getSameLevelOp(headProducer, headConsumer);
        builder.setInsertionPoint(consumerWaitPoint);
        builder.createWithAsyncTaskIds<ttnvws::ConsumerWaitOp>(
            headConsumer->getLoc(), token.second, bufferIdx, phase);
      }

      // Insert ConsumerReleaseOp, if consumer is not a TCGen5MMAOp. For
      // TCGen5MMAOp, TCGen5MMAOp lowering will handle the ConsumerReleaseOp.
      if (commChannel.consumerBarriers.empty()) {
        auto consumerReleasePoint =
            consumerReleaseHeuristic(tailProducer, tailConsumer, token.first);
        builder.setInsertionPointAfter(consumerReleasePoint);
        builder.createWithAsyncTaskIds<ttnvws::ConsumerReleaseOp>(
            consumerReleasePoint->getLoc(), token.second, bufferIdx);
        LLVM_DEBUG({
          LDBG("create ConsumerRelease ");
          token.second.dump();
        });
      }
    }

    // Optimize TMA loads.
    if (tmaLoads.size() > 0) {
      optimizeTMALoads(builder, tmaLoads, buffers, *commChannel.producerBarrier,
                       bufferIdx, bufferIdx, phase, headProducer, headConsumer);
    }
  }
}

void foldLocalLoads(triton::FuncOp funcOp) {
  // If loadResult has a single use which is LocalAlloc, we can get rid of
  // sharedLoad and replace all uses of LocalAlloc with viewLoad.
  DenseMap<Operation *, Value> opsToReplace;
  funcOp.walk([&](ttg::LocalAllocOp localAlloc) {
    if (auto src = localAlloc.getSrc()) {
      if (auto localLoad = dyn_cast<ttg::LocalLoadOp>(src.getDefiningOp())) {
        // Only fold within the same tasks
        if (getAsyncTaskIds(localLoad) == getAsyncTaskIds(localAlloc)) {
          opsToReplace[localAlloc] = localLoad.getSrc();
        }
      }
    }
  });
  OpBuilderWithAsyncTaskIds builder(funcOp.getContext());
  for (auto kv : opsToReplace)
    mlir::triton::replaceUsesAndPropagateType(builder, kv.getFirst(),
                                              kv.getSecond());
}

void doCodePartition(triton::FuncOp &funcOp, unsigned numBuffers) {
  // Step 1: collect all communications between producers and consumers.
  SmallVector<std::unique_ptr<Channel>> channelsOrigin;
  collectAsyncChannels(channelsOrigin, funcOp, numBuffers);
  SmallVector<Channel *> channels;
  for (const auto &c : channelsOrigin) {
    channels.push_back(c.get());
  }
  if (channels.empty()) {
    return;
  }

  // Step 2: group channels
  // -  each entry of the channelsGroupedByProducers is keyed by the srcOp.
  // -  each entry of the channelsGroupedByConsumers is keyed by the dstOp.
  DenseMap<Channel *, SmallVector<Channel *>> channelsGroupedByProducers;
  DenseMap<Channel *, SmallVector<Channel *>> channelsGroupedByConsumers;
  SmallVector<Channel *> orderedChannels;
  groupChannels(channels, channelsGroupedByProducers,
                channelsGroupedByConsumers, orderedChannels);

  // Step 3: reorder producer ops and the backward slices of the producer ops.
  reorderProducerOps(channels);

  // Step 4: find top-level ops that contain a channel, also create new ForOps
  // by adding phase and bufferIdx to the original ForOps, erase the original
  // ForOps.
  SmallVector<Operation *> asyncTaskTopOps = getTaskTopRegion(funcOp, channels);
  SmallVector<Operation *> opList;
  for (auto &op : asyncTaskTopOps) {
    if (auto origIfOp = dyn_cast<scf::IfOp>(op)) {
      opList.push_back(op);
    }
    if (auto origForOp = dyn_cast<scf::ForOp>(op))
      opList.push_back(op);
  }
  DenseSet<Operation *> regionsWithChannels;
  collectRegionsWithChannels(channels, regionsWithChannels);
  appendAccumCntsForOps(asyncTaskTopOps, channels, regionsWithChannels);
  LLVM_DEBUG({
    LDBG("\n\nafter appendAccumCntsForOps");
    funcOp.dump();
  });

  // Step 5: Create buffers. An array of buffers for each channel.
  DenseMap<Channel *, Value> bufferMap =
      createBuffer(channelsGroupedByProducers, channels, funcOp);
  LLVM_DEBUG({
    LDBG("\n\nafter createBuffer");
    funcOp.dump();
  });

  // Step 6: Lower the loads. Also add local copy ops for non-load
  // producers.
  DenseMap<Channel *, std::pair<Operation *, Operation *>> copyOpMap;
  insertAsyncCopy(funcOp, channelsGroupedByProducers, bufferMap, copyOpMap,
                  regionsWithChannels);
  LLVM_DEBUG({
    LDBG("\n\nwith async copy");
    funcOp.dump();
  });

  // Step 7: Create tokens. A set of tokens for each group of channels for
  // each channel.
  DenseMap<Channel *, DenseMap<int, Value>> barrierAllocMap;
  DenseMap<Channel *, CommChannel> tokenMap;
  createToken(channelsGroupedByConsumers, orderedChannels, funcOp, copyOpMap,
              tokenMap);
  LLVM_DEBUG({
    LDBG("\n\nafter createToken");
    funcOp.dump();
  });

  // Step 8: add async communication ops (ProducerAcquire etc). Also lower
  // TMA loads.
  insertAsyncComm(funcOp, channelsGroupedByConsumers, tokenMap, barrierAllocMap,
                  bufferMap, copyOpMap, regionsWithChannels);
  LLVM_DEBUG({
    LDBG("\n\nwith SyncOps");
    funcOp.dump();
  });

  // If loadResult has a single use which is LocalAlloc, we can get rid of
  // sharedLoad and replace all uses of LocalAlloc with viewLoad.
  foldLocalLoads(funcOp);
  LLVM_DEBUG({
    LDBG("\n\nsimplify localLoad + localAlloc");
    funcOp.dump();
  });

  specializeRegion(funcOp, 0 /*requestedRegisters*/);
  LLVM_DEBUG({
    LDBG("\n\nwith specializeRegion");
    funcOp.dump();
  });
}

#define GEN_PASS_DEF_NVGPUTESTWSCODEPARTITION
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

class NVGPUTestWSCodePartitionPass
    : public impl::NVGPUTestWSCodePartitionBase<NVGPUTestWSCodePartitionPass> {
public:
  using impl::NVGPUTestWSCodePartitionBase<
      NVGPUTestWSCodePartitionPass>::NVGPUTestWSCodePartitionBase;

  void runOnFuncOp(triton::FuncOp funcOp) {
    // Disable code partitioning when numBuffers is 0.
    if (numBuffers > 0)
      doCodePartition(funcOp, numBuffers);
  }
  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
    LLVM_DEBUG({
      LDBG("post pass");
      getOperation()->dump();
    });
    return;
  }
};

} // namespace mlir
