#include "Utilities.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/MMAv5PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PartitionBuilder.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include <optional>

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_NVWSINSERTTMEMAREF
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

#define DEBUG_TYPE "nvws-insert-tmem-aref"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

using namespace mlir;
using namespace triton::gpu;
using namespace triton::nvidia_gpu;
using namespace triton::nvws;

int getWsTag(Operation *op) {
  while (op && !hasWarpSpecializeTag(op)) {
    op = op->getParentOfType<scf::ForOp>();
  }
  assert(op);
  return *getWarpSpecializeTag(op);
}

using PartitionId = std::pair<int /* PartitionId*/, int /* WsTag*/>;
std::optional<PartitionId> getPartitionId(Operation *op, int pos = 0) {
  if (!hasPartition(op))
    return std::nullopt;
  auto partitionIds = getPartitionIds(op);
  if (op->getNumRegions() > 0) {
    partitionIds = getPartitionOutputs(op)[pos];
  }
  assert(partitionIds.size() == 1);
  return std::make_pair(*partitionIds.begin(), getWsTag(op));
}

struct TmemAccessDag {
  struct Node {
    // For now we assume there is only one use of generated async tmem token
    std::unique_ptr<Node> user;
    SmallVector<std::unique_ptr<Node>> subDags;
    Node(Operation *op, OpOperand *tokOperand,
         std::optional<PartitionId> partitionId, Node *parent)
        : op(op), tokOperand(tokOperand), partitionId(partitionId),
          parent(parent), parentDag(nullptr) {}

    // ------------------------------------------------------------------------

    Operation *op;
    OpOperand *tokOperand;
    Node *parent;
    Node *parentDag;
    std::optional<int> tokPos;
    std::optional<PartitionId> partitionId;
  };

  TmemAccessDag(std::unique_ptr<Node> dag) : dag(std::move(dag)) {}

  Node *getRootNode() { return dag.get(); }
  TMEMAllocOp getAllocOp() { return cast<TMEMAllocOp>(dag->op); }

  Value addIfOp(Value tok, Node *node) {
    SmallVector<OpOperand *> uses;
    for (auto &use : tok.getUses())
      uses.push_back(&use);
    assert(uses.size() == 2 && "expecting two uses of a token");
    auto useThen = uses[0];
    auto useElse = uses[1];

    auto ifOp = cast<scf::IfOp>(useThen->getOwner()->getParentOp());
    node->user.reset(new Node(ifOp, nullptr, {}, node));
    auto ifOpNode = node->user.get();

    if (ifOp.thenBlock() != useThen->getOwner()->getBlock())
      std::swap(useThen, useElse);
    assert(ifOp.thenBlock() == useThen->getOwner()->getBlock());
    assert(ifOp.elseBlock() == useElse->getOwner()->getBlock());

    // Create access DAGs for then/else blocks.
    auto thenDag =
        std::make_unique<Node>(nullptr, nullptr, std::nullopt, nullptr);
    auto elseDag =
        std::make_unique<Node>(nullptr, nullptr, std::nullopt, nullptr);
    auto thenTok = addOp(*useThen, thenDag.get());
    auto elseTok = addOp(*useElse, elseDag.get());

    auto tokPos =
        *findValuePosInRange(ifOp.thenYield()->getOperands(), thenTok);
    ifOpNode->partitionId = getPartitionId(ifOp, tokPos);

    // find final node in then-branch and assign yieldOp as its user
    // XXX: improve representation later, but for now the user's parentDag
    //      points to the first op in the branch, because we will need to get
    //      stageCluser information later in aref insertion as ifOps don't carry
    //      partition assignment to their results like nvws-branch
    Node *finalThenNode = thenDag.get();
    while (finalThenNode->user)
      finalThenNode = finalThenNode->user.get();
    auto thenYieldOp = ifOp.thenYield();
    finalThenNode->user =
        std::make_unique<Node>(thenYieldOp, &thenYieldOp->getOpOperand(tokPos),
                               ifOpNode->partitionId, finalThenNode);
    finalThenNode->user->parentDag = thenDag->user.get();

    // do the same with else-branch
    Node *finalElseNode = elseDag.get();
    while (finalElseNode->user)
      finalElseNode = finalElseNode->user.get();
    auto elseYieldOp = ifOp.elseYield();
    finalElseNode->user =
        std::make_unique<Node>(elseYieldOp, &elseYieldOp->getOpOperand(tokPos),
                               ifOpNode->partitionId, finalElseNode);
    finalElseNode->user->parentDag = elseDag->user.get();

    // the parent of the first op in the branch is null, but parent dag points
    // to original ifOp
    thenDag->user->parent = nullptr;
    elseDag->user->parent = nullptr;
    thenDag->user->parentDag = ifOpNode;
    elseDag->user->parentDag = ifOpNode;

    ifOpNode->subDags.push_back(std::move(thenDag->user));
    ifOpNode->subDags.push_back(std::move(elseDag->user));

    ifOpNode->tokPos = tokPos;

    auto newTok = ifOp.getResult(tokPos);
    assert(newTok.hasOneUse());
    return addOp(*newTok.getUses().begin(), ifOpNode);
  }

  Value addForOp(OpOperand &tokOperand, Node *forOpNode) {
    auto forOp = cast<scf::ForOp>(tokOperand.getOwner());
    auto tokPos = tokOperand.getOperandNumber() - 3;
    auto tokDefOp = forOp.getYieldedValues()[tokPos].getDefiningOp();
    assert(tokDefOp && "expecting a token definition op");

    // Create access node for the for-loop body. The first op is nullptr,
    // but it has partitionIdx, indicating which partition owns the Tmem when
    // entering the region
    auto subDag =
        std::make_unique<Node>(nullptr, nullptr, std::nullopt, nullptr);
    auto tokArg = forOp.getRegionIterArg(tokPos);
    assert(tokArg.hasOneUse());
    addOp(*tokArg.getUses().begin(), subDag.get());
    forOpNode->partitionId = getPartitionId(forOp, tokPos);

    // finalNode keep track of partition ownership transfer ownership when
    // before exiting the loop-body or re-entering loop body
    // same as in IfOp then/else branches
    Node *finalNode = subDag->user.get();
    while (finalNode->user)
      finalNode = finalNode->user.get();
    auto yieldOp = forOp.getBody()->getTerminator();
    finalNode->user =
        std::make_unique<Node>(yieldOp, &yieldOp->getOpOperand(tokPos),
                               forOpNode->partitionId, finalNode);
    finalNode->user->parentDag = subDag->user.get();
    forOpNode->tokPos = tokPos;

    // subDag->user->parentDag = subDag->user.get();
    subDag->user->parent = nullptr;
    subDag->user->parentDag = forOpNode;

    forOpNode->subDags.push_back(std::move(subDag->user));
    return forOp.getResult(tokPos);
  }

  Value addOp(OpOperand &tokOperand, Node *node) {
    if (isa<scf::YieldOp>(tokOperand.getOwner()))
      return tokOperand.get(); // return token back to the caller

    auto op = tokOperand.getOwner();
    std::optional<PartitionId> partitionId;
    // tmem owning partition for if & for ops are inferred from their regions
    if (op->getNumRegions() == 0)
      partitionId = getPartitionId(op);
    node->user.reset(new Node(op, &tokOperand, partitionId, node));
    auto newNode = node->user.get();
    Value newTok;

    if (auto tmemLoad = dyn_cast<TMEMLoadOp>(op)) {
      newTok = tmemLoad.getToken();
    } else if (auto tmemStore = dyn_cast<TMEMStoreOp>(op)) {
      newTok = tmemStore.getToken();
    } else if (auto mmav5 = dyn_cast<MMAv5OpInterface>(op)) {
      newTok = mmav5.getToken();
    } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      newTok = addForOp(tokOperand, newNode);
    } else {
      llvm_unreachable("unsupported user");
    }

    if (newTok.use_empty())
      return newTok;

    if (newTok.hasOneUse()) {
      auto &use = *newTok.getUses().begin();
      return addOp(use, newNode);
    }

    // Multiple uses of token are expected only in IfOp: one in then and one in
    // else branches.
    return addIfOp(newTok, newNode);
  }

  static TmemAccessDag build(TMEMAllocOp allocOp) {
    std::optional<PartitionId> partitionId;
    if (allocOp.getSrc()) {
      partitionId = getPartitionId(allocOp);
    }
    TmemAccessDag accessDag(
        std::make_unique<Node>(allocOp, nullptr, partitionId, nullptr));

    if (allocOp.getSrc() && !allocOp.getToken()) {
      // Handle tmem_alloc with src operand specially. When a src operand is
      // present, no async tokens are generated, we can't traverse IR,
      // and we directly add the single user operation to the access DAG.
      assert(allocOp->hasOneUse());
      auto user = *allocOp->getUsers().begin();
      accessDag.getRootNode()->user.reset(new Node{
          user, nullptr, getPartitionId(user), accessDag.getRootNode()});
    } else {
      auto tok = allocOp.getToken();
      assert(tok && tok.hasOneUse());
      auto &tokUse = *tok.getUses().begin();
      accessDag.addOp(tokUse, accessDag.getRootNode());
    }
    return accessDag;
  }

  void collectPartitions(
      Node *node, bool &hasRootPartition,
      SmallVector<std::pair<PartitionId, Operation *>> &partitions) {
    if (node->partitionId) {
      partitions.push_back(std::make_pair(*node->partitionId, node->op));
    } else {
      // root partition is considered a real owner only if there are already
      // other partitions owning tmem
      hasRootPartition = !partitions.empty();
    }
    for (auto &subDag : node->subDags) {
      if (subDag) {
        collectPartitions(subDag.get(), hasRootPartition, partitions);
      }
    }
    if (node->user) {
      collectPartitions(node->user.get(), hasRootPartition, partitions);
    }
  };

  std::pair<bool, SmallVector<std::pair<PartitionId, Operation *>>>
  collectPartitionsVec() {
    SmallVector<std::pair<PartitionId, Operation *>> partitions;
    bool hasRootPartition = false;
    auto node = getRootNode();
    auto allocOp = getAllocOp();
    if (allocOp.getSrc() && node->partitionId)
      partitions.push_back(std::make_pair(*node->partitionId, node->op));
    collectPartitions(getRootNode()->user.get(), hasRootPartition, partitions);
    return {hasRootPartition, partitions};
  }

  std::pair<bool, std::set<PartitionId>> collectPartitionsSet() {
    auto [hasRootPartition, partitions] = collectPartitionsVec();
    std::set<PartitionId> partitionSet;
    for (auto [partition, _] : partitions) {
      partitionSet.insert(partition);
    }
    return {hasRootPartition, partitionSet};
  }

  void printNode(Node *node, int indent, llvm::raw_ostream &os) {
    if (!node)
      return;
    for (int i = 0; i < indent; i++) {
      os << " ";
    }
    std::set<PartitionId> partitions;
    os << "|- [" << node->op << "]";
    bool hasRootPartition = false;
    if (node->partitionId)
      partitions.insert(*node->partitionId);
    else
      hasRootPartition = true;
    if (node->op) {
      os << node->op->getName().getStringRef() << " ";
      if (auto tmemAlloc = dyn_cast<TMEMAllocOp>(node->op)) {
        if (tmemAlloc.getSrc()) {
          os << " %src ";
        } else {
          std::tie(hasRootPartition, partitions) = collectPartitionsSet();
        }
      }
      os << "  ";
    }
    os << "[" << (hasRootPartition ? "root" : "");
    for (auto partition : partitions) {
      auto [id, tag] = partition;
      os << " @" << tag << "." << id << " ";
    }
    os << "]";
    os << " prev[" << (node->parent ? node->parent->op : nullptr) << "]";
    os << "\n";
    for (auto &subDag : node->subDags) {
      for (int i = 0; i < indent + 4; i++)
        os << " ";
      os << "|- subDag\n";
      if (subDag)
        printNode(subDag.get(), indent + 8, os);
    }
    if (node->user) {
      printNode(node->user.get(), indent, os);
    }
  };
  void printDag(llvm::raw_ostream &os) {
    os << "TMEMDAG\n";
    printNode(dag.get(), 2, os);
    os << "\n";
  }

  // --------------------------------------------------------------------------

  std::unique_ptr<Node> dag;
  DenseMap<scf::ForOp, TMEMAllocOp> arefTmemAllocs;
};

void assignStage(OpBuilder &b, Operation *op, StageCluster stageCluster) {
  if (stageCluster) {
    op->setAttr(kLoopStageAttrName, b.getI32IntegerAttr(stageCluster->first));
    op->setAttr(kLoopClusterAttrName,
                b.getI32IntegerAttr(stageCluster->second));
  }
}

template <typename OpT, typename... Args>
OpT createInto(
    OpBuilder &b, Location loc,
    std::pair<std::optional<PartitionId>, StageCluster> partitionIdStageCluster,
    Args &&...args) {
  std::optional<SetVector<int>> partitionIds = SetVector<int>();
  std::optional<int> wsTag;
  if (partitionIdStageCluster.first) {
    auto [id, tag] = *partitionIdStageCluster.first;
    wsTag = tag;
    partitionIds->insert(id);
  } else {
    partitionIds = std::nullopt;
  }
  auto op = triton::gpu::createInto<OpT>(b, loc, partitionIds,
                                         partitionIdStageCluster.second,
                                         std::forward<Args>(args)...);
  if (wsTag) {
    auto forOp = op->template getParentOfType<scf::ForOp>();
    while (forOp && !hasWarpSpecializeTag(forOp)) {
      forOp = forOp->template getParentOfType<scf::ForOp>();
    }
    // only set wsTag if op is outside tt.ws loop
    if (!forOp) {
      setWarpSpecializeTag(op, *wsTag);
    }
  }
  return op;
}

struct TMEMAref {
  enum Kind { PUT, GET };

  TMEMAref(Value aref, Value origBuffer, Value replToken)
      : aref(aref), origBuffer(origBuffer), replToken(replToken), kind(PUT) {}

  void acquire(OpBuilder &b, Location loc,
               std::pair<std::optional<PartitionId>, StageCluster>
                   paritionIdStageCluster) {
    auto arefBufType =
        cast<MemDescType>(aref.getDefiningOp()->getOperand(0).getType());
    Type dataBufType = getArefViewBufferType(arefBufType);
    SmallVector<Type> buffers{dataBufType};
    SmallVector<Type> tokens{b.getType<AsyncTokenType>()};
    if (kind == PUT) {
      auto op =
          createInto<ArefPutEnterOp>(b, loc, paritionIdStageCluster, aref,
                                     buffers, b.getType<AsyncTokenType>());
      token = op.getToken();
    } else {
      auto op =
          createInto<ArefGetEnterOp>(b, loc, paritionIdStageCluster, aref,
                                     buffers, b.getType<AsyncTokenType>());
      token = op.getToken();
    }
    partitionId = paritionIdStageCluster.first;
    if (partitionId)
      stageClusters[*partitionId] = paritionIdStageCluster.second;
    buffer = {};
  }
  void release(OpBuilder &b, Location loc) {
    assert(asyncOp[partitionId]);
    StageCluster stageCluster;
    if (partitionId)
      stageCluster = stageClusters[*partitionId];
    if (kind == PUT) {
      createInto<ArefPutExitOp>(
          b, loc, {partitionId, stageCluster}, aref, token,
          b.getArrayAttr(SmallVector<Attribute>{
              AsyncOpAttr::get(b.getContext(), *asyncOp[partitionId])}));
      kind = GET;
    } else {
      createInto<ArefGetExitOp>(
          b, loc, {partitionId, stageCluster}, aref, token,
          b.getArrayAttr(SmallVector<Attribute>{
              AsyncOpAttr::get(b.getContext(), *asyncOp[partitionId])}));
      kind = PUT;
    }
  }
  Value getBuffer(OpBuilder &b, std::optional<PartitionId> partitionId,
                  Operation *op) {
    if (!buffer) {
      auto stageCluster = getStageCluster(op);
      auto arefBufType =
          cast<MemDescType>(aref.getDefiningOp()->getOperand(0).getType());
      Type dataBufType = getArefViewBufferType(arefBufType);
      SmallVector<Type> buffers{dataBufType};
      auto bufferOp = createInto<ArefBufferOp>(
          b, op->getLoc(), {partitionId, stageCluster}, aref, buffers, token);

      buffer = bufferOp.getBuffers()[0];
    }
    return buffer;
  }

  // --------------------------------------------------------------------------

  Value origBuffer;
  Value aref;
  Value replToken;

  Value buffer;
  Value token;
  Kind kind;
  std::optional<PartitionId> partitionId;
  llvm::MapVector<std::optional<PartitionId>, std::optional<AsyncOp>> asyncOp;
  DenseMap<PartitionId, StageCluster> stageClusters;
};

TmemAccessDag::Node *
insertTmemArefImpl(TmemAccessDag::Node *node,
                   std::optional<PartitionId> curPartitionId, TMEMAref &state) {
  // When entering a warp-specialized loop, curPartitionId is std::nullopt.
  // We skip ownership changes here since there's an implicit synchronization
  // barrier when entering the ws-loop that handles the transition safely.
  if (curPartitionId && node->partitionId != curPartitionId) {
    OpBuilder b(node->op);
    Operation *prevOp = nullptr;
    if (node->parent) {
      // release right after the last op which owns the tmem
      prevOp = node->parent->op;
      b.setInsertionPointAfter(prevOp);
    } else {
      // if we are inside if-stmt or for-stmt subdag and need to change
      // ownerhip, release at the top of the block
      // the parentDag op would be if-stmt or for-stmt
      prevOp = node->parentDag->op;
      b.setInsertionPointToStart(node->op->getBlock());
    }
    state.release(b, prevOp->getLoc());

    // acquire right before op that acquires ownership of tmem
    auto curOp = node->op;
    auto partitionId = node->partitionId;
    b.setInsertionPoint(curOp);

    if (isa<scf::YieldOp>(curOp)) {
      // in yieldOp we overload parentDag as the first op in the current subDag
      // so we use its stageCluster to insert acquire
      curOp = node->parentDag->op;
    }
    auto stageCluster = getStageCluster(curOp);
    // if stage-cluster is empty, use the stage-cluster used from the last op
    // that acquired ownership of tmem in a partition
    if (!stageCluster && partitionId)
      stageCluster = state.stageClusters[*partitionId];
    state.acquire(b, curOp->getLoc(), {partitionId, stageCluster});
  }

  for (auto &subDag : node->subDags) {
    auto subdagState = state;
    if (auto forOp = dyn_cast<scf::ForOp>(node->op)) {
      // forOp may have token operand, if so, we need to update the token and
      // and reset buffer
      if (node->tokOperand) {
        subdagState.token =
            forOp.getRegionIterArg(node->tokOperand->getOperandNumber() - 3);
        subdagState.buffer = {};
      }
    }
    insertTmemArefImpl(subDag.get(), node->partitionId, subdagState);

    // subDag may change asyncOp value, update it after inserting arefs
    state.asyncOp = subdagState.asyncOp;
    // store subdag state partitoinId
    state.partitionId = subdagState.partitionId;
  }

  if (isa<MMAv5OpInterface>(node->op)) {
    state.asyncOp[node->partitionId] = AsyncOp::TC5MMA;
  } else if (isa<TMEMLoadOp, TMEMStoreOp>(node->op)) {
    state.asyncOp[node->partitionId] = AsyncOp::NONE;
  }

  OpBuilder b(node->op);
  if (auto tmemLoadOp = dyn_cast<TMEMLoadOp>(node->op)) {
    if (auto id = node->partitionId)
      state.stageClusters[*id] = getStageCluster(node->op);
    tmemLoadOp.getSrcMutable().assign(
        state.getBuffer(b, node->partitionId, node->op));
    tmemLoadOp.getDepMutable().clear();
    tmemLoadOp.getToken().replaceAllUsesWith(state.replToken);
  } else if (auto tmemStoreOp = dyn_cast<TMEMStoreOp>(node->op)) {
    if (auto id = node->partitionId)
      state.stageClusters[*id] = getStageCluster(node->op);
    tmemStoreOp.getDstMutable().assign(
        state.getBuffer(b, node->partitionId, node->op));
    tmemStoreOp.getDepMutable().clear();
    tmemStoreOp.getToken().replaceAllUsesWith(state.replToken);
  } else if (auto mmaOp = dyn_cast<MMAv5OpInterface>(node->op)) {
    if (auto id = node->partitionId)
      state.stageClusters[*id] = getStageCluster(node->op);
    if (mmaOp.getAccumulator() == state.origBuffer) {
      mmaOp.getAccDepMutable().clear();
      mmaOp.getToken().replaceAllUsesWith(state.replToken);
    }
    for (auto &opnd : mmaOp->getOpOperands()) {
      if (opnd.get() == state.origBuffer)
        opnd.set(state.getBuffer(b, node->partitionId, node->op));
    }
  } else if (auto yieldOp = dyn_cast<scf::YieldOp>(node->op)) {
    yieldOp.setOperand(node->tokOperand->getOperandNumber(), state.token);
  } else if (isa<scf::IfOp, scf::ForOp>(node->op)) {
    if (node->tokPos) {
      // forOp/if may return token, if so, update state token, and reset buffer
      if (isa<scf::ForOp>(node->op))
        node->op->setOperand(node->tokOperand->getOperandNumber(), state.token);
      state.token = node->op->getResult(*node->tokPos);
      state.buffer = {};
    }
  } else {
    llvm_unreachable("unsupported tmem op");
  }

  if (node->user)
    return insertTmemArefImpl(node->user.get(), node->partitionId, state);
  return node;
}

bool canDoubleBufferAcc(MMAv5OpInterface mmaOp, int numTmemBlocks) {
  auto tmemDesc = mmaOp.getAccumulator().getType();
  auto blockM = tmemDesc.getShape()[0];
  auto blockN = tmemDesc.getShape()[1];
  constexpr int numTMEMColumns = 512;
  constexpr int numTMEMRows = 128;
  if (numTmemBlocks + (blockM * blockN * 2) > numTMEMRows * numTMEMColumns) {
    return false;
  }
  if (isa<TCGen5MMAScaledOp>(mmaOp) && blockN == 256) {
    return false;
  }
  return true;
};

bool hasProducerConsumerPartitioning(TmemAccessDag &accessDag) {
  // TMEM partitioning follows a producer-consumer pattern if it has this
  // structure:
  //
  //      |alloc
  //      |-- ops
  //    loop (tt.ws)
  //      |----  producer @A
  //      |----  consumer @B
  //      |----  producer @A
  //
  // We have root operations, then enter a warp-specialized loop where:
  // - First, partition A owns TMEM and performs producer operations
  // - Then, partition B owns TMEM and performs consumer operations
  // - Possibly, partition A owns TMEM and performs producer operations
  // - Loop repeats with partition A yielding
  //
  // Here is an example where the producer-consumer pattern is not present:
  //   |alloc
  //   |store
  //   |for  (tt.ws)
  //   |  |store @A
  //   |  |for
  //   |  |   mma @B
  //   |  |load @A
  // The partitions @A & @B are both producers.
  //
  // Compare to the following, where we change ownership of TMEM where partition
  // B is the producer and partition A is the consumer:
  //   |alloc
  //   |store
  //   |for  (tt.ws)
  //   |  |store @B
  //   |  |for
  //   |  |   mma @B
  //   |  |load @A
  // Here, we may double-buffer the accumulator.
  //
  // This is a necessary (but not sufficient) condition for enabling TMEM
  // multi-buffering with arefs. Additional validation will verify sufficient
  // conditions for multi-buffering.

  auto [hasRootPartition, partitions] = accessDag.collectPartitionsVec();
  bool expectProducer = true;
  int changeGroup = 0;
  bool valid = true;

  // Count partition transitions: producer-consumer pattern has exactly two
  // transitions (A->B followed by B->A), where 'A' is producer and 'B' is
  // consumer. More than two transitions (e.g., A-A-B-B-A-A-B-B-A-A) indicate a
  // more complex pattern that doesn't fit the producer-consumer model.
  for (size_t i = 0; i < partitions.size() - 1; ++i) {
    auto op = partitions[i].second;
    if (isa<TMEMLoadOp, TMEMStoreOp, MMAv5OpInterface>(op)) {
      valid = valid && (expectProducer ? isa<TMEMStoreOp, MMAv5OpInterface>(op)
                                       : isa<TMEMLoadOp>(op));
    }
    if (partitions[i].first != partitions[i + 1].first) {
      expectProducer = !expectProducer;
      ++changeGroup;
    }
  }
  valid = valid && changeGroup == 2;

  return valid;
}

int insertTmemAref(TmemAccessDag &accessDag, int numTmemBlocks) {
  auto rootNode = accessDag.getRootNode();
  auto allocOp = cast<TMEMAllocOp>(rootNode->op);

  auto isMultiStaged = hasProducerConsumerPartitioning(accessDag);
  int numTmemBlock = 0;
  if (isMultiStaged) {
    for (auto user : allocOp.getResult().getUsers()) {
      if (auto mmaOp = dyn_cast<MMAv5OpInterface>(user)) {
        if (auto loop = dyn_cast<scf::ForOp>(user->getParentOp())) {
          auto wsLoop = getOuterWSLoop(loop);
          // Determine if the MMA accumulator can be multibuffered.
          bool accIsMultiBuffered =
              // MMAs in subsequent iterations can be overlapped.
              !nvidia_gpu::hasAccReadModifyWrite(mmaOp, loop) &&
              // The accumulator is reset at some point, thus allowing
              // multibuffering.
              isAccMultibufferingPossible(mmaOp, loop) &&
              // The user didn't disable it with a flag.
              !getDisallowAccMultiBuffer(wsLoop) &&
              canDoubleBufferAcc(mmaOp, numTmemBlocks);
          isMultiStaged = isMultiStaged && accIsMultiBuffered;
        }
      }
    }
  }
  auto numStages = 1 + isMultiStaged;

  // update numTmemBlocks for the number of TMEM blocks used by the aref buffer
  auto allocShape = allocOp.getType().getShape();
  numTmemBlocks += allocShape[0] * allocShape[1] * numStages;
  auto arefBufType =
      getArefMultiBufferedType(allocOp.getResult().getType(), numStages);
  OpBuilder b(allocOp);

  // alloc can be inside ws-loop, we need to find the entry point for ws-loop
  auto outerWsLoop = allocOp->getParentOfType<scf::ForOp>();
  while (outerWsLoop && !outerWsLoop->hasAttr(triton::kWarpSpecializeAttrName))
    outerWsLoop = outerWsLoop->getParentOfType<scf::ForOp>();
  if (outerWsLoop)
    b.setInsertionPoint(outerWsLoop);
  auto arefAlloc =
      cast<TMEMAllocOp>(createAlloc(b, allocOp.getLoc(), arefBufType, Value()));
  auto arefOp = createArefCreateOp(b, {arefBufType}, {arefAlloc->getResult(0)},
                                   allocOp.getLoc());

  auto stageCluster = getStageCluster(allocOp);
  auto partitionId = accessDag.getRootNode()->partitionId;
  if (!allocOp.getSrc() && outerWsLoop) {
    // if tmem_alloc inside ws-loop, the first owner is that of the first user
    partitionId = accessDag.getRootNode()->user->partitionId;
  }

  TMEMAref state(
      arefOp, allocOp.getResult(),
      ub::PoisonOp::create(b, allocOp.getLoc(), b.getType<AsyncTokenType>()));
  b.setInsertionPoint(allocOp);
  state.acquire(b, allocOp.getLoc(), {partitionId, stageCluster});

  // If initial acquire is in root partition (no partition annotation), the
  // release must be in the partition of the first owner that has a partition
  // annotation. Find that partition and update state.partitionId accordingly.
  if (!state.partitionId) {
    auto node = rootNode->user.get();
    do {
      state.partitionId = node->partitionId;
      node = node->user.get();
    } while (node && !state.partitionId);
  }

  if (auto src = allocOp.getSrc()) {
    auto buffer = state.getBuffer(b, partitionId, allocOp);
    state.asyncOp[partitionId] = AsyncOp::NONE;
    auto vTrue = createInto<arith::ConstantIntOp>(
        b, allocOp.getLoc(), {partitionId, stageCluster}, true, 1);
    createInto<TMEMStoreOp>(b, allocOp.getLoc(), {partitionId, stageCluster},
                            Type(), buffer, Value(), src, vTrue);
  } else {
    // allocOp w/o src, assume the ownership of tmem belongs to first user
    // partitionId = accessDag.getRootNode()->user->partitionId;
  }

  auto node = insertTmemArefImpl(rootNode->user.get(), partitionId, state);

  if (outerWsLoop) {
    // aref is only used inside ws-loop, so we use the last op to insert
    // matching exit
    b.setInsertionPointAfter(node->op);
  } else {
    // aref is used outside ws-loop, find the last point in the same block as
    // create op to have matching exit
    auto op1 = arefOp->getBlock()->findAncestorOpInBlock(*node->op);
    if (auto id = node->partitionId)
      state.stageClusters[*id] = {};
    b.setInsertionPointAfter(op1);
  }
  state.release(b, node->op->getLoc());

  if (state.kind == TMEMAref::GET) {
    // When the state ends up in a GET operation, we need to acquire and release
    // the corresponding partition to prevent deadlocks. This is necessary
    // because if we're inside an outer loop, re-entering the loop without
    // posting a matching GET operation for the PUT would cause the dead-lock.
    auto [hasRootPartition, partitions] = accessDag.collectPartitionsSet();
    std::optional<PartitionId> otherPartitionId;
    // since we only have two partition, we just pick the other partition for
    // get
    for (auto partitionId : partitions) {
      if (partitionId != state.partitionId) {
        otherPartitionId = partitionId;
        break;
      }
    }
    state.acquire(b, node->op->getLoc(), {otherPartitionId, {}});
    state.release(b, node->op->getLoc());
  }

  return numTmemBlocks;
}

void workaroundForLoopScheduler(triton::FuncOp funcOp) {
  SmallVector<scf::IfOp> ifs;
  funcOp.walk([&](scf::IfOp ifOp) {
    auto firstOp = &*ifOp.thenBlock()->begin();
    auto lastOp = ifOp.thenBlock()->getTerminator()->getPrevNode();
    if (isa<ArefPutExitOp>(firstOp) && isa<ArefPutEnterOp>(lastOp)) {
      ifs.push_back(ifOp);
    }
  });

  // Transform if-statements that contain aref put.exit/put.enter pairs to work
  // around loop scheduler limitations. The transformation splits a single if-op
  // with token-producing operations into three separate if-ops to ensure proper
  // scheduling and token handling.
  //
  // Original pattern:
  //   %results, %token, %more = scf.if %condition {
  //     aref.put.exit                    // Release tensor memory
  //     <computation_code>               // User computation
  //     %new_token = aref.put.enter      // Acquire tensor memory
  //     scf.yield %values, %new_token, %other_values
  //   } else {
  //     scf.yield %alt_values, %old_token, %alt_other_values
  //   }
  //   ... use %token
  //
  // Transformed pattern:
  //   scf.if %condition {
  //     aref.put.exit                    // Separate exit operation
  //   } { .. loop.stage = 1, ttg.partition = {1}, ttg.partition.outputs = [] }
  //   %results, %poison_tok, %more = scf.if %condition {
  //     <computation_code>               // Main computation without token ops
  //     scf.yield %values, %poison_tok, %other_values
  //   } else {
  //     scf.yield %alt_values, %poison_tok, %alt_other_values
  //   } {.. ttg.partition = {0}, ttg.partition.outputs = [{0}, {0}, {0}, ..]}
  //   %token = scf.if %condition {
  //     %new_token = aref.put.enter      // Separate enter operation
  //     scf.yield %new_token
  //   } else {
  //     scf.yield %old_token
  //   } { .. loop.stage = 1, ttg.partition = {1}, ttg.partition.outputs =
  //   [{1}]}
  //   ... use %token

  for (auto ifOp : ifs) {
    ImplicitLocOpBuilder b(ifOp.getLoc(), ifOp);

    // move putExitOp
    b.setInsertionPoint(ifOp);
    auto exitIf =
        scf::IfOp::create(b, SmallVector<Type>{}, ifOp.getCondition(), false);
    auto putExitOp = cast<ArefPutExitOp>(*ifOp.thenBlock()->begin());
    putExitOp->moveBefore(exitIf.thenBlock(), exitIf.thenBlock()->begin());

    // move putEnterOp
    b.setInsertionPointAfter(ifOp);
    auto enterIf =
        scf::IfOp::create(b, SmallVector<Type>{b.getType<AsyncTokenType>()},
                          ifOp.getCondition(), true);
    auto putEnterOp =
        cast<ArefPutEnterOp>(ifOp.thenBlock()->getTerminator()->getPrevNode());
    putEnterOp->moveBefore(enterIf.thenBlock(), enterIf.thenBlock()->begin());

    // replace token uses
    auto tok = putEnterOp.getToken();
    auto pos = *findValuePosInRange(ifOp.thenYield()->getOperands(), tok);
    ifOp.getResult(pos).replaceAllUsesWith(enterIf.getResult(0));

    // insert yield-ops inside enterIf
    b.setInsertionPointToEnd(enterIf.thenBlock());
    scf::YieldOp::create(b, tok);
    b.setInsertionPointToEnd(enterIf.elseBlock());
    scf::YieldOp::create(b, ifOp.elseYield().getOperand(pos));

    // invalid tokens in main ifOp
    b.setInsertionPoint(ifOp);
    auto poisonToken = ub::PoisonOp::create(b, b.getType<AsyncTokenType>());
    ifOp.thenYield().setOperand(pos, poisonToken);
    ifOp.elseYield().setOperand(pos, poisonToken);

    // patch loop.stage=1
    enterIf->setAttrs(ifOp->getAttrs());
    exitIf->setAttrs(ifOp->getAttrs());
    assignStage(b, enterIf, getStageCluster(putEnterOp));
    assignStage(b, exitIf, getStageCluster(putExitOp));

    SetVector<int> enterExitIds, middleIds;
    enterExitIds.insert(1);
    middleIds.insert(0);
    setPartition(enterIf, enterExitIds);
    setPartition(exitIf, enterExitIds);
    setPartition(ifOp, middleIds);

    SetVector<int> p0array, p1array;
    p0array.insert(0);
    p1array.insert(1);
    setPartitionOutputs(exitIf, {});
    setPartitionOutputs(enterIf, {p1array});
    SmallVector<SetVector<int>> outputs(ifOp->getNumResults(), p0array);
    setPartitionOutputs(ifOp, outputs);
  }
}

LogicalResult runOnFunction(triton::FuncOp funcOp) {
  SmallVector<TmemAccessDag> tmemDags;
  funcOp.walk([&](TMEMAllocOp allocOp) {
    tmemDags.push_back(TmemAccessDag::build(allocOp));
  });

  int numTmemBlocks = 0;
  for (auto &accessDag : tmemDags) {
    LLVM_DEBUG({ accessDag.printDag(llvm::dbgs()); });
    auto [hasRootPartition, partitions] = accessDag.collectPartitionsSet();
    assert(partitions.size() <= 2 && "expecting at most 2 partitions");
    auto totalOwners = hasRootPartition + partitions.size();
    if (totalOwners > 1) {
      numTmemBlocks = insertTmemAref(accessDag, numTmemBlocks);
    }
  }

  workaroundForLoopScheduler(funcOp);

  return success();
}

} // namespace

class NVWSTmemArefInsertion
    : public triton::impl::NVWSInsertTmemArefBase<NVWSTmemArefInsertion> {
public:
  void runOnOperation() override {
    getOperation().walk([&](triton::FuncOp funcOp) {
      if (failed(runOnFunction(funcOp)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
  }
};

} // namespace triton
} // namespace mlir
