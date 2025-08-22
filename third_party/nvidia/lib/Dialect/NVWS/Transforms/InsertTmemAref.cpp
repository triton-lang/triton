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

  Node *getNode(Operation *op) { return op2dagMap.lookup(op); }
  Node *getRootNode() { return dag.get(); }
  TMEMAllocOp getAllocOp() { return cast<TMEMAllocOp>(dag->op); }

  Value addIfOp(Value tok, Node *node) {
    SmallVector<OpOperand *> uses;
    for (auto &use : tok.getUses())
      uses.push_back(&use);
    assert(uses.size() == 2);
    assert(uses.size() == 2 && "expecting two uses of a token");
    auto useThen = uses[0];
    auto useElse = uses[1];

    auto ifOp = cast<scf::IfOp>(useThen->getOwner()->getParentOp());
    node->user.reset(new Node(ifOp, nullptr, {}, node));
    auto ifOpNode = node->user.get();
    op2dagMap.insert({ifOp, ifOpNode});

    if (ifOp.thenBlock() != useThen->getOwner()->getBlock())
      std::swap(useThen, useElse);
    assert(ifOp.thenBlock() == useThen->getOwner()->getBlock());
    assert(ifOp.elseBlock() == useElse->getOwner()->getBlock());

    auto partitionId = getPartitionId(tok.getDefiningOp());
    // Create access DAGs for then/else blocks. T
    auto thenDag =
        std::make_unique<Node>(nullptr, nullptr, std::nullopt, nullptr);
    auto elseDag =
        std::make_unique<Node>(nullptr, nullptr, std::nullopt, nullptr);
    auto thenTok = addOp(*useThen, thenDag.get());
    auto elseTok = addOp(*useElse, elseDag.get());

    // heuristic: assign if-Op to partition of the token producer op
    ifOpNode->partitionId = partitionId;

    auto tokPos =
        *findValuePosInRange(ifOp.thenYield()->getOperands(), thenTok);

    // the only use case we have today is
    //   %newTok = if .. {  yield %tok  } else { %tok1= .. ; yield %tok1 }
    // pick correct branch of if-stmt
    bool useThenTok = ifOp.thenYield().getOperand(tokPos) != tok;
    auto yieldOp = useThenTok ? ifOp.thenYield() : ifOp.elseYield();

    partitionId = getPartitionId(yieldOp.getOperand(tokPos).getDefiningOp());
    if (!partitionId) {
      // if op producing token has no partition assigne, use the one from ifOp
      // assigned by scheduler
      partitionId = getPartitionId(ifOp);
      auto newTokOp = yieldOp.getOperand(tokPos).getDefiningOp();
      getNode(newTokOp)->partitionId = partitionId;
    }

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
    auto tok = addOp(*tokArg.getUses().begin(), subDag.get());
    forOpNode->partitionId = subDag->user->partitionId;

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
    node->user.reset(new Node(op, &tokOperand, getPartitionId(op), node));
    auto newNode = node->user.get();
    op2dagMap.insert({op, newNode});
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

    // Mutiple uses of token are expected only in IfOp: one in then and one in
    // else branches.
    return addIfOp(newTok, newNode);
  }

  static TmemAccessDag build(TMEMAllocOp allocOp) {
    TmemAccessDag accessDag(std::make_unique<Node>(
        allocOp, nullptr, getPartitionId(allocOp), nullptr));
    accessDag.op2dagMap.insert({allocOp, accessDag.getRootNode()});

    if (allocOp.getSrc()) {
      // Handle tmem_alloc with src operand specially. When a src operand is
      // present, no async tokens are generated, we can't traverse IR,
      // and we directly add the single user operation to the access DAG.
      assert(!allocOp.getToken());
      assert(allocOp->hasOneUse());
      auto user = *allocOp->getUsers().begin();
      accessDag.getRootNode()->user.reset(new Node{
          user, nullptr, getPartitionId(user), accessDag.getRootNode()});
      accessDag.op2dagMap.insert({user, accessDag.getRootNode()->user.get()});
    } else {
      auto tok = allocOp.getToken();
      assert(tok && tok.hasOneUse());
      auto &tokUse = *tok.getUses().begin();
      accessDag.addOp(tokUse, accessDag.getRootNode());
    }
    return accessDag;
  }

  std::set<PartitionId> collectPartitions(Node *node) {
    std::set<PartitionId> partitions;
    if (node->partitionId)
      partitions.insert(*node->partitionId);

    while (node->user) {
      node = node->user.get();
      if (node->partitionId)
        partitions.insert(*node->partitionId);
      for (auto &subDag : node->subDags) {
        if (subDag) {
          auto ps = collectPartitions(subDag.get());
          partitions.insert(ps.begin(), ps.end());
        }
      }
    }
    return partitions;
  };

  void printNode(Node *node, int indent, llvm::raw_ostream &os) {
    if (!node)
      return;
    for (int i = 0; i < indent; i++) {
      os << " ";
    }
    std::set<PartitionId> partitions;
    os << "|- [" << node->op << "]";
    if (node->partitionId)
      partitions.insert(*node->partitionId);
    if (node->op) {
      os << node->op->getName().getStringRef() << " ";
      if (auto tmemAlloc = dyn_cast<TMEMAllocOp>(node->op)) {
        if (tmemAlloc.getSrc()) {
          os << " %src ";
        } else {
          partitions = collectPartitions(node);
        }
      }
      os << "  ";
    }
    os << "[";
    for (auto partition : partitions) {
      os << " @" << partition.tag() << "." << partition.index() << " ";
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
  DenseMap<Operation *, Node *> op2dagMap;
  DenseMap<scf::ForOp, TMEMAllocOp> arefTmemAllocs;
};

Value intCst(OpBuilder &b, Location loc, int value, unsigned width) {
  return b.create<arith::ConstantIntOp>(loc, value, width);
}

Value boolCst(OpBuilder &b, Location loc, bool value) {
  return intCst(b, loc, value, /*width=*/1);
}
void assignStage(OpBuilder &b, Operation *op, StageCluster stageCluster) {
  if (stageCluster) {
    op->setAttr(kLoopStageAttrName, b.getI32IntegerAttr(stageCluster->first));
    op->setAttr(kLoopClusterAttrName,
                b.getI32IntegerAttr(stageCluster->second));
  }
}

template <typename OpT, typename... Args>
OpT createInto(OpBuilder &b, Location loc,
               std::pair<std::optional<PartitionId>, StageCluster>
                   parititionIdStageCluster,
               Args &&...args) {
  auto op = b.create<OpT>(loc, std::forward<Args>(args)...);
  if (parititionIdStageCluster.first) {
    op->setAttr(kPartitionAttrName,
                b.getI32IntegerAttr(parititionIdStageCluster.first->index()));
    assignStage(b, op, parititionIdStageCluster.second);
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
    Type dataBufType = arefViewBufferType(arefBufType);
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
    buffer = {};
  }
  void release(OpBuilder &b, Location loc,
               std::pair<std::optional<PartitionId>, StageCluster>
                   paritionIdStageCluster) {
    assert(asyncOp);
    if (kind == PUT) {
      createInto<ArefPutExitOp>(
          b, loc, paritionIdStageCluster, aref, token,
          b.getArrayAttr(SmallVector<Attribute>{
              AsyncOpAttr::get(b.getContext(), *asyncOp)}));
      kind = GET;
    } else {
      createInto<ArefGetExitOp>(
          b, loc, paritionIdStageCluster, aref, token,
          b.getArrayAttr(SmallVector<Attribute>{
              AsyncOpAttr::get(b.getContext(), *asyncOp)}));
      kind = PUT;
    }
  }
  Value getBuffer(OpBuilder &b, std::optional<PartitionId> partitionId,
                  Operation *op) {
    if (!buffer) {
      auto stageCluster = getStageCluster(op);
      auto arefBufType =
          cast<MemDescType>(aref.getDefiningOp()->getOperand(0).getType());
      Type dataBufType = arefViewBufferType(arefBufType);
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
  std::optional<AsyncOp> asyncOp;
};

TmemAccessDag::Node *
insertTmemArefImpl(TmemAccessDag::Node *node,
                   std::optional<PartitionId> curPartitionId, TMEMAref &state) {
  if (curPartitionId && node->partitionId != curPartitionId) {
    OpBuilder b(node->op);
    Operation *prevOp = nullptr;
    std::optional<PartitionId> prevPartitionId;
    StageCluster prevStageCluster;
    if (node->parent) {
      // release right after the last op which owns the tmem
      prevOp = node->parent->op;
      b.setInsertionPointAfter(prevOp);
      prevPartitionId = node->parent->partitionId;
      prevStageCluster = getStageCluster(prevOp);
    } else {
      // if we are inside if-stmt or for-stmt subdag and need to change
      // ownerhip, release at the top of the block
      // the parentDag op would be if-stmt or for-stmt
      prevOp = node->parentDag->op;
      b.setInsertionPointToStart(node->op->getBlock());
      prevPartitionId = node->parentDag->partitionId;
    }
    if (!node->partitionId) {
      // if node->partitionId is not set, it means we are outside ws-region
      // reset prevPartitionId and prevStageCluster to defaults
      prevPartitionId = {};
      prevStageCluster = {};
    }
    state.release(b, prevOp->getLoc(), {prevPartitionId, prevStageCluster});

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
  }

  if (isa<MMAv5OpInterface>(node->op)) {
    state.asyncOp = AsyncOp::TC5MMA;
  } else if (isa<TMEMLoadOp, TMEMStoreOp>(node->op)) {
    state.asyncOp = AsyncOp::NONE;
  }

  OpBuilder b(node->op);
  if (auto tmemLoadOp = dyn_cast<TMEMLoadOp>(node->op)) {
    tmemLoadOp.getSrcMutable().assign(
        state.getBuffer(b, node->partitionId, node->op));
    tmemLoadOp.getDepMutable().clear();
    tmemLoadOp.getToken().replaceAllUsesWith(state.replToken);
  } else if (auto tmemStoreOp = dyn_cast<TMEMStoreOp>(node->op)) {
    tmemStoreOp.getDstMutable().assign(
        state.getBuffer(b, node->partitionId, node->op));
    tmemStoreOp.getDepMutable().clear();
    tmemStoreOp.getToken().replaceAllUsesWith(state.replToken);
  } else if (auto mmaOp = dyn_cast<MMAv5OpInterface>(node->op)) {
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

LogicalResult insertTmemAref(TmemAccessDag &accessDag) {
  auto rootNode = accessDag.getRootNode();
  auto allocOp = cast<TMEMAllocOp>(rootNode->op);

  std::optional<bool> isMultiStaged;
  for (auto user : allocOp.getResult().getUsers()) {
    if (auto mmaOp = dyn_cast<MMAv5OpInterface>(user)) {
      if (auto loop = dyn_cast<scf::ForOp>(user->getParentOp())) {
        // Determine if the MMA accumulator can be multibuffered.
        bool accIsMultiBuffered =
            // MMAs in subsequent iterations can be overlapped.
            !nvidia_gpu::hasAccReadModifyWrite(mmaOp, loop) &&
            // The accumulator is reset at some point, thus allowing
            // multibuffering.
            isAccMultibufferingPossible(mmaOp, loop) &&
            // The user didn't disable it with a flag.
            !getDisallowAccMultiBuffer(loop);
        isMultiStaged = isMultiStaged ? *isMultiStaged && accIsMultiBuffered
                                      : accIsMultiBuffered;
      }
    }
  }
  auto numStages = isMultiStaged ? (1 + *isMultiStaged) : 1;
  auto arefBufType =
      arefMultiBufferedType(allocOp.getResult().getType(), numStages);
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

  TMEMAref state(
      arefOp, allocOp.getResult(),
      b.create<ub::PoisonOp>(allocOp.getLoc(), b.getType<AsyncTokenType>()));

  b.setInsertionPoint(allocOp);
  state.acquire(b, allocOp.getLoc(), {partitionId, stageCluster});

  if (auto src = allocOp.getSrc()) {
    auto buffer = state.getBuffer(b, partitionId, allocOp);
    state.asyncOp = AsyncOp::NONE;
    createInto<TMEMStoreOp>(b, allocOp.getLoc(), {partitionId, stageCluster},
                            b.getType<AsyncTokenType>(), buffer, state.token,
                            src, boolCst(b, allocOp.getLoc(), true));
  } else {
    // allocOp w/o src, assume the ownership of tmem belongs to first user
    // partitionId = accessDag.getRootNode()->user->partitionId;
  }

  auto node = insertTmemArefImpl(rootNode->user.get(), partitionId, state);

  if (outerWsLoop) {
    // aref is only used inside ws-loop, so we use the last op to insert
    // matching exit
    partitionId = node->partitionId;
    stageCluster = getStageCluster(node->op);
    b.setInsertionPointAfter(node->op);
  } else {
    // aref is used outside ws-loop, find the last point in the same block as
    // create op to have matching exit
    partitionId = {};
    stageCluster = {};
    auto op1 = arefOp->getBlock()->findAncestorOpInBlock(*node->op);
    b.setInsertionPointAfter(op1);
  }
  state.release(b, node->op->getLoc(), {partitionId, stageCluster});

  return success();
}

LogicalResult runOnFunction(triton::FuncOp funcOp) {
  SmallVector<TmemAccessDag> tmemDags;
  funcOp.walk([&](TMEMAllocOp allocOp) {
    // if allocOp has src and has no partition, we skip it
    if (!allocOp.getSrc() || getPartitionId(allocOp))
      tmemDags.push_back(TmemAccessDag::build(allocOp));
  });

  for (auto &accessDag : tmemDags) {
    LLVM_DEBUG({ accessDag.printDag(llvm::dbgs()); });
    auto partitions = accessDag.collectPartitions(accessDag.getRootNode());
    assert(partitions.size() <= 2 && "expecting at most 2 partitions");
    if (!partitions.empty())
      if (failed(insertTmemAref(accessDag)))
        return failure();
  }
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
