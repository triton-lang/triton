#include <algorithm>

#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Utility.h"

#include "mlir/Analysis/SliceAnalysis.h"

using namespace mlir;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

// Target operations: dot, load, store. Add more when necessary.
#define KEY_TYPES                                                              \
  triton::DotOp, triton::nvidia_gpu::DotAsyncOp, ttg::InsertSliceAsyncOp,      \
      triton::StoreOp

template <typename Head, typename... Tails>
void getKeyTypeId(Operation *op, int &id, bool &found) {
  if (isa<Head>(op))
    found = true;

  if (!found) {
    id++;
    if constexpr (sizeof...(Tails) > 0)
      getKeyTypeId<Tails...>(op, id, found);
  }
}

template <typename... T> int getKeyTypeIdWrapper(Operation *op) {
  bool found = false;
  int id = 0;
  getKeyTypeId<T...>(op, id, found);
  return found ? id : -1;
}

bool isEligible(Operation *agent,
                DenseMap<int, DenseSet<Operation *>> &keyTypeOpMap,
                scf::ForOp &persistentForOp) {
  // metrics:
  //   1. Have more than one key type of operation.
  //   2. persistent (all key operations are in one forOp)
  DenseSet<int> keyTypes;
  DenseSet<Operation *> keyOperations;
  agent->walk([&](Operation *op) {
    auto typeId = getKeyTypeIdWrapper<KEY_TYPES>(op);
    if (typeId >= 0 && op != agent) {
      keyTypes.insert(typeId);
      keyOperations.insert(op);
      keyTypeOpMap[typeId].insert(op);
    }
  });

  if (keyTypes.size() <= 1) {
    return false;
  }

  auto getPersistentFor = [&](DenseSet<Operation *> keyOps,
                              scf::ForOp &innerMostForOp) -> bool {
    DenseSet<scf::ForOp> commonForOps0, commonForOps1;
    DenseSet<scf::ForOp> *commonForOpsPre = &commonForOps0,
                         *commonForOpsPro = &commonForOps1;
    assert(keyOps.size() > 1);
    SmallVector<scf::ForOp> forOps;
    agent->walk<WalkOrder::PreOrder>(
        [&](scf::ForOp forOp) { forOps.push_back(forOp); });

    bool hasCommon = false;
    for (auto &f : forOps) {
      bool isCommon = true;
      for (auto &k : keyOps) {
        if (!f->isAncestor(k)) {
          isCommon = false;
          break;
        }
      }
      if (isCommon) {
        innerMostForOp = f;
        hasCommon = true;
      }
    }
    return hasCommon;
  };

  // Persistent agents with more than one key types are eligible.
  return getPersistentFor(keyOperations, persistentForOp);
}

void mutexSync(ModuleOp &mod, scf::IfOp &ifOp, scf::ForOp &persistentForOp,
               DenseMap<int, DenseSet<Operation *>> &keyTypeOpMap) {
  // Modify keyTypeOpMap: DenseMap<int, DenseSet<Operation *>> --> DenseMap<int,
  // Operation *>. Conservetively, assign each key operation one mutex.
  // =======================detail description (TODO: to be
  // deleted)========================== because it's hard to check if two
  // operations with same typeid can share same mutex, we assign each key
  // operation one mutex. To illustrate the hardness of this analysis, say we
  // have two operations with same typeid: a and b, if there is another
  // operation (say c) of different typeid between a and b, and their locations
  // are a -- c -- b, then if the dependency is:
  //   * b depends on c, then a and b can NOT share the same mutex.
  //   * otherwise, a and b can share after move b before c.
  // It would be more complicated when there are more types and operations.
  DenseMap<int, Operation *> ProxyKeyTypeOpMap;
  for (auto &[id, ops] : keyTypeOpMap) {
    for (auto itr = ops.begin(); itr != ops.end(); ++itr) {
      auto op = *itr;
      ProxyKeyTypeOpMap[ProxyKeyTypeOpMap.size()] = op;
    }
  }

  int numRoles = ProxyKeyTypeOpMap.size();
  auto loc = ifOp.getLoc();
  OpBuilderWithAgentIds builder(ifOp.getContext());
  // Set num-roles for wsmaterialization pass
  ifOp->setAttr("agent.num-roles", builder.getI32IntegerAttr(numRoles));
  builder.setAgentIdsFromOp(ifOp);
  builder.setInsertionPointToStart(&(ifOp.getThenRegion().front()));
  Value _0 = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value curRoleId =
      builder.createWithAgentIds<ttng::GetMutexRoleIdOp>(loc, numRoles);
  Value isNotRole0 = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ne, curRoleId, _0);

  SmallVector<Value> mutexBarriers;
  for (int i = 0; i < numRoles; ++i) {
    auto v = builder.createWithAgentIds<ttng::CreateMutexOp>(loc);
    mutexBarriers.push_back(v);
  }

  // Update lower bound, step and pipelineIdx of persistentForOp
  builder.setInsertionPoint(persistentForOp);
  Value start = builder.createWithAgentIds<arith::MulIOp>(
      loc, persistentForOp.getStep(), curRoleId);
  Value oldLB = persistentForOp.getLowerBound();
  Value pipelineIdx =
      persistentForOp->getOperand(persistentForOp->getNumOperands() - 1);

  start = builder.createWithAgentIds<arith::AddIOp>(loc, oldLB, start);
  persistentForOp.setLowerBound(start);

  Value numRolesValue =
      builder.createWithAgentIds<arith::ConstantIntOp>(loc, numRoles, 32);
  Value step = builder.createWithAgentIds<arith::MulIOp>(
      loc, persistentForOp.getStep(), numRolesValue);
  persistentForOp.setStep(step);

  Value newIdx =
      builder.createWithAgentIds<arith::AddIOp>(loc, pipelineIdx, curRoleId);
  persistentForOp.getInitArgsMutable()
      .slice(persistentForOp.getInitArgs().size() - 1, 1)
      .assign(newIdx);

  pipelineIdx = persistentForOp.getBody()->getArgument(
      persistentForOp.getBody()->getNumArguments() - 1);
  Operation *idxPlusOneOp = nullptr;
  for (OpOperand &v : pipelineIdx.getUses()) {
    if (isa<arith::AddIOp>(v.getOwner())) {
      idxPlusOneOp = v.getOwner();
      break;
    }
  }
  assert(idxPlusOneOp && "idxPlusOneOp should be arith::AddIOp");
  Operation *use = *idxPlusOneOp->getUsers().begin();
  assert(isa<scf::YieldOp>(use) || isa<arith::SelectOp>(use) ||
         isa<arith::CmpIOp>(use));
  idxPlusOneOp->setOperand(1, numRolesValue);

  // Add operations at the start of persistentForOp
  builder.setInsertionPointToStart(persistentForOp.getBody());
  // If( role != 0 || !is_first_tile )
  Value isNotTileId0 = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ne, persistentForOp.getBody()->getArgument(0),
      oldLB);
  Value cond = builder.create<arith::OrIOp>(loc, isNotTileId0, isNotRole0);

  // Determine boundaries: get the largest exclusive op for each key op.
  DenseMap<int, Operation *> lockLocs, unlockLocs;
  DenseMap<int, SmallVector<Operation *>> parentOps;
  for (int i = 0; i < numRoles; ++i) {
    auto op = ProxyKeyTypeOpMap[i]->getParentOp();
    while (op != persistentForOp->getParentOp()) {
      parentOps[i].push_back(op);
      op = op->getParentOp();
    }
  }

  std::map<std::pair<int, int>, std::pair<SmallVector<Operation *>::iterator,
                                          SmallVector<Operation *>::iterator>>
      rangeMap;
  for (auto &[i, opsI] : parentOps) {
    // Check exlusiveness
    auto op = ProxyKeyTypeOpMap[i];
    for (auto &[j, opsJ] : parentOps) {
      if (i == j)
        continue;
      auto pair = std::pair<int, int>(i, j);
      auto pairConj = std::pair<int, int>(j, i);
      auto end0 = rangeMap.count(pair) ? rangeMap[pair].first : opsI.end();
      auto end1 = rangeMap.count(pair) ? rangeMap[pair].second : opsJ.end();
      for (auto m = opsI.begin(); m != end0; ++m) {
        auto itr = std::find(opsJ.begin(), end1, *m);
        if (itr == end1) {
          op = *m;
          rangeMap[pair] = std::make_pair(m, itr);
          rangeMap[pairConj] = rangeMap[pair];
        } else
          goto exit;
      }
    }
  exit:;
    lockLocs[i] = op;
    unlockLocs[i] = op;
  }

  // Only cases where all lock/unlock locations are in same level make sense.
  for (int i = 1; i < numRoles; ++i) {
    if (lockLocs[i]->getParentOp() != lockLocs[i - 1]->getParentOp() ||
        unlockLocs[i]->getParentOp() != unlockLocs[i - 1]->getParentOp()) {
      llvm_unreachable("Only cases where all locl/unlock locations are in same "
                       "level make sense");
    }
  }

  // Extend boundaries: wait and release as early as possible
  DenseMap<int, int> prevTypeIds;
  int prevId = -1;
  persistentForOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
    for (int i = 0; i < numRoles; ++i) {
      if (lockLocs[i] == op) {
        prevTypeIds[i] = prevId;
        prevId = i;
        break;
      }
    }
  });

  // Update lockLocs
  for (int i = 0; i < numRoles; ++i) {
    if (prevTypeIds[i] == -1)
      lockLocs[i] = cond.getDefiningOp();
    else
      lockLocs[i] = unlockLocs[prevTypeIds[i]];
  }

  // Update lockLocs
  // ====================== IR after async launch dots ======================
  // * %0:2 = scf.for %arg0 = %c0 to %1 step %c1 iter_args(%arg1 = %2, arg2 =
  // %3) {
  // *    triton_nvidia_gpu.producer_wait arg2
  // *    %5 = triton_nvidia_gpu.dot_async %4, %5
  // *    triton_nvidia_gpu.dot_wait {pendings = 1}
  // *    %6 = arith.cmpi sgt, arg0, %c0
  // *    scf.if %6 {
  // *      %7 = arith.subi arg2, c1
  // *      triton_nvidia_gpu.consumer_release %7
  // *    }
  // *    %8 = arith.addi arg2, c1
  // *    scf.yield %5, %8
  // * }
  // * triton_nvidia_gpu.dot_wait {pendings = 0}
  // * ...
  // * triton_nvidia_gpu.consumer_release ..
  // * =======================================================================
  // after async launch dots, there will be outstanding consumerReleaseOp after
  // ForOp. we should set the epilogue lockLocs after the outstanding
  // consumerReleaseOp.
  for (int i = 0; i < numRoles; ++i) {
    Operation *lockOp = lockLocs[i];
    if (isa<scf::ForOp>(lockOp)) {
      Operation *loc = nullptr;
      unsigned numOutstandingConsumerRelease = 0;
      for (auto v : lockOp->getResults()) {
        SetVector<Operation *> slices;
        mlir::getForwardSlice(v, &slices);
        auto iter = llvm::find_if(slices, [](Operation *op) {
          return isa<triton::nvidia_gpu::ConsumerReleaseOp>(op);
        });
        if (iter != slices.end()) {
          numOutstandingConsumerRelease++;
          loc = *iter;
        }
      }
      assert(numOutstandingConsumerRelease <= 1 &&
             "should have only one outstanding "
             "consumerReleaseOp after "
             "async launch dots");
      if (loc)
        lockLocs[i] = loc;
    }
  }

  // lock
  for (int i = 0; i < numRoles; ++i) {
    builder.setInsertionPointAfter(lockLocs[i]);
    auto waitIfOp = builder.create<scf::IfOp>(loc, cond);
    builder.setInsertionPointToStart(&(waitIfOp.getThenRegion().front()));
    builder.create<ttng::LockOp>(loc, mutexBarriers[i]);
  }

  // unlock
  for (int i = 0; i < numRoles; ++i) {
    builder.setInsertionPointAfter(unlockLocs[i]);
    builder.create<ttng::UnlockOp>(loc, mutexBarriers[i]);
  }

  // Add attr "agent.mutex_role" for barrier analysis
  int roleId = -1;
  for (Operation &bodyOp : lockLocs[0]->getBlock()->getOperations()) {
    Operation *op = &bodyOp;
    if (roleId != -1)
      op->walk([&](Operation *subOp) {
        if (!isa<scf::YieldOp>(op) && !isa<ttng::LockOp>(op) &&
            !isa<ttng::UnlockOp>(op))
          subOp->setAttr("agent.mutex_role", builder.getI32IntegerAttr(roleId));
      });
    for (int i = 0; i < numRoles; ++i) {
      if (lockLocs[i] == op) {
        if (roleId != -1)
          op->setAttr("agent.mutex_role", builder.getI32IntegerAttr(roleId));
        roleId = i;
        break;
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// WSMaterializationPass
//===----------------------------------------------------------------------===//

struct WSMutexPass : public TritonGPUWSMutexBase<WSMutexPass> {
public:
  WSMutexPass() = default;
  WSMutexPass(int computeCapability) {
    this->computeCapability = computeCapability;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    mod.walk([&](triton::FuncOp funcOp) {
      for (Operation &bodyOp : funcOp.getBody().front().getOperations()) {
        Operation *op = &bodyOp;
        scf::ForOp persistentForOp;
        // premise: agent region is encapsulated with scf.if
        if (isa<scf::IfOp>(op) && getAgentIds(op).size() == 1) {
          DenseMap<int, DenseSet<Operation *>> keyTypeOpMap;
          if (isEligible(op, keyTypeOpMap, persistentForOp)) {
            auto ifOp = cast<scf::IfOp>(op);
            mutexSync(mod, ifOp, persistentForOp, keyTypeOpMap);
          }
        }
      }
    });
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// createTritonNvidiaGPUWSMutexPass
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass>
mlir::createTritonNvidiaGPUWSMutexPass(int computeCapability) {
  return std::make_unique<WSMutexPass>(computeCapability);
}
