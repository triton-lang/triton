#include <fstream>
#include <iostream>
#include <memory>
#include <stack>

#include "WSUtility.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUANALYZEWARPSPECIALIZATION
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritongpu-analyze-warp-specialization"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

triton::nvidia_gpu::MMAv5OpInterface
getMMAv5Op(triton::nvidia_gpu::TMEMLoadOp tmemLoadOp) {
  auto tmemAddr = tmemLoadOp->getOperand(0);
  for (auto user : tmemAddr.getUsers()) {
    if (auto mmaOp = dyn_cast<triton::nvidia_gpu::MMAv5OpInterface>(user)) {
      return mmaOp;
    }
  }
  return nullptr;
};

struct NumWarpsSpec {
  int load;
  int MMA;
  int epilogue;
};

class Traverser {
public:
  Traverser(NumWarpsSpec numWarps) : numWarps(numWarps) {}

  SymbolRefAttr tmaLoadGroup(ModuleOp moduleOp, int idx = 0) {
    assert(idx == 0);
    std::string name =
        std::string(ATTR_WS_TMALOAD) + (idx > 0 ? std::to_string(idx) : "");
    return mkGroup(moduleOp, name,
                   WSGroup{numWarps.epilogue + numWarps.MMA, numWarps.load});
  }

  SymbolRefAttr mmaGroup(ModuleOp moduleOp, int idx = 0) {
    assert(idx == 0);
    std::string name =
        std::string(ATTR_WS_MMA) + (idx > 0 ? std::to_string(idx) : "");
    return mkGroup(moduleOp, name, WSGroup{numWarps.epilogue, numWarps.MMA});
  }

  SymbolRefAttr epilogueGroup(ModuleOp moduleOp, int idx = 0) {
    assert(idx == 0);
    std::string name =
        std::string(ATTR_WS_EPILOGUE) + (idx > 0 ? std::to_string(idx) : "");
    return mkGroup(moduleOp, name, WSGroup{0, numWarps.epilogue});
  }

  void assignGroupsForOp(scf::ForOp forOp) {
    for (auto op = forOp.getBody()->getOperations().rbegin();
         op != forOp.getBody()->getOperations().rend(); ++op) {
      assignGroups(&(*op));
    }
  }
  void assignGroups(Operation *op) {
    if (assignVisited.count(op) > 0) {
      return;
    }
    assignVisited.insert(op);

    OpBuilder builder(op);

    // load ops placed in load warp group
    // FIXME: TMA can also be used for loading bias etc in the epilouge. Those
    // loads must be in a different group than the load group that feeds MMA .
    if (mlir::isa<triton::DescriptorLoadOp>(op)) {
      auto tmaLoad = tmaLoadGroup(op->getParentOfType<ModuleOp>());
      mlir::ArrayAttr list = builder.getArrayAttr({tmaLoad});
      op->setAttr(ATTR_WSGROUPS, list);
    }

    // store and matmulops placed in math warp group
    if (mlir::isa<triton::DotOp, triton::nvidia_gpu::WarpGroupDotOp,
                  triton::StoreOp, triton::DescriptorStoreOp,
                  triton::gpu::MemDescTransOp, triton::nvidia_gpu::TMEMAllocOp,
                  triton::nvidia_gpu::TMEMLoadOp,
                  triton::nvidia_gpu::MMAv5OpInterface>(op)) {
      auto mma = mmaGroup(op->getParentOfType<ModuleOp>());
      mlir::ArrayAttr list = builder.getArrayAttr({mma});
      op->setAttr(ATTR_WSGROUPS, list);
    }

    if (numWarps.epilogue > 0) {
      if (mlir::isa<triton::StoreOp, triton::DescriptorStoreOp>(op)) {
        auto epi = epilogueGroup(op->getParentOfType<ModuleOp>());
        mlir::ArrayAttr list = builder.getArrayAttr({epi});
        op->setAttr(ATTR_WSGROUPS, list);
      } else if (mlir::isa<triton::nvidia_gpu::TMEMLoadOp>(op)) {
        // if tma_load is in the same loop as corresponding mma, it's not part
        // of the epilogue group
        auto tmemLoadOp = dyn_cast<triton::nvidia_gpu::TMEMLoadOp>(op);
        auto mmav5Op = getMMAv5Op(tmemLoadOp);
        if (mmav5Op && mmav5Op->getBlock() != tmemLoadOp->getBlock()) {
          auto epi = epilogueGroup(op->getParentOfType<ModuleOp>());
          mlir::ArrayAttr list = builder.getArrayAttr({epi});
          op->setAttr(ATTR_WSGROUPS, list);
        }
      }
    }

    if (auto if_op = dyn_cast<scf::IfOp>(op)) {
      assignGroupsIfOp(if_op);
    }
  }

  void assignGroupsIfOp(scf::IfOp ifOp) {
    // visit all ops in the then and else block of the if
    // if any of the ops are in parition n, the if is in group n

    if (auto thenBlock = ifOp.thenBlock()) {
      for (auto op = thenBlock->getOperations().rbegin();
           op != ifOp.thenBlock()->getOperations().rend(); ++op) {
        assignGroups(&(*op));
      }
    }
    if (auto elseBlock = ifOp.elseBlock()) {
      for (auto op = elseBlock->getOperations().rbegin();
           op != elseBlock->getOperations().rend(); ++op) {
        assignGroups(&(*op));
      }
    }

    std::set<std::string> groups;
    if (auto thenBlock = ifOp.thenBlock()) {
      for (auto op = thenBlock->getOperations().rbegin();
           op != thenBlock->getOperations().rend(); ++op) {
        if (op->hasAttr(ATTR_WSGROUPS)) {
          auto opGroups = getGroups(&(*op), ATTR_WSGROUPS);
          groups.insert(opGroups.begin(), opGroups.end());
        }
      }
    }
    if (auto elseBlock = ifOp.elseBlock()) {
      for (auto op = elseBlock->getOperations().rbegin();
           op != elseBlock->getOperations().rend(); ++op) {
        if (op->hasAttr(ATTR_WSGROUPS)) {
          auto opGroups = getGroups(&(*op), ATTR_WSGROUPS);
          groups.insert(opGroups.begin(), opGroups.end());
        }
      }
    }
    if (!groups.empty()) {
      OpBuilder builder(ifOp);
      setGroups(ifOp, ATTR_WSGROUPS, groups);
    }
  }

  void propagateGroups(scf::ForOp forOp) {
    DenseSet<Operation *> roots;

    auto isRootOp = [](Operation *op) {
      if (mlir::isa<triton::LoadOp, triton::DescriptorLoadOp, triton::DotOp,
                    triton::nvidia_gpu::WarpGroupDotOp, triton::StoreOp,
                    triton::DescriptorStoreOp, triton::nvidia_gpu::TMEMAllocOp,
                    triton::nvidia_gpu::TMEMLoadOp,
                    triton::nvidia_gpu::MMAv5OpInterface, scf::IfOp>(op)) {
        return true;
      }
      return false;
    };

    for (auto op = forOp.getBody()->getOperations().rbegin();
         op != forOp.getBody()->getOperations().rend(); ++op) {
      if (op->hasAttr(ATTR_WSGROUPS) && isRootOp(&(*op))) {
        // It's a bit hacky. We do not want to propagate beyond the ops listed
        // above, but since partitioning is done recursively across nested
        // loops, ops that we do want to propagate across might already have a
        // partition assigned to it by an eariler call to this function.
        roots.insert(&(*op));
      }
    }
    for (auto op : roots) {
      propagateGroups(forOp, op, roots);
    }

    {
      // Assign groups to loop lower / upper bounds, and step
      std::set<std::string> groupsFor;
      for (auto &op : forOp.getBody()->getOperations()) {
        if (auto attr = op.getAttr(ATTR_WSGROUPS)) {
          auto groups = getGroups(&op, ATTR_WSGROUPS);
          groupsFor.insert(groups.begin(), groups.end());
        }
      }

      OpBuilder builder(forOp);
      setGroups(forOp, ATTR_WSGROUPS, groupsFor);

      for (auto loopProp :
           {forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep()}) {
        auto op = loopProp.getDefiningOp();
        if (!op) {
          continue;
        }
        setGroups(op, ATTR_WSGROUPS, groupsFor);
        propagateGroups(op);
      }
    }

    {
      // propagate from ops that have no groups assigned
      // group won't be assigned if an op is not in the producer chain for a
      // root op but we assume it has a producer op that is chained to a root op
      using groups_t = std::set<std::string>;
      std::function<groups_t(Operation *)> visitor =
          [&](Operation *op) -> groups_t {
        // op has groups, return them
        if (op->hasAttr(ATTR_WSGROUPS)) {
          return getGroups(op, ATTR_WSGROUPS);
        }

        // otherwise visit op's operands, and merge their groups
        groups_t opGroups;
        for (auto operand : op->getOperands()) {
          auto op = operand.getDefiningOp();
          if (op == nullptr) {
            // operand not defined by an op, skip
            continue;
          }
          auto parent = op->getParentOp();
          while (parent != forOp && parent != nullptr) {
            parent = parent->getParentOp();
          }
          if (parent != forOp) {
            // op is not in the for loop body, skip
            continue;
          }
          // get operand groups
          auto groups = visitor(op);
          if (!groups.empty()) {
            // for now we assume that all operands comes form the ops assign to
            // the same group so we break the loop once we get groups
            opGroups.insert(groups.begin(), groups.end());
            break;
          }
        }
        assert(!op->hasAttr(ATTR_WSGROUPS));
        // set groups
        setGroups(op, ATTR_WSGROUPS, opGroups);
        return opGroups;
      };

      // Propagate group IDs from operands to block arguments of the
      // for-loop. Each block argument should have a group assigned.
      // However, it may happen that a block argument, which was not used by the
      // root op, does not have a group assigned. Traverse from the operand
      // all the way to the root op to ensure group are assigned.
      auto yieldOp = mlir::cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      std::function<void(Value operand, groups_t)> visitor_blockarg =
          [&](Value operand, groups_t groups) {
            if (auto blockArg = mlir::dyn_cast<BlockArgument>(operand)) {
              if (blockArg.getOwner()->getParentOp() == forOp) {
                // this is an iter arg for the for loop
                // find the corresponding value from the yield and through the
                // loop body
                // XXX: add comment why we skip argNumber = 0
                if (blockArg.getArgNumber() > 0) {
                  auto idx = blockArg.getArgNumber() - 1;
                  operand = yieldOp.getOperands()[idx];
                  // operand is the yield op operand

                  auto attrName =
                      std::string(std::string(ATTR_WSGROUPS) + ".") +
                      std::to_string(idx);
                  if (!forOp->hasAttr(attrName)) {
                    // assign operand groups to a block arg
                    setGroups(forOp, attrName, groups);
                  }
                }
              }
            } else {

              // now visit operands of defining op
              auto op = operand.getDefiningOp();
              if (op == nullptr) {
                // operand not defined by an op, skip
                return;
              }
              auto parent = op->getParentOp();
              while (parent != forOp && parent != nullptr) {
                parent = parent->getParentOp();
              }
              if (parent != forOp) {
                // op is not in the for loop body, skip
                return;
              }
              for (auto operand1 : op->getOperands()) {
                visitor_blockarg(operand1, groups);
              }
            }
          };

      // find non-roots ops that have no groups assigned
      std::set<Operation *> nonRoots;
      for (auto &op : forOp.getBody()->without_terminator()) {
        if (roots.count(&op) == 0 && !op.hasAttr(ATTR_WSGROUPS)) {
          // assign group to op
          visitor(&op);
          auto groups = getGroups(&op, ATTR_WSGROUPS);
          // assing groups, if needed, to block arg reachable from the op
          for (auto operand : op.getOperands()) {
            visitor_blockarg(operand, groups);
          }
        }
      }
    }
  }

  void propagateGroups(scf::ForOp forOp, Operation *op,
                       const DenseSet<Operation *> &roots) {
    // if forOp is null, just propagate the group ID of op to its operand
    OpBuilder builder(op);
    DenseSet<Operation *> visited;
    std::stack<Operation *> stack;
    stack.push(op);
    visited.insert(op);

    auto groups = getGroups(op, ATTR_WSGROUPS);

    // visit all ops reachable from the initial op, within the body of the for
    // loop propagating group ids, until another "root" op is found Note: we
    // don't propagate forwards to uses, just backwards to operands, as the
    // final ops in the loop body should be root ops (e.g. a store). Anything
    // else is dead code.
    while (!stack.empty()) {
      auto op = stack.top();
      stack.pop();

      auto visitOperand = [&](Value operand) {
        if (auto blockArg = mlir::dyn_cast<BlockArgument>(operand)) {
          if (forOp && blockArg.getOwner()->getParentOp() == forOp) {
            auto yieldOp =
                mlir::cast<scf::YieldOp>(forOp.getBody()->getTerminator());
            // this is an iter arg for the for loop
            // find the corresponding value from the yield and through the loop
            // body
            // XXX: add comment why we skip argNumber = 0
            if (blockArg.getArgNumber() > 0) {
              auto idx = blockArg.getArgNumber() - 1;
              operand = yieldOp.getOperands()[idx];
              // operand is the yield op operand

              auto attrName = std::string(std::string(ATTR_WSGROUPS) + ".") +
                              std::to_string(idx);
              if (forOp->hasAttr(attrName)) {
                // merge with existing group ids
                auto newForOpGroups = getGroups(forOp, attrName);
                newForOpGroups.insert(groups.begin(), groups.end());
                setGroups(forOp, attrName, newForOpGroups);
              } else {
                // just set them
                setGroups(forOp, attrName, groups);
              }
            }
          } else {
            // skip
            return;
          }
        }

        auto op = operand.getDefiningOp();
        if (op == nullptr) {
          // operand not defined by an op, skip
          return;
        }
        if (roots.count(op) > 0) {
          // op is a root, skip
          return;
        }
        if (visited.count(op) > 0) {
          auto opGroups = getGroups(op, ATTR_WSGROUPS);
          bool hasNewID = false;
          for (auto group : groups) {
            if (!opGroups.count(group)) {
              hasNewID = true;
            }
          }

          if (!hasNewID) {
            // already propagated to this op, skip
            return;
          }

          return;
        }
        stack.push(op);
        visited.insert(op);

        // set groups
        if (op->hasAttr(ATTR_WSGROUPS)) {
          // merge with exists partition ids
          auto opGroups = getGroups(op, ATTR_WSGROUPS);
          opGroups.insert(groups.begin(), groups.end());
          setGroups(op, ATTR_WSGROUPS, opGroups);
        } else {
          // just set them
          setGroups(op, ATTR_WSGROUPS, groups);
        }
      };

      for (auto operand : op->getOperands()) {
        visitOperand(operand);
      }

      // FIXME: walk all regions, blocks and ops contained in the op, don't
      // just do this for IfOp
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        if (auto thenBlock = ifOp.thenBlock()) {
          for (auto operand : thenBlock->getTerminator()->getOperands()) {
            visitOperand(operand);
          }
        }
        if (auto elseBlock = ifOp.elseBlock()) {
          for (auto operand : elseBlock->getTerminator()->getOperands()) {
            visitOperand(operand);
          }
        }
      }
    }
  }

  void propagateGroups(Operation *op) { propagateGroups(nullptr, op, {}); }

private:
  NumWarpsSpec numWarps;
  DenseSet<Operation *> assignVisited;
};

bool shouldCreateEpilogueGroup(ModuleOp m) {
  SmallVector<triton::nvidia_gpu::TMEMLoadOp> tmemLoadOps;

  m.walk([&](triton::nvidia_gpu::TMEMLoadOp op) { tmemLoadOps.push_back(op); });

  // create epilogue group if there exist a tmem_load op which is outside the
  // loop enclosing the corresponding mmav5 op
  for (auto tmemLoadOp : tmemLoadOps) {
    auto mmaOp = getMMAv5Op(tmemLoadOp);
    if (mmaOp) {
      auto mmaLoop = mmaOp->getParentOfType<scf::ForOp>();
      if (mmaLoop && mmaLoop->getBlock() == tmemLoadOp->getBlock()) {
        // now check from this tmem_load op to the end of the enclosing block to
        // see if there is any op uses the forOp result
        for (mlir::Operation &op : llvm::make_early_inc_range(llvm::make_range(
                 std::next(tmemLoadOp->getIterator()),
                 mmaLoop->getBlock()->getTerminator()->getIterator()))) {
          for (auto result : mmaLoop->getResults()) {
            if (llvm::any_of(result.getUsers(),
                             [&](Operation *user) { return user == &op; })) {
              return false;
            }
          }
        }
        return true;
      }
    }
  }

  return false;
}

NumWarpsSpec getNumWarps(ModuleOp m) {
  if (shouldCreateEpilogueGroup(m)) {
    return {1, 1, 4};
  }

  const int numLoadWarps = 4;
  const int numMMAWarps =
      mlir::cast<mlir::IntegerAttr>(m->getAttr(AttrNumWarpsName)).getInt();
  return {numLoadWarps, numMMAWarps, 0};
}

} // namespace

class TritonGPUAnalyzeWarpSpecializationPass
    : public impl::TritonGPUAnalyzeWarpSpecializationBase<
          TritonGPUAnalyzeWarpSpecializationPass> {

  using impl::TritonGPUAnalyzeWarpSpecializationBase<
      TritonGPUAnalyzeWarpSpecializationPass>::
      TritonGPUAnalyzeWarpSpecializationBase;

public:
  void runOnOperation() override {
    ModuleOp m = getOperation();

    LLVM_DEBUG({
      DBGS() << "Module before analyzing warp specialization:\n";
      m.dump();
    });

    NumWarpsSpec numWarps = getNumWarps(m);

    for (auto func : m.getOps<triton::FuncOp>()) {
      Traverser t(numWarps);
      mlir::Block &body = func.getBody().front();
      // tag different groups by reversed traversal
      SmallVector<scf::ForOp> forOps;
      for (auto it = body.rbegin(); it != body.rend(); ++it) {
        if (auto forOp = llvm::dyn_cast<scf::ForOp>(*it)) {
          groupForLoops(forOp, m, numWarps);
        } else if (!it->hasAttr(ATTR_WSGROUPS)) {
          // Assign group to epilogue operations of a non-persistent kernel.
          // They are not enclosed in any for loops, so handled here.
          t.assignGroups(&*it);
          if (it->hasAttr(ATTR_WSGROUPS)) {
            t.propagateGroups(&*it);
          }
        }
      }
    }

    LLVM_DEBUG({
      DBGS() << "Module after analyzing warp specialization:\n";
      m.dump();
    });
  }

private:
  void groupForLoops(scf::ForOp forOp, ModuleOp m, NumWarpsSpec numWarps) {
    for (const auto &op : forOp.getBody()->without_terminator()) {
      if (auto innerFor = dyn_cast<scf::ForOp>(op)) {
        groupForLoops(innerFor, m, numWarps);
      }
    }

    Traverser t(numWarps);
    t.assignGroupsForOp(forOp);

    LLVM_DEBUG({
      DBGS() << "Module after assigning groups ids:\n";
      m.dump();
    });

    t.propagateGroups(forOp);

    LLVM_DEBUG({
      DBGS() << "Module after propagating groups ids:\n";
      m.dump();
    });
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
