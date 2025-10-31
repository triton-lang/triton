#include "mlir/Analysis/TopologicalSortUtils.h"
#include "Analysis/ScopeIdAllocation.h"

namespace mlir {
namespace triton::proton {

#define DEBUG_TYPE "proton-scope-id-allocation"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using VirtualBlock = std::pair<Block *, Block::iterator>;

struct BlockInfo {
  llvm::DenseSet<llvm::StringRef> activeScopes;

  BlockInfo() = default;

  /// Unions two BlockInfo objects.
  void join(const BlockInfo &other) {
    for (auto &scope : other.activeScopes) {
      this->activeScopes.insert(scope);
    }
  }

  bool contains(StringRef scopeName) const {
    return this->activeScopes.contains(scopeName);
  }

  void erase(StringRef scopeName) {
    this->activeScopes.erase(scopeName);
  }

  void insert(StringRef scopeName) {
    this->activeScopes.insert(scopeName);
  }

  bool operator ==(const BlockInfo &other) const {
    return this->activeScopes == other.activeScopes;
  }

  void dump() const {
    auto &err = llvm::errs();
    err << "Active Scopes:\n";
    for (auto &scope : activeScopes) {
      err << "  " << scope << "\n";
    }
  }
};

void ScopeIdAllocation::run() {
  // Stage the analysis to match downstream consumers of scope metadata:
  //
  // - reachability(): Track active scopes at CFG boundaries and flag malformed
  //   lifetimes. Example MLIR:
  //     scf.if %cond {
  //       proton.record start @"foo"
  //     }
  //   Because `"foo"` never ends on the `then` branch, reachability() emits
  //   `The scope name 'foo' is not closed properly`.
  //
  // - liveness(): Pair start/end records and assign a shared numeric ID. Example
  //   MLIR:
  //     proton.record start @"foo"
  //     …
  //     proton.record end @"foo"
  //   Both ops are mapped to the same ScopeId in `opToIdMap`.
  //
  // - dominance(): Infer the parent/child hierarchy between scopes via
  //   dominance. Example MLIR:
  //     proton.record start @"outer"
  //     scf.if %cond {
  //       proton.record start @"inner"
  //       …
  //       proton.record end @"inner"
  //     }
  //     proton.record end @"outer"
  //   Because the start of `"outer"` dominates `"inner"`, dominance() records
  //   `(innerId -> outerId)` in `scopeParentIds`.
  reachability();
  liveness();
  dominance();
}

void ScopeIdAllocation::reachability() {
  DenseMap<VirtualBlock, BlockInfo> inputBlockInfoMap;
  DenseMap<VirtualBlock, BlockInfo> outputBlockInfoMap;

  std::deque<VirtualBlock> virtualBlockList;
  funcOp->walk<WalkOrder::PreOrder>([&](Block *block) {
    // Start the analysis from the entry blocks of any nested isolated from
    // above regions.
    if (block->isEntryBlock() &&
        !isa<RegionBranchOpInterface>(block->getParentOp()))
      virtualBlockList.emplace_back(block, Block::iterator());
  });

  while (!virtualBlockList.empty()) {
    VirtualBlock &virtualBlock = virtualBlockList.front();
    virtualBlockList.pop_front();
    // Evaluate the transfer function for this block starting from the cached
    // input state.
    auto inputBlockInfo = inputBlockInfoMap[virtualBlock];
    SmallVector<VirtualBlock> successors;
    Block::iterator startIt =
        virtualBlock.second.isValid() ? std::next(virtualBlock.second) : virtualBlock.first->begin();
    for (Operation &op : llvm::make_range(startIt, virtualBlock.first->end())) {
      if (op.hasTrait<OpTrait::IsTerminator>() ||
          isa<RegionBranchOpInterface>(op)) {
        visitTerminator(&op, successors);
        break;
      }
      if (auto recordOp = dyn_cast<RecordOp>(&op)) {
        auto name = recordOp.getName();
        if (inputBlockInfo.contains(name)) {
          if (!recordOp.getIsStart()) {
            inputBlockInfo.erase(name);
          }
        } else {
          if (recordOp.getIsStart()) {
            inputBlockInfo.insert(name);
          } // else don't handle it right now as the scope might be monotonically closed later
        }
      }
    }
    // Get the reference because we want to update if it changed
    if (outputBlockInfoMap.count(virtualBlock) &&
        inputBlockInfo == outputBlockInfoMap[virtualBlock]) {
      // If we have seen the block before and the inputBlockInfo is the same as
      // the outputBlockInfo, we skip the successors
      continue;
    }
    // Update the current block
    outputBlockInfoMap[virtualBlock].join(inputBlockInfo);
    // Update the successors
    for (VirtualBlock &successor : successors) {
      inputBlockInfoMap[successor].join(outputBlockInfoMap[virtualBlock]);
      virtualBlockList.emplace_back(successor);
    }
  }

  // Go through all blocks, validate reachability analysis results
  for (auto iter : inputBlockInfoMap) {
    auto &virtualBlock = iter.first;
    auto inputBlockInfo = iter.second;
    auto outputBlockInfo = outputBlockInfoMap[virtualBlock];
    DenseSet<llvm::StringRef> unclosedScopes;
    Block::iterator startIt =
        virtualBlock.second.isValid() ? std::next(virtualBlock.second) : virtualBlock.first->begin();
    for (Operation &op : llvm::make_range(startIt, virtualBlock.first->end())) {
      if (auto recordOp = dyn_cast<RecordOp>(&op)) {
        auto name = recordOp.getName();
        if (recordOp.getIsStart()) {
          if (inputBlockInfo.contains(name)) {
            mlir::emitError(recordOp.getLoc(), "The scope name '")
                << name << "' is started without being closed";
          }
          inputBlockInfo.insert(name);
          unclosedScopes.insert(name);
        } else {
          if (inputBlockInfo.contains(name)) {
            inputBlockInfo.erase(name);
            unclosedScopes.erase(name);
          } else {
            mlir::emitError(recordOp.getLoc(), "The scope name '") << name << "' is closed without being opened";
          }
        }
      }
    }
    for (auto &scopeName : unclosedScopes) {
      if (!outputBlockInfo.contains(scopeName)) {
        mlir::emitError(virtualBlock.first->getParentOp()->getLoc(),
                        "The scope name '")
            << scopeName << "' is not closed properly";
      }
    }
  }
}

void ScopeIdAllocation::liveness() {
  // Stage 2: pair start/end records that refer to the same scope name and
  // assign a numeric ID that downstream passes can reuse.
  llvm::DenseMap<StringRef, std::pair</*id=*/size_t, /*isStart=*/bool>> nameToIdMap;
  llvm::DenseMap<ScopeId, RecordOp> idToOpMap;
  ScopeId scopeId = 0;

  funcOp->walk<WalkOrder::PreOrder>([&](RecordOp recordOp) {
    auto name = recordOp.getName();
    LDBG("Processing RecordOp: " << recordOp);
    if (!nameToIdMap.contains(name)) {
      nameToIdMap[name] = {scopeId, /*isStart=*/recordOp.getIsStart()};
      idToNameMap[scopeId] = name;
      LDBG("Assigning new scope scopeId " << scopeId << " to op '" << recordOp << "'");
      opToIdMap[recordOp] = scopeId;
      idToOpMap[scopeId] = recordOp;
      scopeId++;
    } else {
      auto &[existingId, isStart] = nameToIdMap[name];
      if (isStart == recordOp.getIsStart()) {
        // Error: duplicate start or end
        mlir::emitError(recordOp.getLoc(), "The scope name '")
            << name << "' has duplicate "
            << (recordOp.getIsStart() ? "start" : "end") << " record";
      } else {
        // Matching pair found
        LDBG("Found matching pair for scope name '" << name << "' with scopeId " << existingId);
        opToIdMap[recordOp] = existingId;
        idToOpMap[existingId] = recordOp;
        nameToIdMap.erase(name);
      }
    }
  });

  if (!nameToIdMap.empty()) {
    for (auto &[name, idIsStartPair] : nameToIdMap) {
      auto &[id, isStart] = idIsStartPair;
      auto unclosedOp = idToOpMap.lookup(id);
      mlir::emitError(unclosedOp.getLoc(), "The scope name '")
          << name << "' is not properly closed (missing "
          << (isStart ? "end" : "start") << " record)";
    }
  }
}

void ScopeIdAllocation::dominance() {
  // Stage 3: determine parentage between scopes by checking dominance of start
  // operations.
  mlir::DominanceInfo domInfo(funcOp);
  llvm::DenseMap<ScopeId, Operation *> startRecordMap;
  llvm::DenseMap<ScopeId, Operation *> endRecordMap;
  funcOp->walk<WalkOrder::PreOrder>([&](RecordOp recordOp) {
    auto scopeId = opToIdMap.lookup(recordOp);
    if (recordOp.getIsStart())
      startRecordMap[scopeId] = recordOp.getOperation();
    else
      endRecordMap[scopeId] = recordOp.getOperation();
  });

  for (auto &[scopeId, startOp] : startRecordMap) {
    auto *endOp = endRecordMap.lookup(scopeId);
    if (!endOp)
      continue;
    if (domInfo.dominates(endOp, startOp)) {
      auto name = idToNameMap.lookup(scopeId);
      mlir::emitError(endOp->getLoc(), "The scope name '")
          << name << "' has end record that dominates its start record";
    }
  }

  llvm::SetVector<Operation *> startRecordOps;
  for (auto &[scopeId, startOp] : startRecordMap) {
    startRecordOps.insert(startOp);
  }
  auto sortedStartRecordOps = mlir::topologicalSort(startRecordOps);
  for (int i = 0; i < sortedStartRecordOps.size(); ++i) {
    auto *op = sortedStartRecordOps[i];
    for (int j = i - 1; j >= 0; --j) {
      auto *maybeParentOp = sortedStartRecordOps[j];
      auto scopeId = opToIdMap.lookup(op);
      auto endRecordOp = endRecordMap.lookup(scopeId);
      if (domInfo.dominates(maybeParentOp, op) && 
          domInfo.dominates(op, endRecordOp)) {
        auto parentId = opToIdMap.lookup(maybeParentOp);
        auto childId = opToIdMap.lookup(op);
        scopeParentIds.push_back({childId, parentId});
        break;
      }
    }
  }
}

void ScopeIdAllocation::visitTerminator(
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

ModuleScopeIdAllocation::ModuleScopeIdAllocation(ModuleOp moduleOp)
    : CallGraph<ScopeIdAllocation>(moduleOp) {
  ScopeIdAllocation::ScopeId funcScopeId = 0;
  walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
      // Pre-order edge walk callback
      [](CallOpInterface callOp, FunctionOpInterface funcOp) {},
      // Post-order node walk callback
      [&](FunctionOpInterface funcOp) {
        if (funcMap.contains(funcOp)) {
          return;
        }
        auto iter = funcMap.try_emplace(funcOp, ScopeIdAllocation(funcOp));
        funcScopeIdMap[funcOp] = funcScopeId;
        funcScopeId += iter.first->second.getNumScopes();
      });
  // Precompute per-function scope id mappings
  for (auto [funcOp, offset] : funcScopeIdMap) {
    // Names
    auto names = funcMap.lookup(funcOp).getScopeIdNames();
    for (auto &p : names)
      p.first += offset;
    scopeIdNames[funcOp] = std::move(names);
    // Parents
    auto parents = funcMap.lookup(funcOp).getScopeIdParents();
    for (auto &p : parents) {
      p.first += offset;
      p.second += offset;
    }
    scopeIdParents[funcOp] = std::move(parents);
  }
}

ScopeIdAllocation::ScopeId
ModuleScopeIdAllocation::getOpScopeId(Operation *op) const {
  auto funcOp = op->getParentOfType<triton::FuncOp>();
  auto funcOffset = funcScopeIdMap.lookup(funcOp);
  return funcMap.lookup(funcOp).getOpScopeId(op) + funcOffset;
}

ScopeIdAllocation::ScopeIdName
ModuleScopeIdAllocation::getScopeIdNames(triton::FuncOp funcOp) const {
  return scopeIdNames.lookup(funcOp);
}

ScopeIdAllocation::ScopeIdName
ModuleScopeIdAllocation::getScopeIdNames() const {
  ScopeIdAllocation::ScopeIdName combined;
  for (auto &entry : scopeIdNames)
    combined.insert(combined.end(), entry.second.begin(), entry.second.end());
  return combined;
}

ScopeIdAllocation::ScopeIdParent
ModuleScopeIdAllocation::getScopeIdParents(triton::FuncOp funcOp) const {
  return scopeIdParents.lookup(funcOp);
}

ScopeIdAllocation::ScopeIdParent
ModuleScopeIdAllocation::getScopeIdParents() const {
  ScopeIdAllocation::ScopeIdParent combined;
  for (auto &entry : scopeIdParents)
    combined.insert(combined.end(), entry.second.begin(), entry.second.end());
  return combined;
}

} // namespace triton::proton
} // namespace mlir
