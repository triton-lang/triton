#include "Analysis/ScopeIdAllocation.h"
#include "mlir/Analysis/TopologicalSortUtils.h"

namespace mlir {
namespace triton::proton {

#define DEBUG_TYPE "proton-scope-id-allocation"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using VirtualBlock = std::pair<Block *, Block::iterator>;

struct BlockInfo {
  using ScopeId = ScopeIdAllocation::ScopeId;

  llvm::DenseSet<ScopeId> activeScopes;

  BlockInfo() = default;

  /// Unions two BlockInfo objects.
  void join(const BlockInfo &other) {
    for (auto &scope : other.activeScopes) {
      this->activeScopes.insert(scope);
    }
  }

  bool contains(ScopeId scopeId) const {
    return this->activeScopes.contains(scopeId);
  }

  void erase(ScopeId scopeId) { this->activeScopes.erase(scopeId); }

  void insert(ScopeId scopeId) { this->activeScopes.insert(scopeId); }

  bool operator==(const BlockInfo &other) const {
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
  // We execute the following analysis stages in the order to verify if
  // `proton.record` operations are well-formed and associate scope IDs for each
  // pair of start/end records.
  //
  // 1. liveness()
  //
  //    Pair start/end records that share a name and assign a numeric
  //    identifier that later passes reuse. The current implementation pairs
  //    each start with the nearest matching end.
  //
  //      proton.record start @"foo"  // scopeId = 0
  //      …
  //      proton.record end @"foo"    // scopeId = 0
  //      …
  //      proton.record start @"foo"  // scopeId = 1
  //      …
  //      proton.record end @"foo"    // scopeId = 1
  //
  // 2. reachability()
  //
  //    Track active scopes across CFG boundaries and surface
  //    malformed lifetimes once the dataflow converges.
  //
  //      scf.if %cond {
  //        proton.record start @"foo"
  //      }
  //
  //    Because `"foo"` never ends on the `then` branch, reachability() emits
  //    "The scope name 'foo' is not closed properly".
  //
  //      scf.if %cond {
  //        proton.record start @"foo"
  //      }
  //      proton.record end @"foo"
  //
  //    No diagnostic is emitted: the pass assumes the branch may execute and
  //    leaves semantic responsibility to the caller.
  //
  // 3. dominance():
  //
  //    (a) Ensure that each start dominates its matching end.
  //
  //          proton.record end @"foo"
  //          …
  //          proton.record start @"foo"
  //
  //        Because the end dominates the start, dominance() reports an error.
  //
  //    (b) Infer parent/child scope relationships using dominance facts.
  //
  //          proton.record start @"outer"
  //          scf.if %cond {
  //            proton.record start @"inner"
  //            …
  //            proton.record end @"inner"
  //          }
  //          proton.record end @"outer"
  //
  //        `"outer"` dominates `"inner"`, so dominance() records
  //        `(innerId -> outerId)` in `scopeParentIds`.
  liveness();
  reachability();
  dominance();
}

void ScopeIdAllocation::liveness() {
  llvm::DenseMap<StringRef, std::pair</*id=*/size_t, /*isStart=*/bool>>
      nameToIdMap;
  llvm::DenseMap<ScopeId, RecordOp> idToOpMap;
  ScopeId scopeId = 0;

  funcOp->walk<WalkOrder::PreOrder>([&](RecordOp recordOp) {
    auto name = recordOp.getName();
    LDBG("Processing RecordOp: " << recordOp);
    if (!nameToIdMap.contains(name)) {
      nameToIdMap[name] = {scopeId, /*isStart=*/recordOp.getIsStart()};
      idToNameMap[scopeId] = name;
      LDBG("Assigning new scope scopeId " << scopeId << " to op '" << recordOp
                                          << "'");
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
        LDBG("Found matching pair for scope name '" << name << "' with scopeId "
                                                    << existingId);
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

void ScopeIdAllocation::reachability() {
  DenseMap<VirtualBlock, BlockInfo> inputBlockInfoMap;
  DenseMap<VirtualBlock, BlockInfo> outputBlockInfoMap;

  std::deque<VirtualBlock> virtualBlockList;
  virtualBlockList.emplace_back(&funcOp.getBlocks().front(), Block::iterator());

  while (!virtualBlockList.empty()) {
    VirtualBlock virtualBlock = virtualBlockList.front();
    virtualBlockList.pop_front();
    // Evaluate the transfer function for this block starting from the cached
    // input state.
    auto inputBlockInfo = inputBlockInfoMap[virtualBlock];
    SmallVector<VirtualBlock> successors;
    Block::iterator startIt = virtualBlock.second.isValid()
                                  ? std::next(virtualBlock.second)
                                  : virtualBlock.first->begin();
    for (Operation &op : llvm::make_range(startIt, virtualBlock.first->end())) {
      if (op.hasTrait<OpTrait::IsTerminator>() ||
          isa<RegionBranchOpInterface>(op)) {
        visitTerminator(&op, successors);
        break;
      }
      if (auto recordOp = dyn_cast<RecordOp>(&op)) {
        auto scopeId = opToIdMap.lookup(recordOp);
        if (recordOp.getIsStart()) {
          inputBlockInfo.insert(scopeId);
        } else {
          inputBlockInfo.erase(scopeId);
        }
      }
    }
    // Skip successor propagation if the output state is unchanged.
    if (outputBlockInfoMap.count(virtualBlock) &&
        inputBlockInfo == outputBlockInfoMap[virtualBlock]) {
      continue;
    }
    // Update the current block.
    outputBlockInfoMap[virtualBlock].join(inputBlockInfo);
    // Propagate the new facts to successors.
    for (VirtualBlock &successor : successors) {
      inputBlockInfoMap[successor].join(outputBlockInfoMap[virtualBlock]);
      virtualBlockList.emplace_back(successor);
    }
  }

  // Validate the reachability analysis results for each block.
  for (auto iter : inputBlockInfoMap) {
    auto &virtualBlock = iter.first;
    auto inputBlockInfo = iter.second;
    Block::iterator startIt = virtualBlock.second.isValid()
                                  ? std::next(virtualBlock.second)
                                  : virtualBlock.first->begin();
    for (Operation &op : llvm::make_range(startIt, virtualBlock.first->end())) {
      if (auto recordOp = dyn_cast<RecordOp>(&op)) {
        auto scopeId = opToIdMap.lookup(recordOp);
        auto name = idToNameMap.lookup(scopeId);
        if (recordOp.getIsStart()) {
          if (inputBlockInfo.contains(scopeId)) {
            mlir::emitError(recordOp.getLoc(), "The scope name '")
                << name << "' is started without being closed";
          }
          inputBlockInfo.insert(scopeId);
        } else {
          if (inputBlockInfo.contains(scopeId)) {
            inputBlockInfo.erase(scopeId);
          } else {
            mlir::emitError(recordOp.getLoc(), "The scope name '")
                << name << "' is closed without being opened";
          }
        }
      }
    }
  }
}

void ScopeIdAllocation::dominance() {
  // Stage 3: derive scope parentage and verify dominance constraints.
  mlir::DominanceInfo domInfo(funcOp);
  mlir::PostDominanceInfo postDomInfo(funcOp);
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
    auto *startOp = sortedStartRecordOps[i];
    auto scopeId = opToIdMap.lookup(startOp);
    auto endOp = endRecordMap.lookup(scopeId);
    for (int j = i - 1; j >= 0; --j) {
      auto *parentStartOp = sortedStartRecordOps[j];
      auto parentScopeId = opToIdMap.lookup(parentStartOp);
      auto parentEndOp = endRecordMap.lookup(parentScopeId);
      if (domInfo.dominates(parentStartOp, startOp) &&
          postDomInfo.postDominates(parentEndOp, endOp)) {
        auto parentId = opToIdMap.lookup(parentStartOp);
        auto childId = opToIdMap.lookup(startOp);
        scopeParentIds.push_back({childId, parentId});
        break;
      }
    }
  }
}

void ScopeIdAllocation::visitTerminator(Operation *op,
                                        SmallVector<VirtualBlock> &successors) {
  if (isa<BranchOpInterface>(op)) {
    // Collect the block successors of the branch.
    for (Block *successor : op->getSuccessors())
      successors.emplace_back(successor, Block::iterator());
    return;
  }

  if (auto br = dyn_cast<RegionBranchOpInterface>(op)) {
    // Query successors of an op-with-regions. The op can branch to region entry
    // blocks or to the continuation after itself.
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
    // Region branch terminators can jump to another region belonging to the
    // parent operation or to the parent continuation.
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

  // Otherwise, it could be a return-like op.
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
