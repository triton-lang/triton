#include "third_party/proton/dialect/include/Analysis/ScopeIdAllocation.h"

namespace mlir {
namespace triton::proton {

void ScopeIdAllocation::run() {
  llvm::StringMap<size_t> nameCount;
  ScopeId id = 0;

  funcOp->walk<WalkOrder::PreOrder>([&](RecordOp recordOp) {
    auto name = recordOp.getName();
    if (recordOp.getIsStart()) {
      if (!nameCount.contains(name)) {
        nameCount[name] = id;
        idToNameMap[id] = name;
        opToIdMap[recordOp] = id;
        id++;
      } else {
        recordOp->emitError("The scope name must appear in pairs");
      }
    } else {
      if (nameCount.contains(name)) {
        opToIdMap[recordOp] = nameCount.lookup(name);
        nameCount.erase(name);
      } else {
        recordOp->emitError("The scope name must appear in pairs");
      }
    }
  });

  if (nameCount.size() > 0) {
    for (auto &[name, _] : nameCount) {
      funcOp->emitError("Scope name '") << name << "' must appear in pairs";
    }
  }
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
}

ScopeIdAllocation::ScopeId
ModuleScopeIdAllocation::getOpScopeId(Operation *op) const {
  auto funcOp = op->getParentOfType<triton::FuncOp>();
  auto funcOffset = funcScopeIdMap.lookup(funcOp);
  return funcMap.lookup(funcOp).getOpScopeId(op) + funcOffset;
}

ScopeIdAllocation::ScopeIdPairs
ModuleScopeIdAllocation::getScopeIdPairs(triton::FuncOp funcOp) const {
  auto pairs = funcMap.at(funcOp).getScopeIdPairs();
  auto funcOffset = funcScopeIdMap.lookup(funcOp);
  for (auto &[scopeId, name] : pairs) {
    scopeId += funcOffset;
  }
  return pairs;
}

ScopeIdAllocation::ScopeIdPairs
ModuleScopeIdAllocation::getScopeIdPairs() const {
  ScopeIdAllocation::ScopeIdPairs pairs;
  for (auto [funcOp, funcOffset] : funcScopeIdMap) {
    auto funcPairs = getScopeIdPairs(cast<triton::FuncOp>(funcOp));
    pairs.insert(pairs.end(), funcPairs.begin(), funcPairs.end());
  }
  return pairs;
}

} // namespace triton::proton
} // namespace mlir
