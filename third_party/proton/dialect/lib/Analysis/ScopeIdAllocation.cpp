#include "Analysis/ScopeIdAllocation.h"

#include <stack>

namespace mlir {
namespace triton::proton {

void ScopeIdAllocation::run() {
  llvm::StringMap<size_t> nameCount;
  std::stack<ScopeId> scopeIdStack;
  ScopeId id = 0;

  funcOp->walk<WalkOrder::PreOrder>([&](RecordOp recordOp) {
    auto name = recordOp.getName();
    if (recordOp.getIsStart()) {
      if (!nameCount.contains(name)) {
        nameCount[name] = id;
        idToNameMap[id] = name;
        opToIdMap[recordOp] = id;
        if (!scopeIdStack.empty()) {
          scopeParentIds.push_back({id, scopeIdStack.top()});
        }
        scopeIdStack.push(id);
        id++;
      } else {
        recordOp->emitError("The scope name must appear in pairs");
      }
    } else {
      if (nameCount.contains(name)) {
        scopeIdStack.pop();
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
