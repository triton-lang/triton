#ifndef PROTON_ANALYSIS_SCOPE_ID_ALLOCATION_H
#define PROTON_ANALYSIS_SCOPE_ID_ALLOCATION_H

#include "mlir/IR/Operation.h"
#include "third_party/proton/dialect/include/Dialect/Proton/IR/Dialect.h"
#include "triton/Analysis/Utility.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace mlir {

namespace triton::proton {

class ScopeIdAllocation {
public:
  /// A unique identifier for scopes
  using ScopeId = size_t;
  using ScopeIdPairs = std::vector<std::pair<ScopeId, std::string>>;

  ScopeIdAllocation() = default;
  explicit ScopeIdAllocation(Operation *op) : funcOp(op) { run(); }

  ScopeId getOpScopeId(Operation *op) const {
    if (auto recordOp = dyn_cast<RecordOp>(op)) {
      auto name = recordOp.getName();
      return nameToIdMap.lookup(name);
    }
    llvm_unreachable("unexpected operation type");
  }

  ScopeIdPairs getScopeIdPairs() const {
    ScopeIdPairs pairs;
    for (const auto &pair : idToNameMap) {
      pairs.push_back({pair.first, pair.second.str()});
    }
    return pairs;
  }

  size_t getNumScopes() const { return idToNameMap.size(); }

private:
  void run();

  Operation *funcOp;
  llvm::DenseMap<ScopeId, StringRef> idToNameMap;
  llvm::StringMap<ScopeId> nameToIdMap;
};

class ModuleScopeIdAllocation : public CallGraph<ScopeIdAllocation> {
public:
  using FuncOffsetMapT =
      DenseMap<FunctionOpInterface, ScopeIdAllocation::ScopeId>;

  ModuleScopeIdAllocation(ModuleOp moduleOp)
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

  ScopeIdAllocation::ScopeId getOpScopeId(Operation *op) const {
    auto funcOp = op->getParentOfType<triton::FuncOp>();
    auto funcScopeId = funcScopeIdMap.lookup(funcOp);
    return funcMap.lookup(funcOp).getOpScopeId(op) + funcScopeId;
  }

  ScopeIdAllocation::ScopeIdPairs getScopeIdPairs(triton::FuncOp funcOp) const {
    auto pairs = funcMap.at(funcOp).getScopeIdPairs();
    auto funcScopeId = funcScopeIdMap.lookup(funcOp);
    for (auto &[scopeId, name] : pairs) {
      scopeId += funcScopeIdMap.lookup(funcOp);
    }
    return pairs;
  }

  ScopeIdAllocation::ScopeIdPairs getScopeIdPairs() const {
    ScopeIdAllocation::ScopeIdPairs pairs;
    for (auto [funcOp, funcScopeId] : funcScopeIdMap) {
      auto funcScopeIdPairs = getScopeIdPairs(cast<triton::FuncOp>(funcOp));
      pairs.insert(pairs.end(), funcScopeIdPairs.begin(),
                   funcScopeIdPairs.end());
    }
    return pairs;
  }

private:
  FuncOffsetMapT funcScopeIdMap;
};

} // namespace triton::proton

} // namespace mlir

#endif // PROTON_ANALYSIS_SCOPE_ID_ALLOCATION_H
