#ifndef PROTON_ANALYSIS_SCOPE_ID_ALLOCATION_H
#define PROTON_ANALYSIS_SCOPE_ID_ALLOCATION_H

#include "mlir/IR/Operation.h"
#include "third_party/proton/dialect/include/Dialect/Proton/IR/Dialect.h"
#include "triton/Analysis/Utility.h"
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

  ScopeIdAllocation(Operation *op) : funcOp(op) { run(); }

  ScopeId getOpScopeId(Operation *op) const {
    if (auto recordOp = dyn_cast<RecordOp>(op)) {
      auto name = recordOp.getName().str();
      return nameToIdMap.at(name);
    }
    llvm_unreachable("unexpected operation type");
  }

  ScopeIdPairs getScopeIdPairs() const {
    ScopeIdPairs pairs;
    for (auto &pair : idToNameMap) {
      pairs.push_back(pair);
    }
    return pairs;
  }

private:
  void run();

  Operation *funcOp;
  std::map<std::string, ScopeId> nameToIdMap;
  std::map<ScopeId, std::string> idToNameMap;
};

class ModuleScopeIdAllocation : CallGraph<ScopeIdAllocation> {
public:
  ModuleScopeIdAllocation(ModuleOp moduleOp)
      : CallGraph<ScopeIdAllocation>(moduleOp) {
    ScopeIdAllocation::ScopeId scopeId = 0;
    walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
        // Pre-order edge walk callback
        [](CallOpInterface callOp, FunctionOpInterface funcOp) {},
        // Post-order node walk callback
        [&](FunctionOpInterface funcOp) {
          auto scopeIdPairs = ScopeIdAllocation(funcOp).getScopeIdPairs();
          // Adjust offset
          for (auto &pair : scopeIdPairs) {
            pair.first += scopeId;
          }
          scopeId += scopeIdPairs.size();
        });
  }

  ScopeIdAllocation::ScopeIdPairs getScopeIdPairs() const {
    return moduleScopeIdPairs;
  }

private:
  ScopeIdAllocation::ScopeIdPairs moduleScopeIdPairs;
};

} // namespace triton::proton

} // namespace mlir

#endif // PROTON_ANALYSIS_SCOPE_ID_ALLOCATION_H
