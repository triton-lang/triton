#ifndef PROTON_ANALYSIS_SCOPE_ID_ALLOCATION_H
#define PROTON_ANALYSIS_SCOPE_ID_ALLOCATION_H

#include "mlir/IR/Operation.h"
#include "proton/Dialect/include/Dialect/Proton/IR/Dialect.h"
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
  using ScopeId = size_t;
  // id -> name
  using ScopeIdName = std::vector<std::pair<ScopeId, std::string>>;
  // id -> parent id
  using ScopeIdParent = std::vector<std::pair<ScopeId, ScopeId>>;

  ScopeIdAllocation() = default;
  explicit ScopeIdAllocation(Operation *op) : funcOp(op) { run(); }

  ScopeId getOpScopeId(Operation *op) const {
    if (auto recordOp = dyn_cast<RecordOp>(op)) {
      return opToIdMap.lookup(recordOp);
    }
    llvm_unreachable("unexpected operation type");
  }

  ScopeIdName getScopeIdNames() const {
    ScopeIdName scopeIdNames;
    for (const auto &[id, name] : idToNameMap) {
      scopeIdNames.push_back({id, name.str()});
    }
    return scopeIdNames;
  }

  ScopeIdParent getScopeIdParents() const { return scopeParentIds; }

  size_t getNumScopes() const { return idToNameMap.size(); }

private:
  void run();

  Operation *funcOp;
  llvm::DenseMap<ScopeId, StringRef> idToNameMap;
  llvm::DenseMap<Operation *, ScopeId> opToIdMap;
  ScopeIdParent scopeParentIds;
};

class ModuleScopeIdAllocation : public CallGraph<ScopeIdAllocation> {
public:
  using FuncOffsetMapT =
      llvm::DenseMap<FunctionOpInterface, ScopeIdAllocation::ScopeId>;
  // Alias for per-function name and parent maps
  using ScopeIdNameMap =
      llvm::DenseMap<FunctionOpInterface, ScopeIdAllocation::ScopeIdName>;
  using ScopeIdParentMap =
      llvm::DenseMap<FunctionOpInterface, ScopeIdAllocation::ScopeIdParent>;

  explicit ModuleScopeIdAllocation(ModuleOp moduleOp);

  ScopeIdAllocation::ScopeId getOpScopeId(Operation *op) const;
  ScopeIdAllocation::ScopeIdName getScopeIdNames(triton::FuncOp funcOp) const;
  ScopeIdAllocation::ScopeIdName getScopeIdNames() const;
  ScopeIdAllocation::ScopeIdParent
  getScopeIdParents(triton::FuncOp funcOp) const;
  ScopeIdAllocation::ScopeIdParent getScopeIdParents() const;

private:
  FuncOffsetMapT funcScopeIdMap;
  // Precomputed per-function mappings
  ScopeIdNameMap scopeIdNames;
  ScopeIdParentMap scopeIdParents;
};

} // namespace triton::proton
} // namespace mlir

#endif // PROTON_ANALYSIS_SCOPE_ID_ALLOCATION_H
