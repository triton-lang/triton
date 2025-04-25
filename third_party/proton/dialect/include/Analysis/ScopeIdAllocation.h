#ifndef PROTON_ANALYSIS_SCOPE_ID_ALLOCATION_H
#define PROTON_ANALYSIS_SCOPE_ID_ALLOCATION_H

#include "mlir/IR/Operation.h"
#include "proton/dialect/include/Dialect/Proton/IR/Dialect.h"
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

  ScopeIdName getScopeIdName() const {
    ScopeIdName scopeIdName;
    for (const auto &pair : idToNameMap) {
      scopeIdName.push_back({pair.first, pair.second.str()});
    }
    return scopeIdName;
  }

  ScopeIdParent getScopeIdParent() const { return scopeParentId; }

  size_t getNumScopes() const { return idToNameMap.size(); }

private:
  void run();

  Operation *funcOp;
  llvm::DenseMap<ScopeId, StringRef> idToNameMap;
  llvm::DenseMap<Operation *, ScopeId> opToIdMap;
  ScopeIdParent scopeParentId;
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
  ScopeIdAllocation::ScopeIdName getScopeIdName(triton::FuncOp funcOp) const;
  ScopeIdAllocation::ScopeIdName getScopeIdName() const;
  ScopeIdAllocation::ScopeIdParent
  getScopeIdParent(triton::FuncOp funcOp) const;
  ScopeIdAllocation::ScopeIdParent getScopeIdParent() const;

private:
  FuncOffsetMapT funcScopeIdMap;
  // Precomputed per-function mappings
  ScopeIdNameMap scopeIdNames;
  ScopeIdParentMap scopeIdParents;
};

} // namespace triton::proton
} // namespace mlir

#endif // PROTON_ANALYSIS_SCOPE_ID_ALLOCATION_H
