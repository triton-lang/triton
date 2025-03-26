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
  using ScopeId = size_t;
  using ScopeIdPairs = std::vector<std::pair<ScopeId, std::string>>;

  ScopeIdAllocation() = default;
  explicit ScopeIdAllocation(Operation *op) : funcOp(op) { run(); }

  ScopeId getOpScopeId(Operation *op) const {
    if (auto recordOp = dyn_cast<RecordOp>(op)) {
      return opToIdMap.lookup(recordOp);
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
  llvm::DenseMap<Operation *, ScopeId> opToIdMap;
};

class ModuleScopeIdAllocation : public CallGraph<ScopeIdAllocation> {
public:
  using FuncOffsetMapT =
      DenseMap<FunctionOpInterface, ScopeIdAllocation::ScopeId>;

  explicit ModuleScopeIdAllocation(ModuleOp moduleOp);

  ScopeIdAllocation::ScopeId getOpScopeId(Operation *op) const;
  ScopeIdAllocation::ScopeIdPairs getScopeIdPairs(triton::FuncOp funcOp) const;
  ScopeIdAllocation::ScopeIdPairs getScopeIdPairs() const;

private:
  FuncOffsetMapT funcScopeIdMap;
};

} // namespace triton::proton
} // namespace mlir

#endif // PROTON_ANALYSIS_SCOPE_ID_ALLOCATION_H
