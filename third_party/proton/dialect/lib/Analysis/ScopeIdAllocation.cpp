#include "third_party/proton/dialect/include/Analysis/ScopeIdAllocation.h"

namespace mlir {

namespace triton::proton {

void ScopeIdAllocation::run() {
  llvm::StringMap<size_t> nameCount;
  // Iterate over all operations in the function
  funcOp->walk([&](RecordOp recordOp) {
    // If the operation is a RecordOp, assign a unique scope id to it
    auto name = recordOp.getName();
    if (!nameCount.contains(name)) {
      nameCount[name] = 0;
    }
    nameCount[name]++;
    if (!nameToIdMap.contains(name)) {
      auto id = nameToIdMap.size();
      nameToIdMap[name] = id;
      idToNameMap[id] = name;
    }
  });
  funcOp->walk([&](RecordOp recordOp) {
    auto name = recordOp.getName();
    if (nameCount[name] != 2) {
      recordOp->emitError(
          "The scope name must appear exactly twice in each function");
    }
  });
}

} // namespace triton::proton
} // namespace mlir
