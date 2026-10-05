#include "triton/Analysis/Alias.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {

AliasInfo AliasInfo::join(const AliasInfo &lhs, const AliasInfo &rhs) {
  if (lhs == rhs)
    return lhs;
  AliasInfo ret;
  for (auto value : lhs.allocs) {
    ret.insert(value);
  }
  for (auto value : rhs.allocs) {
    ret.insert(value);
  }
  return ret;
}

void SharedMemoryAliasAnalysis::visitCallableOperation(
    CallableOpInterface callable,
    ArrayRef<dataflow::AbstractSparseLattice *> arguments) {
  dataflow::SparseForwardDataFlowAnalysis<
      dataflow::Lattice<AliasInfo>>::visitCallableOperation(callable,
                                                            arguments);
  // Keep the incoming allocation roots and this callee's formal identity.
  for (auto *argument : arguments) {
    auto *lattice = static_cast<dataflow::Lattice<AliasInfo> *>(argument);
    Value value = lattice->getAnchor();
    auto memory = dyn_cast<triton::gpu::MemDescType>(value.getType());
    if (memory &&
        isa<triton::gpu::SharedMemorySpaceAttr>(memory.getMemorySpace()))
      propagateIfChanged(lattice, lattice->join(AliasInfo(value)));
  }
}

LogicalResult SharedMemoryAliasAnalysis::visitOperation(
    Operation *op, ArrayRef<const dataflow::Lattice<AliasInfo> *> operands,
    ArrayRef<dataflow::Lattice<AliasInfo> *> results) {
  AliasInfo aliasInfo;
  bool pessimistic = true;
  auto result = op->getResult(0);
  // skip ops that return memdesc in a different memory space.
  if (auto memdescTy = dyn_cast<triton::gpu::MemDescType>(result.getType())) {
    if (!isa_and_nonnull<triton::gpu::SharedMemorySpaceAttr>(
            memdescTy.getMemorySpace()))
      return success();
  }

  // Only LocalAllocOp creates a new buffer.
  if (isa<triton::gpu::LocalAllocOp>(op)) {
    aliasInfo.insert(result);
    pessimistic = false;
  } else if (op->hasTrait<OpTrait::MemDescViewTrait>()) {
    aliasInfo = AliasInfo(operands[0]->getValue());
    pessimistic = false;
  } else if (isa<arith::SelectOp>(op)) {
    aliasInfo =
        AliasInfo::join(operands[1]->getValue(), operands[2]->getValue());
    pessimistic = false;
  } else if (isa<ub::PoisonOp>(op)) {
    aliasInfo = AliasInfo();
    pessimistic = false;
  } else {
    assert(!isa<triton::gpu::MemDescType>(result.getType()) &&
           "unknown operation creating memory descriptor");
  }

  if (pessimistic) {
    setAllToEntryStates(results);
    return success();
  }
  // Join all lattice elements
  for (auto *result : results)
    propagateIfChanged(result, result->join(aliasInfo));

  return success();
}

AliasResult SharedMemoryAliasAnalysis::alias(Value lhs, Value rhs) {
  // TODO: implement
  return AliasResult::MayAlias;
}

ModRefResult SharedMemoryAliasAnalysis::getModRef(Operation *op,
                                                  Value location) {
  // TODO: implement
  return ModRefResult::getModAndRef();
}

} // namespace mlir
