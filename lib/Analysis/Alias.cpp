#include "triton/Analysis/Alias.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

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

void SharedMemoryAliasAnalysis::visitNonControlFlowArguments(
    Operation *op, const RegionSuccessor &successor,
    ArrayRef<dataflow::Lattice<AliasInfo> *> argLattices, unsigned firstIndex) {
  auto wsOp = dyn_cast<triton::gpu::WarpSpecializePartitionsOp>(op);
  if (!wsOp) {
    setAllToEntryStates(argLattices.take_front(firstIndex));
    setAllToEntryStates(argLattices.drop_front(
        firstIndex + successor.getSuccessorInputs().size()));
    return;
  }

  // Propagate aliases from the parent operation's operands to the block
  // arguments.
  assert(!successor.isParent());
  ProgramPoint *point = getProgramPointAfter(wsOp);

  for (auto [capture, argLattice] :
       llvm::zip(wsOp.getParentOp().getExplicitCaptures(), argLattices)) {
    propagateIfChanged(
        argLattice,
        argLattice->join(getLatticeElementFor(point, capture)->getValue()));
  }
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
