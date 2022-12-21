#include "triton/Analysis/Alias.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "triton/Analysis/Utility.h"
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

ChangeResult SharedMemoryAliasAnalysis::visitOperation(
    Operation *op, ArrayRef<LatticeElement<AliasInfo> *> operands) {
  AliasInfo aliasInfo;
  bool pessimistic = true;
  if (maybeSharedAllocationOp(op)) {
    // These ops may allocate a new shared memory buffer.
    auto result = op->getResult(0);
    // FIXME(Keren): extract and insert are always alias for now
    if (isa<tensor::ExtractSliceOp, triton::TransOp>(op)) {
      // extract_slice %src
      aliasInfo = AliasInfo(operands[0]->getValue());
      pessimistic = false;
    } else if (isa<tensor::InsertSliceOp>(op) ||
               isa<triton::gpu::InsertSliceAsyncOp>(op)) {
      // insert_slice_async %src, %dst, %index
      // insert_slice %src into %dst[%offsets]
      aliasInfo = AliasInfo(operands[1]->getValue());
      pessimistic = false;
    } else if (isSharedEncoding(result)) {
      aliasInfo.insert(result);
      pessimistic = false;
    }
  }

  if (pessimistic) {
    return markAllPessimisticFixpoint(op->getResults());
  }
  // Join all lattice elements
  ChangeResult result = ChangeResult::NoChange;
  for (Value value : op->getResults()) {
    result |= getLatticeElement(value).join(aliasInfo);
  }
  return result;
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
