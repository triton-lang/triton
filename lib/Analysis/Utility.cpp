#include "triton/Analysis/Utility.h"
#include "mlir/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {

bool isSharedEncoding(Value value) {
  auto type = value.getType();
  if (auto tensorType = type.dyn_cast<RankedTensorType>()) {
    auto encoding = tensorType.getEncoding();
    return encoding && encoding.isa<triton::gpu::SharedEncodingAttr>();
  }
  return false;
}

bool maybeSharedAllocationOp(Operation *op) {
  // TODO(Keren): This function can be replaced by adding
  // MemoryEffectOpInterface. We can then use the MemoryEffectOpInterface to
  // query the memory effects of the op.
  auto *dialect = op->getDialect();
  return dialect &&
         (dialect->getTypeID() ==
              mlir::TypeID::get<triton::gpu::TritonGPUDialect>() ||
          dialect->getTypeID() == mlir::TypeID::get<triton::TritonDialect>() ||
          dialect->getTypeID() ==
              mlir::TypeID::get<arith::ArithmeticDialect>() ||
          dialect->getTypeID() == mlir::TypeID::get<tensor::TensorDialect>());
}

std::string getValueOperandName(Value value, AsmState &state) {
  std::string opName;
  llvm::raw_string_ostream ss(opName);
  value.printAsOperand(ss, state);
  return opName;
}

} // namespace mlir
