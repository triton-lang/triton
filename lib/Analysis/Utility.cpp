#include "triton/Analysis/Utility.h"
#include "mlir/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {

bool ReduceOpHelper::isFastReduction() {
  auto srcLayout = srcTy.getEncoding();
  auto axis = op.axis();
  return axis == triton::gpu::getOrder(srcLayout)[0];
}

unsigned ReduceOpHelper::getInterWarpSize() {
  auto srcLayout = srcTy.getEncoding();
  auto srcShape = srcTy.getShape();
  auto axis = op.axis();
  auto srcReduceDimSize = static_cast<unsigned>(srcShape[axis]);
  unsigned sizeIntraWarps = getIntraWarpSize();
  return std::min(srcReduceDimSize / sizeIntraWarps,
                  triton::gpu::getWarpsPerCTA(srcLayout)[axis]);
}

unsigned ReduceOpHelper::getIntraWarpSize() {
  auto srcLayout = srcTy.getEncoding();
  auto srcShape = srcTy.getShape();
  auto axis = op.axis();
  auto srcReduceDimSize = static_cast<unsigned>(srcShape[axis]);
  return std::min(srcReduceDimSize,
                  triton::gpu::getThreadsPerWarp(srcLayout)[axis]);
}

unsigned ReduceOpHelper::getThreadsReductionAxis() {
  auto srcLayout = srcTy.getEncoding();
  auto axis = op.axis();
  return triton::gpu::getThreadsPerWarp(srcLayout)[axis] *
         triton::gpu::getWarpsPerCTA(srcLayout)[axis];
}

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

bool maybeAliasOp(Operation *op) {
  return isa<tensor::ExtractSliceOp>(op) ||
         isa<triton::gpu::InsertSliceAsyncOp>(op) ||
         isa<tensor::InsertSliceOp>(op);
}

std::string getValueOperandName(Value value, AsmState &state) {
  std::string opName;
  llvm::raw_string_ostream ss(opName);
  value.printAsOperand(ss, state);
  return opName;
}

} // namespace mlir
