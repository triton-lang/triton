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

SmallVector<unsigned> ReduceOpHelper::getScratchConfigBasic() {
  auto axis = op.axis();
  auto smemShape = convertType<unsigned>(getSrcShape());
  smemShape[axis] = std::min(smemShape[axis], getThreadsReductionAxis());
  return smemShape;
}

SmallVector<SmallVector<unsigned>> ReduceOpHelper::getScratchConfigsFast() {
  auto axis = op.axis();
  SmallVector<SmallVector<unsigned>> smemShapes(3);

  /// shared memory block0
  smemShapes[0] = convertType<unsigned>(getSrcShape());
  smemShapes[0][axis] = getInterWarpSize();

  /// FIXME(Qingyi): This size is actually larger than required.
  /// shared memory block1:
  auto mod = op.getOperation()->getParentOfType<ModuleOp>();
  unsigned numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
  smemShapes[1].push_back(numWarps * 32);

  return smemShapes;
}

unsigned ReduceOpHelper::getScratchSizeInBytes() {
  unsigned elems = 0;
  if (isFastReduction()) {
    auto smemShapes = getScratchConfigsFast();
    for (const auto &smemShape : smemShapes)
      elems = std::max(elems, product<unsigned>(smemShape));
  } else {
    auto smemShape = getScratchConfigBasic();
    elems = product<unsigned>(smemShape);
  }

  auto tensorType = op.operand().getType().cast<RankedTensorType>();
  unsigned bytes = elems * tensorType.getElementTypeBitWidth() / 8;

  if (triton::ReduceOp::withIndex(op.redOp()))
    bytes += elems * sizeof(int32_t);

  return bytes;
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
  return isa<tensor::ExtractSliceOp>(op) || isa<triton::TransOp>(op) ||
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
