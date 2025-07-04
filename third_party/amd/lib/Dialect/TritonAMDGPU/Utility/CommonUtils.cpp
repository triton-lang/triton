#include "third_party/amd/include/Dialect/TritonAMDGPU/Utility/CommonUtils.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace mlir::triton::AMD {
SmallVector<scf::ForOp> getLeafForOps(triton::FuncOp funcOp) {
  SmallVector<scf::ForOp> allOps;
  funcOp->walk([&](scf::ForOp forOp) { allOps.push_back(forOp); });

  SmallVector<scf::ForOp> leafOps;
  for (scf::ForOp forOp : allOps) {
    auto searchResult = forOp.getBody()->walk(
        [](scf::ForOp) { return WalkResult::interrupt(); });
    if (!searchResult.wasInterrupted())
      leafOps.push_back(forOp);
  }
  return leafOps;
}

template <typename T>
SmallVector<Value> toConstantIntOp(OpBuilder &builder, Location loc,
                                   llvm::SmallVector<T> &intValues) {
  SmallVector<Value> values;
  for (auto intValue : intValues) {
    Value value = builder.create<arith::ConstantIntOp>(loc, intValue, 32);
    values.push_back(value);
  }
  return values;
}

SmallVector<gpu::MemDescSubviewOp>
splitMemDescView(OpBuilder &builder, Location loc,
                 gpu::MemDescSubviewOp parentView,
                 SmallVector<unsigned> numSplits) {
  auto perentType = parentView.getType();
  auto parentShape = perentType.getShape();
  assert(parentShape.size() == numSplits.size());
  SmallVector<int64_t> splitShape(parentShape);
  SmallVector<unsigned> order(parentShape.size(), 0);
  for (size_t dim = 0; dim < parentShape.size(); ++dim) {
    splitShape[dim] /= numSplits[dim];
    order[dim] = parentShape.size() - 1 - dim;
  }

  MLIRContext *ctx = parentView->getContext();
  auto splitType = gpu::MemDescType::get(
      ctx, splitShape, perentType.getElementType(), perentType.getEncoding(),
      perentType.getMemorySpace(), perentType.getMutableMemory(),
      perentType.getAllocShape());

  SmallVector<gpu::MemDescSubviewOp> subSlices;
  auto totalNumSplits = product(numSplits);
  for (unsigned linearIdx = 0; linearIdx < totalNumSplits; ++linearIdx) {
    auto coords = LLVM::delinearize(linearIdx, numSplits, order);

    SmallVector<int32_t> offset(coords.size(), 0);
    for (size_t dim = 0; dim < coords.size(); ++dim) {
      offset[dim] = coords[dim] * splitShape[dim];
    }
    auto valueOffset = toConstantIntOp(builder, loc, offset);
    auto slice = builder.create<gpu::MemDescSubviewOp>(
        loc, splitType, parentView.getResult(), valueOffset);
    subSlices.push_back(slice);
  }

  return subSlices;
}
} // namespace mlir::triton::AMD
