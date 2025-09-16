#ifndef TRITONAMD_ANALYSIS_AMDGPU_ALLOCATION_H
#define TRITONAMD_ANALYSIS_AMDGPU_ALLOCATION_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

namespace mlir::triton::AMD {

constexpr char AttrSharedMemPadded[] = "amdgpu.use_padded_scratch_shmem";

unsigned getConvertLayoutScratchInBytes(RankedTensorType srcTy,
                                        RankedTensorType dstTy,
                                        bool usePadding);

unsigned AMDAllocationAnalysisScratchSizeFn(Operation *op);

// To convert a tensor from one layout to another, we need to allocate a
// temporary buffer (i.e., scratch buffer) in shared memory. The conversion may
// require multiple iterations, with each iteration involving multiple
// vectorized loads/stores. The scratch buffer has a shape (`repShape`) that
// represents the maximum size accessed in each dimension during each iteration.
// It is padded (`paddedRepShape`) to avoid bank conflicts and is accessed in a
// specific `order`.
struct ScratchConfig {
  SmallVector<unsigned> repShape;
  SmallVector<unsigned> paddedRepShape;
  SmallVector<unsigned> order;
  unsigned inVec;
  unsigned outVec;

  ScratchConfig(SmallVector<unsigned> repShape,
                SmallVector<unsigned> paddedRepShape, unsigned inVec = 1,
                unsigned outVec = 1)
      : repShape(repShape), paddedRepShape(paddedRepShape), inVec(inVec),
        outVec(outVec) {}

  void print(llvm::raw_ostream &os) const {
    os << "repShape: [";
    llvm::interleaveComma(repShape, os);
    os << "]";
    os << ", paddedRepShape: [";
    llvm::interleaveComma(paddedRepShape, os);
    os << "]";
    os << ", order: [";
    llvm::interleaveComma(order, os);
    os << "]";
    os << ", inVec: " << inVec << ", outVec: " << outVec << "\n";
  }
};

// For a layout conversion between `srcTy` and `dstTy`, return the vector length
// that can be used for the stores to and loads from shared memory,
// respectively.
std::pair</*inVec*/ unsigned, /*outVec*/ unsigned>
getScratchCvtInOutVecLengths(RankedTensorType srcTy, RankedTensorType dstTy);

ScratchConfig getScratchConfigForCvt(RankedTensorType srcTy,
                                     RankedTensorType dstTy);

} // namespace mlir::triton::AMD

#endif // TRITONAMD_ANALYSIS_AMDGPU_ALLOCATION_H
