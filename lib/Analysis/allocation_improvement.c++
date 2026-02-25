#include "triton/Analysis/Allocation.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "triton/Analysis/Alias.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <optional>
#include <vector>

using namespace mlir::triton::gpu;

//===----------------------------------------------------------------------===//
// Shared Memory Allocation Analysis
//===----------------------------------------------------------------------===//

namespace mlir::triton {

constexpr int kPtrBitWidth = 64;

std::pair<std::vector<unsigned>, std::vector<unsigned>>
getCvtOrder(Attribute srcLayout, Attribute dstLayout) {
    auto srcMmaLayout = srcLayout.dyn_cast<MmaEncodingAttr>();
    auto srcDotLayout = srcLayout.dyn_cast<DotOperandEncodingAttr>();
    auto dstMmaLayout = dstLayout.dyn_cast<MmaEncodingAttr>();
    auto dstDotLayout = dstLayout.dyn_cast<DotOperandEncodingAttr>();
    assert(!(srcMmaLayout && dstMmaLayout) && "Unexpected mma -> mma layout conversion");

    auto inOrd = (srcMmaLayout || srcDotLayout) ? getOrder(dstLayout) : getOrder(srcLayout);
    auto outOrd = (dstMmaLayout || dstDotLayout) ? getOrder(srcLayout) : getOrder(dstLayout);

    return {inOrd, outOrd};
}

std::vector<unsigned> getRepShapeForCvtLayout(triton::gpu::ConvertLayoutOp op) {
    auto srcTy = op.getSrc().getType().cast<RankedTensorType>();
    auto dstTy = op.getResult().getType().cast<RankedTensorType>();
    auto srcLayout = srcTy.getEncoding();
    auto dstLayout = dstTy.getEncoding();

    if (shouldUseDistSmem(srcLayout, dstLayout)) {
        // TODO: Padding to avoid bank conflicts
        return convertType<unsigned, int64_t>(getShapePerCTA(srcTy));
    }

    if (auto srcMmaLayout = srcLayout.dyn_cast<MmaEncodingAttr>()) {
        if (dstLayout.isa<DotOperandEncodingAttr>() && isMmaToDotShortcut(srcTy, dstTy)) {
            return {};
        }
        if (auto dstMmaLayout = dstLayout.dyn_cast<MmaEncodingAttr>();
            isMmaToMmaShortcut(srcTy, dstTy)) {
            return {};
        }
    }

    assert(srcLayout && dstLayout && "Unexpected layout in getRepShape()");

    auto srcShapePerCTA = getShapePerCTA(srcTy);
    auto dstShapePerCTA = getShapePerCTA(dstTy);
    auto srcShapePerCTATile = getShapePerCTATile(srcLayout, srcTy.getShape());
    auto dstShapePerCTATile = getShapePerCTATile(dstLayout, dstTy.getShape());

    unsigned rank = dstTy.getRank();
    std::vector<unsigned> repShape(rank);
    for (unsigned d = 0; d < rank; ++d) {
        repShape[d] = std::max(std::min(srcShapePerCTA[d], srcShapePerCTATile[d]),
                               std::min(dstShapePerCTA[d], dstShapePerCTATile[d]));
    }
    return repShape;
}

std::vector<unsigned>
getScratchConfigForCvtLayout(triton::gpu::ConvertLayoutOp op, unsigned &inVec, unsigned &outVec) {
    auto repShape = getRepShapeForCvtLayout(op);

    auto srcTy = op.getSrc().getType().cast<RankedTensorType>();
    auto dstTy = op.getResult().getType().cast<RankedTensorType>();
    auto srcLayout = srcTy.getEncoding();
    auto dstLayout = dstTy.getEncoding();

    auto [inOrd, outOrd] = getCvtOrder(srcLayout, dstLayout);
    unsigned srcContigPerThread = getUniqueContigPerThread(srcLayout, srcTy.getShape())[inOrd[0]];
    unsigned dstContigPerThread = getUniqueContigPerThread(dstLayout, dstTy.getShape())[outOrd[0]];

    // Handle vectorization issues
    inVec = outOrd[0] == 0 ? 1 : (inOrd[0] == 0 ? 1 : srcContigPerThread);
    outVec = outOrd[0] == 0 ? 1 : dstContigPerThread;

    if (repShape.size() <= 1)
        return repShape;
    
    unsigned paddedDim = 1;
    if (auto dstBlockedLayout = dstLayout.dyn_cast<BlockedEncodingAttr>()) {
        paddedDim = dstBlockedLayout.getOrder()[0];
    }
    
    unsigned pad = std::max(inVec, outVec);
    repShape[paddedDim] += pad;
    return repShape;
}

void AllocationAnalysis::run() {
    getValuesAndSizes();
    resolveLiveness();
    computeOffsets();
}

// Continue refactoring and upgrading other functions similarly...

} // namespace mlir::triton
