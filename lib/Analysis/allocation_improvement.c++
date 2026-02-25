
/**
 * @file allocation_improvement.c++
 * @brief Shared Memory Allocation Analysis for Triton GPU backend.
 * @author Upgraded
 * @date 2026
 */

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

namespace mlir::triton {

/// Pointer bit width for allocations.
constexpr int kPtrBitWidth = 64;

/**
 * @brief Get the conversion order for source and destination layouts.
 * @param srcLayout Source layout attribute.
 * @param dstLayout Destination layout attribute.
 * @return Pair of input and output orders.
 */
auto getCvtOrder(const Attribute &srcLayout, const Attribute &dstLayout)
    -> std::pair<std::vector<unsigned>, std::vector<unsigned>> {
    const auto srcMmaLayout = srcLayout.dyn_cast<MmaEncodingAttr>();
    const auto srcDotLayout = srcLayout.dyn_cast<DotOperandEncodingAttr>();
    const auto dstMmaLayout = dstLayout.dyn_cast<MmaEncodingAttr>();
    const auto dstDotLayout = dstLayout.dyn_cast<DotOperandEncodingAttr>();
    assert(!(srcMmaLayout && dstMmaLayout) && "Unexpected mma -> mma layout conversion");

    const auto inOrd = (srcMmaLayout || srcDotLayout) ? getOrder(dstLayout) : getOrder(srcLayout);
    const auto outOrd = (dstMmaLayout || dstDotLayout) ? getOrder(srcLayout) : getOrder(dstLayout);

    return {inOrd, outOrd};
}

/**
 * @brief Get the representative shape for a layout conversion operation.
 * @param op The ConvertLayoutOp operation.
 * @return Vector of representative shape dimensions.
 */
auto getRepShapeForCvtLayout(const triton::gpu::ConvertLayoutOp &op) -> std::vector<unsigned> {
    const auto srcTy = op.getSrc().getType().cast<RankedTensorType>();
    const auto dstTy = op.getResult().getType().cast<RankedTensorType>();
    const auto srcLayout = srcTy.getEncoding();
    const auto dstLayout = dstTy.getEncoding();

    if (shouldUseDistSmem(srcLayout, dstLayout)) {
        // TODO: Padding to avoid bank conflicts
        return convertType<unsigned, int64_t>(getShapePerCTA(srcTy));
    }

    if (const auto srcMmaLayout = srcLayout.dyn_cast<MmaEncodingAttr>()) {
        if (dstLayout.isa<DotOperandEncodingAttr>() && isMmaToDotShortcut(srcTy, dstTy)) {
            return {};
        }
        if (const auto dstMmaLayout = dstLayout.dyn_cast<MmaEncodingAttr>();
            isMmaToMmaShortcut(srcTy, dstTy)) {
            return {};
        }
    }

    assert(srcLayout && dstLayout && "Unexpected layout in getRepShape()");

    const auto srcShapePerCTA = getShapePerCTA(srcTy);
    const auto dstShapePerCTA = getShapePerCTA(dstTy);
    const auto srcShapePerCTATile = getShapePerCTATile(srcLayout, srcTy.getShape());
    const auto dstShapePerCTATile = getShapePerCTATile(dstLayout, dstTy.getShape());

    const unsigned rank = dstTy.getRank();
    std::vector<unsigned> repShape(rank);
    for (unsigned d = 0; d < rank; ++d) {
        repShape[d] = std::max(std::min(srcShapePerCTA[d], srcShapePerCTATile[d]),
                               std::min(dstShapePerCTA[d], dstShapePerCTATile[d]));
    }
    return repShape;
}

/**
 * @brief Get scratch config for a layout conversion operation.
 * @param op The ConvertLayoutOp operation.
 * @param inVec Reference to input vectorization factor.
 * @param outVec Reference to output vectorization factor.
 * @return Vector of scratch config dimensions.
 */
auto getScratchConfigForCvtLayout(const triton::gpu::ConvertLayoutOp &op, unsigned &inVec, unsigned &outVec)
    -> std::vector<unsigned> {
    const auto repShape = getRepShapeForCvtLayout(op);

    const auto srcTy = op.getSrc().getType().cast<RankedTensorType>();
    const auto dstTy = op.getResult().getType().cast<RankedTensorType>();
    const auto srcLayout = srcTy.getEncoding();
    const auto dstLayout = dstTy.getEncoding();

    const auto [inOrd, outOrd] = getCvtOrder(srcLayout, dstLayout);
    const auto srcContigPerThread = getUniqueContigPerThread(srcLayout, srcTy.getShape())[inOrd[0]];
    const auto dstContigPerThread = getUniqueContigPerThread(dstLayout, dstTy.getShape())[outOrd[0]];

    // Handle vectorization issues
    inVec = outOrd[0] == 0 ? 1 : (inOrd[0] == 0 ? 1 : srcContigPerThread);
    outVec = outOrd[0] == 0 ? 1 : dstContigPerThread;

    if (repShape.size() <= 1)
        return repShape;

    unsigned paddedDim = 1;
    if (const auto dstBlockedLayout = dstLayout.dyn_cast<BlockedEncodingAttr>()) {
        paddedDim = dstBlockedLayout.getOrder()[0];
    }

    const unsigned pad = std::max(inVec, outVec);
    repShape[paddedDim] += pad;
    return repShape;
}

/**
 * @brief Run the allocation analysis.
 */
void AllocationAnalysis::run() {
    getValuesAndSizes();
    resolveLiveness();
    computeOffsets();
}

// Additional modernizations and refactoring can be applied to other functions as needed.

} // namespace mlir::triton
