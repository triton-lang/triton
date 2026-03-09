// AppleMmaEncodingAttr::toLinearLayout
//
// Converts Apple simdgroup MMA encoding → LinearLayout used throughout
// the Triton compiler for layout propagation, conversion insertion,
// and shared memory access analysis.
//
// Verified layout (verify_simdgroup.metal, M1 hardware):
//   lane T, reg R → row = (T >> 3) + R*4,  col = T & 7

#include "Dialect/TritonAppleGPU/IR/Dialect.h"
#include "triton/Tools/LinearLayout.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

#define S(v) StringAttr::get(ctx, (v))

namespace mlir::triton::applegpu {

LinearLayout AppleMmaEncodingAttr::toLinearLayout(
    llvm::ArrayRef<int64_t> shape) const {

    MLIRContext *ctx = getContext();
    int rank = shape.size();
    assert(rank == 2 && "AppleMmaEncoding only supports 2D tensors for now");

    auto dimNames = standardOutDimNames(ctx, rank);
    auto dimRow = dimNames[0];  // "dim0"
    auto dimCol = dimNames[1];  // "dim1"

    // ── Single 8×8 simdgroup tile ─────────────────────────────────────────
    //
    // Basis: lane[0:2]→col, lane[3:4]→row, reg[0]→row+4
    //
    // identity1D(N, inDim, outDim):
    //   consumes log2(N) bits of inDim, maps them to next log2(N) bits of outDim
    LinearLayout ctaLayout =
        LinearLayout::identity1D(8, S("lane"),     dimCol) *  // cols 0-7
        LinearLayout::identity1D(4, S("lane"),     dimRow) *  // rows 0-3
        LinearLayout::identity1D(2, S("register"), dimRow);   // rows 4-7

    // ── Tile simdgroups across M and N ────────────────────────────────────
    auto wpc = getWarpsPerCTA();
    assert(wpc.size() == 2);

    // Row-major warp tiling: [0]=M dim, [1]=N dim
    SmallVector<unsigned> warpOrder{0, 1};
    ctaLayout *=
        identityStandardND(S("warp"), wpc, warpOrder)
            .transposeOuts(llvm::to_vector(ctaLayout.getOutDimNames()));

    // ── Broadcast to full tensor shape ────────────────────────────────────
    // Handles shapes larger than one CTA tile by repeating the pattern.
    // Apple has no CGA (cooperative group arrays) so pass empty CGALayout.
    return combineCtaCgaWithShape(ctaLayout, CGAEncodingAttr{}, shape);
}

} // namespace mlir::triton::applegpu
