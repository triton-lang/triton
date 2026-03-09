// Apple simdgroup_multiply_accumulate LinearLayout basis vectors.
//
// Empirically verified on M1 (verify_simdgroup.metal):
//   lane T, reg R → row = (T >> 3) + R*4,  col = T & 7
//
// XOR basis vectors:
//   lane  bit 0 → col bit 0
//   lane  bit 1 → col bit 1
//   lane  bit 2 → col bit 2  (8 cols)
//   lane  bit 3 → row bit 0
//   lane  bit 4 → row bit 1  (4 row groups)
//   reg   bit 0 → row bit 2  (row halves: 0-3 vs 4-7)

#include "Dialect/TritonAppleGPU/IR/Dialect.h"
#include "triton/Tools/LinearLayout.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::triton;

#define S(v) StringAttr::get(ctx, (v))

namespace mlir::triton::applegpu {

// Returns the LinearLayout for one 8x8 simdgroup_multiply_accumulate tile.
//
// In-dims:  "lane" (0..31), "register" (0..1)
// Out-dims: "dim0"=row (0..7), "dim1"=col (0..7)
LinearLayout appleMmaTile(MLIRContext *ctx) {
    auto dimRow = S("dim0");
    auto dimCol = S("dim1");

    // Build layout as product of identity1D maps:
    //   identity1D(N, inDim, outDim) contributes log2(N) bits:
    //   the next log2(N) bits of inDim XOR-map to the next log2(N) bits of outDim.

    LinearLayout layout =
        // lane bits [0:2] → col bits [0:2]  (8 columns)
        LinearLayout::identity1D(8, S("lane"),     dimCol) *
        // lane bits [3:4] → row bits [0:1]  (4 row groups: rows 0,1,2,3)
        LinearLayout::identity1D(4, S("lane"),     dimRow) *
        // reg  bit  [0]   → row bit  [2]    (row halves: +4)
        LinearLayout::identity1D(2, S("register"), dimRow);

    return layout;
}

// Full CTA layout: tile multiple simdgroups across M and N.
// warpsPerCTA = [warpsM, warpsN]
LinearLayout appleMmaCtaLayout(MLIRContext *ctx,
                                llvm::ArrayRef<unsigned> warpsPerCTA) {
    assert(warpsPerCTA.size() == 2);
    auto layout = appleMmaTile(ctx);

    auto dimRow = S("dim0");
    auto dimCol = S("dim1");

    // Tile warps: warpsPerCTA[1] along cols, [0] along rows (row-major)
    layout *=
        LinearLayout::identity1D(warpsPerCTA[1], S("warp"), dimCol) *
        LinearLayout::identity1D(warpsPerCTA[0], S("warp"), dimRow);

    return layout;
}

} // namespace mlir::triton::applegpu
