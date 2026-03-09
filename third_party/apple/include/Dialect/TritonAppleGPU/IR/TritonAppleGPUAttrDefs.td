#ifndef TRITON_APPLE_GPU_ATTRDEFS
#define TRITON_APPLE_GPU_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"
include "TritonAppleGPUDialect.td"

//===----------------------------------------------------------------------===//
// AppleMmaEncodingAttr
//===----------------------------------------------------------------------===//
//
// Encoding for tensors produced by simdgroup_multiply_accumulate on Apple GPUs.
//
// Empirically verified layout (verify_simdgroup.metal, M1):
//   32 threads per simdgroup, each owns 2 elements of an 8x8 output tile.
//   lane T, reg R → row = (T >> 3) + R*4,  col = T & 7
//
// Layout grid (one 8x8 tile):
//      col: 0  1  2  3  4  5  6  7
// row 0:    0  1  2  3  4  5  6  7   ← lanes 0-7,  reg 0
// row 1:    8  9 10 11 12 13 14 15   ← lanes 8-15, reg 0
// row 2:   16 17 18 19 20 21 22 23
// row 3:   24 25 26 27 28 29 30 31
// row 4:    0  1  2  3  4  5  6  7   ← lanes 0-7,  reg 1
// row 5:    8  9 10 11 12 13 14 15
// row 6:   16 17 18 19 20 21 22 23
// row 7:   24 25 26 27 28 29 30 31   ← lanes 24-31,reg 1
//
// LinearLayout XOR basis vectors:
//   lane  bit 0 → col bit 0 ┐
//   lane  bit 1 → col bit 1 ├ 8 cols
//   lane  bit 2 → col bit 2 ┘
//   lane  bit 3 → row bit 0 ┐
//   lane  bit 4 → row bit 1 ┘ 4 row groups
//   reg   bit 0 → row bit 2   2 halves (rows 0-3 vs 4-7)

def AppleMmaEncodingAttr : AttrDef<TritonAppleGPU_Dialect, "AppleMmaEncoding",
    [DeclareAttrInterfaceMethods<DistributedEncodingTrait>,
     DeclareAttrInterfaceMethods<MmaEncodingTrait>]> {

  let mnemonic = "apple_mma";

  let summary = "Apple simdgroup matmul encoding";

  let description = [{
    Encoding for tensors distributed across Apple simdgroup_multiply_accumulate
    8x8 tiles. Each simdgroup (32 threads) owns one 8x8 output tile.
    warpsPerCTA controls how many simdgroups tile the full tensor.
  }];

  let parameters = (ins
    // How many simdgroups tile the [M, N] dims of the full tensor.
    // e.g. warpsPerCTA=[2,4] → 8 simdgroups cover 16x32 output.
    ArrayRefParameter<"unsigned">:$warpsPerCTA
  );

  let assemblyFormat = "`<` `{` `warpsPerCTA` `=` `[` $warpsPerCTA `]` `}` `>`";

  let extraClassDeclaration = [{
    // Tile shape for one simdgroup — fixed by hardware
    static constexpr unsigned kTileM = 8;
    static constexpr unsigned kTileN = 8;
    static constexpr unsigned kTileK = 8;
    static constexpr unsigned kLanesPerSimdgroup = 32;
    static constexpr unsigned kRegsPerLane = 2;

    SmallVector<unsigned> getInstrShape() const {
      return {kTileM, kTileN};
    }

    // Full CTA tile = instrShape × warpsPerCTA
    SmallVector<unsigned> getCtaTileShape() const {
      auto wpc = getWarpsPerCTA();
      return {kTileM * wpc[0], kTileN * wpc[1]};
    }

    unsigned getRank() const { return getWarpsPerCTA().size(); }

    // Required by DistributedEncodingTrait
    SmallVector<unsigned> getThreadsPerWarp() const {
      // 32 threads per simdgroup = 1 warp on Apple
      return {kLanesPerSimdgroup};
    }

    // toLinearLayout and getWarpsPerCTA declared by tablegen from interfaces/parameters
  }];

  let genVerifyDecl = 1;
}

#endif // TRITON_APPLE_GPU_ATTRDEFS
