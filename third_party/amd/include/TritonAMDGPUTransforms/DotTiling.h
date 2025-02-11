#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "third_party/amd/include/TritonAMDGPUTransforms/MfmaGroup.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-refine-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {
/*
 TODO - this needs to be MUCH more official.
*/
unsigned calcCyclesPerMfma(AMDMfmaEncodingAttr mfmaLayout, DotOp dotOp) {
  // Get mfma op type.
  Value a = dotOp.getA();
  Value b = dotOp.getB();
  auto aTensorTy = cast<RankedTensorType>(a.getType());
  auto bTensorTy = cast<RankedTensorType>(b.getType());
  auto elemTyA = aTensorTy.getElementType();
  auto elemTyB = bTensorTy.getElementType();
  auto mDim = mfmaLayout.getMDim();
  auto nDim = mfmaLayout.getNDim();
  auto mfmaVersion = mfmaLayout.getVersionMajor();
  bool allowXF32 =
      dotOp.getInputPrecision() == InputPrecision::TF32 && mfmaVersion == 3;
  auto maybeMfmaInsn = MfmaInsn::selectMfma(mDim, nDim, elemTyA, elemTyB,
                                              mfmaVersion, allowXF32);
  if (failed(maybeMfmaInsn))
    llvm::report_fatal_error("No match found in MFMA database\n");
  // Estimate rate of mfma op type.
  unsigned maxBitWidth = std::max(maybeMfmaInsn->getElementTypeA().getIntOrFloatBitWidth(),
                                  maybeMfmaInsn->getElementTypeB().getIntOrFloatBitWidth());
  // Estimate throughput as fma's per cycle.
  unsigned opsPerCycle;
  if (maxBitWidth <= 8) { // fp8, bf8, i8
    opsPerCycle = 512;
  } else if (maxBitWidth <= 16) { // fp16, bf16
    opsPerCycle = 256;
  } else if (maxBitWidth <= 32) { // fp32
    opsPerCycle = 128;
  } else {
    opsPerCycle = 64; // fp64
  }
  // total floating point mfmas
  int64_t totalOps = maybeMfmaInsn->getMDim()
      * maybeMfmaInsn->getNDim()
      * maybeMfmaInsn->getKDim();
  unsigned cyclesPerMfma = static_cast<unsigned>(totalOps / opsPerCycle);
  llvm::outs() << maybeMfmaInsn->getInsnName() << " = " << cyclesPerMfma << " cycles\n";
  return cyclesPerMfma;
}

/*
Calculate how many mfmas are in a rep, e.g. 1x1x2.
// TODO - is there a more direct method for this?
*/
SmallVector<unsigned, 3> calcMfmasPerRep(
    SmallVector<int64_t> ctaTile,
    SmallVector<unsigned> warpsPerCta,
    SmallVector<int64_t> numReps,
    SmallVector<unsigned> mfmaShape
) {
  // Tile size per warp.
  SmallVector<int64_t, 3> warpTile = {
    ctaTile[0] / warpsPerCta[0],
    ctaTile[1] / warpsPerCta[1],
    ctaTile[2],
  };
  // Tile size per rep.
  SmallVector<int64_t, 3> repTile = {
    warpTile[0] / numReps[0],
    warpTile[1] / numReps[1],
    warpTile[2] / numReps[2],
  };
  SmallVector<unsigned, 3> mfmasPerRep = {
    static_cast<unsigned>(repTile[0] / mfmaShape[0]),
    static_cast<unsigned>(repTile[1] / mfmaShape[1]),
    static_cast<unsigned>(repTile[2] / mfmaShape[2])};
  return mfmasPerRep;
}

// TODO(guacamoleo) - we also need to detect whether the a or b operands of a tile are already loaded
// into registers, e.g. for FA, and we want to choose the tile shape (small improvement)
// and tile order (rows vs colums which will be a large improvement).

/*
 Returns the dot tile size (in number of reps, not number of mfmas);
 typical sizes when mfmasPerRep = 1x1x2 for b128 and localLoadIssueRate=32 for b128
 2x2 for fp16 (128 mfma cycles per tile) and
 4x4 for fp8 (256 mfma cycles per tile).
 
 Tile size is chosen so that
 (1) A tile's worth of local_load_a or local_load_b can 
     can be issued during a tile's worth of mfmas, and
 (2) A tile's worth of mfmas hides the ds_read data latency.
 Args:
  - mfmasPerRep - 3D shape of number of mfmas in decomposed dot, e.g. 1x1x2.
  - preferLargerM - prefer M > N if not square.
  - cyclesPerMfma - how many cycles does mfma take in total.
  - localLoadIssueRate - cycles between issuing consecutive ds_reads to not overrun hardware queues.
      This is estimated to be b128 -> 32 cycles, b64 -> 16 cycles, b32 -> 8 cycles.
  - localLoadDataLatency - cycles between issuing ds_read and waiting for data; rounded up to pow2.
  Notes:
  The intended scheduling of tiles-of-reps-of-mfmas is 
  --------------------------------
  local_load_a[n]
  MFMA_Tile[n-3] // a can be loaded during tile
  --------------------------------
  local_load_b[n]
  MFMA_Tile[n-2] // b can be loaded during tile
  --------------------------------
  MFMA_Tile[n-1] // load data latency hiding
  --------------------------------
  MFMA_Tile[n] // all a,b data is ready by the first mfma of tile.
  --------------------------------
  This can be further refined if the data latency becomes much larger than
  the issue rate; in this case we can remove the condition that one tile
  hides all the data latency (which could make the tiles huge and waste registers),
  and intead local load issue rate is the only criteria and we retroactively calculate
  how many tiles are needed to hide the data latency.
*/
SmallVector<unsigned, 2> calcDotTileSize(
  SmallVector<unsigned, 3> mfmasPerRep, // = 16x16x64 / 16x16x32 = 1x1x2; served by 1 a,b load
  bool preferLargerM,
  unsigned cyclesPerMfma = 8,
  unsigned localLoadIssueRate = 32,
  unsigned localLoadDataLatency = 128
) {
  SmallVector<unsigned, 2> tileSize = {1, 1};
  int64_t numMfmas = tileSize[0]*tileSize[1]*mfmasPerRep[0]*mfmasPerRep[1]*mfmasPerRep[2];
  int64_t mfmaCycles = numMfmas * cyclesPerMfma;
  int64_t numLoads = std::max(tileSize[0]*mfmasPerRep[0], tileSize[1]*mfmasPerRep[1]);
  int64_t loadIssueCycles = numLoads * localLoadIssueRate;
  while (mfmaCycles < loadIssueCycles || mfmaCycles < localLoadDataLatency) {
    if (tileSize[0]*mfmasPerRep[0] < tileSize[1]*mfmasPerRep[1] || preferLargerM) {
      tileSize[0] *= 2;
    } else {
      tileSize[1] *= 2;
    }
    numMfmas = tileSize[0]*tileSize[1]*mfmasPerRep[0]*mfmasPerRep[1]*mfmasPerRep[2];
    mfmaCycles = numMfmas * cyclesPerMfma;
    numLoads = tileSize[0]*mfmasPerRep[0]+tileSize[1]*mfmasPerRep[1];
    loadIssueCycles = numLoads * localLoadIssueRate;
  };
  return tileSize;
}

/*
  DotTiling creates tiles of mfmas while they are decomposed from a dot operation.
  A tile of mfmas is a set of mfmas that will be co-scheduled
  because they use the same A,B operands; co-scheduling mfmas with same operands
  allows finer control over prefetching from LDS and register usage for these operands.
  Args:
   - inputNumRepM - total number of [decomposed] dot ops along m.
   - inputNumRepN - total number of [decomposed] dot ops along n.
   - inputTileSizeM - number of [decomposed] dot ops along m per tile.
   - inputTileSizeN - number of [decomposed] dot ops along n per tile.
   - inputOuterLoopM - should be set to (warpTileM >= warpTileN). True means m should be
       outer loop of mfma ops so that inner loop is smaller dimension which leads to smallest
       number of registers carrying A,B operands.
  E.g. numRep = 8x4, tileSize=2x2.
*/
struct DotTileOrder {
  const int numRepM;
  const int numRepN;
  const int tileSizeM;
  const int tileSizeN;
  const int numTilesM;
  const int numTilesN;
  bool outerTileM;
  int tileSizeOuter;
  int tileSizeInner;
  int numTilesOuter;
  int numTilesInner;
  explicit DotTileOrder(int inputNumRepM, int inputNumRepN,
                     int inputTileSizeM, int inputTileSizeN,
                     bool inputOuterLoopM)
      : numRepM(inputNumRepM),
        numRepN(inputNumRepN),
        tileSizeM(inputTileSizeM),
        tileSizeN(inputTileSizeN),
        numTilesM(numRepM / tileSizeM),
        numTilesN(numRepN / tileSizeN),
        outerTileM(inputOuterLoopM) {
    // Num mfmas must evenly divide into tiles.
    assert(numTilesM * tileSizeM == numRepM);
    assert(numTilesN * tileSizeN == numRepN);
    // Assign M and N to be outer vs inner tile loop.
    if (outerTileM) {
      // M is tile of outer loop.
      tileSizeOuter = tileSizeM;
      tileSizeInner = tileSizeN;
      numTilesOuter = numTilesM;
      numTilesInner = numTilesN;
    } else {
      // N is tile of outer loop.
      tileSizeOuter = tileSizeN;
      tileSizeInner = tileSizeM;
      numTilesOuter = numTilesN;
      numTilesInner = numTilesM;
    }
  }
  int getTileSizeM() const { return tileSizeM; }
  int getTileSizeN() const { return tileSizeN; }
  int getNumTilesOuter() const { return numTilesOuter; }
  int getNumTilesInner() const { return numTilesInner; }
  int getTileStartM(int tileOuterIdx, int tileInnerIdx) const {
    if (outerTileM) {
      return tileOuterIdx * tileSizeOuter; // M is outer tile loop.
    } else {
      return tileInnerIdx * tileSizeInner; // M is inner tile loop.
    }
  }
  int getTileStartN(int tileOuterIdx, int tileInnerIdx) const {
    if (outerTileM) {
      return tileInnerIdx * tileSizeInner; // N is inner tile loop.
    } else {
      return tileOuterIdx * tileSizeOuter; // N is outer tile loop.
    }
  }
};

} // namespace
