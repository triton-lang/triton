#ifndef TRITON_ANALYSIS_UTILITY_H
#define TRITON_ANALYSIS_UTILITY_H

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/LinearLayout.h"

namespace mlir {

inline bool isZeroConst(Value v) {
  auto constantOp = v.getDefiningOp<arith::ConstantOp>();
  if (!constantOp)
    return false;
  if (auto denseAttr = dyn_cast<DenseFPElementsAttr>(constantOp.getValueAttr()))
    return denseAttr.isSplat() && denseAttr.getSplatValue<APFloat>().isZero();
  if (auto denseAttr =
          dyn_cast<DenseIntElementsAttr>(constantOp.getValueAttr()))
    return denseAttr.isSplat() && denseAttr.getSplatValue<APInt>().isZero();
  return false;
}

class ReduceOpHelper {
public:
  enum class InThreadVectorizeOpKind {
    None,
    AddF,
    MulF,
    MinNumF,
    MaxNumF,
    MinimumF,
    MaximumF,
    AddI,
    MulI,
    MinSI,
    MaxSI,
    MinUI,
    MaxUI,
  };

  explicit ReduceOpHelper(triton::ReduceOp op)
      : op(op), srcTy(op.getInputTypes().front()), srcShape(srcTy.getShape()),
        srcEncoding(srcTy.getEncoding()), axis(op.getAxis()) {}

  RankedTensorType getSrcTy() { return srcTy; }

  unsigned getInterWarpSizeWithUniqueData();

  unsigned getIntraWarpSizeWithUniqueData();

  bool isReduceWithinCTA();

  bool isAssociative();

  // Callback to allow backends to specify a target-specific getter for scratch
  // elements.
  using GetNumScratchElemsFn = std::function<unsigned(
      const triton::LinearLayout &src, const triton::LinearLayout &dst,
      unsigned bitwidth)>;

  // Allocation size for the whole reduction: the maximum sizeInBytes returned
  // by getScratchConfig across its inter-warp/CTA reduction stages.
  unsigned
  getScratchSizeInBytes(GetNumScratchElemsFn numScratchElemsGetter = nullptr);

  InThreadVectorizeOpKind
  getInThreadVectorizeOpKind(bool supportBitwidth16Elementwise,
                             bool supportBitwidth32Elementwise);

  struct ScratchConfig {
    SmallVector<unsigned> offsets;
    unsigned sizeInBytes = 0;
  };

  // Byte offsets in operand order and total storage for one layout conversion.
  ScratchConfig
  getScratchConfig(const triton::LinearLayout &src,
                   const triton::LinearLayout &dst,
                   GetNumScratchElemsFn numScratchElemsGetter = nullptr);

  static triton::ColumnAction
  moveAxisBasesToFront(const triton::LinearLayout &layout, int axis,
                       bool isVectorized = false);

  static triton::LinearLayout
  zeroBasesAlongDimAndReorder(const triton::LinearLayout &layout, unsigned axis,
                              mlir::StringAttr dim);

  static triton::LinearLayout
  getInterWarpReductionLayout(const triton::LinearLayout &layout,
                              unsigned axis);

  // Removes redundant register bases but retains replicated lanes.
  static triton::LinearLayout reducedRegLaneLayout(RankedTensorType srcTy,
                                                   unsigned axis);

private:
  triton::ReduceOp op;
  RankedTensorType srcTy;
  ArrayRef<int64_t> srcShape;
  Attribute srcEncoding;
  int axis;
};

// Layout analysis for scan lowering. The algorithm consists of three stages:
// 1. Compute inclusive prefixes within each thread's groups.
// 2. Scan group totals across lanes within each chunk using warp shuffles.
// 3. Scan chunk totals in logical order and apply the resulting carries to the
//    group prefixes. Use shared memory when the scan spans multiple warps;
//    otherwise, complete this stage within the warp.
//
// Definitions (for fixed coordinates outside the scan axis):
// - thread: A physical lane within a warp, owning a set of register values.
// - group: An ordered set of registers in one thread representing a contiguous
//   axis interval of threadLocalSize elements.
// - chunk: A contiguous axis interval of warpChunkSize elements formed by
//   groups in one warp. Its internal axis bits consist of the group's register
//   bits followed by lane bits; the next register or warp bit starts a new
//   chunk. A warp may own multiple chunks, even for contiguous warp ownership.
// - total: The final inclusive prefix of a group or chunk.
// - carry: The combined total of preceding groups or chunks in scan order.
//
// Example: register=[1,4], lane=[2,8], warp=[16]. In warp 0, lane 0 owns
// [x0,x1,x4,x5] and lane 1 owns [x2,x3,x6,x7]. The register groups are
// [r0,r1] and [r2,r3]; the first two chunks are [x0..x3] and [x4..x7].
// Thus threadLocalSize=2, warpChunkSize=4, and xi has chunk index i / 4.
//
// Swizzled or overlapping register/lane bases use one-element groups and
// chunks. Scans across CTAs are unsupported. Layouts describe increasing axis
// coordinates; reverse traversal is handled by the lowering's reflections.
class ScanLoweringHelper {
public:
  explicit ScanLoweringHelper(triton::ScanOp op);
  bool isSupported() const;
  bool hasInterWarpScan() const;
  unsigned getScratchSizeInElems() const;
  unsigned getScratchSizeInBytes() const;
  unsigned getThreadLocalSize() const { return threadLocalSize; }
  unsigned getWarpChunkSize() const { return warpChunkSize; }
  unsigned getAxisMask(mlir::StringAttr dim, unsigned size) const;
  // Remaining axis XOR mask for reverse scans after reversing registers/lanes.
  unsigned getAxisOffset() const;
  // Logical axis coordinate in chunk units, before reverse-scan reflection.
  unsigned getChunkIndex(unsigned reg, unsigned lane = 0,
                         unsigned warp = 0) const;
  // Inclusive chunk-index bounds over all lanes/warps for this register,
  // accounting for the remaining reverse-scan reflection.
  std::pair<unsigned, unsigned> getChunkBounds(unsigned reg) const;
  // Map a requested chunk index and the current scan's off-axis coordinates to
  // its shared-memory offset (inter-warp) or register/lane owner (intra-warp).
  triton::LinearLayout getChunkLookup() const;
  // Register index -> index into getThreadGroups().
  llvm::ArrayRef<unsigned> getRegisterGroups() const { return registerGroups; }
  // layout: (register, lane, warp, block) -> tensor element coordinates.
  // Register indices exclude redundant register bases; ownership is preserved.
  const triton::LinearLayout &getLayout() const { return layout; }
  // totalsLayout: (register, lane, warp, block) -> tensor coordinates with
  // axis coordinate floor(element / warpChunkSize). All elements of a chunk
  // map to its chunk index; coordinates outside the axis remain unchanged.
  const triton::LinearLayout &getTotalsLayout() const { return *totalsLayout; }
  // scratchAddressLayout: (register, lane, warp, block) -> scratch element
  // offset. Internal group-register and chunk-lane bits map to zero; terminal
  // lanes publish chunk totals, with redundant owners excluded by predicates.
  const triton::LinearLayout &getScratchAddressLayout() const {
    return *scratchAddressLayout;
  }
  // Group index -> register indices in increasing logical axis order.
  llvm::ArrayRef<SmallVector<unsigned>> getThreadGroups() const {
    return threadGroups;
  }

private:
  triton::ScanOp scanOp;
  triton::LinearLayout layout;
  std::optional<triton::LinearLayout> totalsLayout;
  std::optional<triton::LinearLayout> scratchAddressLayout;
  // scratchLayout: scratch element offset -> chunk index and coordinates
  // outside the scan axis, using the same output space as totalsLayout.
  std::optional<triton::LinearLayout> scratchLayout;
  SmallVector<SmallVector<unsigned>> threadGroups;
  SmallVector<unsigned> registerGroups;
  unsigned threadLocalSize;
  unsigned warpChunkSize;
};

// Helper class for lowering `tt.gather` operations. This class shares lowering
// logic between shared memory allocation and LLVM codegen.
class GatherLoweringHelper {
public:
  GatherLoweringHelper(triton::GatherOp gatherOp);

  // Get the shared memory scratch size required by this op.
  unsigned getScratchSizeInBytes();
  // Determine if the gather can be performed completely within a warp.
  bool isWarpLocal();

private:
  triton::GatherOp gatherOp;
  RankedTensorType srcTy;
  RankedTensorType dstTy;
};

// For permutation cases, this struct represents the factorization of a
// warp-local layout conversion into three components: a register-only
// permutation, a lane-only permutation, and a set of swaps between lane and
// register basis vectors. Algebraically, it represents the factorization
// P = P_mixed \circ P_lane \circ P_reg. It is used to aid in the implementation
// of the layout conversion using warp-shuffles.
//
// `pReg` is a square permutation of the padded registers. `shuffleMap` maps
// register/lane/warp/block coordinates to source lanes for the shuffle stage.
// `mixedTranspositions` holds the register bit and source/destination lane bits
// for each exchange, along with 16-bit selectors for byte permute instructions
// (where each of the four nybbles is in the range [0, 7]). A lane bit of -1
// denotes a constant-zero predicate for a one-sided exchange.
// `nPack` gives the number of basis vectors that can be used for register
// packing while ensuring packed elements arrive at the same destination lane.
struct DecomposedWarpConversion {
  struct TranspositionInfo {
    int regBit;
    int srcLane;
    int dstLane;
    uint16_t topPreSel = 0x3210;
    uint16_t botPreSel = 0x7654;
    uint16_t topPostSel = 0x3210;
    uint16_t botPostSel = 0x7654;
  };

  triton::LinearLayout pReg, shuffleMap;
  SmallVector<TranspositionInfo> mixedTranspositions;
  int nPack;
};

// Produces a warp-local decomposition.
//
// For permutation cases, the numbers of register and lane basis vectors may
// differ between the two layouts. This is handled by padding the smaller
// dimension(s) with zero vectors, ensuring that the layout conversion can be
// represented as a permutation.
//
// Supports permutation layouts with warp/CTA-dependent lane selection. Source
// registers must not depend on warp/CTA coordinates.
// The layouts must not contain broadcasted register bases.
DecomposedWarpConversion
getWarpLayoutConvertDecomposition(const triton::LinearLayout &srcLayout,
                                  const triton::LinearLayout &dstLayout,
                                  int bitwidth);

// Decomposes a reshape into simpler pieces.
//
// As an example, suppose we have a reshape from [4,4,4] to [2,2,8,2].
// You might explain what this does as follows.
//
//  - Split the first input dimension into [2,2].
//  - Take the remaining two input dimensions, merge them into a single [16]
//    dim, and then split that into [8,2].
//
// In general, a reshape can be described a sequence of smushing one or more
// input dimensions together and then breaking them apart into one or more
// output dimensions.  So we could represent the example above as follows.
//
//   [
//     ([0], [0, 1]),  # input dim [0] -> output dims [0, 1]
//     ([1, 2], [2, 3]),  # input dims [1, 2] -> output dims [2, 3]
//   ]
//
// Notice that the input dims (first tuple elems) appear in sequential order if
// you read left-to-right-top-to-bottom, and so do the output dims.
//
// This function returns the above decomposition.
SmallVector<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>>
getReshapeDecomposition(ArrayRef<int64_t> srcShape, ArrayRef<int64_t> dstShape);

// Returns the number of elements in the scratch space needed.
// If shape is empty, it means no shared memory is needed.
unsigned getNumScratchElements(ArrayRef<unsigned> shape);

bool supportMMA(triton::DotOp op, int version);

bool supportMMA(triton::DotOpInterface op, int version);

bool supportMMA(Value value, int version);

// Conversion from `srcTy` to `dstTy` only involves reordering of registers.
// There is no need for data exchange across threads, warps, or blocks.
bool cvtReordersRegisters(RankedTensorType srcTy, RankedTensorType dstTy);

// The conversion involves data exchange across threads within a warp, or is
// explicitly forced to use warp shuffles.
bool cvtNeedsWarpShuffle(triton::gpu::ConvertLayoutOp op);

// The conversion requires data exchange through shared memory.
bool cvtNeedsSharedMemory(triton::gpu::ConvertLayoutOp op);

/// Create a basic DataFlowSolver with constant and dead code analysis included.
std::unique_ptr<DataFlowSolver> createDataFlowSolver();

bool isCvtDimSync(const triton::LinearLayout &srcLayout,
                  const triton::LinearLayout &dstLayout, StringAttr dim);

namespace triton {

bool canUseWarpBallotHistogram(HistogramOp op);

struct BarrierStages {
  // Stages are independent: for example, a release atomic with scratch has
  // both a leading ordering barrier and a scratch rendezvous.
  bool beforeMemoryEffects = false;
  bool afterMemoryEffects = false;
  bool betweenMemoryEffects = false;

  bool hasBarrier() const {
    return beforeMemoryEffects || betweenMemoryEffects || afterMemoryEffects;
  }
};

// Classify the barriers required by an atomic at a chosen scope. A result
// broadcast barrier at that scope supplies the post-atomic rendezvous itself.
BarrierStages getAtomicBarrierStages(MemSemantic semantic,
                                     bool hasResultBarrier);

// Lane bits to clear when shuffling an atomic result from its issuing lane.
// Returns nullopt when broadcasting the result requires shared memory.
std::optional<int32_t> getAtomicResultShuffleMask(Value result);

// Whether distributing an atomic result requires communication between CTAs.
bool atomicResultHasCTABroadcast(Operation *op);

} // namespace triton

namespace triton::nvidia_gpu {

bool needsClusterBarrier(Operation *op);

} // namespace triton::nvidia_gpu

} // namespace mlir

#endif // TRITON_ANALYSIS_UTILITY_H
