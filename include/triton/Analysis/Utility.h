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

// Layout coordinates used by scan lowering:
//
// A thread group is a list of registers holding consecutive elements along the
// scan axis in one thread. Registers in each group are ordered by their logical
// axis coordinate. threadLocalSize counts elements, not register-index bits.
//
// A chunk joins these thread-local ranges across lanes for a warp scan. It is a
// contiguous logical axis range of warpChunkSize elements whose internal bits
// are the thread-local register bits followed by lane bits. The next register
// or warp bit starts another chunk. Thus a warp can own several chunks, even
// when all of its elements are contiguous. Swizzled or overlapping
// register/lane bases fall back to one-element groups and chunks.
//
// For a 1D layout with register=[1,4], lane=[2,8], warp=[16]:
//   lane 0, warp 0 owns x0, x1, x4, x5; its thread groups are [r0,r1], [r2,r3].
//   lane 1, warp 0 owns x2, x3, x6, x7.
// A chunk combines two lanes' groups: [x0..x3], then [x4..x7], etc.
// threadLocalSize=2, warpChunkSize=4, and the chunk index of xi is i / 4.
// A chunk total is its final prefix in the scan direction; a carry is the
// combined total of preceding chunks, applied to the chunk's local prefixes.
//
// These definitions apply separately to each scan (coordinates off the axis).
// All scan communication stays within a CTA; axis bits owned by CTAs are not
// supported. Layout queries retain physical ownership and use increasing axis
// coordinates unless explicitly described as accounting for reverse scans.
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
  // Original ownership with redundant register bases removed.
  const triton::LinearLayout &getLayout() const { return layout; }
  // Original owners -> logical coordinates with the axis expressed in chunks.
  // Internal chunk bits map to zero; this describes totals, not a conversion.
  const triton::LinearLayout &getTotalsLayout() const { return *totalsLayout; }
  // Original owners -> scratch element offsets used to publish chunk totals.
  const triton::LinearLayout &getScratchAddressLayout() const {
    return *scratchAddressLayout;
  }
  llvm::ArrayRef<SmallVector<unsigned>> getThreadGroups() const {
    return threadGroups;
  }

private:
  triton::ScanOp scanOp;
  triton::LinearLayout layout;
  std::optional<triton::LinearLayout> totalsLayout;
  std::optional<triton::LinearLayout> scratchAddressLayout;
  // Scratch element offset -> logical coordinates in the totals layout.
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
