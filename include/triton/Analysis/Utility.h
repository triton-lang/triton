#ifndef TRITON_ANALYSIS_UTILITY_H
#define TRITON_ANALYSIS_UTILITY_H

#include <array>

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

// Callback to allow backends to specify a target-specific getter for scratch
// elements.
using GetNumScratchElemsFn =
    std::function<unsigned(const triton::LinearLayout &src,
                           const triton::LinearLayout &dst, unsigned bitwidth)>;

struct LayoutConversionScratchConfig {
  SmallVector<unsigned> offsets;
  unsigned sizeInBytes = 0;
};

// Byte offsets in operand order and total storage for one layout conversion.
LayoutConversionScratchConfig getLayoutConversionScratchConfig(
    const triton::LinearLayout &src, const triton::LinearLayout &dst,
    ArrayRef<Type> elementTypes,
    GetNumScratchElemsFn numScratchElemsGetter = nullptr);

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

  // Allocation size for the whole reduction: the maximum sizeInBytes returned
  // by getLayoutConversionScratchConfig across its inter-warp/CTA stages.
  unsigned
  getScratchSizeInBytes(GetNumScratchElemsFn numScratchElemsGetter = nullptr);

  InThreadVectorizeOpKind
  getInThreadVectorizeOpKind(bool supportBitwidth16Elementwise,
                             bool supportBitwidth32Elementwise);

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

// Lower a scan in three phases: scan thread-local segments, scan their totals
// within each warp-local segment, then exchange and scan warp-local totals.
// Propagate exclusive carries back to the saved thread-local prefixes.
//
// Segment: a contiguous range along the scan axis owned by one thread or warp.
// A totals layout maps physical owners to segment indices. A scan layout
// redistributes those totals for ordered accumulation.
//
// Example: forward sum of x0..x31 with two warps and two distinct lane owners
// per warp. R, L, W list scan-axis register, lane, and warp bases. Four zero
// lane bases and the empty block dimension are omitted. R=[] means one value
// per thread.
//
// 1. originalLayout = permutedLayout: R=[1,4,8], L=[2], W=[16].
//    Coordinates index original elements; the register order is canonical.
//      warp 0, lane 0: [x0,x1], [x4,x5], [x8,x9], [x12,x13]
//      warp 0, lane 1: [x2,x3], [x6,x7], [x10,x11], [x14,x15]
//      warp 1: the same pattern with element indices increased by 16.
//    threadSegmentSize=2, warpSegmentSize=16. Scan each pair locally to obtain
//    [x(2j), Tj], where Tj=x(2j)+x(2j+1). Save these element prefixes.
//
// 2. intraWarpTotalsLayout: R=[2,4], L=[1], W=[8].
//    Extract each pair's total. Divide the element bases by threadSegmentSize
//    and remove zero register bases; coordinates now index thread segments.
//      warp 0, lane 0: T0,T2,T4,T6; lane 1: T1,T3,T5,T7
//      warp 1, lane 0: T8,T10,T12,T14; lane 1: T9,T11,T13,T15
//    intraWarpScanLayout: R=[1,2], L=[4], W=[8].
//    Put the segment's register bits before its lane bits and convert with
//    warp shuffles. Each thread now holds four consecutive totals:
//      warp 0, lane 0: T0,T1,T2,T3; lane 1: T4,T5,T6,T7
//      warp 1, lane 0: T8,T9,T10,T11; lane 1: T12,T13,T14,T15
//    Scan registers, then lanes. Slot j holds Uj=T(8w)+...+Tj in warp w.
//    These warp-local prefixes retain intraWarpScanLayout.
//
// 3. interWarpTotalsLayout: R=[], L=[0], W=[1].
//    Each warp segment contains eight thread totals. Extract register 3:
//      warp 0: lane 0 holds U3; lane 1 holds U7
//      warp 1: lane 0 holds U11; lane 1 holds U15
//    Only terminal lane 1 has the complete warp total: W0=U7, W1=U15.
//    Divide the source axis bases by eight and remove zero register bases.
//    Coordinates now index warp segments. Broadcast each terminal value to
//    the lanes represented by zero lane bases. Keep the Uj prefixes in
//    intraWarpScanLayout for later use.
//
//    interWarpScanLayout: R=[], L=[1], W=[0].
//    Broadcast the full sequence across participating warps. Store W0 and W1
//    from their selected owners, synchronize, then load this layout:
//      each warp, lane 0: W0; lane 1: W1
//    Scan the totals, preserving their layout:
//      each warp, lane 0: S0=W0; lane 1: S1=W0+W1
//
// Carry propagation:
//    Map the consumer's warp-segment index s to the preceding prefix S(s-1)
//    using the inverse of interWarpScanLayout. Warp 0 has no carry; warp 1
//    reads S0 from lane 0 of its own warp. Apply this carry to the saved Uj:
//      warp 0: Qj=Uj                for j=0..7
//      warp 1: Qj=S0+Uj=T0+...+Tj  for j=8..15
//    Qj is a prefix from the scan's start and retains intraWarpScanLayout.
//    The Sj values remain broadcast across warps in interWarpScanLayout.
//    Convert Qj back to intraWarpTotalsLayout:
//      warp 0, lane 0: Q0,Q2,Q4,Q6; lane 1: Q1,Q3,Q5,Q7
//      warp 1, lane 0: Q8,Q10,Q12,Q14; lane 1: Q9,Q11,Q13,Q15
//    Replace each saved pair's terminal value with Qj. Complete its first
//    value with Q(j-1)+x(2j), leaving x0 unchanged. The first pair in warp 1
//    uses its inter-warp carry S0=Q7. Invert the register permutation to return
//    the completed element prefixes from permutedLayout to originalLayout.
//
class ScanLoweringHelper {
public:
  explicit ScanLoweringHelper(triton::ScanOp op);
  ScanLoweringHelper(const triton::LinearLayout &inputLayout, unsigned axis);
  bool isSupported();
  const triton::LinearLayout &getPermutedLayout() const {
    return permutedLayout;
  }
  const triton::ColumnAction &getRegisterOrder() const { return registerOrder; }

  unsigned getThreadSegmentSize() const { return threadSegmentSize; }

  const std::optional<triton::LinearLayout> &getIntraWarpTotalsLayout() const {
    return intraWarpTotalsLayout;
  }

  const std::optional<triton::LinearLayout> &getIntraWarpScanLayout() const {
    return intraWarpScanLayout;
  }

  unsigned getWarpSegmentSize() const { return warpSegmentSize; }

  const std::optional<triton::LinearLayout> &getInterWarpTotalsLayout() const {
    return interWarpTotalsLayout;
  }

  const std::optional<triton::LinearLayout> &getInterWarpScanLayout() const {
    return interWarpScanLayout;
  }

  unsigned
  getScratchSizeInBytes(GetNumScratchElemsFn numScratchElemsGetter = nullptr);

private:
  triton::LinearLayout buildPermutedLayout();
  triton::LinearLayout buildIntraWarpTotalsLayout() const;
  triton::LinearLayout buildIntraWarpScanLayout() const;
  triton::LinearLayout buildInterWarpTotalsLayout() const;
  triton::LinearLayout buildInterWarpScanLayout() const;

  triton::ScanOp op;
  unsigned axis;
  triton::LinearLayout originalLayout;
  // Register zero bases removed, then axis register bits ordered logically.
  triton::LinearLayout permutedLayout;
  triton::ColumnAction registerOrder;
  unsigned threadSegmentSize = 1;
  unsigned warpSegmentSize = 1;
  std::optional<triton::LinearLayout> intraWarpTotalsLayout;
  std::optional<triton::LinearLayout> intraWarpScanLayout;
  std::optional<triton::LinearLayout> interWarpTotalsLayout;
  // Broadcast each scan's full segment sequence across participating warps;
  // distribute it over registers and lanes while preserving independent scans.
  std::optional<triton::LinearLayout> interWarpScanLayout;
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
bool cvtNeedsSharedMemory(const triton::LinearLayout &src,
                          const triton::LinearLayout &dst);

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
