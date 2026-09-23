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
      : op(op.getOperation()), axis(op.getAxis()) {
    auto firstTy = cast<RankedTensorType>(op.getOperands()[0].getType());
    srcTy = firstTy;
    srcShape = firstTy.getShape();
    srcEncoding = firstTy.getEncoding();
    srcElementTypes = op.getElementTypes();

    for (const auto &t : op.getInputTypes()) {
      if (t.getShape() != srcShape) {
        op.emitError() << "shape mismatch";
      }
      if (t.getEncoding() != srcEncoding) {
        op.emitError() << "encoding mismatch";
      }
    }
  }

  RankedTensorType getSrcTy() { return srcTy; }

  unsigned getInterWarpSizeWithUniqueData();

  unsigned getIntraWarpSizeWithUniqueData();

  bool isReduceWithinCTA();

  bool isAssociative();

  // Callback to allow backends to specify target-specific getter for scratch
  // elements.
  using GetNumScratchElemsFn = std::function<unsigned(
      const triton::LinearLayout &src, const triton::LinearLayout &dst,
      unsigned bitwidth)>;

  unsigned
  getScratchSizeInBytes(GetNumScratchElemsFn numScratchElemsGetter = nullptr);

  InThreadVectorizeOpKind
  getInThreadVectorizeOpKind(unsigned axisPack,
                             bool supportBitwidth16Elementwise,
                             bool supportBitwidth32Elementwise);

  static triton::ColumnAction
  moveAxisBasesToFront(const triton::LinearLayout &layout, int axis,
                       bool isVectorized = false);

  static triton::LinearLayout
  zeroBasesAlongDimAndReorder(const triton::LinearLayout &layout, unsigned axis,
                              mlir::StringAttr dim);

  static triton::LinearLayout getInterLayout(const triton::LinearLayout &layout,
                                             unsigned axis);

  static triton::LinearLayout reducedRegLaneLayout(RankedTensorType srcTy,
                                                   unsigned axis);

  static Value createInThreadVectorizedCombineOp(OpBuilder &builder,
                                                 Location loc,
                                                 InThreadVectorizeOpKind kind,
                                                 Value lhs, Value rhs);

private:
  triton::ReduceOp op;
  RankedTensorType srcTy;
  ArrayRef<int64_t> srcShape;
  Attribute srcEncoding;
  SmallVector<Type> srcElementTypes;
  int axis;
};

class ScanLoweringHelper {
public:
  // Masks are indexed by register, lane, and warp. A stage combines the
  // terminal prefix of the lower half with each prefix in the upper half.
  struct Stage {
    std::array<unsigned, 3> lower;
    std::array<unsigned, 3> current;
  };

  explicit ScanLoweringHelper(triton::ScanOp op);
  bool isSupported();
  const triton::LinearLayout &getLayout() const { return layout; }
  const triton::ColumnAction &getRegisterOrder() const { return registerOrder; }
  unsigned getLocalScanSize() const { return localScanSize; }
  // Length of a contiguous logical segment contained in one warp.
  unsigned getSegmentSize() const { return segmentSize; }
  const std::optional<triton::LinearLayout> &getScratchLayout() const {
    return scratchLayout;
  }
  ArrayRef<Stage> getStages() const { return stages; }
  unsigned getScratchSizeInElems() const;
  unsigned getScratchSizeInBytes();

private:
  triton::ScanOp scanOp;
  triton::LinearLayout layout;
  triton::ColumnAction registerOrder;
  unsigned localScanSize = 1;
  unsigned segmentSize = 1;
  SmallVector<Stage> stages;
  std::optional<triton::LinearLayout> scratchLayout;
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

bool supportWMMA(triton::DotOp op);

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

// TODO: Move utility functions that belong to ConvertLayoutOp to class
// ConvertLayoutOpHelper in the future
bool shouldUseDistSmem(Attribute srcLayout, Attribute dstLayout);

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
