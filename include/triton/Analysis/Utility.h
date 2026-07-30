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
  explicit ScanLoweringHelper(triton::ScanOp op);
  // Return true if the lowering of the scan op is supported.
  bool isSupported();
  // Return the number of elements per thread along axis dim.
  unsigned getAxisNumElementsPerThread();
  // Return the number of elements per thread along non-axis dims.
  unsigned getNonAxisNumElementsPerThread();
  // Return the number of threads per warp along non-axis dims.
  unsigned getNonAxisNumThreadsPerWarp();
  // Return the flat numbers of threads computing independent scan results.
  unsigned getNonAxisNumThreadsPerCTA();
  // Return the number of warps per CTA along axis dim with unique data.
  unsigned getAxisNumWarpsWithUniqueData();
  // Return the number of threads per warp along axis dim with unique data.
  unsigned getAxisNumThreadsPerWarpWithUniqueData();
  // Return the number of blocks along axis dim.
  unsigned getAxisNumBlocks();
  // Return the number of blocks along non axis dim.
  unsigned getNonAxisNumBlocks();
  // Return the size of the scratch space needed for scan lowering.
  unsigned getScratchSizeInBytes();
  // Return the number of elements of the scratch space needed for scan
  // lowering.
  unsigned getScratchSizeInElems();

  // Stride between contiguous element along axis dim.
  unsigned getAxisElementStride();
  // Stride between contiguous threads along axis dim.
  unsigned getAxisThreadStride();
  // Stride between contiguous blocks along axis dim.
  unsigned getAxisBlockStride();

  Location getLoc() { return scanOp.getLoc(); }
  unsigned getAxis() { return scanOp.getAxis(); }
  bool getReverse() { return scanOp.getReverse(); }
  triton::gpu::LinearEncodingAttr getEncoding() { return srcEncoding; }
  llvm::ArrayRef<int64_t> getShape() { return srcShape; }
  unsigned getNumOperands() { return scanOp.getNumOperands(); }
  SmallVector<Type> getElementTypes() { return srcElementTypes; }
  SmallVector<unsigned> getOrder() { return order; }
  Region &getCombineOp();

private:
  triton::ScanOp scanOp;
  triton::gpu::LinearEncodingAttr srcEncoding;
  Attribute legacyEncoding;
  llvm::ArrayRef<int64_t> srcShape;
  SmallVector<Type> srcElementTypes;
  SmallVector<unsigned> order;
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

// This struct represents the factorization of a warp-local layout conversion
// into three components: a register-only permutation, a lane-only permutation,
// and a set of swaps between lane and register basis vectors. Algebraically, it
// represents the factorization P = P_mixed \circ P_lane \circ P_reg. It is used
// to aid in the implementation of the layout conversion using warp-shuffles.
//
// `pReg` and `pLane` are square layouts each with only one input and output
// dimension. `mixedTranspositions` holds pairs of integers (i, j)
// corresponding to the transposition (r_i l_j) of the i-th register basis
// vector with the j-th lane basis vector along with 16-bit selectors for byte
// permute instructions (where each of the four nybbles is in the range [0, 7]).
// `nPack` gives the number of basis vectors that can be used for register
// packing while ensuring packed elements arrive at the same destination lane.
struct DecomposedWarpConversion {
  struct TranspositionInfo {
    std::pair<int, int> transposition;
    uint16_t topPreSel = 0x3210;
    uint16_t botPreSel = 0x7654;
    uint16_t topPostSel = 0x3210;
    uint16_t botPostSel = 0x7654;
  };

  triton::LinearLayout pReg, pLane;
  SmallVector<TranspositionInfo> mixedTranspositions;
  int nPack;
};

// Produces a decomposition of a permutation describing a warp-local layout
// conversion as described in `DecomposedWarpConversion` above.
//
// This function handles cases where the numbers of register and lane basis
// vectors differ between the two layouts. This is done by padding the smaller
// dimension(s) with zero vectors, ensuring that the layout conversion can be
// represented as a permutation. The layouts must not contain broadcasted
// register bases.
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

// Conversion from `srcLayout` to `dstLayout` involving the minimum amount of
// data transfer. The output will be such that layout.getInDimNames() ==
// layout.getOutDimNames() and the conversion will not include kBlock (resp.
// kWarp or kLane) if it can be avoided.
triton::LinearLayout minimalCvtLayout(const triton::LinearLayout &srcLayout,
                                      const triton::LinearLayout &dstLayout);

// Type-based convenience overload for layouts that can be converted to LL.
triton::LinearLayout minimalCvtLayout(Type srcTy, Type dstTy);

// Conversion from `srcTy` to `dstTy` only involves reordering of registers.
// There is no need for data exchange across threads, warps, or blocks.
bool cvtReordersRegisters(RankedTensorType srcTy, RankedTensorType dstTy);

// Conversion from `srcTy` to `dstTy` involves data exchange across threads
// within a warp.  No data exchange across warps or blocks is needed.
bool cvtNeedsWarpShuffle(RankedTensorType srcTy, RankedTensorType dstTy);

// Conversion from `srcTy` to `dstTy` involves data exchange across threads,
// warps, and possibly blocks.
bool cvtNeedsSharedMemory(RankedTensorType srcTy, RankedTensorType dstTy);

// TODO: Move utility functions that belong to ConvertLayoutOp to class
// ConvertLayoutOpHelper in the future
bool shouldUseDistSmem(Attribute srcLayout, Attribute dstLayout);

/// Create a basic DataFlowSolver with constant and dead code analysis included.
std::unique_ptr<DataFlowSolver> createDataFlowSolver();

bool isCvtDimSync(const triton::LinearLayout &srcLayout,
                  const triton::LinearLayout &dstLayout, StringAttr dim);

} // namespace mlir

#endif // TRITON_ANALYSIS_UTILITY_H
