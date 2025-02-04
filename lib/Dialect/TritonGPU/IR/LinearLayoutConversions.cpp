#include <vector>

#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::triton::gpu {
namespace {

// We use the following nomenclature in this file.
//
//  - ctaLayout: A layout for one block, i.e. input dims [register, lane, warp]
//    for register layouts, and input dims [offset] for shared layouts.
//  - cgaLayout: Arrangement of multiple blocks, i.e. input dims [block].
//
// Note that this is inconsistent with the type name CTALayoutAttr.  That type
// is equivalent to our cgaLayout.
//
// IMO the name CTALayoutAttr is wrong.  If we tried to be consistent anyway,
// then we'd have to rename ctaLayout to "warpLayout".  I think that's more
// confusing than being inconsistent about "cgaLayout", especially when we have
// to consider the size of the warpLayout (surely that's not the "warpSize").

#define S(v) StringAttr::get(ctx, (v))

// TODO Have order be a mandatory argument of standardOutDimNames.
SmallVector<StringAttr> permuteDimNames(const SmallVector<StringAttr> &names,
                                        const SmallVector<unsigned> &order) {
  assert(names.size() == order.size());
  SmallVector<StringAttr> ret;
  for (unsigned i : order) {
    ret.push_back(names[i]);
  }
  return ret;
}

// Make a LinearLayout that maps a block-id to an N-dimensional index.
//
// The tensor is split up into CTAsPerCGA pieces, which are distributed among
// the CTAsPerCGA CTAs (i.e. blocks) in the CGA (i.e. groups).
//
// See the nomenclature note at the top of the file for an explanation of why
// this is called makeCgaLayout when it accepts a CTALayoutAttr.
LinearLayout makeCgaLayout(CTALayoutAttr layout) {
  MLIRContext *ctx = layout.getContext();
  StringAttr kBlock = S("block");

  int rank = layout.getCTAOrder().size();
  SmallVector<StringAttr> outDimNames = standardOutDimNames(ctx, rank);

  LinearLayout ret = LinearLayout::empty();
  for (int i = 0; i < rank; i++) {
    // Start with the most minor dimension, which is order[0].
    int dim = layout.getCTAOrder()[i];
    int split = layout.getCTASplitNum()[dim];
    int ctas = layout.getCTAsPerCGA()[dim];
    assert(ctas % split == 0);
    ret *= LinearLayout::identity1D(split, kBlock, outDimNames[dim]) *
           LinearLayout::zeros1D(ctas / split, kBlock, outDimNames[dim]);
  }

  // Transpose to standard order (dim0, dim1, ...).
  return ret.transposeOuts(outDimNames);
}

// Combines the layout of a CTA (input dims [register, lane, warp]) with the
// layout of a CGA (i.e. a block), and ensures that the resulting layout has the
// given shape.
//
// See the nomenclature note at the top of the file for why the variable with
// type CTALayoutAttr is called cgaLayoutAttr.
LinearLayout combineCtaCgaWithShape(LinearLayout ctaLayout,
                                    CTALayoutAttr cgaLayoutAttr,
                                    ArrayRef<int64_t> shape) {
  int rank = shape.size();
  assert(ctaLayout.getNumOutDims() == rank);
  assert(cgaLayoutAttr.getCTAOrder().size() == rank);
  MLIRContext *ctx = cgaLayoutAttr.getContext();

  SmallVector<StringAttr> outDimNames = standardOutDimNames(ctx, rank);

  llvm::SmallDenseMap<StringAttr, int64_t> labeledShape;
  for (auto [dim, size] : llvm::zip(outDimNames, shape)) {
    labeledShape[dim] = size;
  }

  LinearLayout cgaLayout =
      ensureLayoutNotLargerThan(makeCgaLayout(cgaLayoutAttr), labeledShape)
          .transposeOuts(llvm::to_vector(ctaLayout.getOutDimNames()));

  // Calculate the shape of the ctaLayout, which is `shape` divided by the
  // cgaLayout's size.
  llvm::SmallDenseMap<StringAttr, int64_t> ctaShape;
  assert(llvm::to_vector(ctaLayout.getOutDimNames()) ==
         llvm::to_vector(cgaLayout.getOutDimNames()));
  for (auto dim : ctaLayout.getOutDimNames()) {
    ctaShape[dim] =
        std::max(int64_t{1}, labeledShape[dim] / cgaLayout.getOutDimSize(dim));
  }

  ctaLayout = ensureLayoutNotSmallerThan(ctaLayout, ctaShape);
  ctaLayout = ensureLayoutNotLargerThan(ctaLayout, ctaShape);

  LinearLayout ret = (ctaLayout * cgaLayout).transposeOuts(outDimNames);
  for (auto dim : ret.getOutDimNames()) {
    assert(ret.getOutDimSize(dim) == labeledShape[dim]);
  }
  return ret;
}

LinearLayout
sharedToLinearLayoutNoLeadingOffset(ArrayRef<int64_t> shape,
                                    SwizzledSharedEncodingAttr shared) {
  MLIRContext *ctx = shared.getContext();
  int rank = shape.size();
  if (rank == 1) {
    return combineCtaCgaWithShape(
        LinearLayout::identity1D(shape[0], S("offset"), S("dim0")),
        shared.getCTALayout(), shape);
  }

  auto outDimNames = standardOutDimNames(ctx, rank);

  // Construct bases for the 2 most minor dimensions of the layout.  These are
  // the dims that get swizzled.
  assert(shape.size() >= 2);
  int colDim = shared.getOrder()[0];
  int rowDim = shared.getOrder()[1];
  int numCols = shape[colDim];
  int numRows = shape[rowDim];
  StringAttr colDimName = outDimNames[colDim];
  StringAttr rowDimName = outDimNames[rowDim];

  std::vector<std::vector<int>> bases2D;
  for (int logCol = 0; logCol < llvm::Log2_32(numCols); logCol++) {
    bases2D.push_back({0, 1 << logCol});
  }
  for (int logRow = 0; logRow < llvm::Log2_32(numRows); logRow++) {
    int row = 1 << logRow;
    int vec = shared.getVec();
    int perPhase = shared.getPerPhase();
    int maxPhase = shared.getMaxPhase();
    bases2D.push_back({row, (vec * ((row / perPhase) % maxPhase)) % numCols});
  }
  LinearLayout ctaLayout =
      LinearLayout({{S("offset"), bases2D}}, {rowDimName, colDimName});

  // Add the remaining dimensions.
  for (int i = 2; i < rank; i++) {
    int dim = shared.getOrder()[i];
    ctaLayout *=
        LinearLayout::identity1D(shape[dim], S("offset"), outDimNames[dim]);
  }

  return combineCtaCgaWithShape(ctaLayout, shared.getCTALayout(), shape);
}

} // namespace

LinearLayout sharedToLinearLayoutLeadingOffset(ArrayRef<int64_t> shape,
                                               NVMMASharedEncodingAttr shared,
                                               bool disableSwizzle) {
  MLIRContext *ctx = shared.getContext();
  int rank = shape.size();
  if (rank == 1) {
    // TODO: Not sure if this is correct.
    return combineCtaCgaWithShape(
        LinearLayout::identity1D(shape[0], S("offset"), S("dim0")),
        shared.getCTALayout(), shape);
  }
  int elemBitWidth = shared.getElementBitWidth();
  int tileWidthBytes = shared.getSwizzlingByteWidth();
  int vec = 128 / elemBitWidth;
  int perPhase = 0;
  int maxPhase = 0;
  if (tileWidthBytes == 32) {
    perPhase = 4;
    maxPhase = 2;
  } else if (tileWidthBytes == 64) {
    perPhase = 2;
    maxPhase = 4;
  } else if (tileWidthBytes == 128) {
    perPhase = 1;
    maxPhase = 8;
  }
  auto outDimNames = standardOutDimNames(ctx, rank);

  // Construct bases for a the layout's 2-dimensional tile.
  assert(shape.size() >= 2);
  int colDim = shared.getTransposed() ? 0 : 1;
  int rowDim = shared.getTransposed() ? 1 : 0;

  int tileRows = 8;
  int tileCols = 8 * tileWidthBytes / elemBitWidth;
  bool isFp4Padded = false;
  if (auto sharedMMALayout =
          dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(shared)) {
    if (sharedMMALayout.getFp4Padded()) {
      isFp4Padded = true;
    }
  }
  int packingFactor = isFp4Padded ? 2 : 1;

  if (shape[colDim] * packingFactor < tileCols || shape[rowDim] < tileRows) {
    llvm::errs() << "Illegal shared layout; expected shape to be at least ["
                 << tileRows << ", " << tileCols << "], shape: ["
                 << shape[rowDim] << ", " << shape[colDim] << "]\n";
    llvm::report_fatal_error("Illegal shared layout");
  }

  StringAttr colDimName = outDimNames[colDim];
  StringAttr rowDimName = outDimNames[rowDim];

  std::vector<std::vector<int>> bases2D;
  for (int logCol = 0; logCol < llvm::Log2_32(tileCols); logCol++) {
    if (isFp4Padded) {
      int colPadded = 1 << logCol;
      // Each group of 16 offsets consists of 8 "real" and 8 "padded" offsets.
      // We represent the padded layout by mapping 8 padded offsets to the same
      // coordinates as the real ones. When computing the inverse of this LL,
      // the offsets correspoding to the real ones are picked in the image by
      // invertAndCompose.
      int colPacked = colPadded / 16 * 8 + colPadded % 8;
      bases2D.push_back({0, colPacked});
    } else {
      bases2D.push_back({0, 1 << logCol});
    }
  }
  for (int logRow = 0; logRow < llvm::Log2_32(tileRows); logRow++) {
    int row = 1 << logRow;
    if (disableSwizzle) {
      bases2D.push_back({row, 0});
      continue;
    }
    if (isFp4Padded) {
      int colPadded = vec * ((row / perPhase) % maxPhase);
      int colPacked = colPadded / 16 * 8 + colPadded % 8;
      bases2D.push_back({row, colPacked});
    } else {
      bases2D.push_back({row, vec * ((row / perPhase) % maxPhase)});
    }
  }
  LinearLayout tileLayout =
      LinearLayout({{S("offset"), bases2D}}, {rowDimName, colDimName});

  // Add the remaining dimensions.
  for (int i = 2; i < rank; i++) {
    int dim = shared.getTransposed() ? i : 1 - i;
    tileLayout *=
        LinearLayout::identity1D(shape[dim], S("offset"), outDimNames[dim]);
  }

  return combineCtaCgaWithShape(tileLayout, shared.getCTALayout(), shape);
}

/// Function to generate lane and warp layout for dot operands.
static LinearLayout broadcastedDotOperandLayout(MLIRContext *ctx,
                                                ArrayRef<unsigned> shape,
                                                ArrayRef<unsigned> order,
                                                unsigned kDim,
                                                StringAttr inDimName) {
  // Let warpsPerCTAMma = {2, 2}, then
  // warpsPerCTA = {2, 1} for opA and warpsPerCTA = {1, 2} for opB
  // assume warpOrder = {1, 0}
  // Assume that C is tiled by 2x2 tiles. Since warpOrder={1, 0}, we have that
  // the C is owned as per the following layout:
  // C: 0 | 1
  //    - | -
  //    2 | 3
  // In order to be able to compute C, we need the following warp tiling of
  // A and B:
  // A: 0 1 | 0 1    B: 0 2 | 1 3
  //    - - | - -       - - | - -
  //    2 3 | 2 3       0 2 | 1 3
  // In other words, we need to broadcast along K
  auto rank = shape.size();
  auto dimNames = standardOutDimNames(ctx, rank);
  LinearLayout layout = LinearLayout::empty();

  // We have to broadcast along the inner dimension
  // For A, when moving along M we go from 0 to 2.
  // For B, when moving along N we go from 0 to 1.
  // As such, choosing the order of A {1, 0}, gives us the correct broadcasting
  // Same happens if the warpOrder is {0, 1}, like in Hopper
  for (auto d : order) {
    if (d == kDim) {
      layout *= LinearLayout::zeros1D(shape[d], inDimName, dimNames[d]);
    } else {
      layout *= LinearLayout::identity1D(shape[d], inDimName, dimNames[d]);
    }
  }
  return layout;
}

LinearLayout
AMDMfmaEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  int rank = shape.size();
  assert(rank == getWarpsPerCTA().size());

  bool hasBatchDim = rank == 3;
  int mIndex = 0 + hasBatchDim;
  int nIndex = 1 + hasBatchDim;
  (void)mIndex, (void)nIndex;

  assert(((getMDim() == 32 && getNDim() == 32) ||
          (getMDim() == 16 && getNDim() == 16)) &&
         "Unsupported mfma type");

  MLIRContext *ctx = getContext();
  SmallVector<StringAttr> outDimNames = standardOutDimNames(ctx, rank);

  StringAttr kRegister = S("register");
  StringAttr kLane = S("lane");

  // https://github.com/ROCm/amd_matrix_instruction_calculator can print the
  // register and lane layout for mfma instructions.

  // We use the order from fastest varying to slowest varying. So each base
  // vector is a tuple of values mapping to matrix C's (N, M[, B]) indices.
  SmallVector<unsigned> order = triton::gpu::getOrder(*this);
  auto tileLayout = LinearLayout::empty();

  if (getMDim() == 32) {
    // For mfma with 32x32 output, each of the 64 threads holds 16 elements.
    //
    // For the register (i.e., element) dimension, these 16 elements are along
    // the matrix C's M dimension, with 4 consecutive elements spanning 4 rows
    // and then the next 4 rows being a gap.
    //
    // For the lane (i.e., thread) dimension, these threads are along the
    // matrix C's N dimension, with 32 consecutive threads covering a whole
    // row and the next 32 threads start after a gap spanning 4 rows.
    tileLayout = LinearLayout(
        {{kRegister, {{0, 1}, {0, 2}, {0, 8}, /*gap*/ {0, 16}}},
         {kLane, {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, /*gap*/ {0, 4}}}},
        {outDimNames[order[0]], outDimNames[order[1]]});
    // For mfma.transposed layout, the element ownership among threads are
    // "transposed" within each warp.
    if (getIsTransposed())
      tileLayout = LinearLayout(
          {{kRegister, {{1, 0}, {2, 0}, {8, 0}, /*gap*/ {16, 0}}},
           {kLane, {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, /*gap*/ {4, 0}}}},
          {outDimNames[order[0]], outDimNames[order[1]]});
  } else {
    assert(getMDim() == 16);
    // For mfma with 16x16 output, each of the 64 threads holds 4 elements.
    //
    // For the register (i.e., element) dimension, these 4 elements are along
    // the matrix C's M dimension, with 4 consecutive elements spanning 4 rows.
    //
    // For the lane (i.e., thread) dimension, these threads are along the
    // matrix C's N dimension, with 16 consecutive threads covering a whole
    // row and the next 16 threads start after a gap spanning 4 rows.
    tileLayout = LinearLayout(
        {{kRegister, {{0, 1}, {0, 2}}},
         {kLane, {{1, 0}, {2, 0}, {4, 0}, {8, 0}, /*gap*/ {0, 4}, {0, 8}}}},
        {outDimNames[order[0]], outDimNames[order[1]]});
    // For mfma.transposed layout, the element ownership among threads are
    // "transposed" within each warp.
    if (getIsTransposed())
      tileLayout = LinearLayout(
          {{kRegister, {{1, 0}, {2, 0}}},
           {kLane, {{0, 1}, {0, 2}, {0, 4}, {0, 8}, /*gap*/ {4, 0}, {8, 0}}}},
          {outDimNames[order[0]], outDimNames[order[1]]});
  }
  if (hasBatchDim) {
    assert(order[2] == 0);
    // Extend the base vector with one value to accommodate for the batch
    // dimension, which appears at the last.
    tileLayout *= LinearLayout::identity1D(1, kRegister, outDimNames[order[2]]);
    tileLayout *= LinearLayout::identity1D(1, kLane, outDimNames[order[2]]);
  }

  // And each warp takes the same register and lane sub-layout. So multiply with
  // an identity layout for the warp.
  LinearLayout warpLayout =
      identityStandardND(S("warp"), getWarpsPerCTA(), order);
  LinearLayout ctaLayout = tileLayout * warpLayout;

  return combineCtaCgaWithShape(ctaLayout, getCTALayout(), shape);
}

LinearLayout mfmaDotToLinearLayout(DotOperandEncodingAttr dotMfmaLayout,
                                   ArrayRef<int64_t> shape) {

  // Current linear layout conversion for dot operand is only necessary to
  // enable LDS bypass for operand B in the MFMA dot path. To achieve
  // performance gains from bypassing LDS, the following conditions must be met:
  //
  // 1) opIdx == 1: Currently, only the B tensor (e.g. weights in moe-like
  //    kernels) bypasses LDS. This constraint is not strict and support for
  //    bypassing operand A (e.g. Q tensor in flash attention) will be added in
  //    the future.
  //
  // 2) B tensor must be column major: This is required to support vectorized
  //    global load instructions, as MFMA instructions expect threads to hold B
  //    operand elements along the K dimension.
  //
  // 3) kWidth == 8: Ensures maximum global load vectorization for fp16
  //    operations.
  //    TODO: Generalize conversion to handle maximum kWidth for other types
  //    (i.e. fp8).
  //
  // 4) warpsPerCTA[mDim] == 1: This guarantees that every B tensor element is
  //    held by exactly one thread, maintaining the same number of global loads
  //    as in a blocked layout.
  //
  // Other use of Linear layout is a support of rare corner cases,
  // for example one instruction tile is larger than tensor
  auto mfmaLayout = llvm::cast<AMDMfmaEncodingAttr>(dotMfmaLayout.getParent());

  auto rank = shape.size();
  bool hasBatchDim = rank == 3;
  int mIndex = 0 + hasBatchDim;

  int32_t kWidth = dotMfmaLayout.getKWidth();
  auto kDim = dotMfmaLayout.getOpIdx() == 0 ? rank - 1 : rank - 2;
  int32_t kSize = shape[kDim];
  auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();

  MLIRContext *ctx = dotMfmaLayout.getContext();
  SmallVector<StringAttr> outDimNames = standardOutDimNames(ctx, rank);

  StringAttr kRegister = S("register");
  StringAttr kLane = S("lane");
  StringAttr kWarp = S("warp");

  // register order
  // operand A: [1, 0] / [2, 1, 0]
  // operand B: [0, 1] / [1, 2, 0]
  // for both cases it is [k, nonk]/[k, nonk, batch]
  SmallVector<unsigned> order = triton::gpu::getOrder(dotMfmaLayout);
  // warp order
  // common for both operand A and B: [0, 1] / [0, 1, 2]
  // in both cases it is [M dim, N dim]/[batch, M dim, N dim]
  SmallVector<unsigned> warpOrder = triton::gpu::getWarpOrder(dotMfmaLayout);

  // Lane holds kWidth consecutive elements along k dimension, so
  // base register vectors for one tile are initialized in following way:
  // {1, 0}, {2, 0} ... {kWidth/2, 0}
  std::vector<std::vector<int32_t>> registerBase;
  for (int32_t elem = 1; elem < kWidth; elem *= 2)
    registerBase.emplace_back(std::vector<int32_t>{elem, 0});

  std::vector<std::vector<int32_t>> laneBase;
  int32_t kTileSize = -1;

  if (mfmaLayout.getMDim() == 32) {
    // Canonical MFMA linear layout handles 4 consecutive elements along
    // the register dimension. Dot operand handles variable kWidth consecutive
    // elements. For lane dim, since the MFMA thread arrangement is {K, N} = {2,
    // 32}, this means that mapping of first 5 base (up to thread 16) vectors
    // will be an identity along N dim. Thread 32 will be mapped to element
    // kWidth in K dimension.
    laneBase = {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {kWidth, 0}};
    kTileSize = kWidth * 2;
  } else {
    assert(mfmaLayout.getMDim() == 16);
    // For lane dim, since the MFMA thread arrangement is {K, N} = {4, 16}, this
    // means that mapping of first 4 base (up to thread 16) vectors will be an
    // identity along N dim. Thread 16 will be mapped to element kWisth in K
    // dimension. Thread 32 is mapped to element 2*kWidth in K dim.
    laneBase = {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {kWidth, 0}, {kWidth * 2, 0}};
    kTileSize = kWidth * 4;
  }
  assert(kTileSize != -1);
  // Add repeats of registers along K dimension to register base vectors
  for (int32_t elem = kTileSize; elem < kSize; elem *= 2)
    registerBase.emplace_back(std::vector<int32_t>{elem, 0});

  // Base vectors above are defined in a fixed order [non-k-dim, k-dim].
  // To assign them to actual matrix dimensions `order` array is used.
  // For operand A: non-k-dim -> dim0, k-dim -> dim1
  // For operand B: non-k-dim -> dim1, k-dim -> dim0
  LinearLayout tileLayout({{kRegister, registerBase}, {kLane, laneBase}},
                          {outDimNames[order[0]], outDimNames[order[1]]});

  if (hasBatchDim) {
    assert(order[2] == 0);
    // Extend the base vector with one value to accommodate for the batch
    // dimension, which appears at the last.
    tileLayout *= LinearLayout::identity1D(1, kRegister, outDimNames[order[2]]);
    tileLayout *= LinearLayout::identity1D(1, kLane, outDimNames[order[2]]);
  }

  LinearLayout warpLayout = identityStandardND(kWarp, warpsPerCTA, warpOrder);

  LinearLayout ctaLayout = tileLayout.transposeOuts(outDimNames) *
                           warpLayout.transposeOuts(outDimNames);

  return combineCtaCgaWithShape(ctaLayout, mfmaLayout.getCTALayout(), shape);
}

LinearLayout
AMDWmmaEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  int rank = shape.size();
  assert(rank == getWarpsPerCTA().size());

  bool hasBatchDim = rank == 3;
  int mIndex = 0 + hasBatchDim;
  int nIndex = 1 + hasBatchDim;
  (void)mIndex, (void)nIndex;

  SmallVector<unsigned> mnkDim = getMNKDimPerInstr();
  unsigned mDim = mnkDim[0], nDim = mnkDim[1];
  (void)mDim, (void)nDim;

  assert(((shape[mIndex] == 1 || shape[mIndex] >= mDim) &&
          (shape[nIndex] == 1 || shape[nIndex] >= nDim)) &&
         "Unsupported tensor shape for given wmma layout");

  MLIRContext *ctx = getContext();
  SmallVector<StringAttr> outDimNames = standardOutDimNames(ctx, rank);

  StringAttr kRegister = S("register");
  StringAttr kLane = S("lane");

  // https://github.com/ROCm/amd_matrix_instruction_calculator can print the
  // register and lane layout for mfma instructions.

  // We use the order from fastest varying to slowest varying. So each base
  // vector is a tuple of values mapping to matrix C's (N, M[, B]) indices.
  SmallVector<unsigned> threadOrder = getThreadOrder();
  assert(threadOrder[0] == mIndex || threadOrder[0] == nIndex);
  assert(threadOrder[1] == mIndex || threadOrder[1] == nIndex);

  // For wmma with 16x16 output, each of the 32 threads holds 8 elements.
  //
  // The first version of WMMA layout has following specific:
  // for the register (i.e., element) dimension, these 8 elements are
  // along the matrix C's M dimension, with 1 consecutive elements
  // spanning 1 row and then the next 1 row being a gap.
  //
  // For the lane (i.e., thread) dimension, these threads are along the
  // matrix C's N dimension, with 16 consecutive threads covering a whole
  // row and the next 16 threads start at the next row.
  //
  // The second version of wmma layout is less tricky:
  // for the register dimension 8 elements are along the matrix C's M
  // dimension. First 16 lanes take 0-8 elems along M, second 16 take 8-15.
  // We have 16 pair of threads in each warp, one pair covers the whole
  // column.
  //
  // Please also check explaining comments in TritonGPUAttrDefs.td at the
  // AMDWmmaEncodingAttr section.
  unsigned ver = getVersion();
  assert(ver == 1 || ver == 2);
  LinearLayout tileLayout =
      ver == 1
          ? LinearLayout(
                {{kRegister, {/*gap*/ {0, 2}, {0, 4}, {0, 8}}},
                 {kLane, {{1, 0}, {2, 0}, {4, 0}, {8, 0}, /*gap*/ {0, 1}}}},
                {outDimNames[threadOrder[0]], outDimNames[threadOrder[1]]})
          : LinearLayout(
                {{kRegister, {{0, 1}, {0, 2}, {0, 4}}},
                 {kLane, {{1, 0}, {2, 0}, {4, 0}, {8, 0}, /*gap*/ {0, 8}}}},
                {outDimNames[threadOrder[0]], outDimNames[threadOrder[1]]});

  if (hasBatchDim) {
    int batchIndex = 0;
    // Extend the base vector with one value to accommodate for the batch
    // dimension, which appears at the last.
    tileLayout *=
        LinearLayout::identity1D(1, kRegister, outDimNames[batchIndex]);
    tileLayout *= LinearLayout::identity1D(1, kLane, outDimNames[batchIndex]);
  }

  // And each warp takes the same register and lane sub-layout. So multiply with
  // an identity layout for the warp.
  auto warpOrder = getWarpOrder();
  LinearLayout warpLayout =
      identityStandardND(S("warp"), getWarpsPerCTA(), warpOrder);
  // reorder dim names in rep order, so combineCtaCgaWithShape generate proper
  // extension of layout
  auto repOrder = getRepOrder();
  SmallVector<StringAttr> repDimNames;
  for (auto dim : repOrder)
    repDimNames.push_back(outDimNames[dim]);
  LinearLayout ctaLayout = tileLayout.transposeOuts(repDimNames) *
                           warpLayout.transposeOuts(repDimNames);

  return combineCtaCgaWithShape(ctaLayout, getCTALayout(), shape);
}

LinearLayout wmmaDotOperandToLinearLayout(DotOperandEncodingAttr dotWmmaLayout,
                                          ArrayRef<int64_t> shape) {
  auto wmmaLayout = llvm::cast<AMDWmmaEncodingAttr>(dotWmmaLayout.getParent());
  auto rank = shape.size();
  bool hasBatchDim = rank == 3;
  auto kDim = dotWmmaLayout.getOpIdx() == 0 ? rank - 1 : rank - 2;
  int32_t kSize = shape[kDim];
  MLIRContext *ctx = dotWmmaLayout.getContext();
  SmallVector<StringAttr> outDimNames = standardOutDimNames(ctx, rank);
  StringAttr kRegister = S("register");
  StringAttr kLane = S("lane");
  StringAttr kWarp = S("warp");
  // lane order
  // operand A: [1, 0] / [2, 1, 0]
  // operand B: [0, 1] / [1, 2, 0]
  // for both cases it is [k, nonk]/[k, nonk, batch]
  SmallVector<unsigned> laneOrder = triton::gpu::getOrder(dotWmmaLayout);
  // generate continuous part of register bases(i.e. kWidth)
  std::vector<std::vector<int32_t>> registerBase;
  const int32_t kWidth = dotWmmaLayout.getKWidth();
  for (int i = 1; i < kWidth; i *= 2)
    registerBase.push_back(std::vector<int32_t>{i, 0});
  std::vector<std::vector<int32_t>> laneBase = {{0, 1}, {0, 2}, {0, 4}, {0, 8}};
  switch (wmmaLayout.getVersion()) {
  case 1:
    // WMMA version 1 duplicates values in lanes 0-15 and 16-31
    laneBase.push_back({0, 0});
    break;
  case 2:
    // WMMA version 2 offset values in lanes 0-15 and 16-31 across k dimensions
    laneBase.push_back({kWidth, 0});
    break;
  default:
    assert(false && "unexpected version");
  }
  // Generate layout for one wmma instruction
  LinearLayout tileLayout(
      {{kRegister, registerBase}, {kLane, laneBase}},
      {outDimNames[laneOrder[0]], outDimNames[laneOrder[1]]});
  if (hasBatchDim) {
    assert(laneOrder[2] == 0);
    // Extend the base vector with one value to accommodate for the batch
    // dimension, which appears at the last.
    tileLayout *=
        LinearLayout::identity1D(1, kRegister, outDimNames[laneOrder[2]]);
    tileLayout *= LinearLayout::identity1D(1, kLane, outDimNames[laneOrder[2]]);
  }

  // Generate warp layout
  auto warpsPerCTA = wmmaLayout.getWarpsPerCTA();
  auto warpOrder = triton::gpu::getWarpOrder(dotWmmaLayout);
  LinearLayout warpLayout =
      broadcastedDotOperandLayout(ctx, warpsPerCTA, warpOrder, kDim, S("warp"));

  // reorder dim names in rep order, so combineCtaCgaWithShape generate proper
  // extension of layout
  auto repOrder = wmmaLayout.getRepOrderForOperand(dotWmmaLayout.getOpIdx());
  SmallVector<StringAttr> repDimNames;
  for (auto dim : repOrder)
    repDimNames.push_back(outDimNames[dim]);

  // join instruction layout and warps using repetition order of dimensions
  LinearLayout ctaLayout = tileLayout.transposeOuts(repDimNames) *
                           warpLayout.transposeOuts(repDimNames);

  return combineCtaCgaWithShape(ctaLayout, wmmaLayout.getCTALayout(), shape);
}

LinearLayout
BlockedEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  assert(shape.size() == getOrder().size());
  MLIRContext *ctx = getContext();

  const auto &order = getOrder();
  LinearLayout ctaLayout =
      identityStandardND(S("register"), getSizePerThread(), order) *
      identityStandardND(S("lane"), getThreadsPerWarp(), order) *
      identityStandardND(S("warp"), getWarpsPerCTA(), order);

  return combineCtaCgaWithShape(ctaLayout, getCTALayout(), shape);
}

LinearLayout fmaDotToLinearLayout(DotOperandEncodingAttr operandLayout,
                                  ArrayRef<int64_t> shape) {
  int rank = shape.size();
  auto blocked = cast<BlockedEncodingAttr>(operandLayout.getParent());
  MLIRContext *ctx = operandLayout.getContext();

  // TODO: introduce registerOrder or use getOrder(operandLayout)
  // Currently this order is used in legacy converter, because we do not
  // have access to full dot operand layout, only parent part.
  auto regOrder = blocked.getOrder();
  // TODO: use operandLayout.getThreadOrder()
  auto threadOrder = blocked.getThreadOrder();
  auto warpOrder = blocked.getWarpOrder();
  auto repOrder = blocked.getRepOrder();

  StringAttr kReg = S("register");
  StringAttr kLane = S("lane");
  StringAttr kWarp = S("warp");

  SmallVector<unsigned> threadSize = blocked.getSizePerThread();
  auto kDimIdx = operandLayout.getOpIdx() == 0 ? rank - 1 : rank - 2;
  threadSize[kDimIdx] = shape[kDimIdx];
  auto threadShape = blocked.getThreadsPerWarp();
  auto warpShape = blocked.getWarpsPerCTA();

  SmallVector<StringAttr> repDimNames =
      permuteDimNames(standardOutDimNames(ctx, rank), repOrder);

  auto registersLayout = identityStandardND(kReg, threadSize, regOrder);
  auto lanesLayout = broadcastedDotOperandLayout(ctx, threadShape, threadOrder,
                                                 kDimIdx, kLane);
  auto warpsLayout =
      broadcastedDotOperandLayout(ctx, warpShape, warpOrder, kDimIdx, kWarp);

  LinearLayout ctaLayout = registersLayout.transposeOuts(repDimNames) *
                           lanesLayout.transposeOuts(repDimNames) *
                           warpsLayout.transposeOuts(repDimNames);

  return combineCtaCgaWithShape(ctaLayout, getCTALayout(operandLayout), shape);
}

LinearLayout nvidiaMmaTile(MLIRContext *ctx, ArrayRef<unsigned> tileShape,
                           unsigned kWidth, ArrayRef<unsigned> order,
                           ArrayRef<unsigned> repOrder) {
  // Trivial layout mapping 0 -> (0, 0), but we set the order to repOrder
  // Like LinearLayout::empty() but with a rank and an order
  int rank = repOrder.size();
  auto dimNames = standardOutDimNames(ctx, rank);
  auto trivialShape = SmallVector<unsigned>(rank, 1);
  LinearLayout ctaLayout =
      identityStandardND(S("register"), trivialShape, repOrder);

  assert(rank >= 2);
  auto inner = order[0];
  auto outer = order[1];

  assert(tileShape.size() == rank);
  int m = tileShape[outer];
  int n = tileShape[inner];

  // The relative order of registers and lanes is given by:
  // - Inner dim: kWidth registers
  // - Inner dim: 4 lanes
  // - Outer dim: 8 lanes
  // - Outer dim: repeat m / 8 times
  // - Inner dim: repeat n / (kWidth * 4) times
  assert(m % 8 == 0);
  assert(n % (kWidth * 4) == 0);
  // There is at least one subtile on the inner-most dimension
  // FIXME. We should implement operator* in terms of operator*=
  // and chain *= instead of using *
  auto outDimNames = llvm::to_vector(ctaLayout.getOutDimNames());
  ctaLayout = ctaLayout *
              LinearLayout::identity1D(kWidth, S("register"), dimNames[inner]) *
              LinearLayout::identity1D(4, S("lane"), dimNames[inner]) *
              LinearLayout::identity1D(8, S("lane"), dimNames[outer]) *
              LinearLayout::identity1D(m / 8, S("register"), dimNames[outer]) *
              LinearLayout::identity1D(n / (kWidth * 4), S("register"),
                                       dimNames[inner]);
  return ctaLayout;
}

LinearLayout
NvidiaMmaEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  auto ctx = getContext();
  int rank = shape.size();

  SmallVector<unsigned> tileShape;
  if (isAmpere()) {
    // Ampere.getInstrShape() returns the tile shape
    tileShape = SmallVector<unsigned>(getInstrShape());
  } else {
    assert(isHopper());
    auto instrShapeMNK = getInstrShape();
    tileShape = SmallVector<unsigned>({instrShapeMNK[0], instrShapeMNK[1]});
  }
  // nvidiamma layout always assumes kWidth = 2
  constexpr auto kWidth = 2;
  auto ctaLayout =
      nvidiaMmaTile(ctx, tileShape, kWidth, getOrder(*this), getRepOrder());

  // The triton orders are defined on [dim0, dim1, ...], so we need to pass
  // those dims Then, for some reason, operator* requires the orders to match
  // so we need to reorder the outs to match
  ctaLayout *= identityStandardND(S("warp"), getWarpsPerCTA(), getWarpOrder())
                   .transposeOuts(llvm::to_vector(ctaLayout.getOutDimNames()));

  return combineCtaCgaWithShape(ctaLayout, getCTALayout(), shape);
}

LinearLayout nvidiaDotToLinearLayout(ArrayRef<int64_t> shape,
                                     DotOperandEncodingAttr dot) {
  int rank = shape.size();
  auto mma = cast<NvidiaMmaEncodingAttr>(dot.getParent());
  int kWidth = dot.getKWidth();
  bool isA = dot.getOpIdx() == 0;
  MLIRContext *ctx = mma.getContext();

  SmallVector<unsigned> tileShape(rank, 1);
  if (isA) {
    tileShape[rank - 2] = 16;
    tileShape[rank - 1] = kWidth * 8;
  } else {
    // Hopper takes the rhs via shared memory
    assert(mma.isAmpere());
    tileShape[rank - 2] = kWidth * 8;
    tileShape[rank - 1] = 8;
  }
  auto ctaLayout =
      nvidiaMmaTile(ctx, tileShape, kWidth, getOrder(dot), dot.getRepOrder());
  auto kDim = isA ? rank - 1 : rank - 2;
  ctaLayout *= broadcastedDotOperandLayout(ctx, mma.getWarpsPerCTA(),
                                           mma.getWarpOrder(), kDim, S("warp"))
                   .transposeOuts(llvm::to_vector(ctaLayout.getOutDimNames()));

  return combineCtaCgaWithShape(ctaLayout, getCTALayout(dot), shape);
}

LinearLayout
DotOperandEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  auto parent = getParent();
  if (auto blockedLayout = mlir::dyn_cast<BlockedEncodingAttr>(parent)) {
    return fmaDotToLinearLayout(*this, shape);
  } else if (auto mfmaLayout = mlir::dyn_cast<AMDMfmaEncodingAttr>(parent)) {
    return mfmaDotToLinearLayout(*this, shape);
  } else if (auto wmmaLayout = mlir::dyn_cast<AMDWmmaEncodingAttr>(parent)) {
    return wmmaDotOperandToLinearLayout(*this, shape);
  } else {
    auto mma = mlir::cast<NvidiaMmaEncodingAttr>(parent);
    return nvidiaDotToLinearLayout(shape, *this);
  }
}

LinearLayout SliceEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  MLIRContext *ctx = getContext();

  // First compute the linear layout for this layout's parent.
  SmallVector<int64_t> parentShape(shape);
  parentShape.insert(parentShape.begin() + getDim(), 1);
  LinearLayout parentLL = triton::gpu::toLinearLayout(parentShape, getParent());

  // Remove dimension getDim() from the parent layout.
  //
  //  1. Construct a layout `transform` from parent-out-dims to slice-out-dims
  //     that removes the relevant out-dim.
  //  2. Compute linearSlice = parent.compose(transform).  Now linearSlice maps
  //     from parent in-dims to slice out-dims.
  //  3. Fix up duplicate registers introduced by slicing.
  auto outDimNames = standardOutDimNames(ctx, shape.size() + 1);
  LinearLayout transform = LinearLayout::empty();
  for (auto [idx, outDim] : llvm::enumerate(parentLL.getOutDimNames())) {
    if (idx == getDim()) {
      // Because we're multiplying by all zeros, we could replace outDimNames[0]
      // with any other valid out-dim; the layout will be the same.
      transform *= LinearLayout::zeros1D(parentLL.getOutDimSize(outDim), outDim,
                                         outDimNames[0]);
    } else {
      transform *=
          LinearLayout::identity1D(parentLL.getOutDimSize(outDim), outDim,
                                   outDimNames[idx - (idx < getDim() ? 0 : 1)]);
    }
  }
  LinearLayout sliceLL = parentLL.compose(transform);

  // Step 3: Along the "register" dim, remove any all-zero bases.
  auto bases = sliceLL.getBases();
  std::vector<std::vector<int>> newRegBases;
  for (const auto &basis : bases[S("register")]) {
    if (llvm::any_of(basis, [](int b) { return b != 0; })) {
      newRegBases.push_back(basis);
    }
  }
  bases[S("register")] = newRegBases;

  LinearLayout ret =
      LinearLayout(std::move(bases), llvm::to_vector(sliceLL.getOutDimNames()));

  // Match a hack in the legacy code that ensures that the number of registers
  // matches getTotalElemsPerThread.  Yup: We just removed all the zeros, now
  // we're (maybe) adding some back.  :)
  //
  // TODO(jlebar): Once getTotalElemsPerThread uses LLs instead of the existing
  // legacy code, I think we can remove this.
  int expectedNumRegisters =
      triton::gpu::getTotalElemsPerThread(RankedTensorType::get(
          shape, IntegerType::get(ctx, 32) /*dummy type*/, *this));
  if (ret.getInDimSize(S("register")) != expectedNumRegisters) {
    int extraZeros = expectedNumRegisters / ret.getInDimSize(S("register"));
    // Our use of "dim0" here is arbitrary; because we're adding zeros, any
    // output dimension would work.
    ret *= LinearLayout::zeros1D(extraZeros, S("register"), S("dim0"));
  }
  return ret;
}

LinearLayout TritonGPUDialect::toLinearLayout(ArrayRef<int64_t> shape,
                                              Attribute layout) {
  CacheKey key{std::vector<int64_t>(shape.begin(), shape.end()), layout};
  if (auto result = llCache.get(key)) {
    return *result;
  }

  // Layouts are distributed or shared in triton core
  // To add a new layout add an else-if clause
  LinearLayout result = LinearLayout::empty();
  if (auto distributed = dyn_cast<DistributedEncodingTrait>(layout)) {
    result = distributed.toLinearLayout(shape);
  } else {
    if (auto shared = dyn_cast<SwizzledSharedEncodingAttr>(layout)) {
      result = sharedToLinearLayoutNoLeadingOffset(shape, shared);
    } else if (auto shared = dyn_cast<NVMMASharedEncodingAttr>(layout)) {
      result = sharedToLinearLayoutLeadingOffset(shape, shared);
    } else {
      assert(0 && "unknown layout");
    }
  }

  llCache.set(std::move(key), result);
  return result;
}

LinearLayout toLinearLayout(ArrayRef<int64_t> shape, Attribute layout) {
  auto *ctx = layout.getContext();
  return ctx->getLoadedDialect<TritonGPUDialect>()->toLinearLayout(shape,
                                                                   layout);
}

LinearLayout getLayoutWithinBlock(const LinearLayout &layout) {
  assert(!layout.getInDimNames().empty());
  MLIRContext *ctx = layout.getInDimNames().begin()->getContext();

  StringAttr kBlock = S("block");
  assert(layout.hasInDim(kBlock));
  auto bases = layout.getBases();
  bases[kBlock] = {};
  return LinearLayout(bases, llvm::to_vector<4>(layout.getOutDimNames()));
}

LinearLayout chooseShemLayoutForRegToRegConversion(
    MLIRContext *ctx, ArrayRef<unsigned> tensorShape,
    ArrayRef<unsigned> repShape, ArrayRef<unsigned> order) {
  auto outDimNames = standardOutDimNames(ctx, tensorShape.size());
  LinearLayout layout = LinearLayout::empty();
  SmallVector<StringAttr> kRepDims;
  SmallVector<StringAttr> kOffsetDims;
  auto totalIters = 1;
  auto totalOffsets = 1;
  for (int i = 0; i < tensorShape.size(); i++) {
    int dim = order[i];
    StringAttr kIteration = S("iteration" + std::to_string(dim));
    StringAttr kOffset = S("offset" + std::to_string(dim));
    kRepDims.push_back(kIteration);
    kOffsetDims.push_back(kOffset);
    assert(llvm::isPowerOf2_32(repShape[dim]));
    assert(llvm::isPowerOf2_32(tensorShape[dim]));
    auto numIters = tensorShape[dim] / repShape[dim];
    layout *=
        LinearLayout::identity1D(repShape[dim], kOffset, outDimNames[dim]);
    layout *= LinearLayout::identity1D(numIters, kIteration, outDimNames[dim]);
    totalIters *= numIters;
    totalOffsets *= repShape[dim];
  }
  StringAttr kOffset = S("offset");
  StringAttr kIteration = S("iteration");
  StringAttr kBlock = S("block");
  SmallVector<StringAttr> newDims;
  newDims.append(kOffsetDims.begin(), kOffsetDims.end());
  newDims.append(kRepDims.begin(), kRepDims.end());
  // Transpose layout from [offset0, rep0, offset1, rep1, ...] to
  // [offset0, offset1, ..., rep0, rep1, ...]
  auto ret = layout.transposeIns(newDims);
  // Reshape layout from [offset0, offset1, ..., rep0, rep1, ...] to
  // [offset, rep, block]
  return ret.reshapeIns(
      {{kOffset, totalOffsets}, {kIteration, totalIters}, {kBlock, 1}});
}

namespace {
LinearLayout chooseStMatrixLayoutLeadingOffset(MLIRContext *ctx,
                                               RankedTensorType tensorTy,
                                               int swizzleByteSize) {
  int perPhase;
  int maxPhase;
  if (swizzleByteSize == 32) {
    perPhase = 4;
    maxPhase = 2;
  } else if (swizzleByteSize == 64) {
    perPhase = 2;
    maxPhase = 4;
  } else if (swizzleByteSize == 128) {
    perPhase = 1;
    maxPhase = 8;
  } else {
    llvm::errs() << "Illegal swizzleByteSize: " << swizzleByteSize << "\n";
    llvm::report_fatal_error("Illegal swizzleByteSize");
  }

  // stmatrix only supports 16-bit elements, and each vector has 8 elements
  int elemBitWidth = 16;
  int vecSize = 8;
  int numRowsPerTile = 16;
  int numColsPerChunk = 8 * swizzleByteSize / elemBitWidth;

  // Construct a single stmatrix.x4 (16x16) tile
  std::vector<std::vector<int>> basesReg = {{1, 0}, {2, 0}, {4, 0}};
  std::vector<std::vector<int>> basesLane;
  for (int logRow = 0; logRow < llvm::Log2_32(numRowsPerTile); logRow++) {
    int row = 1 << logRow;
    basesLane.push_back({vecSize * ((row / perPhase) % maxPhase), row});
  }
  basesLane.push_back({8, 0});

  auto mma = cast<NvidiaMmaEncodingAttr>(tensorTy.getEncoding());
  assert(mma.getVersionMajor() >= 3 && "Only MMAv3 is supported");
  int instrM = mma.getInstrShape()[0];
  int instrN = mma.getInstrShape()[1];

  // TODO(Keren): The following logic can be simplified by using the
  // `divideLeft` function in `LinearLayout` once it's available.
  // Construct the bases for a single chunk
  // In theory the following situation is valid but it will be
  // suboptimal. Swizzling should happen within a warp.
  assert(instrN >= numColsPerChunk &&
         "Each chunk is filled in with a single warp");
  for (int logCol = 0; logCol < llvm::Log2_32(numColsPerChunk / 16); logCol++) {
    int col = 1 << logCol;
    basesReg.push_back({16 * col, 0});
  }

  // Construct the bases for warpsPerCTA[0]
  std::vector<std::vector<int>> basesWarp;
  auto warpsPerCTA = mma.getWarpsPerCTA();
  auto shape = tensorTy.getShape();
  for (int logWarp = 0; logWarp < llvm::Log2_32(warpsPerCTA[0]); logWarp++) {
    int warp = 1 << logWarp;
    basesWarp.push_back({0, warp * instrM});
  }

  // Expand the `register` dimension so the size of columns matches `shape[1] /
  // warpsPerCTA[1]`
  auto numColsPerWarp = std::max<int>(instrN, shape[1] / warpsPerCTA[1]);
  assert(warpsPerCTA[1] * instrN >= shape[1] &&
         "There must be enough columns to use MMAv3");
  auto logNumCols = llvm::Log2_32(numColsPerWarp / numColsPerChunk);
  for (int logCol = 0; logCol < logNumCols; logCol++) {
    int chunk = 1 << logCol;
    int basis = chunk * shape[0];
    basesReg.push_back({0, basis});
  }

  // Expand the `register` dimension so that the size of rows matches `shape[0]`
  assert(warpsPerCTA[0] * instrM <= shape[0] &&
         "There must be enough rows to use MMAv3");
  auto logNumRows = llvm::Log2_32(shape[0] / (warpsPerCTA[0] * instrM));
  for (int logRow = 0; logRow < logNumRows; logRow++) {
    int chunk = 1 << logRow;
    int basis = chunk * warpsPerCTA[0] * instrM;
    basesReg.push_back({0, basis});
  }

  // Expand the `warp` dimension so that the size of cols matches `shape[1]`
  for (int logWarp = 0; logWarp < llvm::Log2_32(warpsPerCTA[1]); logWarp++) {
    int warp = 1 << logWarp;
    if (warp * numColsPerWarp >= shape[1]) {
      basesWarp.push_back({0, 0});
    } else {
      int basis = (warp * numColsPerWarp) / numColsPerChunk * shape[0];
      basesWarp.push_back({0, basis});
    }
  }

  auto layout = LinearLayout({{S("register"), basesReg},
                              {S("lane"), basesLane},
                              {S("warp"), basesWarp},
                              {S("block"), {}}},
                             {S("offset1"), S("offset0")});
  return layout.reshapeOuts(
      {{S("offset"), layout.getTotalOutDimSize()}, {S("iteration"), 1}});
}

LinearLayout chooseStMatrixLayoutNoLeadingOffset(MLIRContext *ctx,
                                                 Attribute encoding,
                                                 ArrayRef<int64_t> shape) {
  StringAttr kReg = S("register");
  StringAttr kLane = S("lane");
  StringAttr kWarp = S("warp");
  StringAttr kCol = S("dim1");
  StringAttr kRow = S("dim0");
  StringAttr kBlock = S("block");

  std::vector<std::vector<int>> basesReg = {{1, 0}, {2, 0}, {4, 0}};
  std::vector<std::vector<int>> basesLane = {
      {0, 1}, {0, 2}, {0, 4}, {0, 8}, {8, 0}};
  LinearLayout layout =
      LinearLayout({{kReg, basesReg}, {kLane, basesLane}}, {kCol, kRow});

  // Expand the `register` dimension so the size of columns matches `n`.
  auto mma = cast<NvidiaMmaEncodingAttr>(encoding);
  int n = mma.getInstrShape()[1];
  layout *=
      LinearLayout::identity1D(n / layout.getOutDimSize(kCol), kReg, kCol);

  // Expand the `warp` dimension according to warpsPerCTA.
  layout *= identityStandardND(kWarp, mma.getWarpsPerCTA(), /*order=*/{0, 1})
                .transposeOuts(llvm::to_vector(layout.getOutDimNames()));
  auto ret = combineCtaCgaWithShape(layout, mma.getCTALayout(), shape);
  auto tensorShapePerCTA = getShapePerCTA(mma, shape);
  llvm::SmallDenseMap<StringAttr, int64_t> namedTensorShape;
  namedTensorShape[kRow] = tensorShapePerCTA[0];
  namedTensorShape[kCol] = tensorShapePerCTA[1];
  ret = ensureLayoutNotSmallerThan(ret, namedTensorShape);
  ret = ensureLayoutNotLargerThan(ret, namedTensorShape);
  return ret.transposeOuts(llvm::to_vector(layout.getOutDimNames()))
      .reshapeOuts(
          {{S("offset"), ret.getTotalOutDimSize()}, {S("iteration"), 1}});
}

LinearLayout chooseDotLdMatrixLayout(DotOperandEncodingAttr dot,
                                     ArrayRef<int64_t> shape, bool needTrans,
                                     int32_t elemBitWidth) {
  auto ctx = dot.getContext();
  auto mma = cast<NvidiaMmaEncodingAttr>(dot.getParent());
  auto rank = shape.size();
  auto opIdx = dot.getOpIdx();
  int kDim = (opIdx == 0) ? rank - 1 : rank - 2;
  int nonKDim = (opIdx == 0) ? rank - 2 : rank - 1;

  StringAttr kReg = S("register");
  StringAttr kLane = S("lane");
  StringAttr kWarp = S("warp");
  StringAttr kBlock = S("block");
  StringAttr kInner = opIdx == 0 ? (needTrans ? S("dim0") : S("dim1"))
                                 : (needTrans ? S("dim1") : S("dim0"));
  StringAttr kOuter = opIdx == 0 ? (needTrans ? S("dim1") : S("dim0"))
                                 : (needTrans ? S("dim0") : S("dim1"));

  std::vector<std::vector<int>> basesReg;
  for (int logReg = 0; logReg < llvm::Log2_32(8 * 16 / elemBitWidth);
       logReg++) {
    auto reg = 1 << logReg;
    basesReg.push_back({0, reg});
  }
  std::vector<std::vector<int>> basesLane = {
      {1, 0}, {2, 0}, {4, 0}, {0, 0}, {0, 0}};
  bool kX2 = shape[kDim] > 8 * 16 / elemBitWidth;
  bool kX4 = shape[kDim] > 16 * 16 / elemBitWidth;
  bool nonKX2 = shape[nonKDim] > 8;
  // Construct a tile consisting of 4 8x8x16bits sub-tiles to use ldmatrix
  // efficiently. opIdx=0 and opIdx=1 are handled differently.
  if (opIdx == 0) {
    // The matrix elements of thread 0 are distributed in the following pattern
    // (fp16):
    //
    //           col0       col8
    //   row0  reg[0-1]   reg[4-5]
    //   row8  reg[2-3]   reg[6-7]
    if (needTrans) {
      assert(elemBitWidth <= 16 && "Only elements smaller than 16 bits are "
                                   "supported in the transposed mode");
      if (nonKX2)
        basesLane[3] = {0, 8};
      if (kX2)
        basesLane[4] = {8 * 16 / elemBitWidth, 0};
    } else {
      if (nonKX2)
        basesLane[3] = {8, 0};
      if (kX2)
        basesLane[4] = {0, 8 * 16 / elemBitWidth};
    }
  } else {
    // The matrix elements of thread 0 are distributed in the following pattern
    // (fp16):
    //
    //           col0       col8      col16    col24
    //   row0  reg[0-1]   reg[2-3]  reg[4-5]  reg[6-7]
    if (needTrans) {
      assert(elemBitWidth <= 16 && "Only elements smaller than 16 bits are "
                                   "supported in the transposed mode");
      if (kX2)
        basesLane[3] = {8, 0};
      if (kX4)
        basesLane[4] = {16, 0};
    } else {
      if (kX2)
        basesLane[3] = {0, 8 * 16 / elemBitWidth};
      if (kX4)
        basesLane[4] = {0, 16 * 16 / elemBitWidth};
    }
  }
  int numTileCols =
      (8 * 16 / elemBitWidth)
      << (static_cast<int>(kX2) + static_cast<int>(kX4 && opIdx == 1));
  // Expand the `register` dimension so the size of columns matches `K`.
  auto layout =
      LinearLayout({{kReg, basesReg}, {kLane, basesLane}, {kWarp, {}}},
                   {kOuter, kInner}) *
      LinearLayout::identity1D(shape[kDim] / numTileCols, kReg,
                               S("dim" + std::to_string(kDim)));
  // Expand the `warp` dimension according to warpsPerCTA.
  auto warpsPerCTA = mma.getWarpsPerCTA();
  layout *= broadcastedDotOperandLayout(ctx, warpsPerCTA, mma.getWarpOrder(),
                                        kDim, kWarp)
                .transposeOuts(llvm::to_vector(layout.getOutDimNames()));
  return combineCtaCgaWithShape(layout, getCTALayout(dot), shape);
}

} // anonymous namespace

LinearLayout chooseStMatrixLayout(MLIRContext *ctx, RankedTensorType tensorTy,
                                  int swizzleByteSize) {
  if (swizzleByteSize == 0)
    return chooseStMatrixLayoutNoLeadingOffset(ctx, tensorTy.getEncoding(),
                                               tensorTy.getShape());
  else
    return chooseStMatrixLayoutLeadingOffset(ctx, tensorTy, swizzleByteSize);
}

LinearLayout chooseLdMatrixLayout(Attribute enc, ArrayRef<int64_t> shape,
                                  bool needTrans, int32_t elemBitWidth) {
  auto dot = cast<DotOperandEncodingAttr>(enc);
  return chooseDotLdMatrixLayout(dot, shape, needTrans, elemBitWidth);
}

} // namespace mlir::triton::gpu
