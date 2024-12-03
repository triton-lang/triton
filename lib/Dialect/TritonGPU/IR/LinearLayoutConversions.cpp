#include <vector>

#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
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

LinearLayout sharedToLinearLayoutNoLeadingOffset(ArrayRef<int64_t> shape,
                                                 SharedEncodingAttr shared) {
  assert(!shared.getHasLeadingOffset());

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

LinearLayout sharedToLinearLayoutLeadingOffset(ArrayRef<int64_t> shape,
                                               SharedEncodingAttr shared,
                                               int32_t elemBitWidth) {
  assert(shared.getHasLeadingOffset());

  MLIRContext *ctx = shared.getContext();
  int rank = shape.size();
  if (rank == 1) {
    // TODO: Not sure if this is correct.
    return combineCtaCgaWithShape(
        LinearLayout::identity1D(shape[0], S("offset"), S("dim0")),
        shared.getCTALayout(), shape);
  }

  int tileWidthBytes;
  if (shared.getPerPhase() == 4 && shared.getMaxPhase() == 2) {
    tileWidthBytes = 32;
  } else if (shared.getPerPhase() == 2 && shared.getMaxPhase() == 4) {
    tileWidthBytes = 64;
  } else if (shared.getPerPhase() == 1 && shared.getMaxPhase() == 8) {
    tileWidthBytes = 128;
  } else {
    llvm::errs()
        << "Illegal shared encoding.  If hasLeadingOffset is true, "
           "then (perPhase, maxPhase) must be either (4,2), (2,4), or (1,8): "
        << shared << "\n";
    llvm_unreachable("Illegal shared encoding");
  }

  auto outDimNames = standardOutDimNames(ctx, rank);

  // Construct bases for a the layout's 2-dimensional tile.
  assert(shape.size() >= 2);
  int colDim = shared.getOrder()[0];
  int rowDim = shared.getOrder()[1];

  int tileRows = 8;
  int tileCols = 8 * tileWidthBytes / elemBitWidth;

  if (shape[colDim] < tileCols || shape[rowDim] < tileRows) {
    llvm::errs() << "Illegal shared layout; expected shape to be at least ["
                 << tileRows << ", " << tileCols << "], shape: ["
                 << shape[rowDim] << ", " << shape[colDim] << "]\n";
    llvm::report_fatal_error("Illegal shared layout");
  }

  int vec = shared.getVec();

  StringAttr colDimName = outDimNames[colDim];
  StringAttr rowDimName = outDimNames[rowDim];

  std::vector<std::vector<int>> bases2D;
  for (int logCol = 0; logCol < llvm::Log2_32(tileCols); logCol++) {
    bases2D.push_back({0, 1 << logCol});
  }
  for (int logRow = 0; logRow < llvm::Log2_32(tileRows); logRow++) {
    int row = 1 << logRow;
    int perPhase = shared.getPerPhase();
    int maxPhase = shared.getMaxPhase();
    bases2D.push_back({row, vec * ((row / perPhase) % maxPhase)});
  }
  LinearLayout tileLayout =
      LinearLayout({{S("offset"), bases2D}}, {rowDimName, colDimName});

  // Add the remaining dimensions.
  for (int i = 2; i < rank; i++) {
    int dim = shared.getOrder()[i];
    tileLayout *=
        LinearLayout::identity1D(shape[dim], S("offset"), outDimNames[dim]);
  }

  return combineCtaCgaWithShape(tileLayout, shared.getCTALayout(), shape);
}

} // anonymous namespace

std::optional<LinearLayout>
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
    // Extend the base vector with one value to accomodate for the batch
    // dimension, which appears at the last.
    tileLayout *= LinearLayout::identity1D(1, kRegister, outDimNames[order[2]]);
    tileLayout *= LinearLayout::identity1D(1, kLane, outDimNames[order[2]]);
  }

  // And each warp takes the same register and lane sub-layout. So mulitply with
  // an identity layout for the warp.
  LinearLayout warpLayout =
      identityStandardND(S("warp"), getWarpsPerCTA(), order);
  LinearLayout ctaLayout = tileLayout * warpLayout;

  return combineCtaCgaWithShape(ctaLayout, getCTALayout(), shape);
}

std::optional<LinearLayout>
mfmaDotToLinearLayout(DotOperandEncodingAttr dotMfmaLayout,
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
    // the register dimension. Dot operand handles varaible kWidth consecutive
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
    // Extend the base vector with one value to accomodate for the batch
    // dimension, which appears at the last.
    tileLayout *= LinearLayout::identity1D(1, kRegister, outDimNames[order[2]]);
    tileLayout *= LinearLayout::identity1D(1, kLane, outDimNames[order[2]]);
  }

  LinearLayout warpLayout = identityStandardND(kWarp, warpsPerCTA, warpOrder);

  LinearLayout ctaLayout = tileLayout.transposeOuts(outDimNames) *
                           warpLayout.transposeOuts(outDimNames);

  return combineCtaCgaWithShape(ctaLayout, mfmaLayout.getCTALayout(), shape);
}

std::optional<LinearLayout>
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
  SmallVector<unsigned> order = triton::gpu::getOrder(*this);

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
                {outDimNames[order[0]], outDimNames[order[1]]})
          : LinearLayout(
                {{kRegister, {{0, 1}, {0, 2}, {0, 4}}},
                 {kLane, {{1, 0}, {2, 0}, {4, 0}, {8, 0}, /*gap*/ {0, 8}}}},
                {outDimNames[order[0]], outDimNames[order[1]]});

  if (hasBatchDim) {
    assert(order[2] == 0);
    // Extend the base vector with one value to accomodate for the batch
    // dimension, which appears at the last.
    tileLayout *= LinearLayout::identity1D(1, kRegister, outDimNames[order[2]]);
    tileLayout *= LinearLayout::identity1D(1, kLane, outDimNames[order[2]]);
  }

  // And each warp takes the same register and lane sub-layout. So mulitply with
  // an identity layout for the warp.
  LinearLayout warpLayout =
      identityStandardND(S("warp"), getWarpsPerCTA(), order);
  LinearLayout ctaLayout = tileLayout * warpLayout;

  return combineCtaCgaWithShape(ctaLayout, getCTALayout(), shape);
}

std::optional<LinearLayout>
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

std::optional<LinearLayout>
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

LinearLayout warpsNvidiaDot(MLIRContext *ctx, ArrayRef<unsigned> mmaWarpShape,
                            ArrayRef<unsigned> mmaWarpOrder, bool isA) {
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
  auto rank = mmaWarpOrder.size();
  auto inner = isA ? rank - 1 : rank - 2;
  auto dimNames = standardOutDimNames(ctx, rank);
  LinearLayout warpLayout = LinearLayout::empty();

  // We have to broadcast along the inner dimension
  // For A, when moving along M we go from 0 to 2.
  // For B, when moving along N we go from 0 to 1.
  // As such, choosing the order of A {1, 0}, gives us the correct broadcasting
  // Same happens if the mmaWarpOrder is {0, 1}, like in Hopper
  for (auto d : mmaWarpOrder) {
    if (d == inner) {
      warpLayout *=
          LinearLayout::zeros1D(mmaWarpShape[d], S("warp"), dimNames[d]);
    } else {
      warpLayout *=
          LinearLayout::identity1D(mmaWarpShape[d], S("warp"), dimNames[d]);
    }
  }
  return warpLayout;
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
  ctaLayout *=
      warpsNvidiaDot(ctx, mma.getWarpsPerCTA(), mma.getWarpOrder(), isA)
          .transposeOuts(llvm::to_vector(ctaLayout.getOutDimNames()));

  return combineCtaCgaWithShape(ctaLayout, getCTALayout(dot), shape);
}

std::optional<LinearLayout>
DotOperandEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  auto parent = getParent();
  if (auto mfmaLayout = llvm::dyn_cast<AMDMfmaEncodingAttr>(parent)) {
    return mfmaDotToLinearLayout(*this, shape);
  } else if (auto mma = mlir::dyn_cast<NvidiaMmaEncodingAttr>(parent)) {
    return nvidiaDotToLinearLayout(shape, *this);
  }
  return std::nullopt;
}

std::optional<LinearLayout>
SliceEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  MLIRContext *ctx = getContext();

  // First compute the linear layout for this layout's parent.
  SmallVector<int64_t> parentShape(shape);
  parentShape.insert(parentShape.begin() + getDim(), 1);
  std::optional<LinearLayout> parentLL =
      triton::gpu::toLinearLayout(parentShape, getParent());
  if (!parentLL.has_value()) {
    if (mlir::isa<DotOperandEncodingAttr>(getParent()))
      return std::nullopt;
    llvm::report_fatal_error(
        "Failed to compute parent layout for slice layout.");
  }

  // Remove dimension getDim() from the parent layout.
  //
  //  1. Construct a layout `transform` from parent-out-dims to slice-out-dims
  //     that removes the relevant out-dim.
  //  2. Compute linearSlice = parent.compose(transform).  Now linearSlice maps
  //     from parent in-dims to slice out-dims.
  //  3. Fix up duplicate registers introduced by slicing.
  auto outDimNames = standardOutDimNames(ctx, shape.size() + 1);
  LinearLayout transform = LinearLayout::empty();
  for (auto [idx, outDim] : llvm::enumerate(parentLL->getOutDimNames())) {
    if (idx == getDim()) {
      // Because we're multiplying by all zeros, we could replace outDimNames[0]
      // with any other valid out-dim; the layout will be the same.
      transform *= LinearLayout::zeros1D(parentLL->getOutDimSize(outDim),
                                         outDim, outDimNames[0]);
    } else {
      transform *=
          LinearLayout::identity1D(parentLL->getOutDimSize(outDim), outDim,
                                   outDimNames[idx - (idx < getDim() ? 0 : 1)]);
    }
  }
  LinearLayout sliceLL = parentLL->compose(transform);

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

std::optional<LinearLayout>
toLinearLayout(ArrayRef<int64_t> shape, Attribute layout,
               std::optional<int32_t> elemBitWidth /*= std::nullopt*/) {
  // Layouts are distributed or shared
  if (auto distributed = dyn_cast<DistributedEncodingTrait>(layout)) {
    return distributed.toLinearLayout(shape);
  } else if (auto shared = dyn_cast<SharedEncodingAttr>(layout)) {
    if (shared.getHasLeadingOffset()) {
      assert(elemBitWidth.has_value());
      return sharedToLinearLayoutLeadingOffset(shape, shared, *elemBitWidth);
    } else {
      return sharedToLinearLayoutNoLeadingOffset(shape, shared);
    }
  }

  // Third party layouts
  return std::nullopt;
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
LinearLayout chooseStMatrixLayoutLeadingOffset(
    MLIRContext *ctx, RankedTensorType tensorTy, ArrayRef<unsigned> repShape,
    ArrayRef<unsigned> paddedRepShape, ArrayRef<unsigned> order,
    int swizzleByteSize) {
  StringAttr kReg = S("register");
  StringAttr kLane = S("lane");
  StringAttr kWarp = S("warp");
  StringAttr kCol = S("dim1");
  StringAttr kRow = S("dim0");
  StringAttr kOffset = S("offset");

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
  int numRows = 16;
  int numCols = 8 * swizzleByteSize / elemBitWidth;

  // Construct a single stmatrix.x4 (16x16) tile
  std::vector<std::vector<int>> basesReg = {{1, 0}, {2, 0}, {4, 0}};
  std::vector<std::vector<int>> basesLane;
  for (int logRow = 0; logRow < llvm::Log2_32(numRows); logRow++) {
    int row = 1 << logRow;
    basesLane.push_back({vecSize * ((row / perPhase) % maxPhase), row});
  }
  basesLane.push_back({8, 0});

  // Expand the tile's register dimension to fit swizzleByteSize, which is a
  // "chunk"
  for (int logChunk = 0; logChunk < llvm::Log2_32(numCols / 16); logChunk++) {
    int chunk = 1 << logChunk;
    basesReg.push_back({16 * chunk, 0});
  }

  // Construct the layout for a single chunk
  LinearLayout layout =
      LinearLayout({{kReg, basesReg}, {kLane, basesLane}}, {kCol, kRow});

  // Expand the `warp` dimension according to warpsPerCTA.
  auto mma = cast<NvidiaMmaEncodingAttr>(tensorTy.getEncoding());
  layout *= identityStandardND(kWarp, mma.getWarpsPerCTA(), /*order=*/{0, 1})
                .transposeOuts(llvm::to_vector(layout.getOutDimNames()));

  // Expand the `register` dimension so the size of columns matches `n`.
  int n = mma.getInstrShape()[1];
  int numWarpRows = layout.getOutDimSize(kRow);
  layout = (layout.reshapeOuts({{kOffset, layout.getTotalOutDimSize()}}) *
            LinearLayout::identity1D(n / numCols, kReg, kOffset))
               .reshapeOuts({{kCol, n}, {kRow, numWarpRows}});

  auto ret =
      combineCtaCgaWithShape(layout, mma.getCTALayout(), tensorTy.getShape());
  return ret.transposeOuts(llvm::to_vector(layout.getOutDimNames()))
      .reshapeOuts({{kOffset, ret.getTotalOutDimSize()}, {S("iteration"), 1}});
}

LinearLayout chooseStMatrixLayoutNoLeadingOffset(
    MLIRContext *ctx, RankedTensorType tensorTy, ArrayRef<unsigned> repShape,
    ArrayRef<unsigned> paddedRepShape, ArrayRef<unsigned> order) {
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
  auto mma = cast<NvidiaMmaEncodingAttr>(tensorTy.getEncoding());
  int n = mma.getInstrShape()[1];
  layout *=
      LinearLayout::identity1D(n / layout.getOutDimSize(kCol), kReg, kCol);

  // Expand the `warp` dimension according to warpsPerCTA.
  layout *= identityStandardND(kWarp, mma.getWarpsPerCTA(), /*order=*/{0, 1})
                .transposeOuts(llvm::to_vector(layout.getOutDimNames()));
  auto ret =
      combineCtaCgaWithShape(layout, mma.getCTALayout(), tensorTy.getShape());
  auto tensorShapePerCTA = getShapePerCTA(mma, tensorTy.getShape());
  llvm::SmallDenseMap<StringAttr, int64_t> namedTensorShape;
  namedTensorShape[kRow] = tensorShapePerCTA[0];
  namedTensorShape[kCol] = tensorShapePerCTA[1];
  ret = ensureLayoutNotSmallerThan(ret, namedTensorShape);
  ret = ensureLayoutNotLargerThan(ret, namedTensorShape);
  return ret.transposeOuts(llvm::to_vector(layout.getOutDimNames()))
      .reshapeOuts(
          {{S("offset"), ret.getTotalOutDimSize()}, {S("iteration"), 1}});
}

} // anonymous namespace

LinearLayout chooseStMatrixLayout(MLIRContext *ctx, RankedTensorType tensorTy,
                                  ArrayRef<unsigned> repShape,
                                  ArrayRef<unsigned> paddedRepShape,
                                  ArrayRef<unsigned> order,
                                  int swizzleByteSize) {
  if (swizzleByteSize == 0)
    return chooseStMatrixLayoutNoLeadingOffset(ctx, tensorTy, repShape,
                                               paddedRepShape, order);
  else
    return chooseStMatrixLayoutLeadingOffset(
        ctx, tensorTy, repShape, paddedRepShape, order, swizzleByteSize);
}

} // namespace mlir::triton::gpu
