#include <vector>

#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
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

// Returns ["out0", "out1", ..., "out<rank-1>"].
SmallVector<StringAttr> standardOutDimNames(MLIRContext *ctx, int rank) {
  SmallVector<StringAttr> ret;
  for (int i = 0; i < rank; i++) {
    ret.push_back(S("dim" + llvm::Twine(i)));
  }
  return ret;
}

void assertIsRegisterLayout(const LinearLayout &layout) {
  assert(layout.getNumInDims() > 0);
  MLIRContext *ctx = layout.getInDimNames().begin()->getContext();
  StringAttr kRegister = S("register");
  StringAttr kLane = S("lane");
  StringAttr kWarp = S("warp");
  StringAttr kBlock = S("block");

  const auto &ins = layout.getInDimNames();
  assert(llvm::SmallVector<StringAttr>(ins.begin(), ins.end()) ==
         llvm::SmallVector<StringAttr>({kRegister, kLane, kWarp, kBlock}));

  const auto &outs = layout.getOutDimNames();
  const auto &expectedOuts = standardOutDimNames(ctx, layout.getNumOutDims());
  assert(llvm::SmallDenseSet<StringAttr>(outs.begin(), outs.end()) ==
         llvm::SmallDenseSet<StringAttr>(expectedOuts.begin(),
                                         expectedOuts.end()));
}

// Returns a 1D -> ND layout that's equivalent to creating a 1D -> 1D mapping of
// size product(shape) and then reshaping to permute(shape, order).
LinearLayout identityND(StringAttr inDimName, ArrayRef<unsigned> shape,
                        ArrayRef<unsigned> order,
                        ArrayRef<StringAttr> outDimNames) {
  assert(shape.size() == order.size());

  MLIRContext *ctx = inDimName.getContext();
  LinearLayout ret = LinearLayout::empty();
  for (int i = 0; i < shape.size(); i++) {
    // Start with the most-minor dimension, which is order[0].
    int dim = order[i];
    ret *= LinearLayout::identity1D(shape[dim], inDimName, outDimNames[dim]);
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

// For each output dimension d, ensure that the layout's output size (i.e., its
// codomain) does not exceed shape[d]. Do this without changing the size of the
// layout's inputs (i.e., leave its domain unchanged).
//
// This function is invariant to the order of the layout's input and output
// dimensions.
//
// We achieve this by setting the largest value in each output dimension d to 0
// because bases that map to a location larger than shape[d]
// effectively duplicate along that dimension.  For example, consider a layout
// with an output dimension size of 32, and we call ensureLayoutNotLargerThan to
// shrink the output dimension size to 8:
//
//   L(register=1) = 8
//   L(register=2) = 4
//   L(register=4) = 1
//   L(lane=1) = 2
//   L(lane=2) = 16
//
// In the first step, we shrink the output dimension size to 16 by setting
// L(lane=2) to 0:
//
//   L(register=1) = 8
//   L(register=2) = 4
//   L(register=4) = 1
//   L(lane=1) = 2
//   L(lane=2) = 0
//
// This means that lane=2 has the same data as lane=0.
//
// Now the output dimension of this layout has a size of 16, which is still
// larger than 8.  We find the current largest value in the output dimension,
// which is L(register=1) = 8, and we set L(register=1) to 0:
//
//   L(register=1) = 0
//   L(register=2) = 4
//   L(register=4) = 1
//   L(lane=1) = 2
//   L(lane=2) = 0
//
// Now the output dimension of this layout has a size of 8, which is the desired
// size.  Note that this method works only because the bases are powers of two.
// It is unclear what to do when they are not.
LinearLayout ensureLayoutNotLargerThan(
    const LinearLayout &layout,
    const llvm::SmallDenseMap<StringAttr, int64_t> &shape) {
  assert(shape.size() == layout.getNumOutDims());
  if (shape.empty()) {
    return layout;
  }
  MLIRContext *ctx = shape.begin()->first.getContext();

  auto bases = layout.getBases();
  for (auto outDim : llvm::enumerate(layout.getOutDimNames())) {
    auto outDimName = outDim.value();
    int32_t actualSize = layout.getOutDimSize(outDimName);
    int32_t desiredSize = shape.lookup(outDimName);
    if (actualSize <= desiredSize) {
      continue;
    }
    assert(actualSize % desiredSize == 0);
    // <inDimName, basisIdx, outValue>
    std::vector<std::tuple<StringAttr, int, int>> sortedBases;
    for (auto [inDimName, basis] : bases) {
      for (size_t basisIdx = 0; basisIdx < basis.size(); basisIdx++) {
        auto outValue = basis[basisIdx][outDim.index()];
        if (outValue == 0) {
          continue;
        }
        assert(llvm::isPowerOf2_32(outValue));
        sortedBases.emplace_back(inDimName, basisIdx, outValue);
      }
    }
    // From the largest basis to the smallest.
    llvm::sort(sortedBases,
               [](auto a, auto b) { return std::get<2>(a) > std::get<2>(b); });
    for (auto [inDimName, basisIdx, outValue] : sortedBases) {
      if (actualSize <= desiredSize) {
        break;
      }
      bases[inDimName][basisIdx][outDim.index()] = 0;
      actualSize >>= 1;
    }
  }
  return LinearLayout(std::move(bases),
                      llvm::to_vector(layout.getOutDimNames()));
}

// For each out-dim d, ensure the layout's out-size (i.e. its codomain) is no
// smaller than shape[d].  Do this by increasing the size of the layout's inputs
// along its most-minor dimension ("register" for register layouts, "offset" for
// shared layouts).
//
// This function is invariant to the order of the layout's input dimensions, but
// it cares about the order of the output dims, which should be minor-to-major.
LinearLayout ensureLayoutNotSmallerThan(
    const LinearLayout &layout,
    const llvm::SmallDenseMap<StringAttr, int64_t> &shape) {
  assert(shape.size() == layout.getNumOutDims());
  if (shape.empty()) {
    return layout;
  }

  MLIRContext *ctx = shape.begin()->first.getContext();
  StringAttr kDim = *layout.getInDimNames().begin();
  assert(kDim == "register" || kDim == "offset");

  LinearLayout ret = layout;
  for (StringAttr outDimName : layout.getOutDimNames()) {
    int32_t actualSize = layout.getOutDimSize(outDimName);
    int32_t desiredSize = shape.lookup(outDimName);
    assert(actualSize > desiredSize || desiredSize % actualSize == 0);
    ret *= LinearLayout::identity1D(desiredSize / actualSize, kDim, outDimName);
    assert(ret.getOutDimSize(outDimName) >= desiredSize);
  }
  return ret;
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

LinearLayout ampereMmaToLinearLayout(ArrayRef<int64_t> shape,
                                     NvidiaMmaEncodingAttr mma) {
  int rank = shape.size();

  assert(mma.isAmpere());
  assert(rank == 2 || rank == 3);
  assert(mma.getInstrShape().size() == rank);
  assert((rank == 2 && mma.getInstrShape() == ArrayRef<unsigned>({16, 8})) ||
         (rank == 3 && mma.getInstrShape() == ArrayRef<unsigned>({1, 16, 8})));

  MLIRContext *ctx = mma.getContext();
  SmallVector<StringAttr> dimNames = standardOutDimNames(ctx, rank);

  LinearLayout ctaLayout(
      {{S("register"), {{1, 0}, {0, 8}}},
       {S("lane"), {{2, 0}, {4, 0}, {0, 1}, {0, 2}, {0, 4}}}},
      llvm::to_vector(llvm::reverse(ArrayRef(dimNames).take_back(2))));

  ctaLayout *= identityND(
      S("warp"), mma.getWarpsPerCTA(),
      llvm::to_vector(llvm::reverse(llvm::seq<unsigned>(rank))), dimNames);

  return combineCtaCgaWithShape(ctaLayout, mma.getCTALayout(), shape);
}

LinearLayout hopperMmaToLinearLayout(ArrayRef<int64_t> shape,
                                     NvidiaMmaEncodingAttr mma) {
  int rank = shape.size();
  assert(mma.isHopper());
  assert(rank == 2);

  // wgmma operates on groups of 4 warps.
  assert(product(mma.getWarpsPerCTA()) % 4 == 0);

  // Check that it's a known MMA layout.
  assert(mma.getInstrShape().size() == 3);
  int m = mma.getInstrShape()[0];
  int n = mma.getInstrShape()[1];
  int k = mma.getInstrShape()[2];
  assert(m == 16);
  assert(n == 8 || n == 16 || n == 32 || n == 64 || n == 128 || n == 256);
  assert(k == 8 || k == 16 || k == 32);

  MLIRContext *ctx = mma.getContext();
  LinearLayout ctaLayout(
      {{S("register"), {{1, 0}, {0, 8}}},
       {S("lane"), {{2, 0}, {4, 0}, {0, 1}, {0, 2}, {0, 4}}}},
      {S("dim1"), S("dim0")});

  // Expand the `register` dimension so the size of dim1 matches `n`.
  ctaLayout *= LinearLayout::identity1D(n / ctaLayout.getOutDimSize(S("dim1")),
                                        S("register"), S("dim1"));

  // Expand the `warp` dimension according to warpsPerCTA.
  //
  // It's weird that this is order [0,1] when MMAv2's warpsPerCTA is [1,0], but
  // this really does seem to be correct.
  ctaLayout *= identityND(S("warp"), mma.getWarpsPerCTA(), /*order=*/{0, 1},
                          {S("dim0"), S("dim1")})
                   .transposeOuts(llvm::to_vector(ctaLayout.getOutDimNames()));

  return combineCtaCgaWithShape(ctaLayout, mma.getCTALayout(), shape);
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

  int vec = 8 * 16 / elemBitWidth;
  if (vec != shared.getVec()) {
    llvm::errs() << "Illegal shared layout; expected `vec` to be " << vec
                 << ": " << shared << "\n";
    llvm::report_fatal_error("Illegal shared layout");
  }

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

  assert(((shape[mIndex] == 1 || shape[mIndex] >= getMDim()) &&
          (shape[nIndex] == 1 || shape[nIndex] >= getNDim())) &&
         "Unsupported tensor shape for given mfma layout");

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
      identityND(S("warp"), getWarpsPerCTA(), order, outDimNames);
  LinearLayout ctaLayout = tileLayout * warpLayout;

  return combineCtaCgaWithShape(ctaLayout, getCTALayout(), shape);
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
      identityND(S("warp"), getWarpsPerCTA(), order, outDimNames);
  LinearLayout ctaLayout = tileLayout * warpLayout;

  return combineCtaCgaWithShape(ctaLayout, getCTALayout(), shape);
}

std::optional<LinearLayout>
BlockedEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  assert(shape.size() == getOrder().size());

  int rank = shape.size();
  MLIRContext *ctx = getContext();
  SmallVector<StringAttr> outDimNames = standardOutDimNames(ctx, rank);

  const auto &order = getOrder();
  LinearLayout ctaLayout =
      identityND(S("register"), getSizePerThread(), order, outDimNames) *
      identityND(S("lane"), getThreadsPerWarp(), order, outDimNames) *
      identityND(S("warp"), getWarpsPerCTA(), order, outDimNames);

  return combineCtaCgaWithShape(ctaLayout, getCTALayout(), shape);
}

std::optional<LinearLayout>
NvidiaMmaEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  if (isAmpere()) {
    return ampereMmaToLinearLayout(shape, *this);
  }
  if (isHopper()) {
    return hopperMmaToLinearLayout(shape, *this);
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

// TODO: DotOperandEncoding doesn't support LinearLayout conversion yet.
std::optional<LinearLayout>
DotOperandEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  return std::nullopt;
}

std::optional<LinearLayout>
toLinearLayout(ArrayRef<int64_t> shape, Attribute layout,
               std::optional<int32_t> elemBitWidth /*= std::nullopt*/) {
  if (auto distributed = dyn_cast<DistributedEncodingTrait>(layout)) {
    return distributed.toLinearLayout(shape);
  }
  if (auto shared = dyn_cast<SharedEncodingAttr>(layout)) {
    if (shared.getHasLeadingOffset()) {
      assert(elemBitWidth.has_value());
      return sharedToLinearLayoutLeadingOffset(shape, shared, *elemBitWidth);
    } else {
      return sharedToLinearLayoutNoLeadingOffset(shape, shared);
    }
  }

  // TODO(jlebar): Other layouts
  return std::nullopt;
}

bool isCrossCTAConversion(const LinearLayout &layout) {
  assert(!layout.getInDimNames().empty());
  MLIRContext *ctx = layout.getInDimNames().begin()->getContext();

  StringAttr kBlock = S("block");
  assert(layout.hasInDim(kBlock));
  assert(layout.hasOutDim(kBlock));

  SetVector<StringAttr> nonBlockInDims(layout.getInDimNames().begin(),
                                       layout.getInDimNames().end());
  nonBlockInDims.remove(kBlock);

  // This layout moves data between CTAs if
  // - the value for any input dim other than block affects the output block, or
  // - input (0, ..., block=i) does not map to output (0, ..., block=i).
  return !layout.sublayoutIsZero(nonBlockInDims.getArrayRef(), {kBlock}) ||
         !layout.sublayoutIsIdentity({kBlock}, {kBlock});
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

// TODO (Keren): Currently, we have more restrictions than necessary when using
// stmatrix.  These restrictions are retained from legacy code, and we could
// relax some of them in the future.
bool canUseStMatrix(RankedTensorType tensorTy, ArrayRef<unsigned> repShape,
                    ArrayRef<unsigned> paddedRepShape,
                    ArrayRef<unsigned> order) {
  auto mmaLayout =
      mlir::dyn_cast<NvidiaMmaEncodingAttr>(tensorTy.getEncoding());
  if (!mmaLayout || !mmaLayout.isHopper())
    return false;
  if (tensorTy.getElementType().getIntOrFloatBitWidth() != 16)
    return false;
  if (order[0] != 1)
    return false;

  auto tensorShapePerCTA = getShapePerCTA(mmaLayout, tensorTy.getShape());
  if (tensorShapePerCTA.size() != 2)
    return false;
  auto numIterations = ceil<unsigned>(tensorShapePerCTA[1], repShape[1]) *
                       ceil<unsigned>(tensorShapePerCTA[0], repShape[0]);
  if (numIterations > 1)
    return false;
  if (paddedRepShape[1] % 8 != 0)
    return false;
  return true;
}

} // anonymous namespace

std::optional<LinearLayout> chooseStMatrixLayoutForRegToRegConversion(
    MLIRContext *ctx, RankedTensorType tensorTy, ArrayRef<unsigned> repShape,
    ArrayRef<unsigned> paddedRepShape, ArrayRef<unsigned> order) {
  if (!canUseStMatrix(tensorTy, repShape, paddedRepShape, order))
    return std::nullopt;

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
  layout *=
      identityND(kWarp, mma.getWarpsPerCTA(), /*order=*/{0, 1}, {kRow, kCol})
          .transposeOuts(llvm::to_vector(layout.getOutDimNames()));
  auto ret =
      combineCtaCgaWithShape(layout, mma.getCTALayout(), tensorTy.getShape());

  return ret.transposeOuts(llvm::to_vector(layout.getOutDimNames()))
      .reshapeOuts(
          {{S("offset"), ret.getTotalOutDimSize()}, {S("iteration"), 1}});
}

} // namespace mlir::triton::gpu
