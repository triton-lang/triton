#include <vector>

#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LinearLayout.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::triton::gpu {
namespace {

// We use the following nomenclature in this file.
//
//  - ctaLayout: A layout for one block, i.e. input dims (register, lane, warp).
//  - cgaLayout: Arrangement of multiple blocks, i.e. input dims (block).
//
// Note that this is inconsistent with the type name CTALayoutAttr.  That type
// is equivalent to our cgaLayout.
//
// IMO the type name is wrong.  If we tried to be consistent anyway, then we'd
// have to rename ctaLayout to "warpLayout".  I think that's more confusing than
// being inconsistent about "cgaLayout", especially when we have to consider the
// size of the warpLayout (surely that's not the "warpSize").

#define S(v) StringAttr::get(ctx, (v))

// Returns ["out0", "out1", ..., "out<rank-1>"].
SmallVector<StringAttr> standardOutDimNames(MLIRContext *ctx, int rank) {
  SmallVector<StringAttr> ret;
  for (int i = 0; i < rank; i++) {
    ret.push_back(S("dim" + llvm::Twine(i)));
  }
  return ret;
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

// Shrinks the output set of a layout function while leaving the input set
// unchanged, by making high-order inputs in inDimName map to the same output.
// Attempts to shrink down to desiredSize, but this is not always possible just
// by modifying one the specified input dimension.
//
// We do this by making the most-major inputs to the layout map to 0.  This
// effectively duplicates data along that input dimension.  For example, this
// layout has out-dim size 32:
//
//   L(register=1) = 8
//   L(register=2) = 4
//   L(register=4) = 1
//   L(lane=1) = 2
//   L(lane=2) = 16.
//
// If we shrink it to size 16 along the `lane` dimension, we set L(lane=2) to 0:
//
//   L(register=1) = 8
//   L(register=2) = 4
//   L(register=4) = 1
//   L(lane=1) = 2
//   L(lane=2) = 0.
//
// This means that lane=2 has the same data as lane=0.
//
// If we shrink to size 8 along the lane dimension, we set L(lane=1) = 0 as
// well.  But when we do this, we have to remove bit 1 (the value of L(lane=1))
// from all other bases:
//
//   L(register=1) = 4
//   L(register=2) = 2
//   L(register=1) = 1
//   L(lane=1) = 0
//   L(lane=2) = 0.
//
// Note this only works because the bases are powers of two.  I don't quite know
// what to do when they're not.
LinearLayout shrinkCodomain(const LinearLayout &layout, StringAttr inDimName,
                            StringAttr outDimName, int desiredSize) {
  assert(llvm::isPowerOf2_32(desiredSize));
  int outDimIdx = layout.getOutDimIndex(outDimName);
  int desiredZeros =
      llvm::Log2_32(layout.getOutDimSize(outDimName) / desiredSize);
  if (desiredZeros == 0) {
    return layout;
  }

  // Find the desiredZeros most-major basis vectors that are not already zero.
  // These are the ones we will set to zero.
  SmallVector<int> basesToZero;
  for (int i = layout.getInDimSizeLog2(inDimName) - 1;
       i >= 0 && basesToZero.size() < desiredZeros; i--) {
    int basis = layout.getBasis(inDimName, i, outDimName);
    if (basis != 0) {
      basesToZero.push_back(basis);
    }
  }

  // Bail if all the bases are already zero; nothing more we can do.
  if (basesToZero.empty()) {
    return layout;
  }

  // The algorithm below only works because the bases are powers of two.  I'm
  // not sure what to do otherwise.
  assert(llvm::all_of(basesToZero,
                      [&](int basis) { return llvm::isPowerOf2_32(basis); }));

  // We want to zero out the bases in `basesToZero`, and also "shift out" the
  // corresponding bits from all other bases.  For example if we remove the
  // basis with value 8 = 0b100, then if another basis has value 26 = 0b11010,
  // the 1 in its 3rd position gets removed and it becomes 10 = 0b1010.
  //
  // We could manually alter the bases in `layout` to achieve this, but it's
  // perhaps simpler to use the linearity of LLs to our advantage.
  //
  // Consider the function O which is the identity map from out-dims to
  // out-dims.  We can easily calculate what happens when we remove the relevant
  // bases from O.  Call this new function O'.
  //
  // Because of linearity, removing the bases from L is equivalent to composing
  // L with O'.  So that's what we do below.

  // Construct the out-dims -> out-dims identity layout O.
  LinearLayout outputIdentity = LinearLayout::empty();
  for (StringAttr dim : layout.getOutDimNames()) {
    outputIdentity *=
        LinearLayout::identity1D(layout.getOutDimSize(dim), dim, dim);
  }

  // Modify O to remove the relevant bases.
  //
  // TODO(jlebar): I don't like manually modifying bases here.  Perhaps this
  // should be a function on LinearLayout.
  LinearLayout::BasesT newBases = outputIdentity.getBases();
  llvm::sort(basesToZero);
  for (int basis : basesToZero) {
    int idx = llvm::Log2_32(basis);
    for (int i = newBases[outDimName].size() - 1; i > idx; i--) {
      newBases[outDimName][i][outDimIdx] =
          newBases[outDimName][i - 1][outDimIdx];
    }
    newBases[outDimName][idx][outDimIdx] = 0;
  }

  // Construct O'.
  LinearLayout transform(std::move(newBases), layout.getOutDimNames());

  // Compose O' with L.
  return layout.compose(transform);
}

// For each out-dim d, ensure the layout's out-size (i.e. its codomain) is no
// larger than shape[d].  Do this without changing the size of the layout's
// inputs (i.e. leave its domain unchanged).
//
// This function is invariant to the order of the layout's input and output
// dimensions.
LinearLayout ensureLayoutNotLargerThan(
    const LinearLayout &layout,
    const llvm::SmallDenseMap<StringAttr, int64_t> &shape) {
  assert(shape.size() == layout.getNumOutDims());
  if (shape.empty()) {
    return layout;
  }
  MLIRContext *ctx = shape.begin()->first.getContext();

  // For the purposes of this function, "block" is the "most-minor" dimension.
  // This is just a consequence of how legacy layouts work: We only put the same
  // tensor element into two different blocks as a last resort, only after all
  // the registers in all the lanes in all the warps in a block already have the
  // same tensor element.
  SmallVector<StringAttr> inDimNames = {
      S("block"),
      S("register"),
      S("lane"),
      S("warp"),
  };

  LinearLayout ret = layout;
  for (auto outDimName : layout.getOutDimNames()) {
    int32_t actualSize = layout.getOutDimSize(outDimName);
    int32_t desiredSize = shape.lookup(outDimName);
    if (actualSize <= desiredSize) {
      continue;
    }
    assert(actualSize % desiredSize == 0);
    // TODO: We claim this is invariant to the order of dims, so can we get rid
    // of llvm::reverse?
    for (StringAttr inDimName : llvm::reverse(inDimNames)) {
      if (ret.hasInDim(inDimName)) {
        ret = shrinkCodomain(ret, inDimName, outDimName, desiredSize);
      }
    }
    assert(ret.getOutDimSize(outDimName) == desiredSize);
  }
  return ret;
}

// For each out-dim d, ensure the layout's out-size (i.e. its codomain) is no
// smaller than shape[d].  Do this by increasing the size of the layout's inputs
// along the "register" dimension.
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
  StringAttr kRegister = S("register");

  LinearLayout ret = layout;
  for (StringAttr outDimName : layout.getOutDimNames()) {
    int32_t actualSize = layout.getOutDimSize(outDimName);
    int32_t desiredSize = shape.lookup(outDimName);
    assert(actualSize > desiredSize || desiredSize % actualSize == 0);
    ret *= LinearLayout::identity1D(desiredSize / actualSize, kRegister,
                                    outDimName);
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
          .transposeOuts(ctaLayout.getOutDimNames());

  // Calculate the shape of the ctaLayout, which is `shape` divided by the
  // cgaLayout's size.
  llvm::SmallDenseMap<StringAttr, int64_t> ctaShape;
  assert(ctaLayout.getOutDimNames() == cgaLayout.getOutDimNames());
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

LinearLayout blockedToLinearLayout(ArrayRef<int64_t> shape,
                                   BlockedEncodingAttr blocked) {
  assert(shape.size() == blocked.getOrder().size());

  int rank = shape.size();
  MLIRContext *ctx = blocked.getContext();
  SmallVector<StringAttr> outDimNames = standardOutDimNames(ctx, rank);

  const auto &order = blocked.getOrder();
  LinearLayout ctaLayout =
      identityND(S("register"), blocked.getSizePerThread(), order,
                 outDimNames) *
      identityND(S("lane"), blocked.getThreadsPerWarp(), order, outDimNames) *
      identityND(S("warp"), blocked.getWarpsPerCTA(), order, outDimNames);

  return combineCtaCgaWithShape(ctaLayout, blocked.getCTALayout(), shape);
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
  assert(n == 16 || n == 32 || n == 64 || n == 128 || n == 256);
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
                   .transposeOuts(ctaLayout.getOutDimNames());

  return combineCtaCgaWithShape(ctaLayout, mma.getCTALayout(), shape);
}

LinearLayout mfmaToLinearLayout(ArrayRef<int64_t> shape,
                                AMDMfmaEncodingAttr mfma) {
  int rank = shape.size();
  assert(rank == mfma.getWarpsPerCTA().size());
  MLIRContext *ctx = mfma.getContext();
  SmallVector<StringAttr> outDimNames = standardOutDimNames(ctx, rank);

  assert(((shape[rank - 2] == 1 || shape[rank - 2] >= mfma.getMDim()) &&
          (shape[rank - 1] == 1 || shape[rank - 1] >= mfma.getNDim())) &&
         "Unsupported tensor shape for given mfma layout");

  assert(((mfma.getMDim() == 32 && mfma.getNDim() == 32) ||
          (mfma.getMDim() == 16 || mfma.getNDim() == 16)) &&
         "unsupported mfma type");

  auto rDim = S("register");
  auto lDim = S("lane");
  auto wDim = S("warp");

  auto order = triton::gpu::getOrder(mfma);
  auto tileLayout = LinearLayout::empty();
  if (mfma.getMDim() == 32) {
    tileLayout = LinearLayout(
        {{rDim, {{0, 1}, {0, 2}, {0, 8}, {0, 16}}},
         {lDim, {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {0, 4}}}},
        {outDimNames[order[0]], outDimNames[order[1]]});
  } else {
    assert(mfma.getMDim() == 16);
    tileLayout =
        LinearLayout({{rDim, {{0, 1}, {0, 2}}},
                      {lDim, {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 4}, {0, 8}}}},
                     {outDimNames[order[0]], outDimNames[order[1]]});
  }
  if (rank == 3) {
    assert(order[2] == 0);
    tileLayout *= LinearLayout::identity1D(1, rDim, outDimNames[order[2]]);
    tileLayout *= LinearLayout::identity1D(1, lDim, outDimNames[order[2]]);
  }

  LinearLayout warpLayout =
      identityND(wDim, mfma.getWarpsPerCTA(), order, outDimNames);

  LinearLayout ctaLayout = tileLayout * warpLayout;

  return combineCtaCgaWithShape(ctaLayout, mfma.getCTALayout(), shape);
}

std::optional<LinearLayout> toLinearLayout(ArrayRef<int64_t> shape,
                                           SliceEncodingAttr slice) {
  MLIRContext *ctx = slice.getContext();

  // First compute the linear layout for this layout's parent.
  SmallVector<int64_t> parentShape(shape);
  parentShape.insert(parentShape.begin() + slice.getDim(), 1);
  std::optional<LinearLayout> parentLL =
      triton::gpu::toLinearLayout(parentShape, slice.getParent());
  if (!parentLL) {
    return std::nullopt;
  }

  // Remove dimension slice.getDim() from the parent layout.
  //
  //  1. Construct a layout `transform` from parent-out-dims to slice-out-dims
  //     that removes the relevant out-dim.
  //  2. Compute linearSlice = parent.compose(transform).  Now linearSlice maps
  //     from parent in-dims to slice out-dims.
  //  3. Fix up duplicate registers introduced by slicing.
  auto outDimNames = standardOutDimNames(ctx, shape.size() + 1);
  LinearLayout transform = LinearLayout::empty();
  for (auto [idx, outDim] : llvm::enumerate(parentLL->getOutDimNames())) {
    if (idx == slice.getDim()) {
      // Because we're multiplying by all zeros, we could replace outDimNames[0]
      // with any other valid out-dim; the layout will be the same.
      transform *= LinearLayout::zeros1D(parentLL->getOutDimSize(outDim),
                                         outDim, outDimNames[0]);
    } else {
      transform *= LinearLayout::identity1D(
          parentLL->getOutDimSize(outDim), outDim,
          outDimNames[idx - (idx < slice.getDim() ? 0 : 1)]);
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

  LinearLayout ret = LinearLayout(std::move(bases), sliceLL.getOutDimNames());

  // Match a hack in the legacy code that ensures that the number of registers
  // matches getTotalElemsPerThread.  Yup: We just removed all the zeros, now
  // we're (maybe) adding some back.  :)
  //
  // TODO(jlebar): Once getTotalElemsPerThread uses LLs instead of the existing
  // legacy code, I think we can remove this.
  int expectedNumRegisters = getTotalElemsPerThread(RankedTensorType::get(
      shape, IntegerType::get(ctx, 32) /*dummy type*/, slice));
  if (ret.getInDimSize(S("register")) != expectedNumRegisters) {
    int extraZeros = expectedNumRegisters / ret.getInDimSize(S("register"));
    // Our use of "dim0" here is arbitrary; because we're adding zeros, any
    // output dimension would work.
    ret *= LinearLayout::zeros1D(extraZeros, S("register"), S("dim0"));
  }
  return ret;
}

} // anonymous namespace

std::optional<LinearLayout> toLinearLayout(ArrayRef<int64_t> shape,
                                           Attribute layout) {
  if (auto blocked = dyn_cast<BlockedEncodingAttr>(layout)) {
    return blockedToLinearLayout(shape, blocked);
  }
  if (auto mfma = dyn_cast<AMDMfmaEncodingAttr>(layout)) {
    return mfmaToLinearLayout(shape, mfma);
  }
  if (auto mma = dyn_cast<NvidiaMmaEncodingAttr>(layout)) {
    if (mma.isAmpere()) {
      return ampereMmaToLinearLayout(shape, mma);
    }
    if (mma.isHopper()) {
      return hopperMmaToLinearLayout(shape, mma);
    }
  }
  if (auto slice = dyn_cast<SliceEncodingAttr>(layout)) {
    return toLinearLayout(shape, slice);
  }

  // TODO(jlebar): Other layouts
  return std::nullopt;
}

} // namespace mlir::triton::gpu
