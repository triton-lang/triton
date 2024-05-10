#include <vector>

#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::triton::gpu {
namespace {

LinearLayout blockedToLinearLayout(ArrayRef<int64_t> shape,
                                   BlockedEncodingAttr blocked) {
  MLIRContext *ctx = blocked.getContext();

  assert(shape.size() == blocked.getOrder().size());
  const int rank = shape.size();

  // Create the StringAttrs we'll need for our layout.
  StringAttr kRegister = StringAttr::get(ctx, "register");
  StringAttr kLane = StringAttr::get(ctx, "lane");
  StringAttr kWarp = StringAttr::get(ctx, "warp");
  StringAttr kBlock = StringAttr::get(ctx, "block");
  std::vector<StringAttr> outDimNames;
  for (int i = 0; i < rank; i++) {
    outDimNames.push_back(StringAttr::get(ctx, "dim" + llvm::Twine(i)));
  }

  // The size of `blocked` (i.e. its register * lane * warp * block size) may
  // be different than the shape's size.  If `blocked` is larger than the shape,
  // it means that some data elements will be stored twice (or more) in
  // registers, i.e. the LinearLayout will map two or more (reg, lane, warp,
  // block) input tuples to the same (dim0,...,dimN) output tuple.
  //
  // We keep track of this in `shapeRemaining`, which tells us how much of the
  // shape has been "covered" by the layout.  We call layoutForDim starting with
  // the most minor input dimension (i.e. "register") and once we've "used up"
  // all of shapeRemaining, the LinearLayout broadcasts the remaining input
  // dimensions.
  std::vector<int32_t> shapeRemaining(shape.begin(), shape.end());
  auto layoutForDim = [&](StringAttr inDimName, ArrayRef<unsigned> sizes,
                          ArrayRef<unsigned> order,
                          ArrayRef<unsigned> extraZeros = {}) {
    LinearLayout ret = LinearLayout::empty();

    // Start with the most minor dimension, which is order[0].
    for (int i = 0; i < rank; i++) {
      int dim = order[i];

      int32_t size, zeros;
      if (shapeRemaining[dim] >= sizes[dim]) {
        size = sizes[dim];
        zeros = 1;
        shapeRemaining[dim] /= sizes[dim];
      } else {
        size = shapeRemaining[dim];
        zeros = size > 0 ? sizes[dim] / size : sizes[dim];
        shapeRemaining[dim] = 0;
      }

      if (!extraZeros.empty()) {
        zeros *= extraZeros[dim];
      }

      ret *= LinearLayout::identity1D(size, inDimName, outDimNames[dim]) *
             LinearLayout::zeros1D(zeros, inDimName, outDimNames[dim]);
    }
    return ret;
  };

  // First the shape is split into CTASplitNum pieces, which are distributed
  // among the NumCTAs in the CTG.  Then it's distributed among the threads in
  // the block.
  SmallVector<unsigned> ctgDupes;
  for (int i = 0; i < rank; i++) {
    ctgDupes.push_back(blocked.getCTAsPerCGA()[i] /
                       blocked.getCTASplitNum()[i]);
  }
  LinearLayout ctgLayout =
      layoutForDim(kBlock, blocked.getCTASplitNum(), blocked.getCTAOrder(),
                   /*extraZeros=*/ctgDupes);

  // Split the shape among the register+lane+warp.
  LinearLayout ctaLayout =
      layoutForDim(kRegister, blocked.getSizePerThread(), blocked.getOrder()) *
      layoutForDim(kLane, blocked.getThreadsPerWarp(), blocked.getOrder()) *
      layoutForDim(kWarp, blocked.getWarpsPerCTA(), blocked.getOrder());

  // If the shape per CTA is larger than the layout, we repeat the layout by
  // having each lane hold multiple elements, i.e. adding to the register
  // dimension.  This happens *before* we multiply the CTG, so CTG repetition is
  // always the most-major step.
  //
  // The `block` dimension is always more major than the repeats.  That is, we
  // repeat enough so that then when we tack on the multi-block dimension, we
  // fill the shape exactly.
  for (int i = 0; i < rank; i++) {
    int dim = blocked.getOrder()[i];
    int32_t layoutSize = ctaLayout.getOutDimSize(outDimNames[dim]);

    int32_t shapeSize = shape[dim] / ctgLayout.getOutDimSize(outDimNames[dim]);
    if (shapeSize <= layoutSize) {
      continue;
    }
    assert(shapeSize % layoutSize == 0);
    ctaLayout *= LinearLayout::identity1D(shapeSize / layoutSize, kRegister,
                                          outDimNames[dim]);
  }

  // Join the layouts, with the CTG layout being more major and being transposed
  // to match the order of the CTA layout.  (You can't multiply two layouts with
  // different relative orders for the dims they have in common.)
  LinearLayout ret =
      ctaLayout *
      ctgLayout.transposeOuts(llvm::to_vector(ctaLayout.getOutDimNames()));

  // Transpose out dims to 0, 1, 2, ...
  ret = ret.transposeOuts(outDimNames);

  for (int i = 0; i < rank; i++) {
    if (ret.getOutDimSize(outDimNames[i]) != shape[i]) {
      llvm::errs() << "Bug in blockedToLinearLayout; wrong output sizes\n";
      llvm::errs() << "input shape: " << triton::join(shape, ",") << "\n";
      llvm::errs() << "input layout: " << blocked << "\n";
      llvm::errs() << "output linear layout:\n" << ret << "\n";
      llvm::report_fatal_error(
          "Bug in blockedToLinearLayout; wrong output sizes.");
    }
  }

  return ret;
}

} // anonymous namespace

LinearLayout toLinearLayout(ArrayRef<int64_t> shape, Attribute layout) {
  if (auto blocked = dyn_cast<BlockedEncodingAttr>(layout)) {
    return blockedToLinearLayout(shape, blocked);
  }

  // TODO(jlebar): Other layouts
  llvm::llvm_unreachable_internal("Unsupported layout");
}

} // namespace mlir::triton::gpu
