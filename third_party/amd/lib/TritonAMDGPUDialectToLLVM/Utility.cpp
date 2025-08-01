#include "Utility.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::LLVM::AMD {
using namespace mlir;

SmallVector<unsigned> getCTATileOrder(MLIRContext *ctx,
                                      const triton::LinearLayout &layout) {
  auto llEnc = triton::gpu::LinearEncodingAttr::get(ctx, layout);
  auto regDim = StringAttr::get(ctx, "register");
  auto &bases = layout.getBases().find(regDim)->second;

  // Compute number of CTA tiles in a layout.
  unsigned totalElems = layout.getTotalOutDimSize();
  auto ctaShape = llEnc.getShapePerCTATile();
  unsigned elemsPerCTA =
      std::accumulate(ctaShape.begin(), ctaShape.end(), 1, std::multiplies<>());
  assert((totalElems % elemsPerCTA) == 0 &&
         "Total elements must be divisible by elemsPerCTA");
  unsigned numCTAs = totalElems / elemsPerCTA;

  // To determine the CTA tile order, start by identifying the register basis
  // vector that corresponds to the first element of the second CTA tile. The
  // nonzero index in the logical tensor it maps to indicates the fastest
  // varying dimension. Then, for each subsequent basis register (first element
  // of some CTA tile), extract the next nonzero index to build the full
  // dimension order.
  unsigned registersPerThreadPerCTA =
      product(llEnc.basesPerDim(regDim, /*skipBroadcast=*/false)) / numCTAs;
  unsigned startIndex =
      static_cast<unsigned>(std::log2(registersPerThreadPerCTA));

  llvm::SmallSetVector<unsigned, 8> order;
  for (unsigned i = startIndex; i < bases.size(); ++i) {
    auto range = llvm::make_range(bases[i].begin(), bases[i].end());
    auto it = llvm::find_if(range, [](unsigned v) { return v != 0; });
    if (it != bases[i].end())
      order.insert(std::distance(bases[i].begin(), it));
  }

  // Append any dims missing from our default order.
  for (unsigned dim : llEnc.getOrder())
    order.insert(dim);

  return order.takeVector();
}

/*
 *
 The bases vectors for all in-dims are truncated on the most-fast changing
 order, for exampe: {register = [[1, 0], [2, 0], [4, 0], [0, 16], [0, 32], [0,
 64]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [8, 0], [16, 0]], warp = [[32,
 0], [64, 0], [128, 0]], block = []} and the order is [1, 0] and if warpsPerCTA
 is 8, then the biggest 3 vectors are [0, 16], [0, 32], [0, 64] with "register"
 are removed in the output bases vectors, so the output will be {register = [[1,
 0], [2, 0], [4, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [8, 0], [16, 0]],
 warp = [[32, 0], [64, 0], [128, 0]], block = []}
 */
mlir::triton::LinearLayout
inferLinearLayoutFromExtractSlice(MLIRContext *ctx,
                                  RankedTensorType tensorType) {
  auto encoding = mlir::triton::gpu::toLinearEncoding(tensorType);
  auto ctaNum = encoding.getCTAsPerCGA();
  assert(llvm::all_of(ctaNum, [](unsigned dim) { return dim == 1; }) &&
         "Only SplitNum with 1 per CGA supported for now");

  auto order = encoding.getOrder().front();
  const auto &ll = encoding.getLinearLayout();
  auto sizePerCTA = encoding.getWarpsPerCTA();
  auto slicedInLog2 = llvm::Log2_32(std::reduce(
      sizePerCTA.begin(), sizePerCTA.end(), 1, std::multiplies<unsigned>()));

  triton::LinearLayout::BasesT bases = ll.getBases();
  std::vector<std::vector<int>> baseVecs;
  for (const auto &[_, inDimBases] : bases) {
    baseVecs.insert(baseVecs.end(), inDimBases.begin(), inDimBases.end());
  }

  llvm::sort(baseVecs, [&](const auto &v1, const auto &v2) {
    return v1[order] < v2[order];
  });

  std::vector<std::vector<int>> removedVec{baseVecs.end() - slicedInLog2,
                                           baseVecs.end()};

  triton::LinearLayout::BasesT newBases;
  for (const auto &[inDim, inDimBases] : bases) {
    auto &newInDimBases = newBases[inDim];
    for (const auto &inDimBase : inDimBases) {
      auto it = std::find(removedVec.begin(), removedVec.end(), inDimBase);
      if (it != removedVec.end())
        continue;
      newInDimBases.emplace_back(inDimBase);
    }
  }

  llvm::MapVector<StringAttr, int32_t> outDimSizesLog2;
  llvm::SmallVector<std::pair<StringAttr, int32_t>> outDimSizes;
  for (StringAttr outDim : ll.getOutDimNames()) {
    outDimSizesLog2[outDim] = ll.getOutDimSizeLog2(outDim) - slicedInLog2;
  }

  for (auto [outDim, sizeLog2] : outDimSizesLog2) {
    outDimSizes.push_back({outDim, 1 << sizeLog2});
  }

  return triton::LinearLayout(std::move(newBases), outDimSizes,
                              ll.isSurjective());
}

// Infer LinearLayout from the amd::extract_slice
mlir::triton::LinearLayout
inferLinearLayoutFromConcat(MLIRContext *ctx,
                            mlir::triton::LinearLayout layout) {
  return layout;
}

} // namespace mlir::LLVM::AMD
