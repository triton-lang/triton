#include "Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

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
} // namespace mlir::LLVM::AMD
