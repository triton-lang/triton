#include "triton/Dialect/TritonNvidiaGPU/IR/NvmmaSmemAttrs.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"

#include <cassert>

namespace ttg = mlir::triton::gpu;

namespace mlir::triton::nvidia_gpu {

namespace {

// Builds the core matrix implied by `attrs` and checks that it tiles
// `nvmmaSmemLL`. Returns the core matrix, as a map from the two logical dims to
// offsets, when it does.
std::optional<LinearLayout> matchCoreMatrix(const LinearLayout &nvmmaSmemLL,
                                            unsigned bitwidth,
                                            NvmmaSmemAttrs attrs) {
  auto dims = llvm::to_vector<2>(nvmmaSmemLL.getInDimNames());
  auto *ctx = dims[0].getContext();
  auto offset = StringAttr::get(ctx, "offset");
  auto block = StringAttr::get(ctx, "block");

  auto enc = ttg::NVMMASharedEncodingAttr::get(
      ctx, attrs.swizzlingByteWidth, attrs.transposed, std::max(8u, bitwidth),
      attrs.fp4Padded, ttg::CGAEncodingAttr::get1CTALayout(ctx, /*rank=*/2));
  auto coreMatrixLL =
      ttg::getCoreMatrixLinearLayout(enc, /*disableSwizzle=*/false);
  auto outDims = llvm::to_vector(coreMatrixLL.getOutDims());
  outDims[0].first = dims[0];
  outDims[1].first = dims[1];
  coreMatrixLL = LinearLayout(coreMatrixLL.getBases(), outDims,
                              /*requireSurjective=*/false);
  if (bitwidth == 4)
    coreMatrixLL = LinearLayout::identity1D(2, offset, dims[1]) * coreMatrixLL;
  if (attrs.transposed)
    coreMatrixLL = transposeLinearLayout(coreMatrixLL, {1, 0});
  auto candidateLL = coreMatrixLL.pseudoinvert();
  // Add a trivial block dimension as getReps expects both layouts to
  // have the same outdims
  auto matchLL = candidateLL * LinearLayout::identity1D(1, dims[0], block);
  if (!getReps(nvmmaSmemLL, matchLL).has_value())
    return std::nullopt;
  return candidateLL;
}

struct CoreMatrixGeometry {
  unsigned swizzlingByteWidth;
  bool transposed;
};

// Recovers the core matrix geometry from the two lowest bases of
// `nvmmaSmemLL`, so that only the implied core matrix has to be matched.
//
// getReps compares a tile's leading bases exactly, so those bases are the core
// matrix's own. Inverting getCoreMatrixLinearLayout, the contiguous axis maps
// col 2^k to offset 2^k, and the strided axis maps row 1 to
// tileCols + swizzle(1), which is distinct for every swizzling mode. fp4Padded
// only halves swizzle(1), and only for sw=128, so it is left to matching.
std::optional<CoreMatrixGeometry>
deriveCoreMatrixGeometry(const LinearLayout &nvmmaSmemLL, unsigned bitwidth) {
  auto dims = llvm::to_vector<2>(nvmmaSmemLL.getInDimNames());
  auto *ctx = dims[0].getContext();
  auto offset = StringAttr::get(ctx, "offset");

  auto lowestBasisIsOne = [&](StringAttr dim) {
    return nvmmaSmemLL.getInDimSizeLog2(dim) > 0 &&
           nvmmaSmemLL.getBasis(dim, 0, offset) == 1;
  };
  // Only a degenerate layout lets both axes claim offset 1; prefer the
  // untransposed reading, which matchCoreMatrix then has to confirm.
  int contig =
      lowestBasisIsOne(dims[1]) ? 1 : (lowestBasisIsOne(dims[0]) ? 0 : -1);
  if (contig < 0)
    return std::nullopt;
  auto strided = dims[1 - contig];
  if (nvmmaSmemLL.getInDimSizeLog2(strided) == 0)
    return std::nullopt;

  int64_t stridedBasis = nvmmaSmemLL.getBasis(strided, 0, offset);
  // A 4-bit element type is stored as packed pairs, doubling every offset.
  if (bitwidth == 4) {
    if (stridedBasis % 2 != 0)
      return std::nullopt;
    stridedBasis /= 2;
  }
  bool transposed = contig == 0;
  // In bit units the basis is tileCols * ebw, which is 8 * max(16, swizzling),
  // plus swizzle(1) * ebw, which is zero below sw=128.
  switch (stridedBasis * std::max(8u, bitwidth)) {
  case 8 * 16:
    return CoreMatrixGeometry{0u, transposed};
  case 8 * 32:
    return CoreMatrixGeometry{32u, transposed};
  case 8 * 64:
    return CoreMatrixGeometry{64u, transposed};
  case 8 * 128 + 128: // swizzle(1) is one full vec of 128 bits
  case 8 * 128 + 64:  // ... which the fp4Padded column packing halves
    return CoreMatrixGeometry{128u, transposed};
  default:
    return std::nullopt;
  }
}

} // namespace

std::optional<std::pair<NvmmaSmemAttrs, LinearLayout>>
getNvmmaSmemAttrs(const LinearLayout &nvmmaSmemLL, unsigned bitwidth) {
  assert(nvmmaSmemLL.getNumInDims() == 2);
  assert(nvmmaSmemLL.getNumOutDims() == 2);

  [[maybe_unused]] auto dims = llvm::to_vector<2>(nvmmaSmemLL.getInDimNames());
  [[maybe_unused]] auto *ctx = dims[0].getContext();
  assert(nvmmaSmemLL.hasOutDim(StringAttr::get(ctx, "offset")) &&
         nvmmaSmemLL.hasOutDim(StringAttr::get(ctx, "block")));

  // TODO: sw=0 is only matched for the MMA "core-matrices" operand form. The
  // other sw=0 interpretation (nvmma_shared<sw=0> as a flat TMA destination) is
  // not supported here.
  auto geometry = deriveCoreMatrixGeometry(nvmmaSmemLL, bitwidth);
  if (!geometry)
    return std::nullopt;
  for (bool fp4Padded : {false, true}) {
    NvmmaSmemAttrs attrs{geometry->swizzlingByteWidth, geometry->transposed,
                         fp4Padded};
    if (auto coreMatrix = matchCoreMatrix(nvmmaSmemLL, bitwidth, attrs))
      return std::make_pair(attrs, std::move(*coreMatrix));
  }
  return std::nullopt;
}

std::optional<NvmmaSmemAttrs> getNvmmaSmemAttrs(ttg::MemDescType memTy) {
  if (auto nvmma = dyn_cast<ttg::NVMMASharedEncodingAttr>(memTy.getEncoding()))
    return NvmmaSmemAttrs{nvmma.getSwizzlingByteWidth(), nvmma.getTransposed(),
                          nvmma.getFp4Padded()};

  auto ll = ttg::toLinearLayout(memTy).pseudoinvert();
  unsigned bitwidth = memTy.getElementType().getIntOrFloatBitWidth();
  auto attrsAndCandidate = getNvmmaSmemAttrs(ll, bitwidth);
  if (!attrsAndCandidate)
    return std::nullopt;
  return attrsAndCandidate->first;
}

} // namespace mlir::triton::nvidia_gpu
