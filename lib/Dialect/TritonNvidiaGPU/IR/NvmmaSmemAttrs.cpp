#include "triton/Dialect/TritonNvidiaGPU/IR/NvmmaSmemAttrs.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"

#include <cassert>

namespace ttg = mlir::triton::gpu;

namespace mlir::triton::nvidia_gpu {

std::optional<std::pair<NvmmaSmemAttrs, LinearLayout>>
getNvmmaSmemAttrs(const LinearLayout &nvmmaSmemLL, unsigned bitwidth) {
  assert(nvmmaSmemLL.getNumInDims() == 2);
  assert(nvmmaSmemLL.getNumOutDims() == 2);

  auto dims = llvm::to_vector<2>(nvmmaSmemLL.getInDimNames());
  auto *ctx = dims[0].getContext();
  auto offset = StringAttr::get(ctx, "offset");
  auto block = StringAttr::get(ctx, "block");
  assert(nvmmaSmemLL.hasOutDim(offset) && nvmmaSmemLL.hasOutDim(block));

  // TODO: sw=0 is only matched for the MMA "core-matrices" operand form. The
  // other sw=0 interpretation (nvmma_shared<sw=0> as a flat TMA destination) is
  // not supported here.
  for (bool fp4Padded : {false, true}) {
    for (unsigned swizzling : {0u, 32u, 64u, 128u}) {
      for (bool transposed : {false, true}) {
        auto enc = ttg::NVMMASharedEncodingAttr::get(
            ctx, swizzling, transposed, std::max(8u, bitwidth), fp4Padded,
            ttg::CGAEncodingAttr::get1CTALayout(ctx, /*rank=*/2));
        auto coreMatrixLL =
            ttg::getCoreMatrixLinearLayout(enc, /*disableSwizzle=*/false);
        auto outDims = llvm::to_vector(coreMatrixLL.getOutDims());
        outDims[0].first = dims[0];
        outDims[1].first = dims[1];
        coreMatrixLL = LinearLayout(coreMatrixLL.getBases(), outDims,
                                    /*requireSurjective=*/false);
        if (bitwidth == 4)
          coreMatrixLL =
              LinearLayout::identity1D(2, offset, dims[1]) * coreMatrixLL;
        if (transposed)
          coreMatrixLL = transposeLinearLayout(coreMatrixLL, {1, 0});
        auto candidateLL = coreMatrixLL.pseudoinvert();
        // Add a trivial block dimension as getReps expects both layouts to
        // have the same outdims
        auto matchLL =
            candidateLL * LinearLayout::identity1D(1, dims[0], block);
        if (getReps(nvmmaSmemLL, matchLL).has_value())
          return std::make_pair(
              NvmmaSmemAttrs{swizzling, transposed, fp4Padded},
              std::move(candidateLL));
      }
    }
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
