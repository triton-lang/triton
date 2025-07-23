#include "mlir/Pass/Pass.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"

namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUCONVERTTOTRANSPOSELOADS
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

static void loadTranspose(tt::LoadOp loadOp) {
  IRRewriter b(loadOp);
  OpResult loadedData = loadOp->getResult(0);
  // We only use this instruction for load<#blocked> -> convert_layout ->
  // dot_op<#wmma> chains for now
  if (loadOp.getMask())
    return;
  auto loadedTensorType = dyn_cast<RankedTensorType>(loadedData.getType());
  if (!loadedTensorType || !loadedTensorType.getElementType().isFloat(16) ||
      !isa<triton::gpu::BlockedEncodingAttr>(loadedTensorType.getEncoding()))
    return;
  auto shape = loadedTensorType.getShape();
  if (shape.size() != 2 || shape[0] % 16 != 0 || shape[1] % 16 != 0)
    return;
  auto convertLayout =
      dyn_cast<ttg::ConvertLayoutOp>(*loadedData.getUsers().begin());
  if (!convertLayout || !loadedData.hasOneUse())
    return;

  auto convertLayoutResultType =
      dyn_cast<RankedTensorType>(convertLayout->getResult(0).getType());
  if (!convertLayoutResultType)
    return;
  triton::gpu::DotOperandEncodingAttr resultLayout =
      dyn_cast<triton::gpu::DotOperandEncodingAttr>(
          convertLayoutResultType.getEncoding());
  if (!resultLayout)
    return;
  ttg::AMDWmmaEncodingAttr wmmaEncoding =
      dyn_cast<triton::gpu::AMDWmmaEncodingAttr>(resultLayout.getParent());
  if (!wmmaEncoding || wmmaEncoding.getVersion() != 2)
    return;

  auto loadedEncoding =
      cast<triton::gpu::BlockedEncodingAttr>(loadedTensorType.getEncoding());
  if (resultLayout.getOpIdx() == 0 && loadedEncoding.getOrder()[0] == 1 ||
      resultLayout.getOpIdx() == 1 && loadedEncoding.getOrder()[0] == 0)
    // k is the contiguous dimension -> No need to transpose
    return;

  // Found a suboptimal load that would require shared memory transposing
  auto wmmaLayoutBases =
      wmmaEncoding.toLinearLayout(convertLayoutResultType.getShape())
          .getBases();
  auto [layoutAddr, layoutData] = triton::gpu::chooseGlobalLoadTrLayout(
      resultLayout, convertLayoutResultType.getShape());
  // Eight lanes load an 8x8 block of values and transpose them before storing
  // into registers. For details on this instruction and layout, see 11.6.2. in
  // https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna4-instruction-set-architecture.pdf
  // We use a simplified version here that is more similar to the wmma layout
  // used in triton
  auto transposedLoadEncodingAddr = triton::gpu::LinearEncodingAttr::get(
      loadedEncoding.getContext(), layoutAddr);
  auto transposedLoadEncodingData = triton::gpu::LinearEncodingAttr::get(
      loadedEncoding.getContext(), layoutData);
  auto newAddrType = cast<RankedTensorType>(loadOp.getPtr().getType())
                         .cloneWithEncoding(transposedLoadEncodingAddr);
  auto newLoadedTensorType =
      loadedTensorType.cloneWithEncoding(transposedLoadEncodingData);
  // We first need to convert the addresses to the wmma format
  b.setInsertionPoint(loadOp);
  auto newPtr = b.create<ttg::ConvertLayoutOp>(loadOp->getLoc(), newAddrType,
                                               loadOp.getPtr());
  b.replaceOpWithNewOp<triton::amdgpu::GlobalLoadTransposeOp>(
      loadOp, newLoadedTensorType, newPtr);
}

} // anonymous namespace

struct TritonAMDGPUConvertToTransposeLoadsPass
    : public impl::TritonAMDGPUConvertToTransposeLoadsBase<
          TritonAMDGPUConvertToTransposeLoadsPass> {
public:
  using impl::TritonAMDGPUConvertToTransposeLoadsBase<
      TritonAMDGPUConvertToTransposeLoadsPass>::
      TritonAMDGPUConvertToTransposeLoadsBase;

  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();

    SmallVector<tt::LoadOp> loadOps;
    moduleOp.walk([&](tt::LoadOp loadOp) { loadOps.push_back(loadOp); });

    for (auto loadOp : loadOps)
      loadTranspose(loadOp);
  }
};

} // namespace mlir
