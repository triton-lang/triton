#include "mlir/Pass/Pass.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"

namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;

#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUCONVERTTOTRANSPOSELOADS
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

class ConvertToTransposeLoadsPattern : public OpRewritePattern<tt::LoadOp> {
public:
  ConvertToTransposeLoadsPattern(MLIRContext *context, PatternBenefit benefit)
      : OpRewritePattern(context, benefit) {}

  LogicalResult matchAndRewrite(tt::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    // We check tt.load --> ttg.convert_layout chains if they introduce any
    // shared memory transposing and replace the global_load with
    // amdgpu.global_load_transpose to improve the efficiency of the
    // convert_layout.

    // Check if the tensor loaded by this tt.load can support global_load_tr
    if (loadOp.getMask())
      return failure();
    auto dataType = dyn_cast<RankedTensorType>(loadOp.getType());
    if (!dataType)
      return failure();
    auto elementType = dataType.getElementType();
    if (!elementType.isIntOrFloat() ||
        elementType.getIntOrFloatBitWidth() != 16)
      return failure();
    auto shape = dataType.getShape();
    // Each global_load_tr loads four 8x8 blocks, so we cannot load tensors with
    // smaller dimensions
    if (shape.size() != 2 || shape[0] % 8 != 0 || shape[1] % 8 != 0 ||
        shape[0] * shape[1] < 256)
      return failure();

    // find a single convert_layout user of this tensor
    if (!loadOp->hasOneUse())
      return failure();
    ttg::ConvertLayoutOp convertOp =
        dyn_cast<ttg::ConvertLayoutOp>(*loadOp->getUsers().begin());
    if (!convertOp)
      return failure();

    // Found a convert_layout. Check the order of input and output layout if it
    // makes sense to use transposed global loads.
    auto convertResType = dyn_cast<RankedTensorType>(convertOp.getType());
    auto dataEncoding =
        dyn_cast<ttg::BlockedEncodingAttr>(dataType.getEncoding());
    if (!convertResType || !dataEncoding)
      return failure();
    auto convertResEncoding = ttg::LinearEncodingAttr::get(
        loadOp.getContext(), ttg::toLinearLayout(convertResType));
    if (dataEncoding.getOrder()[0] == convertResEncoding.getOrder()[0] &&
        dataEncoding.getOrder()[1] == convertResEncoding.getOrder()[1])
      // Order is not changed by convert_layout -> No need to transpose
      return failure();

    // Found a suboptimal load that would require shared memory transposing
    // Replace the tt.load with amdgpu.global_load_transpose and insert
    // convert_layout operations for the new layout
    auto [layoutAddr, layoutData] =
        ttg::chooseGlobalLoadTrLayout(dataEncoding, convertResType.getShape());
    auto transposedLoadEncodingAddr =
        ttg::LinearEncodingAttr::get(dataEncoding.getContext(), layoutAddr);
    auto transposedLoadEncodingData =
        ttg::LinearEncodingAttr::get(dataEncoding.getContext(), layoutData);
    auto newAddrType = cast<RankedTensorType>(loadOp.getPtr().getType())
                           .cloneWithEncoding(transposedLoadEncodingAddr);
    auto newDataType = dataType.cloneWithEncoding(transposedLoadEncodingData);

    rewriter.setInsertionPoint(loadOp);
    auto newPtr = rewriter.create<ttg::ConvertLayoutOp>(
        loadOp->getLoc(), newAddrType, loadOp.getPtr());
    auto loadTrOp = rewriter.create<tt::amdgpu::GlobalLoadTransposeOp>(
        loadOp->getLoc(), newDataType, newPtr);
    rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(loadOp, dataType,
                                                      loadTrOp);
    return success();
  }
};

} // anonymous namespace

struct TritonAMDGPUConvertToTransposeLoadsPass
    : public impl::TritonAMDGPUConvertToTransposeLoadsBase<
          TritonAMDGPUConvertToTransposeLoadsPass> {

public:
  void runOnOperation() override {
    tt::FuncOp f = getOperation();

    auto ctx = f.getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ConvertToTransposeLoadsPattern>(ctx, /*benefit=*/1);
    walkAndApplyPatterns(f, std::move(patterns));
  }
};

} // namespace mlir
