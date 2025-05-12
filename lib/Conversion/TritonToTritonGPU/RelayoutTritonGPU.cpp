#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir::triton {
#define GEN_PASS_DEF_RELAYOUTTRITONGPU
#include "triton/Conversion/TritonToTritonGPU/Passes.h.inc"
} // namespace mlir::triton

namespace {

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
namespace ttng = triton::nvidia_gpu;

// Given a tensor and its representation in tensor memory, determine its
// distributed layout.
RankedTensorType getTMEMTensorLayout(const TypeConverter *tc,
                                     RankedTensorType type, MemDescType memdesc,
                                     unsigned numWarps) {
  Attribute encoding;
  type = cast<RankedTensorType>(tc->convertType(type));
  if (isa<ttng::TensorMemoryScalesEncodingAttr>(memdesc.getEncoding())) {
    encoding = LinearEncodingAttr::get(
        type.getContext(), getScaleTMEMStoreLinearLayout(type, numWarps));
  } else {
    auto tmemEnc = cast<ttng::TensorMemoryEncodingAttr>(memdesc.getEncoding());
    encoding = ttng::getTmemCompatibleLayout(
        tmemEnc.getBlockM(), tmemEnc.getBlockN(), type, numWarps);
  }
  return RankedTensorType::get(type.getShape(), type.getElementType(),
                               encoding);
}

struct TMEMLoadOpPattern : public OpConversionPattern<ttng::TMEMLoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttng::TMEMLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType type = getTMEMTensorLayout(
        typeConverter, op.getType(), op.getSrc().getType(), lookupNumWarps(op));
    rewriter.modifyOpInPlace(op, [&] { op.getResult().setType(type); });
    Type resultType = getTypeConverter()->convertType(op.getType());
    rewriter.setInsertionPointAfter(op);
    auto cvt = rewriter.create<ConvertLayoutOp>(op.getLoc(), resultType,
                                                op.getResult());
    rewriter.replaceAllUsesExcept(op.getResult(), cvt, cvt);
    return success();
  }
};

struct TMEMStoreOpPattern : public OpConversionPattern<ttng::TMEMStoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttng::TMEMStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType type =
        getTMEMTensorLayout(typeConverter, op.getSrc().getType(),
                            op.getDst().getType(), lookupNumWarps(op));
    Value src =
        rewriter.create<ConvertLayoutOp>(op.getLoc(), type, adaptor.getSrc());
    rewriter.modifyOpInPlace(op, [&] { op.getSrcMutable().assign(src); });
    return success();
  }
};

struct TMEMAllocOpPattern : public OpConversionPattern<ttng::TMEMAllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttng::TMEMAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getSrc())
      return success();
    RankedTensorType type = getTMEMTensorLayout(
        typeConverter, op.getSrc().getType(), op.getType(), lookupNumWarps(op));
    Value src =
        rewriter.create<ConvertLayoutOp>(op.getLoc(), type, adaptor.getSrc());
    rewriter.modifyOpInPlace(op, [&] { op.getSrcMutable().assign(src); });
    return success();
  }
};

class RelayoutTritonGPU
    : public triton::impl::RelayoutTritonGPUBase<RelayoutTritonGPU> {
public:
  using RelayoutTritonGPUBase::RelayoutTritonGPUBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    int numWarps = lookupNumWarps(mod);
    int threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(mod);
    int numCTAs = TritonGPUDialect::getNumCTAs(mod);

    // type converter
    TritonGPUTypeConverter typeConverter(context, numWarps, threadsPerWarp,
                                         numCTAs, /*enableSourceRemat=*/true);
    TritonGPUConversionTarget target(*context, typeConverter);
    target.addDynamicallyLegalDialect<ttng::TritonNvidiaGPUDialect>(
        [&](Operation *op) {
          return TritonGPUConversionTarget::isDynamicallyLegal(op,
                                                               typeConverter);
        });

    // rewrite patterns
    RewritePatternSet patterns(context);
    // add rules
    patterns.insert<
        // clang-format off
        GatherScatterOpPattern<ttng::AsyncTMAGatherOp>,
        GatherScatterOpPattern<ttng::AsyncTMAScatterOp>,
        TMEMLoadOpPattern,
        TMEMStoreOpPattern,
        TMEMAllocOpPattern
        // clang-format on
        >(typeConverter, context);

    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
