#include "../lib/Dialect/TritonGPU/Transforms/WSUtility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace {

using namespace mlir;
using namespace mlir::triton;
struct MakeRangeOpConversion
    : public ConvertOpToLLVMPattern<triton::MakeRangeOp> {
  MakeRangeOpConversion(LLVMTypeConverter &converter,
                        const TargetInfoBase &targetInfo,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::MakeRangeOp>(converter, benefit),
        targetInfo(targetInfo) {}
  LogicalResult
  matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    RankedTensorType ty = op.getType();
    auto shape = ty.getShape();
    auto layout = ty.getEncoding();
    auto elemTy = ty.getElementType();
    assert(elemTy.isInteger(32));
    Value start = createIndexAttrConstant(rewriter, loc, elemTy, op.getStart());
    std::optional<int> warpGroupStart;
    if (!getWarpGroupStart(rewriter.getInsertionBlock())) {
      // If make_range is not in ttng.warp_group, use WAR to compute the correct
      // index range (https://jirasw.nvidia.com/browse/OT-112)
      // make_range needs to be placed in the warp-group where it is being used
      // to ensure the index range is correct. Currently, this is not always
      // done, so we assume it is used in the MMA group and use the respective
      // starting warp.
      auto mod = op->getParentOfType<ModuleOp>();
      auto use_ttg_ws_attr =
          mod->getAttrOfType<BoolAttr>(triton::gpu::AttrUseTtgWsName);
      auto use_ttg_ws = use_ttg_ws_attr ? use_ttg_ws_attr.getValue() : false;
      if (mod->hasAttr(triton::gpu::ATTR_WS_MMA) && !use_ttg_ws) {
        auto attrMma =
            SymbolRefAttr::get(mod->getContext(), triton::gpu::ATTR_WS_MMA);
        auto wsMma = triton::gpu::getGroupFromSymbolRefAttr(mod, attrMma);
        if (wsMma.startWarp > 0)
          warpGroupStart = wsMma.startWarp;
      }
    }
    auto idxs = emitIndices(loc, rewriter, targetInfo, layout, ty, true,
                            warpGroupStart);
    unsigned elems = idxs.size();
    SmallVector<Value> retVals(elems);
    // TODO: slice layout has more elements than expected.
    // Unexpected behavior for make range, but generally OK when followed by
    // expand dims + broadcast. very weird behavior otherwise potentially.
    for (const auto &multiDim : llvm::enumerate(idxs)) {
      assert(multiDim.value().size() == 1);
      retVals[multiDim.index()] = b.add(multiDim.value()[0], start);
    }
    auto typeConverter = getTypeConverter();
    Value result = packLLElements(loc, typeConverter, retVals, rewriter, ty);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::populateMakeRangeOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<MakeRangeOpConversion>(typeConverter, targetInfo, benefit);
}
