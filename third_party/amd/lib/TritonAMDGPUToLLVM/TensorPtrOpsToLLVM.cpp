#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TDMUtility.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {
Attribute findEncodingFromUsers(Operation *op) {
  Attribute sharedEnc;

  for (auto use : op->getUsers()) {
    Attribute userEnc;
    if (auto load = llvm::dyn_cast<amdgpu::AsyncTDMCopyGlobalToLocalOp>(use)) {
      userEnc = load.getResult().getType().getEncoding();
    } else if (auto store =
                   llvm::dyn_cast<amdgpu::AsyncTDMCopyLocalToGlobalOp>(use)) {
      userEnc = store.getSrc().getType().getEncoding();
    }
    if (!userEnc)
      continue;

    // Assign first encoding found; or error out if different encoding is found
    if (!sharedEnc)
      sharedEnc = userEnc;
    else if (sharedEnc != userEnc) {
      op->emitError("Descriptor is used with different shared encodings.");
      return {};
    }
  }
  return sharedEnc;
}

struct MakeTensorDescOpConversion
    : public ConvertOpToLLVMPattern<triton::MakeTensorDescOp> {
  using ConvertOpToLLVMPattern<
      triton::MakeTensorDescOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::MakeTensorDescOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto basePtr = adaptor.getBase();
    auto tensorShape = adaptor.getShape();
    auto tensorStride = adaptor.getStrides();
    auto result = op.getResult();

    auto tensorDescTy = result.getType();
    auto blockTy = tensorDescTy.getBlockType();
    auto sharedEnc = blockTy.getEncoding();
    if (!sharedEnc) {
      // TODO: add an extra pass to assign layout to descriptors
      sharedEnc = findEncodingFromUsers(op);
      if (!sharedEnc)
        return rewriter.notifyMatchFailure(op, "Descriptor has no layout.");
    }
    auto paddedEnc = llvm::dyn_cast<PaddedSharedEncodingAttr>(sharedEnc);

    unsigned padInterval = 0;
    unsigned padAmount = 0;
    if (paddedEnc) {
      if (paddedEnc.getIntervals().size() != 1 ||
          paddedEnc.getPaddings().size() != 1)
        return rewriter.notifyMatchFailure(
            op, "NYI: Multiple interval-padding pairs in TDM.");
      padInterval = paddedEnc.getIntervals()[0];
      padAmount = paddedEnc.getPaddings()[0];
    }

    Type elementType =
        getTypeConverter()->convertType(blockTy.getElementType());
    SmallVector<int64_t> blockShape = to_vector(blockTy.getShape());
    int numWarps = lookupNumWarps(op);
    auto shapePerCTA = triton::gpu::getShapePerCTA(sharedEnc, blockShape);

    // Create TDM descriptor for 2D-5D tensors
    auto tdmDesc = LLVM::AMD::createTDMDescriptor(
        rewriter, loc, getTypeConverter(), elementType, shapePerCTA, numWarps,
        padInterval, padAmount, tensorShape, tensorStride, basePtr);

    SmallVector<Value> groups = tdmDesc.getAllGroups();

    auto desc =
        packLLElements(loc, getTypeConverter(), groups, rewriter, tensorDescTy);

    rewriter.replaceOp(op, desc);
    return success();
  }
};
} // namespace

void mlir::triton::AMD::populateTensorPtrOpsToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<MakeTensorDescOpConversion>(typeConverter, benefit);
  return;
}
