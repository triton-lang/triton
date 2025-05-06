#include <triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h>

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton::nvidia_gpu {

ttg::CTALayoutAttr updateCTALayoutForShape(ttg::CTALayoutAttr ctaLayout,
                                           ArrayRef<int64_t> shape) {
  auto rank = shape.size();
  if (ctaLayout.getRank() == rank)
    return ctaLayout;

  auto ctx = ctaLayout.getContext();
  if (ctaLayout.getRank() > rank) {
    unsigned rankDiff = ctaLayout.getRank() - rank;
    return ttg::CTALayoutAttr::get(
        ctx, ctaLayout.getCTAsPerCGA().drop_front(rankDiff),
        ctaLayout.getCTASplitNum().drop_front(rankDiff),
        ctaLayout.getCTAOrder().drop_front(rankDiff));
  }
  // For rank-reducing loads, we need to rank-increase the CTA Layout
  auto rankDiff = rank - ctaLayout.getRank();
  for (unsigned i = 0; i < rankDiff; ++i) {
    assert(shape[i] == 1 && "Should only happen for rank-reducing loads");
  }
  SmallVector<unsigned> CTAsPerCGA(rank, 1);
  SmallVector<unsigned> CTASplitNum(rank, 1);
  SmallVector<unsigned> CTAOrder(rank, 1);

  llvm::copy(ctaLayout.getCTAsPerCGA(), CTAsPerCGA.begin() + rankDiff);
  llvm::copy(ctaLayout.getCTASplitNum(), CTASplitNum.begin() + rankDiff);
  for (unsigned i = 0; i < rankDiff; ++i) {
    CTAOrder[i] = rank - i;
  }
  llvm::copy(ctaLayout.getCTAOrder(), CTAOrder.begin() + rankDiff);
  return ttg::CTALayoutAttr::get(ctx, CTAsPerCGA, CTASplitNum, CTAOrder);
}

ttg::SharedEncodingTrait
updateEncodingForShape(Operation *op, ttg::SharedEncodingTrait encoding,
                       RankedTensorType tensorType) {
  auto ctx = encoding.getContext();
  auto ctaLayout = ttg::getCTALayout(encoding);
  if (auto nvmmaEnc = dyn_cast<ttg::NVMMASharedEncodingAttr>(encoding)) {
    auto existingCta = nvmmaEnc.getCTALayout();
    if (!existingCta)
      return nvmmaEnc;

    auto newCtaEnc = updateCTALayoutForShape(ctaLayout, tensorType.getShape());
    return ttg::NVMMASharedEncodingAttr::get(
        ctx, nvmmaEnc.getSwizzlingByteWidth(), nvmmaEnc.getTransposed(),
        nvmmaEnc.getElementBitWidth(), nvmmaEnc.getFp4Padded(), newCtaEnc);
  }
  if (auto swizEnc = dyn_cast<ttg::SwizzledSharedEncodingAttr>(encoding)) {
    auto existingCta = swizEnc.getCTALayout();
    if (!existingCta)
      return swizEnc;

    auto rank = tensorType.getRank();
    auto oldOrder = swizEnc.getOrder();
    SmallVector<unsigned> order;
    for (int i = 0; i + oldOrder.size() < rank; ++i)
      order.push_back(rank - i - 1);
    for (int i = 0; i < oldOrder.size(); ++i) {
      // If it is a rank-reducing load, we need to drop the last dimensions.
      if (oldOrder[i] >= rank)
        continue;
      order.push_back(oldOrder[i]);
    }
    auto newCtaEnc = updateCTALayoutForShape(ctaLayout, tensorType.getShape());
    return ttg::SwizzledSharedEncodingAttr::get(
        ctx, swizEnc.getVec(), swizEnc.getPerPhase(), swizEnc.getMaxPhase(),
        order, newCtaEnc);
  }

  constexpr auto msg = "Internal Error: Unhandled tensor descriptor encoding";
  if (op)
    op->emitError() << msg;
  llvm::report_fatal_error(msg);
}

ttg::SharedEncodingTrait getEncodingFromDescriptor(Operation *op,
                                                   RankedTensorType tensorType,
                                                   Value desc) {
  auto descBlockType = cast<TensorDescType>(desc.getType()).getBlockType();
  Attribute encoding = descBlockType.getEncoding();
  if (!encoding) {
    constexpr auto msg =
        "Internal Error: Tensor descriptor should have encoding set";
    if (op)
      op->emitError() << msg;
    llvm::report_fatal_error(msg);
  }
  auto sharedEnc = cast<ttg::SharedEncodingTrait>(encoding);
  if (descBlockType.getShape() == tensorType.getShape())
    return sharedEnc;

  return updateEncodingForShape(op, sharedEnc, tensorType);
}

int64_t getTMAContigDim(Attribute encoding, ArrayRef<int64_t> shape) {
  assert(encoding);
  auto mmaEncoding =
      llvm::dyn_cast_or_null<ttg::NVMMASharedEncodingAttr>(encoding);

  // The bounding box inner dimension must be less than or equal to the
  // swizzle size.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7
  // We clamp the block size and the codegen will emit multiple copy
  // operations.
  if (mmaEncoding && mmaEncoding.getSwizzlingByteWidth() != 0) {
    auto elemSize = mmaEncoding.getElementBitWidth() / 8;
    return mmaEncoding.getSwizzlingByteWidth() / elemSize;
  }

  auto shapePerCTA = ttg::getShapePerCTA(encoding, shape);
  return shapePerCTA.back();
}

std::optional<int> getTMASwizzleMode(Operation *op, TensorDescType ty) {
  auto encoding = ty.getBlockType().getEncoding();
  auto mmaEncoding = dyn_cast<ttg::NVMMASharedEncodingAttr>(encoding);
  unsigned swizzleBytes = mmaEncoding ? mmaEncoding.getSwizzlingByteWidth() : 0;
  if (!mmaEncoding) {
    auto swizzledEnc = dyn_cast<ttg::SwizzledSharedEncodingAttr>(encoding);
    if (!swizzledEnc || swizzledEnc.getVec() != 1 ||
        swizzledEnc.getPerPhase() != 1 || swizzledEnc.getMaxPhase() != 1) {
      if (op)
        op->emitError("Unhandled encoding type");
      return std::nullopt;
    }
  }

  bool fp4Padded = isFp4Padded(encoding);
  assert(!fp4Padded || swizzleBytes == 128 &&
                           "elem type .b4x16_p64 supports only 128B swizzling");

  int32_t swizzleMode = 0;
  if (swizzleBytes == 128) {
    swizzleMode = 3;
  } else if (swizzleBytes == 64) {
    swizzleMode = 2;
  } else if (swizzleBytes == 32) {
    swizzleMode = 1;
  } else {
    assert(swizzleBytes == 0);
  }
  return swizzleMode;
}

enum TMA_ELEMENT_TYPES {
  TMA_U8 = 0,
  TMA_U16 = 1,
  TMA_U32 = 2,
  TMA_S32 = 3,
  TMA_U64 = 4,
  TMA_S64 = 5,
  TMA_F16 = 6,
  TMA_F32 = 7,
  TMA_F32_FTZ = 8,
  TMA_F64 = 9,
  TMA_BF16 = 10,
  TMA_TF32 = 11,
  TMA_TF32_FTZ = 12,
  TMA_B4X16 = 13,
  TMA_B4X16_P64 = 14,
  TMA_B6X16_P32 = 15,
  TMA_B6P2X16 = 15,
};

std::optional<int> getTMAElementType(Operation *op, TensorDescType ty) {
  auto encoding = ty.getBlockType().getEncoding();
  auto mmaEncoding = dyn_cast<ttg::NVMMASharedEncodingAttr>(encoding);
  bool fp4Padded = isFp4Padded(encoding);

  if (fp4Padded)
    return TMA_B4X16_P64;

  auto elemTy = ty.getBlockType().getElementType();
  if (elemTy.isBF16()) {
    return TMA_BF16;
  } else if (elemTy.isF16()) {
    return TMA_F16;
  } else if (elemTy.isF32()) {
    return TMA_F32;
  } else if (elemTy.isF64()) {
    return TMA_F64;
  }

  auto elemSize = elemTy.getIntOrFloatBitWidth() / 8;
  switch (elemSize) {
  case 1:
    return TMA_U8;
  case 2:
    return TMA_U16;
  case 4:
    return elemTy.isSignedInteger() ? TMA_S32 : TMA_U32;
  case 8:
    return elemTy.isSignedInteger() ? TMA_S64 : TMA_U64;
  default:
    break;
  }
  if (op) {
    op->emitError()
        << "Tensor descriptor element type must have size 1, 2, or 4 but got "
        << elemSize;
  }
  return std::nullopt;
}

} // namespace mlir::triton::nvidia_gpu
