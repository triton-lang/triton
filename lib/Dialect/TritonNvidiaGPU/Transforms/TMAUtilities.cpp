#include <triton/Dialect/TritonNvidiaGPU/IR/Dialect.h>
#include <triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h>
#include <triton/Tools/LayoutUtils.h>

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton::nvidia_gpu {

SmallVector<Value> translateTMAIndices(OpBuilder &builder, Location loc,
                                       Attribute encoding,
                                       SmallVector<Value> indices) {
  if (isFp4Padded(encoding)) {
    auto two = arith::ConstantIntOp::create(builder, loc, 2, 32);
    indices.back() = arith::MulIOp::create(builder, loc, indices.back(), two);
  }
  return indices;
}

ttg::CGAEncodingAttr updateCGALayoutForShape(ttg::CGAEncodingAttr cgaLayout,
                                             ArrayRef<int64_t> shape) {
  auto rank = shape.size();
  if (cgaLayout.getRank() == rank)
    return cgaLayout;

  auto ctx = cgaLayout.getContext();
  if (cgaLayout.getRank() > rank) {
    auto ll = cgaLayout.getLinearLayout();
    // Broadcast over the first rankDiff dims
    unsigned rankDiff = cgaLayout.getRank() - rank;
    for (int i = 0; i < rankDiff; ++i) {
      ll = removeStandardDim(ll, 0);
    }
    return ttg::CGAEncodingAttr::get(ctx, std::move(ll));
  }
  // For rank-reducing loads, we need to rank-increase the CTA Layout
  auto rankDiff = rank - cgaLayout.getRank();
  for (unsigned i = 0; i < rankDiff; ++i) {
    assert(shape[i] == 1 && "Should only happen for rank-reducing loads");
  }
  auto ll = cgaLayout.getLinearLayout();
  auto kBlock = *ll.getInDimNames().begin();
  auto standardOuts = standardOutDimNames(ctx, rank);
  // Append to front
  for (int i = cgaLayout.getRank(); i < rank; ++i) {
    ll = LinearLayout::identity1D(1, kBlock, standardOuts[i]) * ll;
  }
  // Rename out dims to dim0..dimn-1
  auto dimSizes = ll.getOutDims();
  for (auto [i, dim] : llvm::enumerate(standardOuts)) {
    dimSizes[i].first = dim;
  }
  ll = LinearLayout(ll.getBases(), dimSizes, false);
  return ttg::CGAEncodingAttr::get(ctx, std::move(ll));
}

ttg::SharedEncodingTrait
updateEncodingForShape(Operation *op, ttg::SharedEncodingTrait encoding,
                       RankedTensorType tensorType) {
  auto ctx = encoding.getContext();
  auto cgaLayout = ttg::getCGALayout(encoding);
  if (auto nvmmaEnc = dyn_cast<ttg::NVMMASharedEncodingAttr>(encoding)) {
    auto existingCga = nvmmaEnc.getCGALayout();
    if (!existingCga)
      return nvmmaEnc;

    auto newCgaEnc = updateCGALayoutForShape(cgaLayout, tensorType.getShape());
    return ttg::NVMMASharedEncodingAttr::get(
        ctx, nvmmaEnc.getSwizzlingByteWidth(), nvmmaEnc.getTransposed(),
        nvmmaEnc.getElementBitWidth(), nvmmaEnc.getFp4Padded(), newCgaEnc);
  }
  if (auto swizEnc = dyn_cast<ttg::SwizzledSharedEncodingAttr>(encoding)) {
    auto existingCga = swizEnc.getCGALayout();
    if (!existingCga)
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
    auto newCgaEnc = updateCGALayoutForShape(cgaLayout, tensorType.getShape());
    return ttg::SwizzledSharedEncodingAttr::get(
        ctx, swizEnc.getVec(), swizEnc.getPerPhase(), swizEnc.getMaxPhase(),
        order, newCgaEnc);
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

SmallVector<int64_t> getTMABlockShape(ArrayRef<int64_t> shapePerCTA,
                                      int elementBitWidth, int swizzleBytes,
                                      bool fp4Padded, bool isTransposed,
                                      bool packedSize) {
  SmallVector<int64_t> blockShape(shapePerCTA);
  int contigDim = isTransposed ? 0 : blockShape.size() - 1;
  if (fp4Padded) {
    blockShape[contigDim] *= 2;
  }
  // All dimensions must be at most 256
  constexpr int64_t dimMax = 256;
  for (auto &size : blockShape) {
    size = std::min(size, dimMax);
  }
  // Last dim must equal the swizzle byte size
  if (swizzleBytes != 0) {
    auto contigDimSize = (8 * swizzleBytes) / elementBitWidth;
    if (blockShape[contigDim] < contigDimSize) {
      llvm::report_fatal_error("Block shape is too small for the swizzle byte "
                               "size in NVMMA Shared Layout.");
    }
    blockShape[contigDim] = contigDimSize;
  }
  if (fp4Padded && packedSize) {
    blockShape[contigDim] /= 2;
  }
  return blockShape;
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

LogicalResult createTMADesc(Value tmaPtr, MakeTensorDescOp op,
                            OpBuilder &builder) {
  using namespace mlir;
  MLIRContext *ctx = op.getContext();
  auto loc = op.getLoc();
  auto mkI32Constant = [&](int32_t val) {
    return arith::ConstantOp::create(builder, loc, builder.getI32Type(),
                                     builder.getI32IntegerAttr(val));
  };

  auto elemType = op.getBase().getType().getPointeeType();
  auto elemSize = elemType.getIntOrFloatBitWidth() / 8;
  auto encoding = op.getType().getBlockType().getEncoding();
  auto mmaEncoding =
      llvm::dyn_cast_or_null<gpu::NVMMASharedEncodingAttr>(encoding);
  bool fp4Padded = mmaEncoding && mmaEncoding.getFp4Padded();

  int paddingScale = fp4Padded ? 2 : 1;
  auto shapePerCTA = gpu::getShapePerCTA(encoding, op.getTensorShape());
  auto blockShape =
      getTMABlockShape(encoding, shapePerCTA, /*packedSize=*/false);
  auto contigDimSize = blockShape.back();

  llvm::SmallVector<Value> boxDim;
  if (fp4Padded && contigDimSize != 128) {
    return op->emitError(
        "FP4 padded loads require 128 elements or more in the last dim");
  }
  boxDim.push_back(mkI32Constant(contigDimSize));
  for (int k = shapePerCTA.size() - 2; k >= 0; --k)
    boxDim.push_back(mkI32Constant(blockShape[k]));

  unsigned swizzleBytes = mmaEncoding ? mmaEncoding.getSwizzlingByteWidth() : 0;
  if (!mmaEncoding) {
    auto swizzledEnc = dyn_cast<gpu::SwizzledSharedEncodingAttr>(
        op.getType().getBlockType().getEncoding());
    if (!swizzledEnc || swizzledEnc.getVec() != 1 ||
        swizzledEnc.getPerPhase() != 1 || swizzledEnc.getMaxPhase() != 1) {
      op->emitError() << "Unhandled encoding type";
      return failure();
    }
  }

  auto maybeSwizzleMode = getTMASwizzleMode(op, op.getType());
  if (!maybeSwizzleMode)
    return failure();
  auto swizzleMode = *maybeSwizzleMode;

  Value elemSizeVal = arith::ConstantOp::create(
      builder, loc, builder.getI64Type(), builder.getI64IntegerAttr(elemSize));

  SmallVector<Value> globalDim(llvm::reverse(op.getShape()));
  SmallVector<Value> globalStride;
  for (int k = op.getStrides().size() - 2; k >= 0; --k) {
    globalStride.push_back(op.getStrides()[k]);
  }

  if (fp4Padded) {
    // Convert number of bytes to number of mxfp4 elements
    globalDim[0] =
        arith::MulIOp::create(builder, loc, globalDim[0], mkI32Constant(2));
  }

  SmallVector<Value> elementStride(globalDim.size(), mkI32Constant(1));

  for (int i = 0; i < globalStride.size(); ++i)
    globalStride[i] =
        arith::MulIOp::create(builder, loc, globalStride[i], elemSizeVal);

  auto elemTypeEnum = getTMAElementType(op, op.getType());
  if (!elemTypeEnum) {
    return failure();
  }

  auto fillMode = (op.getPadding() == triton::PaddingOption::PAD_NAN) ? 1 : 0;

  TensormapCreateOp::create(
      builder, loc,
      /*desc_ptr=*/tmaPtr,
      /*global_address=*/op.getBase(),
      /*box_dim=*/boxDim,
      /*global_dim=*/globalDim,
      /*global_stride=*/globalStride,
      /*element_strides=*/elementStride,
      /*elem_type*/ builder.getI32IntegerAttr(*elemTypeEnum),
      /*interleave_layout*/ builder.getI32IntegerAttr(0),
      /*swizzle_mode=*/builder.getI32IntegerAttr(swizzleMode),
      /*fill_mode=*/builder.getI32IntegerAttr(fillMode));
  return success();
}

} // namespace mlir::triton::nvidia_gpu
