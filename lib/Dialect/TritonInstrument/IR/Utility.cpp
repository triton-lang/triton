#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton::gpu;
using namespace mlir::triton::instrument;

namespace mlir::triton::instrument {

TypedValue<RankedTensorType> createConstIntTensor(OpBuilder &builder,
                                                  Location loc, int val,
                                                  RankedTensorType tensorType) {
  auto denseAttr = DenseElementsAttr::get(
      tensorType, APInt(tensorType.getElementType().getIntOrFloatBitWidth(),
                        val, /*isSigned=*/true));
  return cast<TypedValue<RankedTensorType>>(
      builder.create<arith::ConstantOp>(loc, tensorType, denseAttr)
          .getResult());
}

static DistributedEncodingTrait
getSingleDimSliceEncoding(BlockedEncodingAttr encoding, int dim) {
  int rank = encoding.getOrder().size();
  MLIRContext *ctx = encoding.getContext();
  assert(dim < rank && "Expected dim to be less than rank");
  DistributedEncodingTrait sliceEncoding = encoding;
  for (int i = 0; i < rank; ++i) {
    if (i != dim) {
      sliceEncoding = SliceEncodingAttr::get(ctx, i, encoding);
    }
  }
  return sliceEncoding;
}

Value expandOuterSlicedDim(OpBuilder &b, Location loc, Value tensor) {
  auto type = cast<RankedTensorType>(tensor.getType());
  auto sliceEncoding = dyn_cast<SliceEncodingAttr>(type.getEncoding());
  if (sliceEncoding) {
    int dim = sliceEncoding.getDim();
    auto shape = type.getShape();
    auto newShape = SmallVector<int64_t>(shape);
    newShape.insert(newShape.begin() + dim, 1);
    auto newType = RankedTensorType::get(newShape, type.getElementType(),
                                         sliceEncoding.getParent());
    tensor = b.create<ExpandDimsOp>(loc, newType, tensor, dim);
  }
  return tensor;
}

static Value expandAllSlicedDims(OpBuilder &b, Location loc, Value tensor) {
  auto type = cast<RankedTensorType>(tensor.getType());
  auto sliceEncoding = dyn_cast<SliceEncodingAttr>(type.getEncoding());
  while (sliceEncoding) {
    tensor = expandOuterSlicedDim(b, loc, tensor);
    type = cast<RankedTensorType>(tensor.getType());
    sliceEncoding = dyn_cast<SliceEncodingAttr>(type.getEncoding());
  }
  return tensor;
}

static Value createPointerTensor(OpBuilder &b, Location loc, Value base,
                                 RankedTensorType tensorType) {
  auto encoding = cast<BlockedEncodingAttr>(tensorType.getEncoding());
  Value ptrTensor = b.create<SplatOp>(
      loc,
      RankedTensorType::get(tensorType.getShape(), base.getType(), encoding),
      base);
  auto offsetsType =
      RankedTensorType::get(tensorType.getShape(), b.getI32Type(), encoding);
  SmallVector<int> strides(tensorType.getRank());
  strides[0] = 1;
  for (int i = 1; i < tensorType.getRank(); ++i) {
    strides[i] = strides[i - 1] * tensorType.getShape()[i - 1];
  }
  for (int i = 0; i < tensorType.getRank(); ++i) {
    auto partialEncoding = getSingleDimSliceEncoding(encoding, i);
    auto arangeType = RankedTensorType::get({tensorType.getShape()[i]},
                                            b.getI32Type(), partialEncoding);
    auto arange =
        b.create<MakeRangeOp>(loc, arangeType, 0, arangeType.getShape()[0]);
    auto cstStride = createConstIntTensor(b, loc, strides[i], arangeType);
    auto arangeTimesStride =
        b.create<arith::MulIOp>(loc, arangeType, arange, cstStride);
    auto expandDims = expandAllSlicedDims(b, loc, arangeTimesStride);
    if (cast<RankedTensorType>(expandDims.getType()).getShape() !=
        tensorType.getShape()) {
      expandDims = b.create<BroadcastOp>(loc, offsetsType, expandDims);
    }
    ptrTensor =
        b.create<AddPtrOp>(loc, ptrTensor.getType(), ptrTensor, expandDims);
  }
  return ptrTensor;
}

Operation *createStoreScratchMemory(OpBuilder &b, Location loc, Value alloc,
                                    Value tensor, RankedTensorType tensorType) {
  auto ptrTensor = createPointerTensor(b, loc, alloc, tensorType);
  return b.create<StoreOp>(loc, ptrTensor, tensor, CacheModifier::NONE,
                           EvictionPolicy::NORMAL);
}

Operation *createLoadScratchMemory(OpBuilder &b, Location loc, Value alloc,
                                   RankedTensorType tensorType) {
  auto ptrTensor = createPointerTensor(b, loc, alloc, tensorType);
  return b.create<LoadOp>(loc, ptrTensor, CacheModifier::NONE,
                          EvictionPolicy::NORMAL, false);
}

} // namespace mlir::triton::instrument
