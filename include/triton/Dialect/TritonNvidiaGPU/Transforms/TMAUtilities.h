#pragma once
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "llvm/Support/Casting.h"

namespace mlir::triton::nvidia_gpu {

constexpr inline int TMA_SIZE_BYTES = 128;
constexpr inline int TMA_ALIGN = 128;

inline bool isFp4Padded(Attribute encoding) {
  auto mmaEnc = dyn_cast<gpu::NVMMASharedEncodingAttr>(encoding);
  return mmaEnc && mmaEnc.getFp4Padded();
}

template <typename BuilderT>
inline SmallVector<Value> translateTMAIndices(BuilderT &builder, Location loc,
                                              Attribute encoding,
                                              SmallVector<Value> indices) {
  if (isFp4Padded(encoding)) {
    auto two = builder.template create<arith::ConstantIntOp>(loc, 2, 32);
    indices.back() =
        builder.template create<arith::MulIOp>(loc, indices.back(), two);
  }
  return indices;
}

gpu::CTALayoutAttr updateCTALayoutForShape(gpu::CTALayoutAttr ctaLayout,
                                           ArrayRef<int64_t> shape);

gpu::SharedEncodingTrait
updateEncodingForShape(Operation *op, gpu::SharedEncodingTrait encoding,
                       RankedTensorType tensorType);

triton::gpu::SharedEncodingTrait
getEncodingFromDescriptor(Operation *op, RankedTensorType tensorType,
                          Value desc);

int64_t getTMAContigDim(Attribute encoding, ArrayRef<int64_t> shape);

inline int64_t getTMAContigDim(RankedTensorType tensorType) {
  return getTMAContigDim(tensorType.getEncoding(), tensorType.getShape());
}

inline int64_t getTMAContigDim(gpu::MemDescType memDescType) {
  return getTMAContigDim(memDescType.getEncoding(), memDescType.getShape());
}

std::optional<int> getTMASwizzleMode(Operation *op, TensorDescType ty);

std::optional<int> getTMAElementType(Operation *op, TensorDescType ty);

template <typename BuilderT>
mlir::LogicalResult createTMADesc(mlir::Value tmaPtr,
                                  mlir::triton::MakeTensorDescOp op,
                                  BuilderT &builder) {
  using namespace mlir;
  MLIRContext *ctx = op.getContext();
  auto loc = op.getLoc();
  auto mkI32Constant = [&](int32_t val) {
    return builder.template create<arith::ConstantOp>(
        loc, builder.getI32Type(), builder.getI32IntegerAttr(val));
  };

  auto elemType = op.getBase().getType().getPointeeType();
  auto elemSize = elemType.getIntOrFloatBitWidth() / 8;
  auto encoding = op.getType().getBlockType().getEncoding();
  auto mmaEncoding =
      llvm::dyn_cast_or_null<gpu::NVMMASharedEncodingAttr>(encoding);
  bool fp4Padded = mmaEncoding && mmaEncoding.getFp4Padded();

  int paddingScale = fp4Padded ? 2 : 1;
  auto shapePerCTA = gpu::getShapePerCTA(encoding, op.getTensorShape());
  int32_t contig_dim_size = getTMAContigDim(encoding, op.getTensorShape());

  llvm::SmallVector<Value> boxDim;
  if (fp4Padded && contig_dim_size != 128) {
    return op->emitError(
        "FP4 padded loads require 128 elements or more in the last dim");
  }
  boxDim.push_back(mkI32Constant(contig_dim_size));
  for (int k = shapePerCTA.size() - 2; k >= 0; --k)
    boxDim.push_back(mkI32Constant(shapePerCTA[k]));

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

  Value elemSizeVal = builder.template create<arith::ConstantOp>(
      loc, builder.getI64Type(), builder.getI64IntegerAttr(elemSize));

  SmallVector<Value> globalDim(llvm::reverse(op.getShape()));
  SmallVector<Value> globalStride;
  for (int k = op.getStrides().size() - 2; k >= 0; --k) {
    globalStride.push_back(op.getStrides()[k]);
  }

  if (fp4Padded) {
    // Convert number of bytes to number of mxfp4 elements
    globalDim[0] = builder.template create<arith::MulIOp>(loc, globalDim[0],
                                                          mkI32Constant(2));
  }

  SmallVector<Value> elementStride(globalDim.size(), mkI32Constant(1));

  for (int i = 0; i < globalStride.size(); ++i)
    globalStride[i] = builder.template create<arith::MulIOp>(
        loc, globalStride[i], elemSizeVal);

  auto elemTypeEnum = getTMAElementType(op, op.getType());
  if (!elemTypeEnum) {
    return failure();
  }

  builder.template create<triton::ExperimentalTensormapCreateOp>(
      loc,
      /*desc_ptr=*/tmaPtr,
      /*global_address=*/op.getBase(),
      /*box_dim=*/boxDim,
      /*global_dim=*/globalDim,
      /*global_stride=*/globalStride,
      /*element_strides=*/elementStride,
      /*elem_type*/ builder.getI32IntegerAttr(*elemTypeEnum),
      /*interleave_layout*/ builder.getI32IntegerAttr(0),
      /*swizzle_mode=*/builder.getI32IntegerAttr(swizzleMode),
      /*fill_mode=*/builder.getI32IntegerAttr(0));
  return success();
}

} // namespace mlir::triton::nvidia_gpu
