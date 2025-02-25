#pragma once
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "llvm/Support/Casting.h"

namespace mlir::triton::nvidia_gpu {

constexpr inline int TMA_SIZE_BYTES = 128;
constexpr inline int TMA_ALIGN = 128;

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
  auto mmaEncoding = llvm::dyn_cast_or_null<gpu::NVMMASharedEncodingAttr>(
      op.getType().getBlockType().getEncoding());
  bool fp4Padded = mmaEncoding && mmaEncoding.getFp4Padded();

  int32_t contig_dim_size = op.getTensorShape().back();
  int32_t contig_dim_size_in_bytes = contig_dim_size * elemSize;
  if (contig_dim_size_in_bytes > 128) {
    contig_dim_size = 128 / elemSize;
  }
  llvm::SmallVector<Value> boxDim;
  if (fp4Padded) {
    boxDim.push_back(mkI32Constant(128));
  } else {
    boxDim.push_back(mkI32Constant(contig_dim_size));
  }
  for (int k = op.getTensorShape().size() - 2; k >= 0; --k) {
    boxDim.push_back(mkI32Constant(op.getTensorShape()[k]));
  }

  unsigned swizzleBytes = 0;
  if (mmaEncoding) {
    swizzleBytes = mmaEncoding.getSwizzlingByteWidth();
    if (fp4Padded) {
      assert(swizzleBytes == 128 &&
             "elem type .b4x16_p64 supports only 128B swizzling");
    }
  } else {
    op->emitError() << "Unhandled encoding type";
    return failure();
  }

  int32_t swizzle_mode;
  if (swizzleBytes == 128) {
    swizzle_mode = 3;
  } else if (swizzleBytes == 64) {
    swizzle_mode = 2;
  } else if (swizzleBytes == 32) {
    swizzle_mode = 1;
  }

  Value elemSizeVal = builder.template create<arith::ConstantOp>(
      loc, builder.getI64Type(), builder.getI64IntegerAttr(elemSize));

  SmallVector<Value> globalDim(llvm::reverse(op.getShape()));
  SmallVector<Value> globalStride;
  for (int k = op.getStrides().size() - 2; k >= 0; --k) {
    globalStride.push_back(op.getStrides()[k]);
  }

  SmallVector<Value> elementStride(globalDim.size(), mkI32Constant(1));

  for (int i = 0; i < globalStride.size(); ++i)
    globalStride[i] = builder.template create<arith::MulIOp>(
        loc, globalStride[i], elemSizeVal);

  int elemTypeEnum;

  if (fp4Padded) {
    elemTypeEnum = 14; // .b4x16_p64
  } else {
    switch (elemSize) {
    case 1: {
      elemTypeEnum = 0;
      break;
    }
    case 2: {
      elemTypeEnum = 1;
      break;
    }
    case 4: {
      elemTypeEnum = 2;
      break;
    }
    default: {
      op->emitError()
          << "Tensor descriptor element type must have size 1, 2, or 4 but got "
          << elemSize;
      return failure();
    }
    }
  }

  builder.template create<triton::ExperimentalTensormapCreateOp>(
      loc,
      /*desc_ptr=*/tmaPtr,
      /*global_address=*/op.getBase(),
      /*box_dim=*/boxDim,
      /*global_dim=*/globalDim,
      /*global_stride=*/globalStride,
      /*element_strides=*/elementStride,
      /*elem_type*/ builder.getI32IntegerAttr(elemTypeEnum),
      /*interleave_layout*/ builder.getI32IntegerAttr(0),
      /*swizzle_mode=*/builder.getI32IntegerAttr(swizzle_mode),
      /*fill_mode=*/builder.getI32IntegerAttr(0));
  return success();
}

} // namespace mlir::triton::nvidia_gpu
