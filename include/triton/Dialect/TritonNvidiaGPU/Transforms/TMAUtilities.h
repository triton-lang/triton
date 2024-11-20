#pragma once
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

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

  int32_t contig_dim_size = op.getTensorShape().back();
  int32_t contig_dim_size_in_bytes = contig_dim_size * elemSize;
  if (contig_dim_size_in_bytes > 128) {
    contig_dim_size = 128 / elemSize;
  }
  llvm::SmallVector<Value> boxDim;
  boxDim.push_back(mkI32Constant(contig_dim_size));
  for (int k = op.getTensorShape().size() - 2; k >= 0; --k) {
    boxDim.push_back(mkI32Constant(op.getTensorShape()[k]));
  }

  int32_t swizzle_mode;
  if (contig_dim_size_in_bytes >= 128) {
    swizzle_mode = 3;
  } else if (contig_dim_size_in_bytes == 64) {
    swizzle_mode = 2;
  } else if (contig_dim_size_in_bytes == 32) {
    swizzle_mode = 1;
  } else {
    op->emitError()
        << "contiguous box dimension must be at least 32 bytes but got "
        << contig_dim_size_in_bytes;
    return failure();
  }

  Value elemSizeVal = builder.template create<arith::ConstantOp>(
      loc, builder.getI64Type(), builder.getI64IntegerAttr(elemSize));
  Value globalStride = builder.template create<arith::MulIOp>(
      loc, op.getStrides()[0], elemSizeVal);
  // TODO: Workaround for ptxas bug, remove when we update ptxas
  Value four = builder.template create<arith::ConstantOp>(
      loc, builder.getI64Type(), builder.getI64IntegerAttr(4));
  globalStride =
      builder.template create<arith::ShRSIOp>(loc, globalStride, four);

  int elemTypeEnum;
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

  auto one = mkI32Constant(1);
  builder.template create<triton::ExperimentalTensormapCreateOp>(
      loc,
      /*desc_ptr=*/tmaPtr,
      /*global_address=*/op.getBase(),
      /*box_dim=*/boxDim,
      /*global_dim=*/ValueRange{op.getShape()[1], op.getShape()[0]},
      /*global_stride=*/ValueRange{globalStride},
      /*element_strides=*/ValueRange{one, one},
      /*elem_type*/ builder.getI32IntegerAttr(elemTypeEnum),
      /*interleave_layout*/ builder.getI32IntegerAttr(0),
      /*swizzle_mode=*/builder.getI32IntegerAttr(swizzle_mode),
      /*fill_mode=*/builder.getI32IntegerAttr(0));
  return success();
}

} // namespace mlir::triton::nvidia_gpu
