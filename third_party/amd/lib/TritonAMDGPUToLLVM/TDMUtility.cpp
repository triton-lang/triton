#include "TDMUtility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::LLVM::AMD {

std::pair<Value, Value>
createTDMDescriptor(RewriterBase &rewriter, Location loc,
                    const LLVMTypeConverter *typeConverter, Type elementType,
                    SmallVector<int64_t> blockShape,
                    SmallVector<Value> tensorShape,
                    SmallVector<Value> tensorStride, SmallVector<Value> offset,
                    Value srcPtr, Value dstPtr, Value pred, int numWarps,
                    unsigned padInterval, unsigned padAmount) {
  assert(tensorShape.size() == 2 && tensorStride.size() == 2 &&
         blockShape.size() == 2 && offset.size() == 2 &&
         "NYI: TDM > 2D cases.");
  auto ctx = rewriter.getContext();
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  auto elementBitWidth = elementType.getIntOrFloatBitWidth();
  auto elementSizeInBytes = elementBitWidth / 8;

  Type globalPtrTy = ptr_ty(ctx, 1);
  Type sharedPtrTy = ptr_ty(ctx, 3);

  // Cast strides from i64 to i32
  tensorStride[0] = b.trunc(i32_ty, tensorStride[0]);
  tensorStride[1] = b.trunc(i32_ty, tensorStride[1]);

  // For block shape [M, N], each warp will handle shape [M/numWarps, N].
  auto warpId = getLaneAndWarpId(rewriter, loc).second;
  int outerBlockShape = blockShape[0];
  int outerBlockShapePerWarp = ceil(outerBlockShape, numWarps);
  int outerBlockStride = blockShape[1];

  // Shift global pointer by offset
  Value outerOffset = b.mul(b.i32_val(outerBlockShapePerWarp), warpId);
  offset[0] = b.add(offset[0], outerOffset);

  Value baseOffset = b.add(b.mul(tensorStride[0], offset[0]),
                           b.mul(tensorStride[1], offset[1]));
  srcPtr = b.gep(globalPtrTy, elementType, srcPtr, baseOffset);

  // Shift shared pointer by offset
  Value dstOffset = b.mul(b.i32_val(outerBlockStride), outerOffset);
  if (padInterval > 0 && padAmount > 0) {
    Value iVal = b.i32_val(log2(padInterval));
    Value pVal = b.i32_val(log2(padAmount));
    Value padOffset = b.shl(i32_ty, b.ashr(dstOffset, iVal), pVal);
    dstOffset = b.add(dstOffset, padOffset);
  }
  dstPtr = b.gep(sharedPtrTy, elementType, dstPtr, dstOffset);

  // Update tensor shape and block shape based on offset
  Value zero = b.i32_val(0);
  tensorShape[0] = b.smax(zero, b.sub(tensorShape[0], offset[0]));
  tensorShape[1] = b.smax(zero, b.sub(tensorShape[1], offset[1]));

  blockShape[0] = outerBlockShapePerWarp;

  // group0 (128 bits / 4 dwords) effective bit encoding:
  // [1:0]:     pred
  // [63:32]:   lds address
  // [120:64]:  global address
  // [127:126]: type - currently always set to 0x2
  SmallVector<Value, 4> group0(4, b.i32_val(0));
  Value globalAddr = b.ptrtoint(i64_ty, srcPtr);
  Value ldsAddr = b.ptrtoint(i32_ty, dstPtr);
  group0[0] = b.zext(i32_ty, pred);
  group0[1] = ldsAddr;
  group0[2] = b.trunc(i32_ty, globalAddr);
  group0[3] = b.trunc(i32_ty, b.lshr(globalAddr, b.i64_val(32)));
  group0[3] = b.or_(group0[3], b.i32_val(0x80000000));

  VectorType vecTy0 = vec_ty(i32_ty, 4);
  Value group0Vec = b.undef(vecTy0);
  for (unsigned ii = 0; ii < 4; ++ii) {
    Value vecIdx = rewriter.create<LLVM::ConstantOp>(
        loc, typeConverter->getIndexType(), rewriter.getI32IntegerAttr(ii));
    group0Vec = b.insert_element(vecTy0, group0Vec, group0[ii], vecIdx);
  }

  // group1 (256 bits / 8 dwords) effective bit encoding:
  // [15:0]:    multicast mask
  // [17:16]:   data size - log2(element size in bytes)
  // [20]:      enable padding
  // [24:22]:   pad interval - log2(pad interval in dwords) - 1
  // [31:25]:   pad amount - pad amount in dwords - 1
  // [79:48]:   tensor shape dim inner
  // [111:80]:  tensor shape dim outer
  // [127:112]: block shape dim inner
  // [143:128]: block shape dim outer
  // [207:160]: tensor stride dim outer (we only use 32 bits)
  SmallVector<Value, 8> group1(8, b.i32_val(0));
  int32_t dataSize = log2(elementSizeInBytes);
  unsigned dwordSize = 32;
  auto padIntervalInDwords = padInterval * elementBitWidth / dwordSize;
  auto padAmountInDwords = padAmount * elementBitWidth / dwordSize;
  group1[0] = b.or_(group1[0], b.i32_val(dataSize << 16));
  if (padIntervalInDwords > 0 && padAmountInDwords > 0) {
    assert(llvm::isPowerOf2_32(padIntervalInDwords));
    int32_t log2PadInterval = log2(padIntervalInDwords);
    group1[0] = b.or_(group1[0], b.i32_val(1 << 20));
    group1[0] = b.or_(group1[0], b.i32_val((log2PadInterval - 1) << 22));
    group1[0] = b.or_(group1[0], b.i32_val((padAmountInDwords - 1) << 25));
  }
  group1[1] = b.shl(tensorShape[1], b.i32_val(16));
  group1[2] = b.lshr(tensorShape[1], b.i32_val(16));
  group1[2] = b.or_(group1[2], b.shl(tensorShape[0], b.i32_val(16)));
  group1[3] = b.lshr(tensorShape[0], b.i32_val(16));
  group1[3] = b.or_(group1[3], b.i32_val(blockShape[1] << 16));
  group1[4] = b.i32_val(blockShape[0] & 0xFFFF);
  group1[5] = tensorStride[0];

  VectorType vecTy1 = vec_ty(i32_ty, 8);
  Value group1Vec = b.undef(vecTy1);
  for (unsigned ii = 0; ii < 8; ++ii) {
    Value vecIdx = rewriter.create<LLVM::ConstantOp>(
        loc, typeConverter->getIndexType(), rewriter.getIndexAttr(ii));
    group1Vec = b.insert_element(vecTy1, group1Vec, group1[ii], vecIdx);
  }

  return {group0Vec, group1Vec};
}

Value packTensorDesc(RewriterBase &rewriter, Location loc,
                     const LLVMTypeConverter *typeConverter, Value base,
                     ValueRange tensorShape, ValueRange tensorStride,
                     Type resultTy) {
  SmallVector<Value> elems;

  elems.push_back(base);
  llvm::append_range(elems, tensorShape);
  llvm::append_range(elems, tensorStride);
  return packLLElements(loc, typeConverter, elems, rewriter, resultTy);
}

std::tuple<Value, SmallVector<Value>, SmallVector<Value>>
unpackTensorDesc(RewriterBase &rewriter, Location loc, Value desc) {
  SmallVector<Value> descriptorFields = unpackLLElements(loc, desc, rewriter);
  auto length = descriptorFields.size();
  assert(length >= 5 && "invalid tensor descriptor");

  Value base = descriptorFields[0];
  SmallVector<Value> tensorShape;
  SmallVector<Value> tensorStride;
  for (int i = 1; i < (length - 1) / 2 + 1; i++)
    tensorShape.push_back(descriptorFields[i]);
  for (int i = (length - 1) / 2 + 1; i < length; i++)
    tensorStride.push_back(descriptorFields[i]);
  return {base, tensorShape, tensorStride};
}

} // namespace mlir::LLVM::AMD
