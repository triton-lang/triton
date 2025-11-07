#include "TDMUtility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::LLVM::AMD {

namespace {

// Decode a TDM descriptor from group vectors into
// (base, [shape0, shape1], [stride0, stride1]).
std::tuple<Value, SmallVector<Value>, SmallVector<Value>>
decodeTDMDescriptor(RewriterBase &rewriter, Location loc,
                    ArrayRef<Value> group0, ArrayRef<Value> group1) {
  auto ctx = rewriter.getContext();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type globalPtrTy = ptr_ty(ctx, 1);

  Value globalAddrLow = group0[2];
  Value globalAddrHigh = b.and_(group0[3], b.i32_val(0x7FFFFFFF));
  globalAddrLow = b.zext(i64_ty, globalAddrLow);
  globalAddrHigh = b.shl(b.zext(i64_ty, globalAddrHigh), b.i64_val(32));
  Value globalAddr = b.or_(globalAddrLow, globalAddrHigh);
  Value srcPtr = b.inttoptr(globalPtrTy, globalAddr);

  Value tensorStride0 = group1[5];
  Value tensorStride1 = b.i32_val(1);
  SmallVector<Value> tensorStride = {tensorStride0, tensorStride1};

  Value tensorShape1Low = b.lshr(group1[1], b.i32_val(16));
  Value tensorShape1High = b.shl(group1[2], b.i32_val(16));
  Value tensorShape1 = b.or_(tensorShape1Low, tensorShape1High);
  Value tensorShape0Low = b.lshr(group1[2], b.i32_val(16));
  Value tensorShape0High = b.shl(group1[3], b.i32_val(16));
  Value tensorShape0 = b.or_(tensorShape0Low, tensorShape0High);
  SmallVector<Value> tensorShape = {tensorShape0, tensorShape1};

  return {srcPtr, tensorShape, tensorStride};
}

SmallVector<int> getWarpDistribution(ArrayRef<int64_t> blockShape,
                                     int numWarps) {
  int numWarpsDim0 = numWarps;
  for (; numWarpsDim0 > blockShape[0]; numWarpsDim0 /= 2)
    ;
  int numWarpsDim1 = numWarps / numWarpsDim0;

  assert(numWarpsDim0 > 0 && blockShape[1] % numWarpsDim1 == 0 &&
         "Can't distribute warps in TDM");

  return {numWarpsDim0, numWarpsDim1};
}
} // namespace

std::pair<SmallVector<Value>, SmallVector<Value>>
createTDMDescriptor(RewriterBase &rewriter, Location loc,
                    const LLVMTypeConverter *typeConverter, Type elementType,
                    SmallVector<int64_t> blockShape, int numWarps,
                    unsigned padInterval, unsigned padAmount,
                    SmallVector<Value> tensorShape,
                    SmallVector<Value> tensorStride, Value srcPtr) {
  assert(tensorShape.size() == 2 && tensorStride.size() == 2 &&
         blockShape.size() == 2 && "NYI: TDM > 2D cases.");
  auto ctx = rewriter.getContext();
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  auto elementBitWidth = elementType.getIntOrFloatBitWidth();
  auto elementSizeInBytes = elementBitWidth / 8;

  // Cast strides from i64 to i32
  tensorStride[0] = b.trunc(i32_ty, tensorStride[0]);
  tensorStride[1] = b.trunc(i32_ty, tensorStride[1]);

  // Distribute block among warps
  auto warps = getWarpDistribution(blockShape, numWarps);
  blockShape[0] = ceil(blockShape[0], int64_t(warps[0]));
  blockShape[1] = ceil(blockShape[1], int64_t(warps[1]));

  // group0 (128 bits / 4 dwords) effective bit encoding:
  // [1:0]:     pred (to be filled later)
  // [63:32]:   lds address (to be filled later)
  // [120:64]:  global address
  // [127:126]: type - currently always set to 0x2
  SmallVector<Value> group0(4, b.i32_val(0));
  Value globalAddr = b.ptrtoint(i64_ty, srcPtr);
  group0[2] = b.trunc(i32_ty, globalAddr);
  group0[3] = b.trunc(i32_ty, b.lshr(globalAddr, b.i64_val(32)));
  group0[3] = b.or_(group0[3], b.i32_val(1 << 31));

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
  SmallVector<Value> group1(8, b.i32_val(0));
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

  return {group0, group1};
}

void fillTDMDescriptor(RewriterBase &rewriter, Location loc,
                       const LLVMTypeConverter *typeConverter, Type elementType,
                       SmallVector<int64_t> blockShape, int numWarps,
                       unsigned padInterval, unsigned padAmount,
                       SmallVector<Value> &group0, SmallVector<Value> &group1,
                       SmallVector<Value> offset, Value dstPtr, Value pred) {
  assert(offset.size() == 2 && "NYI: TDM > 2D cases.");
  auto ctx = rewriter.getContext();
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  Type globalPtrTy = ptr_ty(ctx, 1);
  Type sharedPtrTy = ptr_ty(ctx, 3);

  auto [srcPtr, tensorShape, tensorStride] =
      decodeTDMDescriptor(rewriter, loc, group0, group1);

  auto warpId = getLaneAndWarpId(rewriter, loc).second;
  auto warps = getWarpDistribution(blockShape, numWarps);

  // Shift global pointer by offset
  Value warpDim0 = b.i32_val(warps[0]);
  SmallVector<Value, 2> warpCoord = {b.urem(warpId, warpDim0),
                                     b.udiv(warpId, warpDim0)};

  SmallVector<Value, 2> globalOffset;
  for (int i = 0; i < 2; i++) {
    int64_t blockShapePerWarp = ceil(blockShape[i], int64_t(warps[i]));
    globalOffset.push_back(b.mul(b.i32_val(blockShapePerWarp), warpCoord[i]));
    offset[i] = b.add(offset[i], globalOffset[i]);
  }

  Value baseOffset = b.add(b.mul(tensorStride[0], offset[0]),
                           b.mul(tensorStride[1], offset[1]));
  srcPtr = b.gep(globalPtrTy, elementType, srcPtr, baseOffset);

  // Shift shared pointer by offset
  Value dstOffset =
      b.add(b.mul(b.i32_val(blockShape[1]), globalOffset[0]), globalOffset[1]);
  if (padInterval > 0 && padAmount > 0) {
    Value iVal = b.i32_val(log2(padInterval));
    Value pVal = b.i32_val(log2(padAmount));
    Value padOffset = b.shl(i32_ty, b.ashr(dstOffset, iVal), pVal);
    dstOffset = b.add(dstOffset, padOffset);
  }
  dstPtr = b.gep(sharedPtrTy, elementType, dstPtr, dstOffset);

  // Update tensor shape and block shape based on offset
  tensorShape[0] = b.smax(b.i32_val(0), b.sub(tensorShape[0], offset[0]));
  tensorShape[1] = b.smax(b.i32_val(0), b.sub(tensorShape[1], offset[1]));

  // group0 changed fields:
  // [1:0]:     pred
  // [63:32]:   lds address
  // [120:64]:  global address
  Value globalAddr = b.ptrtoint(i64_ty, srcPtr);
  Value ldsAddr = b.ptrtoint(i32_ty, dstPtr);
  group0[0] = b.zext(i32_ty, pred);
  group0[1] = ldsAddr;
  group0[2] = b.trunc(i32_ty, globalAddr);
  group0[3] = b.and_(group0[3], b.i32_val(1 << 31));
  group0[3] =
      b.or_(group0[3], b.trunc(i32_ty, b.lshr(globalAddr, b.i64_val(32))));

  // group1 changed fields:
  // [79:48]:   tensor shape dim inner
  // [111:80]:  tensor shape dim outer
  group1[1] = b.shl(tensorShape[1], b.i32_val(16));
  group1[2] = b.lshr(tensorShape[1], b.i32_val(16));
  group1[2] = b.or_(group1[2], b.shl(tensorShape[0], b.i32_val(16)));
  group1[3] = b.and_(group1[3], b.i32_val(0xFFFF << 16));
  group1[3] = b.or_(group1[3], b.lshr(tensorShape[0], b.i32_val(16)));
}

} // namespace mlir::LLVM::AMD
