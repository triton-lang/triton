#include "TDMUtility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include <optional>

// Include shared C-compatible TDM utilities
#include "../../backend/include/TDMCommon.h"

namespace mlir::LLVM::AMD {
namespace {

// Helper to encode a 48-bit value: 32 bits in first word, 16 bits in second
// word
static void encode48BitValue(RewriterBase &rewriter, TritonLLVMOpBuilder &b,
                             Value value, SmallVector<Value> &group,
                             int startIdx) {
  // Lower 32 bits go into the first word
  group[startIdx] = b.trunc(i32_ty, value);
  // Upper 16 bits go into the lower 16 bits of the second word
  Value upperBits = b.trunc(i32_ty, b.lshr(value, b.i32_val(32)));
  group[startIdx + 1] =
      b.or_(group[startIdx + 1], b.and_(upperBits, b.i32_val(0xFFFF)));
}

// Helper to decode a value spanning two 32-bit words
static Value decode48BitValue(RewriterBase &rewriter, TritonLLVMOpBuilder &b,
                              ArrayRef<Value> group, int startIdx) {
  Value low = b.lshr(group[startIdx], b.i32_val(16));
  Value high = b.shl(group[startIdx + 1], b.i32_val(16));
  return b.or_(low, high);
}

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

  Value tensorShape1 = decode48BitValue(rewriter, b, group1, 1);
  Value tensorShape0 = decode48BitValue(rewriter, b, group1, 2);
  SmallVector<Value> tensorShape = {tensorShape0, tensorShape1};

  return {srcPtr, tensorShape, tensorStride};
}

// C++ wrapper for the shared tdmGetWarpDistribution function
SmallVector<int> getWarpDistribution(ArrayRef<int64_t> blockShape,
                                     int numWarps) {
  int numDims = blockShape.size();
  SmallVector<int> warps(numDims);
  tdmGetWarpDistribution(blockShape.data(), numDims, numWarps, warps.data());

  // Verify the distribution is valid
  int totalWarps = 1;
  for (int i = 0; i < numDims; ++i) {
    totalWarps *= warps[i];
    assert(blockShape[i] % warps[i] == 0 &&
           "Block shape must be divisible by warp distribution");
  }
  assert(totalWarps == numWarps && "Warp distribution mismatch");

  return warps;
}
} // namespace

SmallVector<Value> TDMDescriptor::getAllGroups() const {
  SmallVector<Value> result;
  llvm::append_range(result, group0);
  llvm::append_range(result, group1);
  if (group2.has_value()) {
    llvm::append_range(result, group2.value());
  }
  if (group3.has_value()) {
    llvm::append_range(result, group3.value());
  }
  return result;
}

// Decode a full TDM descriptor from all 4 group vectors for 3D-5D tensors
// Returns (base, tensorShape[], tensorStride[], blockShape[])
std::tuple<Value, SmallVector<Value>, SmallVector<Value>, SmallVector<Value>>
decodeTDMDescriptorFull(RewriterBase &rewriter, Location loc,
                        ArrayRef<Value> group0, ArrayRef<Value> group1,
                        std::optional<ArrayRef<Value>> group2,
                        std::optional<ArrayRef<Value>> group3, size_t numDims) {
  auto ctx = rewriter.getContext();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type globalPtrTy = ptr_ty(ctx, 1);

  // Decode base address from group0
  Value globalAddrLow = group0[2];
  Value globalAddrHigh = b.and_(group0[3], b.i32_val(0x7FFFFFFF));
  globalAddrLow = b.zext(i64_ty, globalAddrLow);
  globalAddrHigh = b.shl(b.zext(i64_ty, globalAddrHigh), b.i64_val(32));
  Value globalAddr = b.or_(globalAddrLow, globalAddrHigh);
  Value srcPtr = b.inttoptr(globalPtrTy, globalAddr);

  SmallVector<Value> tensorShape(numDims);
  SmallVector<Value> tensorStride(numDims);
  SmallVector<Value> blockShape(numDims);

  // Decode dimensions from the end (inner dimensions first)
  tensorShape[numDims - 1] = decode48BitValue(rewriter, b, group1, 1);

  if (numDims >= 2) {
    tensorShape[numDims - 2] = decode48BitValue(rewriter, b, group1, 2);

    // Strides are loaded in opposite order of shapes
    // tensor_dim0_stride from group1[5]
    tensorStride[numDims - 2] = group1[5];

    // tensor_dim1_stride is encoded in group1[6] (48-bit value across group1[6]
    // and group1[7])
    if (numDims >= 3) {
      Value stride1Low =
          b.and_(b.lshr(group1[6], b.i32_val(16)), b.i32_val(0xFFFF));
      Value stride1High = b.and_(group1[7], b.i32_val(0xFFFF));
      tensorStride[numDims - 3] =
          b.or_(stride1Low, b.shl(stride1High, b.i32_val(16)));
    }
  }

  // tensor_dim2_stride from group2[2]
  if (numDims >= 4) {
    tensorStride[numDims - 4] = group2.value()[2];
  }

  // tensor_dim3_stride from group3[0]
  if (numDims == 5) {
    tensorStride[numDims - 5] = group3.value()[0];
  }

  // The innermost dimension always has stride 1
  tensorStride[numDims - 1] = b.i32_val(1);

  // Block shapes from group1
  blockShape[numDims - 1] =
      b.and_(b.lshr(group1[3], b.i32_val(16)), b.i32_val(0xFFFF));
  if (numDims >= 2) {
    blockShape[numDims - 2] = b.and_(group1[4], b.i32_val(0xFFFF));
  }

  // 3rd dimension from group2 if present
  if (numDims >= 3) {
    tensorShape[numDims - 3] = group2.value()[0];
    blockShape[numDims - 3] =
        b.and_(b.lshr(group1[4], b.i32_val(16)), b.i32_val(0xFFFF));
  }

  // 4th dimension from group2/group3 if present
  if (numDims >= 4) {
    tensorShape[numDims - 4] = group2.value()[1];
    blockShape[numDims - 4] =
        b.and_(b.lshr(group2.value()[3], b.i32_val(16)), b.i32_val(0xFFFF));
  }

  // 5th dimension from group3 if present
  if (numDims == 5) {
    // tensor_dim4 is encoded across group3[1] and group3[2]
    Value tensorDim4Low =
        b.and_(b.lshr(group3.value()[1], b.i32_val(16)), b.i32_val(0xFFFF));
    Value tensorDim4High = b.and_(group3.value()[2], b.i32_val(0xFFFF));
    tensorShape[0] = b.or_(tensorDim4Low, b.shl(tensorDim4High, b.i32_val(16)));

    blockShape[0] =
        b.and_(b.lshr(group3.value()[2], b.i32_val(16)), b.i32_val(0xFFFF));
  }

  return {srcPtr, tensorShape, tensorStride, blockShape};
}

TDMDescriptor createTDMDescriptor(RewriterBase &rewriter, Location loc,
                                  const LLVMTypeConverter *typeConverter,
                                  Type elementType,
                                  SmallVector<int64_t> blockShape, int numWarps,
                                  unsigned padInterval, unsigned padAmount,
                                  SmallVector<Value> tensorShape,
                                  SmallVector<Value> tensorStride,
                                  Value srcPtr) {
  size_t numDims = tensorShape.size();
  assert(numDims >= 1 && numDims <= 5 && tensorStride.size() == numDims &&
         "TDM only supported for 1D-5D tensors.");
  assert(blockShape.size() == tensorStride.size() &&
         blockShape.size() == numDims &&
         "Block/tensor/stride dim count must all be equal.");
  auto ctx = rewriter.getContext();
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  // Define common values for better readability
  Value v16 = b.i32_val(16);
  Value v32 = b.i64_val(32);
  Value mask16 = b.i32_val(0xFFFF);
  Value mask31 = b.i32_val(0x7FFFFFFF);

  auto elementBitWidth = elementType.getIntOrFloatBitWidth();
  auto elementSizeInBytes = elementBitWidth / 8;

  // Cast strides from i64 to i32
  for (size_t i = 0; i < numDims; ++i) {
    tensorStride[i] = b.trunc(i32_ty, tensorStride[i]);
  }

  // Distribute block among warps
  {
    int64_t blkShapePerWarp[5];
    tdmGetAdjustedBlockShape(blockShape.data(), numDims, numWarps,
                             &blkShapePerWarp[0]);
    blockShape.assign(blkShapePerWarp, blkShapePerWarp + blockShape.size());
  }

  // group0 (128 bits / 4 dwords) effective bit encoding:
  // [1:0]:     pred (to be filled later)
  // [63:32]:   lds address (to be filled later)
  // [120:64]:  global address
  // [127:126]: type - currently always set to 0x2
  SmallVector<Value> group0(4, b.i32_val(0));
  Value globalAddr = b.ptrtoint(i64_ty, srcPtr);
  group0[2] = b.trunc(i32_ty, globalAddr);
  group0[3] = b.trunc(i32_ty, b.lshr(globalAddr, v32));
  group0[3] = b.or_(group0[3], b.i32_val(1 << 31));

  /* group1 bit-field definition:

    NOTE that in this chart
    - {tensor|tile}-dim0 for means innermost dimension.
    - stride-dim0 refers to the stride of the 2nd innermost dimension.
      FIXME: Is the stride for innermost dimension always 1, and hence no
      need to set in the descriptor

    ================================================================
     dword | dword     | bit-size | field
           | -bit-ofst |
     ------------------------------------------------
      0      0          16         multicast mask
             16         2          data size - log2(element size in bytes)
             18         1          atomic barrier enable
             19         1          iterate enable
             20         1          pad enable
             22         3          pad interval
                                   (log2(pad interval in dwords) - 1)
             25         7          pad amount - pad amount in dwords - 1
                                   (pad amount in dwords - 1)
     ---------------------------------------------------------
     1       0          16         atomic barrier address
             16         16         tensor_dim0 (low-16-bit)
     --------------------------------------------------------
     2       0           16        tensor_dim0 (high-16-bit)
             16          16        tensor_dim1 (low-16-bit)
     ----------------------------------------------------------
     3       0           16        tensor_dim1 (high-16-bit)
             16          16        tile_dim0
     -------------------------------------------------------
     4       0           16        tile_dim1
             16          16        tile_dim2
     -------------------------------------------------------
     5       0           32        tensor_dim0_stride(low-32-bit)
     -------------------------------------------------------
     6       0           16        tensor_dim0_stride(high-16-bit)
            16           16        tensor_dim1_stride(low-16-bit)
     -------------------------------------------------------------
     7       0           32        tensor_dim1_stride(high-16-bit)
     ================================================================
  */
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
  // Encode tensor shapes using 48-bit encoding
  group1[1] = b.shl(tensorShape[numDims - 1], v16);
  group1[2] = b.lshr(tensorShape[numDims - 1], v16);

  if (numDims >= 2) {
    group1[2] = b.or_(group1[2], b.shl(tensorShape[numDims - 2], v16));
    group1[3] = b.lshr(tensorShape[numDims - 2], v16);
  }

  // Block shapes
  group1[3] = b.or_(group1[3], b.i32_val(blockShape[numDims - 1] << 16));
  if (numDims >= 2) {
    group1[4] = b.i32_val(blockShape[numDims - 2]);
  }
  // tile_dim2 (upper 16 bits of group1[4])
  if (numDims >= 3) {
    group1[4] = b.or_(group1[4], b.i32_val(blockShape[numDims - 3] << 16));
  }

  // Handle strides
  if (numDims >= 2) {
    group1[5] = tensorStride[numDims - 2];
    if (numDims >= 3) {
      group1[6] = b.or_(group1[6], b.shl(tensorStride[numDims - 3], v16));
      group1[7] = b.lshr(tensorStride[numDims - 3], v16);
    }
  }

  if (numDims <= 2) {
    return TDMDescriptor{group0, group1, std::nullopt, std::nullopt};
  }

  // For 3D-5D tensors, fill group2 and group3
  // group2 (128 bits / 4 dwords) effective bit encoding:
  // [31:0]:    tensor_dim2 (3rd dimension from the end)
  // [63:32]:   tensor_dim3 (4th dimension from the end) (or lds_addr_increment
  // if iterate_enable) [111:64]:  tensor_dim2_stride (or global_addr_increment
  // if iterate_enable) [127:112]: tile_dim3 (or iterate_count if
  // iterate_enable)
  SmallVector<Value> group2(4, b.i32_val(0));
  if (numDims >= 3) {
    // tensor_dim2 (3rd dimension from the end)
    group2[0] = tensorShape[numDims - 3];

    // tensor_dim3 (4th dimension from the end)
    if (numDims >= 4) {
      group2[1] = tensorShape[numDims - 4];
      // tensor_dim2_stride (48 bits: lower 32 bits in group2[2], upper 16 bits
      // in group2[3])
      group2[2] = tensorStride[numDims - 4];
    }

    // tile_dim3 (upper 16 bits of group2[3])
    if (numDims >= 4) {
      group2[3] =
          b.or_(group2[3], b.shl(b.i32_val(blockShape[numDims - 4]), v16));
    }
  }

  /* group3 bit-field definition
    ================================================================
     dword | dword     | bit-size | field
           | -bit-ofst |
     ---------------------------------------------------------------
         0           0          32 tensor_dim3_stride LSB-32
         1           0          16 tensor_dim3_stride MSB-16
                    16          16 tensor_dim4 LSB-16
         2          00          16 tensor_dim4 MSB-16
                    16          16 tile_dim4
         3           0          32 reserved
    ================================================================
  */
  SmallVector<Value> group3(4, b.i32_val(0));
  if (numDims >= 4) {

    // tensor_dim4 (5th dimension from the end) (32 bits starting at bit 48:
    // upper 16 bits of group3[1] and lower 16 bits of group3[2])
    if (numDims == 5) {
      // Lower 16 bits go into upper 16 bits of group3[1]
      group3[1] = b.or_(group3[1], b.shl(tensorShape[numDims - 5], v16));
      // Upper 16 bits go into lower 16 bits of group3[2]
      group3[2] = b.or_(group3[2], b.lshr(tensorShape[numDims - 5], v16));
    }

    // tile_dim4 (16 bits starting at bit 80: upper 16 bits of group3[2])
    if (numDims == 5) {
      // tensor_dim3_stride (4th dimension from the end) (48 bits split across
      // group3[0] and lower 16 bits of group3[1])
      group3[0] = tensorStride[numDims - 5];
      group3[2] =
          b.or_(group3[2], b.shl(b.i32_val(blockShape[numDims - 5]), v16));
    }
  }

  return TDMDescriptor{group0, group1, group2, group3};
}

void fillTDMDescriptor(
    RewriterBase &rewriter, Location loc,
    const LLVMTypeConverter *typeConverter, Type elementType,
    SmallVector<int64_t> blockShape, int numWarps, unsigned padInterval,
    unsigned padAmount, SmallVector<Value> &group0, SmallVector<Value> &group1,
    std::optional<std::reference_wrapper<SmallVector<Value>>> group2,
    std::optional<std::reference_wrapper<SmallVector<Value>>> group3,
    SmallVector<Value> offset, Value dstPtr, Value pred, Value multicastMask,
    Value barrierPtr, const triton::LinearLayout &cgaLayout, Value ctaId) {
  size_t numDims = offset.size();
  assert(numDims >= 1 && numDims <= 5 && "TDM supports 1D to 5D tensors.");

  auto ctx = rewriter.getContext();
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  Type globalPtrTy = ptr_ty(ctx, 1);
  Type sharedPtrTy = ptr_ty(ctx, 3);

  // Decode the full TDM descriptor to get all values
  auto [srcPtr, tensorShape, tensorStride, decodedBlockShape] =
      decodeTDMDescriptorFull(
          rewriter, loc, group0, group1,
          group2.has_value()
              ? std::optional<ArrayRef<Value>>(group2.value().get())
              : std::nullopt,
          group3.has_value()
              ? std::optional<ArrayRef<Value>>(group3.value().get())
              : std::nullopt,
          numDims);

  auto warpId = getLaneAndWarpId(rewriter, loc).second;
  auto warps = getWarpDistribution(blockShape, numWarps);

  // Compute warp coordinates for each dimension
  SmallVector<Value> warpCoord(numDims);
  Value remainingId = warpId;
  for (size_t i = 0; i < numDims - 1; ++i) {
    warpCoord[i] = b.urem(remainingId, b.i32_val(warps[i]));
    remainingId = b.udiv(remainingId, b.i32_val(warps[i]));
  }
  // Last dimension gets the remaining warp id
  warpCoord[numDims - 1] = remainingId;

  // Apply warp offsets to each dimension
  SmallVector<Value> globalOffset(numDims);
  for (size_t i = 0; i < numDims; ++i) {
    int64_t blockShapePerWarp = ceil(blockShape[i], int64_t(warps[i]));
    globalOffset[i] = b.mul(b.i32_val(blockShapePerWarp), warpCoord[i]);
    offset[i] = b.add(offset[i], globalOffset[i]);
  }

  // We need to adjust the outer strides based on our CTAId and the block layout
  auto kBlock = str_attr("block");
  auto cgaOffsets =
      applyLinearLayout(loc, rewriter, cgaLayout, {{kBlock, ctaId}});
  // Apply CTA offsets to the base pointer
  // Compute the global address offset: sum(ctaOffsets[i] * tensorStride[i])
  Value cgaBaseOffset = b.i32_val(0);
  for (size_t i = 0; i < numDims; ++i) {
    Value dimOffset = b.mul(cgaOffsets[i].second, tensorStride[i]);
    cgaBaseOffset = b.add(cgaBaseOffset, dimOffset);
  }
  srcPtr = b.gep(globalPtrTy, elementType, srcPtr, cgaBaseOffset);

  // Calculate the full global address offset based on all dimensions
  Value baseOffset = b.i32_val(0);
  for (size_t i = 0; i < numDims; ++i) {
    Value dimOffset = b.mul(offset[i], tensorStride[i]);
    baseOffset = b.add(baseOffset, dimOffset);
  }

  srcPtr = b.gep(globalPtrTy, elementType, srcPtr, baseOffset);

  // Calculate shared memory offset using row-major layout
  Value dstOffset = b.i32_val(0);
  Value dstStride = b.i32_val(1);

  // Calculate offset from right to left
  for (int i = numDims - 1; i >= 0; --i) {
    Value dimOffset = b.mul(globalOffset[i], dstStride);
    dstOffset = b.add(dstOffset, dimOffset);
    if (i > 0) {
      dstStride = b.mul(dstStride, b.i32_val(blockShape[i]));
    }
  }

  // Apply padding if needed
  if (padInterval > 0 && padAmount > 0) {
    Value iVal = b.i32_val(log2(padInterval));
    Value pVal = b.i32_val(log2(padAmount));
    Value padOffset = b.shl(i32_ty, b.ashr(dstOffset, iVal), pVal);
    dstOffset = b.add(dstOffset, padOffset);
  }
  dstPtr = b.gep(sharedPtrTy, elementType, dstPtr, dstOffset);

  // Update tensor shapes based on offset
  for (size_t i = 0; i < numDims; ++i) {
    tensorShape[i] = b.smax(b.i32_val(0), b.sub(tensorShape[i], offset[i]));
  }

  Value globalAddr = b.ptrtoint(i64_ty, srcPtr);
  Value ldsAddr = b.ptrtoint(i32_ty, dstPtr);
  group0[0] = b.zext(i32_ty, pred);
  group0[1] = ldsAddr;
  group0[2] = b.trunc(i32_ty, globalAddr);
  group0[3] = b.and_(group0[3], b.i32_val(1 << 31));
  group0[3] =
      b.or_(group0[3], b.trunc(i32_ty, b.lshr(globalAddr, b.i64_val(32))));

  if (multicastMask)
    group1[0] = b.or_(group1[0], multicastMask);
  // Update groups with adjusted tensor shapes
  group1[1] = b.shl(tensorShape[numDims - 1], b.i32_val(16));
  group1[2] = b.lshr(tensorShape[numDims - 1], b.i32_val(16));

  if (numDims >= 2) {
    group1[2] =
        b.or_(group1[2], b.shl(tensorShape[numDims - 2], b.i32_val(16)));
    group1[3] = b.and_(group1[3], b.i32_val(0xFFFF << 16));
    group1[3] =
        b.or_(group1[3], b.lshr(tensorShape[numDims - 2], b.i32_val(16)));
  }

  if (barrierPtr) {
    group1[0] = b.or_(group1[0], b.shl(b.i32_val(1), b.i32_val(18)));
    group1[1] = b.or_(
        group1[1], b.and_(b.lshr(b.ptrtoint(i32_ty, barrierPtr), b.i32_val(3)),
                          b.i32_val(0x00FFFF)));
  } else {
    // Disable atomic_barrier_enable in case it was set before
    group1[0] = b.and_(group1[0], b.i32_val(0xFFFBFFFF));
  }

  if (numDims >= 3) {
    group2.value().get()[0] = tensorShape[numDims - 3];
  }

  if (numDims >= 4) {
    group2.value().get()[1] = tensorShape[numDims - 4];
  }

  if (numDims == 5) {
    group3.value().get()[1] =
        b.and_(group3.value().get()[1], b.i32_val(0xFFFF));
    group3.value().get()[1] =
        b.or_(group3.value().get()[1], b.shl(tensorShape[0], b.i32_val(16)));
    group3.value().get()[2] =
        b.and_(group3.value().get()[2], b.i32_val(0xFFFF << 16));
    group3.value().get()[2] =
        b.or_(group3.value().get()[2], b.lshr(tensorShape[0], b.i32_val(16)));
  }
}

// Helper function to handle TDM operations for both load and store
void emitTDMOperation(RewriterBase &rewriter, Location loc,
                      const LLVMTypeConverter *typeConverter,
                      ArrayRef<Value> desc, ArrayRef<int64_t> blockShape,
                      int numWarps, unsigned padInterval, unsigned padAmount,
                      ArrayRef<Value> offset, Value dstPtr, Value pred,
                      Value multicastMask, Type elementType, Value barrierPtr,
                      bool isLoad, const triton::LinearLayout &cgaLayout,
                      Value ctaId) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  assert(blockShape.size() <= 5);

  if (blockShape.size() > 2) {
    // Use full variant for >2D tensors
    auto group0Vec = SmallVector<Value>(desc.begin(), desc.begin() + 4);
    auto group1Vec = SmallVector<Value>(desc.begin() + 4, desc.begin() + 12);
    auto group2Vec = SmallVector<Value>(desc.begin() + 12, desc.begin() + 16);
    auto group3Vec = SmallVector<Value>(desc.begin() + 16, desc.end());

    fillTDMDescriptor(rewriter, loc, typeConverter, elementType,
                      to_vector(blockShape), numWarps, padInterval, padAmount,
                      group0Vec, group1Vec, std::ref(group2Vec),
                      std::ref(group3Vec), to_vector(offset), dstPtr, pred,
                      multicastMask, barrierPtr, cgaLayout, ctaId);

    auto group0 = packLLVector(loc, group0Vec, rewriter);
    auto group1 = packLLVector(loc, group1Vec, rewriter);
    auto group2 = packLLVector(loc, group2Vec, rewriter);
    auto group3 = packLLVector(loc, group3Vec, rewriter);

    const char *intrinsicName = isLoad ? "llvm.amdgcn.tensor.load.to.lds"
                                       : "llvm.amdgcn.tensor.store.from.lds";
    LLVM::createLLVMIntrinsicCallOp(
        rewriter, loc, intrinsicName, {},
        {group0, group1, group2, group3, b.i32_val(0)});
  } else {
    // Use d2 variant for 1D-2D tensors
    auto group0Vec = SmallVector<Value>(desc.begin(), desc.begin() + 4);
    auto group1Vec = SmallVector<Value>(desc.begin() + 4, desc.end());

    fillTDMDescriptor(rewriter, loc, typeConverter, elementType,
                      to_vector(blockShape), numWarps, padInterval, padAmount,
                      group0Vec, group1Vec, std::nullopt, std::nullopt,
                      to_vector(offset), dstPtr, pred, multicastMask,
                      barrierPtr, cgaLayout, ctaId);

    auto group0 = packLLVector(loc, group0Vec, rewriter);
    auto group1 = packLLVector(loc, group1Vec, rewriter);

    const char *intrinsicName = isLoad ? "llvm.amdgcn.tensor.load.to.lds.d2"
                                       : "llvm.amdgcn.tensor.store.from.lds.d2";
    LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsicName, {},
                                    {group0, group1, b.i32_val(0)});
  }
}

} // namespace mlir::LLVM::AMD
