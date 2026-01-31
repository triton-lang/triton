#include "TDMUtility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
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
  // [30]:      Scatter/gather index size (0=16-bit, 1=32-bit)
  // [31]:      Scatter/gather enable (0=disabled, 1=enabled)
  // [63:32]:   lds address (to be filled later)
  // [120:64]:  global address
  // [127:126]: type - currently always set to 0x2
  // NOTE: Currently only scatter is implemented; gather (load) is TODO.
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

  /* For 3D-5D tensors, fill group2 and group3
     group2 bit-field definition
    ================================================================
     dword | dword     | bit-size | field
           | -bit-ofst |
     ---------------------------------------------------------------
          0           0         32 tensor_dim2 (3rd inner dimension)
          1           0         32 tensor_dim3 (4th inner dimension)
                                   (or lds_addr_increment if iterate_enable)
          2           0         32 tensor_dim2_stride low-32-bit
                                   (or global_addr_increment low-32-bit
                                   if iterate_enable)
          3           0         16 tensor_dim2_stride high-16-bit
                                   (or global_addr_increment high-16-bit
                                   if iterate_enable)
                     16         16 tile_dim3 (or iterate_count
                                   if iterate_enable)
    ================================================================

     group2 bit-field definition (Gather/Scatter mode, 16-bit indices)
    ================================================================
     dword | dword     | bit-size | field
           | -bit-ofst |
     ---------------------------------------------------------------
          0    0         16        row_index_0
              16         16        row_index_1
          1    0         16        row_index_2
              16         16        row_index_3
          2    0         16        row_index_4
              16         16        row_index_5
          3    0         16        row_index_6
              16         16        row_index_7
    ================================================================

     group2 bit-field definition (Gather/Scatter mode, 32-bit indices)
    ================================================================
     dword | dword     | bit-size | field
           | -bit-ofst |
     ---------------------------------------------------------------
          0    0         32        row_index_0
          1    0         32        row_index_1
          2    0         32        row_index_2
          3    0         32        row_index_3
    ================================================================
  */
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

     group3 bit-field definition (Gather/Scatter mode, 16-bit indices)
    ================================================================
     dword | dword     | bit-size | field
           | -bit-ofst |
     ---------------------------------------------------------------
          0    0         16        row_index_8
              16         16        row_index_9
          1    0         16        row_index_10
              16         16        row_index_11
          2    0         16        row_index_12
              16         16        row_index_13
          3    0         16        row_index_14
              16         16        row_index_15
    ================================================================

     group3 bit-field definition (Gather/Scatter mode, 32-bit indices)
    ================================================================
     dword | dword     | bit-size | field
           | -bit-ofst |
     ---------------------------------------------------------------
          0    0         32        row_index_4
          1    0         32        row_index_5
          2    0         32        row_index_6
          3    0         32        row_index_7
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

// Fill TDM descriptor for regular load/store operations (1D-5D tensors)
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

  // Distribute warps across the block
  auto warpId = getLaneAndWarpId(rewriter, loc).second;
  auto warps = getWarpDistribution(blockShape, numWarps);

  // Compute warp coordinates for each dimension
  SmallVector<Value> warpCoord(numDims);
  Value remainingId = warpId;
  for (size_t i = 0; i < numDims - 1; ++i) {
    warpCoord[i] = b.urem(remainingId, b.i32_val(warps[i]));
    remainingId = b.udiv(remainingId, b.i32_val(warps[i]));
  }
  warpCoord[numDims - 1] = remainingId;

  // Apply warp offsets to each dimension
  SmallVector<Value> globalOffset(numDims);
  for (size_t i = 0; i < numDims; ++i) {
    int64_t blockShapePerWarp = ceil(blockShape[i], int64_t(warps[i]));
    globalOffset[i] = b.mul(b.i32_val(blockShapePerWarp), warpCoord[i]);
    offset[i] = b.add(offset[i], globalOffset[i]);
  }

  // Adjust strides based on CTAId and the block layout
  auto kBlock = str_attr("block");
  auto cgaOffsets =
      applyLinearLayout(loc, rewriter, cgaLayout, {{kBlock, ctaId}});
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

  // Update tensor shapes based on offset and cgaOffset
  for (size_t i = 0; i < numDims; ++i) {
    auto fullOffset = b.add(offset[i], cgaOffsets[i].second);
    tensorShape[i] = b.smax(b.i32_val(0), b.sub(tensorShape[i], fullOffset));
  }

  // Update group0 with addresses
  Value globalAddr = b.ptrtoint(i64_ty, srcPtr);
  Value ldsAddr = b.ptrtoint(i32_ty, dstPtr);
  group0[0] = pred;
  group0[1] = ldsAddr;
  group0[2] = b.trunc(i32_ty, globalAddr);
  group0[3] = b.and_(group0[3], b.i32_val(1 << 31));
  group0[3] =
      b.or_(group0[3], b.trunc(i32_ty, b.lshr(globalAddr, b.i64_val(32))));

  // Update group1 with tensor shapes
  if (multicastMask)
    group1[0] = b.or_(group1[0], multicastMask);
  group1[1] = b.shl(tensorShape[numDims - 1], b.i32_val(16));
  group1[2] = b.lshr(tensorShape[numDims - 1], b.i32_val(16));

  if (numDims >= 2) {
    group1[2] =
        b.or_(group1[2], b.shl(tensorShape[numDims - 2], b.i32_val(16)));
    group1[3] = b.and_(group1[3], b.i32_val(0xFFFF << 16));
    group1[3] =
        b.or_(group1[3], b.lshr(tensorShape[numDims - 2], b.i32_val(16)));
  }

  // Configure barrier
  if (barrierPtr) {
    group1[0] = b.or_(group1[0], b.shl(b.i32_val(1), b.i32_val(18)));
    group1[1] = b.or_(
        group1[1], b.and_(b.lshr(b.ptrtoint(i32_ty, barrierPtr), b.i32_val(3)),
                          b.i32_val(0x00FFFF)));
  } else {
    group1[0] = b.and_(group1[0], b.i32_val(0xFFFBFFFF));
  }

  // Update group2/group3 for higher dimensions
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

// Fill TDM descriptor for scatter operation (2D only).
// Scatter writes data from LDS to non-contiguous rows in global memory.
void fillTDMDescriptorForScatter(
    RewriterBase &rewriter, Location loc,
    const LLVMTypeConverter *typeConverter, Type elementType,
    SmallVector<int64_t> blockShape, SmallVector<Value> &group0,
    SmallVector<Value> &group1, SmallVector<Value> &group2,
    SmallVector<Value> &group3, Value ldsRowOffset, Value globalColOffset,
    Value ldsPtr, Value pred, Value barrierPtr,
    const triton::LinearLayout &cgaLayout, Value ctaId,
    ArrayRef<Value> rowIndices, bool use32BitIndices) {
  assert(!rowIndices.empty() && "Scatter requires row indices.");

  auto ctx = rewriter.getContext();
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  Type globalPtrTy = ptr_ty(ctx, 1);
  Type sharedPtrTy = ptr_ty(ctx, 3);

  // Decode descriptor to get tensor info
  auto [globalPtr, tensorShape, tensorStride, decodedBlockShape] =
      decodeTDMDescriptorFull(rewriter, loc, group0, group1, group2, group3,
                              /*numDims=*/2);

  // Apply CTA offsets to the base pointer
  auto kBlock = str_attr("block");
  auto cgaOffsets =
      applyLinearLayout(loc, rewriter, cgaLayout, {{kBlock, ctaId}});
  Value cgaBaseOffset = b.i32_val(0);
  for (size_t i = 0; i < 2; ++i) {
    Value dimOffset = b.mul(cgaOffsets[i].second, tensorStride[i]);
    cgaBaseOffset = b.add(cgaBaseOffset, dimOffset);
  }
  globalPtr = b.gep(globalPtrTy, elementType, globalPtr, cgaBaseOffset);

  // For scatter, only apply column offset to global address
  // Row positions are specified by rowIndices
  Value colOffset = b.mul(globalColOffset, tensorStride[1]);
  globalPtr = b.gep(globalPtrTy, elementType, globalPtr, colOffset);

  // Calculate LDS offset based on row offset only (column always starts at 0)
  Value ldsOffset = b.mul(ldsRowOffset, b.i32_val(blockShape[1]));
  ldsPtr = b.gep(sharedPtrTy, elementType, ldsPtr, ldsOffset);

  // Update group0 with addresses and enable scatter
  Value globalAddr = b.ptrtoint(i64_ty, globalPtr);
  Value ldsAddr = b.ptrtoint(i32_ty, ldsPtr);

  // Set scatter bits: bit 31 = enable, bit 30 = 32-bit indices
  Value predWithScatter = b.or_(pred, b.i32_val(1 << 31));
  if (use32BitIndices) {
    predWithScatter = b.or_(predWithScatter, b.i32_val(1 << 30));
  }

  group0[0] = predWithScatter;
  group0[1] = ldsAddr;
  group0[2] = b.trunc(i32_ty, globalAddr);

  // group0[3]: preserve type bits, set global_addr upper 25 bits
  Value globalAddrHigh = b.trunc(i32_ty, b.lshr(globalAddr, b.i64_val(32)));
  globalAddrHigh = b.and_(globalAddrHigh, b.i32_val(0x01FFFFFF));
  Value typeBits = b.and_(group0[3], b.i32_val(0xC0000000));
  group0[3] = b.or_(typeBits, globalAddrHigh);

  // Update group1 with tensor shapes (keep original for stride calculation)
  group1[1] = b.shl(tensorShape[1], b.i32_val(16));
  group1[2] = b.lshr(tensorShape[1], b.i32_val(16));
  group1[2] = b.or_(group1[2], b.shl(tensorShape[0], b.i32_val(16)));
  group1[3] = b.and_(group1[3], b.i32_val(0xFFFF << 16));
  group1[3] = b.or_(group1[3], b.lshr(tensorShape[0], b.i32_val(16)));

  // Configure barrier
  if (barrierPtr) {
    group1[0] = b.or_(group1[0], b.shl(b.i32_val(1), b.i32_val(18)));
    group1[1] = b.or_(
        group1[1], b.and_(b.lshr(b.ptrtoint(i32_ty, barrierPtr), b.i32_val(3)),
                          b.i32_val(0x00FFFF)));
  } else {
    group1[0] = b.and_(group1[0], b.i32_val(0xFFFBFFFF));
  }

  // Set tile_dim1 (number of valid indices) in lower 16 bits of group1[4]
  size_t numIndices = rowIndices.size();
  group1[4] = b.and_(group1[4], b.i32_val(0xFFFF0000));
  group1[4] = b.or_(group1[4], b.i32_val(numIndices & 0xFFFF));

  // Fill group2 and group3 with row indices
  if (use32BitIndices) {
    // 32-bit indices: 4 in group2, 4 in group3
    for (size_t i = 0; i < 4 && i < numIndices; ++i) {
      group2[i] = rowIndices[i];
    }
    for (size_t i = 4; i < 8 && i < numIndices; ++i) {
      group3[i - 4] = rowIndices[i];
    }
  } else {
    // 16-bit indices: pack 2 per dword
    // Indices are i16, so zero-extend to i32 before packing
    auto packIndices = [&](MutableArrayRef<Value> group, size_t baseIdx) {
      for (size_t i = 0; i < 4; ++i) {
        Value dword = b.i32_val(0);
        size_t idx0 = baseIdx + i * 2;
        size_t idx1 = baseIdx + i * 2 + 1;
        if (idx0 < numIndices) {
          // Zero-extend i16 to i32, then mask (mask is technically redundant
          // but makes intent clear)
          Value idx0_i32 = b.zext(i32_ty, rowIndices[idx0]);
          dword = b.and_(idx0_i32, b.i32_val(0xFFFF));
        }
        if (idx1 < numIndices) {
          Value idx1_i32 = b.zext(i32_ty, rowIndices[idx1]);
          dword = b.or_(
              dword, b.shl(b.and_(idx1_i32, b.i32_val(0xFFFF)), b.i32_val(16)));
        }
        group[i] = dword;
      }
    };
    packIndices(group2, 0);
    packIndices(group3, 8);
  }
}

// Emit a TDM load or store operation for regular (non-scatter) transfers.
void emitTDMLoadStore(RewriterBase &rewriter, Location loc,
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

// Emit a TDM scatter operation to write non-contiguous rows from LDS to global.
void emitTDMScatter(RewriterBase &rewriter, Location loc,
                    const LLVMTypeConverter *typeConverter,
                    ArrayRef<Value> desc, ArrayRef<int64_t> blockShape,
                    Value srcPtr, Value pred, Type elementType,
                    Value barrierPtr, const triton::LinearLayout &cgaLayout,
                    Value ctaId, ArrayRef<Value> rowIndices, Value colOffset,
                    bool use32BitIndices) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  assert(!rowIndices.empty() && "Scatter requires row indices");
  assert(colOffset && "Scatter requires column offset");

  // Determine max indices per instruction based on index size
  size_t maxIndicesPerInstr = use32BitIndices ? 8 : 16;
  size_t numIndices = rowIndices.size();

  // Get the descriptor groups (scatter uses 2D format: 12 dwords)
  auto group0Vec = SmallVector<Value>(desc.begin(), desc.begin() + 4);
  auto group1Vec = SmallVector<Value>(desc.begin() + 4, desc.end());

  // For TDM scatter, we need group2 and group3 for indices
  SmallVector<Value> group2Vec(4, b.i32_val(0));
  SmallVector<Value> group3Vec(4, b.i32_val(0));

  // Issue multiple TDM instructions if needed
  for (size_t startIdx = 0; startIdx < numIndices;
       startIdx += maxIndicesPerInstr) {
    size_t endIdx = std::min(startIdx + maxIndicesPerInstr, numIndices);

    // Get the subset of indices for this batch
    SmallVector<Value> batchIndices(rowIndices.begin() + startIdx,
                                    rowIndices.begin() + endIdx);

    // Make copies of the descriptor groups for this iteration
    auto g0 = group0Vec;
    auto g1 = group1Vec;
    auto g2 = group2Vec;
    auto g3 = group3Vec;

    // Fill the descriptor for scatter:
    // - ldsRowOffset: row offset within shared memory for this batch
    // - colOffset: starting column in global memory
    fillTDMDescriptorForScatter(
        rewriter, loc, typeConverter, elementType, to_vector(blockShape), g0,
        g1, g2, g3, b.i32_val(startIdx), colOffset, srcPtr, pred, barrierPtr,
        cgaLayout, ctaId, batchIndices, use32BitIndices);

    // Pack and emit the instruction
    auto group0 = packLLVector(loc, g0, rewriter);
    auto group1 = packLLVector(loc, g1, rewriter);
    auto group2 = packLLVector(loc, g2, rewriter);
    auto group3 = packLLVector(loc, g3, rewriter);

    // Scatter uses tensor.store.from.lds (not the d2 variant) because it
    // needs group2/group3 for indices
    LLVM::createLLVMIntrinsicCallOp(
        rewriter, loc, "llvm.amdgcn.tensor.store.from.lds", {},
        {group0, group1, group2, group3, b.i32_val(0)});
  }
}

SmallVector<Value> emitTDMPrefetch(RewriterBase &rewriter, Location loc,
                                   ArrayRef<Value> desc,
                                   ArrayRef<int64_t> blockShape, int numLanes,
                                   int numWarps, int numCTAs,
                                   ArrayRef<Value> offset, Value pred,
                                   Type elementType, Value laneId, Value warpId,
                                   Value ctaId, bool isSpeculative) {
  // TDM prefetch uses the same syntax as a regular load. Each lane can prefetch
  // a different address; hardware aligns to a 256-byte boundary and makes that
  // 256-byte region available in L2. We distribute the nD tile (blockShape)
  // across CTAs, warps, and lanes so the whole tile is covered by prefetches.
  // Speculative prefetches may go out-of-bounds; non-speculative prefetches
  // need bounds checks. We currently only guard based on the whole tensor
  // extent, so some prefetched chunks might never be used if masking trims
  // inner dimensions. To add inner-dimension bounds checks we would need to
  // expose the CTA offsets from the tensor descriptor, which is currenlty
  // directly applied to the base pointer.
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  int numDims = blockShape.size();
  Type globalPtrTy = ptr_ty(loc.getContext(), 1);

  // Decode TDM descriptor to get the base pointer, shape, and strides
  auto group0Vec = SmallVector<Value>(desc.begin(), desc.begin() + 4);
  auto group1Vec = SmallVector<Value>(desc.begin() + 4, desc.begin() + 12);
  std::optional<SmallVector<Value>> group2Vec;
  std::optional<SmallVector<Value>> group3Vec;
  if (numDims > 2) {
    group2Vec = SmallVector<Value>(desc.begin() + 12, desc.begin() + 16);
    group3Vec = SmallVector<Value>(desc.begin() + 16, desc.end());
  }
  auto [basePtr, tensorShape, tensorStride, decodedBlockShape] =
      mlir::LLVM::AMD::decodeTDMDescriptorFull(
          rewriter, loc, group0Vec, group1Vec,
          group2Vec.has_value()
              ? std::optional<ArrayRef<Value>>(group2Vec.value())
              : std::nullopt,
          group3Vec.has_value()
              ? std::optional<ArrayRef<Value>>(group3Vec.value())
              : std::nullopt,
          numDims);

  auto dot64 = [&](ArrayRef<Value> indices, ArrayRef<Value> strides) {
    Value ret = b.i64_val(0);
    for (auto [index, stride] : llvm::zip(indices, strides)) {
      ret = b.add(ret, b.mul(b.zext(i64_ty, index), b.zext(i64_ty, stride)));
    }
    return ret;
  };

  // Apply the passed offsets to the base pointer.
  Value tileOffset = dot64(offset, tensorStride);
  auto tilePtr = b.gep(globalPtrTy, elementType, basePtr, tileOffset);

  // Calculate the total tensor size for bounds checking.
  Value linearTensorSize =
      b.mul(b.zext(i64_ty, tensorShape[0]), b.zext(i64_ty, tensorStride[0]));

  // Calculate maximum allowed offset from tilePtr before going out of bounds
  Value maxOffsetFromTile = b.sub(linearTensorSize, tileOffset);

  // Prefetches 256 bytes into L2
  const int bytesPerPrefetch = 256;
  int elemPerPrefetch =
      (bytesPerPrefetch * 8) / elementType.getIntOrFloatBitWidth();

  // Scale the block shape by the number of elements per prefetch
  SmallVector<int64_t> scaledBlockShape(blockShape.begin(), blockShape.end());
  scaledBlockShape.back() =
      ceil<int64_t>(scaledBlockShape.back(), elemPerPrefetch);

  // Use the default blocked encoding to unroll the TDM tile
  auto blockedEnc = triton::gpu::getDefaultBlockedEncoding(
      loc.getContext(), scaledBlockShape, numWarps, numLanes, numCTAs);
  auto ll = triton::gpu::toLinearLayout(scaledBlockShape, blockedEnc);

  auto kRegister = rewriter.getStringAttr("register");
  auto kLane = rewriter.getStringAttr("lane");
  auto kWarp = rewriter.getStringAttr("warp");
  auto kBlock = rewriter.getStringAttr("block");

  // Adjust the inner stride (always 1) to the number of elements per prefetch
  auto scaledStride = tensorStride;
  scaledStride.back() = b.i32_val(elemPerPrefetch);

  auto baseIndices = applyLinearLayout(loc, rewriter, ll,
                                       {{kRegister, b.i32_val(0)},
                                        {kLane, laneId},
                                        {kWarp, warpId},
                                        {kBlock, ctaId}});
  // Iterate over each register and emit a prefetch intrinsic
  SmallVector<Value> offsets(ll.getInDimSize(kRegister));
  for (int reg = 0; reg < ll.getInDimSize(kRegister); reg++) {
    auto regIndices =
        ll.apply({{kRegister, reg}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});

    // XOR the base indices with the register specific indices
    SmallVector<std::pair<StringAttr, Value>> indices;
    for (auto [base, regIdx] : llvm::zip(baseIndices, regIndices)) {
      assert(base.first == regIdx.first);
      Value combined = b.xor_(base.second, b.i32_val(regIdx.second));
      indices.emplace_back(base.first, combined);
    }

    // Compute the local offset from tile ptr for this prefetch based on the
    // computed indices
    Value localOffset =
        dot64(to_vector(make_second_range(indices)), scaledStride);
    auto prefetchPtr = b.gep(globalPtrTy, elementType, tilePtr, localOffset);

    // Mask the prefetch if the offset is out of bounds
    Value inBounds = b.icmp_slt(localOffset, maxOffsetFromTile);
    // Only predicate based in inBounds for non-speculative prefetches.
    Value combinedPred = isSpeculative ? pred : b.and_(pred, inBounds);

    // Predicate and emit prefetch
    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterPrefetch =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *prefetchBlock = rewriter.createBlock(afterPrefetch);
    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::CondBrOp::create(rewriter, loc, combinedPred, prefetchBlock,
                           afterPrefetch);

    rewriter.setInsertionPointToStart(prefetchBlock);
    int cache_scope = 8; // (8) = L2 scope
    int speculative = isSpeculative;
    int llvmTemporalHint = cache_scope | speculative;
    Value scope = LLVM::ConstantOp::create(
        rewriter, loc, i32_ty, rewriter.getI32IntegerAttr(llvmTemporalHint));
    LLVM::createLLVMIntrinsicCallOp(
        rewriter, loc, "llvm.amdgcn.global.prefetch", {}, {prefetchPtr, scope});

    rewriter.setInsertionPointToEnd(prefetchBlock);
    LLVM::BrOp::create(rewriter, loc, afterPrefetch);
    rewriter.setInsertionPointToStart(afterPrefetch);

    // We return the offsets for unit testing
    offsets[reg] =
        b.select(combinedPred, b.add(localOffset, tileOffset), b.i64_val(0));
  }
  return offsets;
}

} // namespace mlir::LLVM::AMD
