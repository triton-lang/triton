#include "TDMUtility.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include <numeric>
#include <optional>

// Include shared C-compatible TDM utilities
#include "../../backend/include/TDMCommon.h"

// Override the inherited DEBUG_TYPE from Utility.h so LLVM_DEBUG output for
// the TDM merge analysis is selectable via `-debug-only=tdm-merge`.
#undef DEBUG_TYPE
#define DEBUG_TYPE "tdm-merge"

namespace mlir::LLVM::AMD {

WarpHintInfo extractWarpHintInfo(uint32_t hint, int numWarps) {
  assert(hint != 0 && "hint must be non-zero (verifier checked)");
  assert(llvm::isPowerOf2_64(numWarps) && "numWarps must be a power of two");

  WarpHintInfo info;
  info.K = static_cast<unsigned>(llvm::popcount(hint));
  assert(llvm::isPowerOf2_32(info.K) &&
         "popcount(hint) must be a power of two");
  // `hint` bit positions are warp INDICES.  i0 is the smallest active
  // warp index (an integer in [0, numWarps)), NOT the bitmask form
  // `hint & -hint` -- the lowering XORs runtime warpId values with this
  // i0 (integer XOR), which only matches the verifier-side coset math
  // when both sides agree on warp-index arithmetic.
  info.i0 = llvm::countr_zero(hint);

  // For each active warp w, compute the basis vector (w XOR i0) as an
  // integer; the OR of these is `support` -- a bitmask whose set bits
  // are the basis bit POSITIONS within the warp index.  The verifier
  // (axis-aligned coset rule) guarantees popcount(support) == log2(K).
  uint32_t support = 0;
  for (uint32_t mask = hint; mask != 0; mask &= mask - 1) {
    unsigned w = llvm::countr_zero(mask);
    support |= static_cast<uint32_t>(w ^ info.i0);
  }
  for (uint32_t s = support; s != 0; s &= s - 1)
    info.basisBits.push_back(static_cast<int32_t>(llvm::countr_zero(s)));
  assert(info.basisBits.size() == llvm::Log2_32(info.K) &&
         "axis-aligned hint must have log2(K) single-bit basis vectors");

  return info;
}

namespace {

// Helper to decode a value spanning two 32-bit words
static Value decode48BitValue(RewriterBase &rewriter, TritonLLVMOpBuilder &b,
                              ArrayRef<Value> group, int startIdx) {
  Value low = b.lshr(group[startIdx], b.i32_val(16));
  Value high = b.shl(group[startIdx + 1], b.i32_val(16));
  return b.or_(low, high);
}

// C++ wrapper for the shared tdmGetWarpDistribution function.
SmallVector<unsigned> distributeTDMWarps(ArrayRef<int64_t> blockShape,
                                         int numWarps) {
  int numDims = blockShape.size();
  SmallVector<int> warps(numDims);
  tdmGetWarpDistribution(blockShape.data(), numDims, numWarps, warps.data());
  return SmallVector<unsigned>(warps.begin(), warps.end());
}

// Compute a warp distribution that respects LDS partition boundaries.
//
// The default distributeTDMWarps may assign a warp a chunk that spans
// multiple partitions along partitionDim. This function ensures each warp's
// chunk stays within a single partition piece.
//
// Algorithm:
//   1. Start with numLogicalPieces warps along partitionDim (one per piece).
//      If numWarps < numLogicalPieces, use all warps and emit multiple TDM
//      instructions.
//   2. Subdivide partition pieces along partitionDim (adding more warps), but
//      ONLY when partitionDim is not the innermost dimension.
//   3. Distribute remaining warps to non-partition, non-inner dimensions.
//
// The innermost dimension (numDims-1) is never subdivided beyond the initial
// partition assignment.  TDM writes data as a linear element stream with LDS
// padding applied every padInterval elements.  If the per-warp innermost
// extent differs from the inner layout's row width, padding boundaries in
// the linear stream won't align with the shared memory row structure.
//
// Returns: (warpsPerCTA, numTDMInstructions)
std::pair<SmallVector<unsigned>, unsigned> distributeTDMWarpsAlignToPartition(
    ArrayRef<int64_t> blockShape, int numWarps,
    PartitionedSharedEncodingAttr partitionedEnc) {
  unsigned numDims = blockShape.size();
  unsigned partitionDim = partitionedEnc.getPartitionDim();
  unsigned numLogicalPieces = partitionedEnc.getNumLogicalPieces();
  int64_t pieceSize =
      blockShape[partitionDim] / static_cast<int64_t>(numLogicalPieces);
  unsigned innerDim = numDims - 1;

  assert(partitionDim < numDims && "partitionDim out of range");
  assert(blockShape[partitionDim] % numLogicalPieces == 0 &&
         "block shape must be divisible by numLogicalPieces");

  unsigned warpsAlongPartition =
      std::gcd(static_cast<unsigned>(numWarps), numLogicalPieces);
  unsigned numTDMInstructions =
      llvm::divideCeil(numLogicalPieces, warpsAlongPartition);

  // Subdivide partition pieces to absorb more warps, but only when
  // partitionDim != innerDim.  Subdividing the innermost dimension would
  // make the per-warp row width narrower than the inner layout row,
  // misaligning TDM's linear padding with the shared memory row structure.
  int remainingWarps = numWarps / warpsAlongPartition;
  if (partitionDim != innerDim) {
    while (remainingWarps >= 2 &&
           pieceSize % ((warpsAlongPartition / numLogicalPieces) * 2) == 0 &&
           static_cast<int64_t>(warpsAlongPartition * 2) <=
               blockShape[partitionDim]) {
      warpsAlongPartition *= 2;
      remainingWarps /= 2;
    }
  }

  SmallVector<unsigned> warps(numDims, 1);
  warps[partitionDim] = warpsAlongPartition;

  // Distribute remaining warps to non-partition, non-inner dimensions.
  for (unsigned i = 0; i < numDims && remainingWarps > 1; ++i) {
    if (i == partitionDim || i == innerDim)
      continue;
    while (remainingWarps > 1 &&
           static_cast<int64_t>(warps[i] * 2) <= blockShape[i]) {
      warps[i] *= 2;
      remainingWarps /= 2;
    }
  }

  return {warps, numTDMInstructions};
}

} // namespace

std::pair<SmallVector<unsigned>, unsigned>
distributeTDMWarpsAlignToPartition(ArrayRef<int64_t> blockShape, int numWarps,
                                   Attribute encoding) {
  if (auto partitionedEnc = dyn_cast<PartitionedSharedEncodingAttr>(encoding))
    return distributeTDMWarpsAlignToPartition(blockShape, numWarps,
                                              partitionedEnc);
  return {distributeTDMWarps(blockShape, numWarps), 1};
}

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
                                  Type elementType, size_t numDims,
                                  unsigned padInterval, unsigned padAmount,
                                  SmallVector<Value> tensorShape,
                                  SmallVector<Value> tensorStride,
                                  Value srcPtr) {
  assert(numDims >= 1 && numDims <= 5 && tensorShape.size() == numDims &&
         tensorStride.size() == numDims &&
         "TDM only supported for 1D-5D tensors.");
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  // Define common values for better readability
  Value v16 = b.i32_val(16);
  Value v32 = b.i64_val(32);

  auto elementBitWidth = elementType.getIntOrFloatBitWidth();
  auto elementSizeInBytes = elementBitWidth / 8;

  // Cast strides from i64 to i32
  for (size_t i = 0; i < numDims; ++i) {
    tensorStride[i] = b.trunc(i32_ty, tensorStride[i]);
  }

  // Per-instruction fields (tile_dim*) are intentionally not written here;
  // they are encoded later by fillTDMDescriptor / fillTDMDescriptorForGather
  // Scatter, which know the warp distribution for the specific copy op.

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
  // Encode 32-bit tensor shapes
  group1[1] = b.shl(tensorShape[numDims - 1], v16);
  group1[2] = b.lshr(tensorShape[numDims - 1], v16);

  if (numDims >= 2) {
    group1[2] = b.or_(group1[2], b.shl(tensorShape[numDims - 2], v16));
    group1[3] = b.lshr(tensorShape[numDims - 2], v16);
  }

  // tile_dim0/1/2 (group1[3] high 16 bits, group1[4]) are filled in by the
  // per-op descriptor filler from the actual warp distribution.

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

    // tile_dim3 (upper 16 bits of group2[3]) is filled in later by the
    // per-op descriptor filler.
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

    // tensor_dim3_stride (4th dimension from the end) (48 bits split across
    // group3[0] and lower 16 bits of group3[1]).  tile_dim4 (upper 16 bits of
    // group3[2]) is filled in later by the per-op descriptor filler.
    if (numDims == 5) {
      group3[0] = tensorStride[numDims - 5];
    }
  }

  return TDMDescriptor{group0, group1, group2, group3};
}

// Fill TDM descriptor for regular load/store operations (1D-5D tensors).
// Computes the per-warp tile shape from `shapePerCTA / warpsPerCTA` and
// encodes the tile_dim* fields into the descriptor; createTDMDescriptor
// only sets per-tensor fields, so this is the sole owner of tile dimensions.
// When the layout's "warp" input dim has free variable bits (i.e. a
// `warp_used_hint` shrunk warpsPerCTA below numWarps), those redundant
// warps get pred=0 so their TDM instruction is a hardware no-op.
void fillTDMDescriptor(
    RewriterBase &rewriter, Location loc,
    const LLVMTypeConverter *typeConverter, Type elementType,
    SmallVector<int64_t> shapePerCTA, unsigned padInterval, unsigned padAmount,
    SmallVector<Value> &group0, SmallVector<Value> &group1,
    std::optional<std::reference_wrapper<SmallVector<Value>>> group2,
    std::optional<std::reference_wrapper<SmallVector<Value>>> group3,
    SmallVector<Value> offset, ArrayRef<Value> dstPtrs, Value pred,
    Value multicastMask, Value barrierPtr,
    const triton::LinearLayout &sharedLayout, Value ctaId, bool isStore,
    ArrayRef<unsigned> warpsPerCTA, int numWarps,
    const std::optional<WarpHintInfo> &warpHint) {
  size_t numDims = offset.size();
  assert(numDims >= 1 && numDims <= 5 && "TDM supports 1D to 5D tensors.");
  assert(!dstPtrs.empty() && "dstPtrs cannot be empty");
  assert(warpsPerCTA.size() == numDims &&
         "warpsPerCTA must have one entry per tensor dim");

  auto ctx = rewriter.getContext();
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  Type globalPtrTy = ptr_ty(ctx, 1);
  Type sharedPtrTy = ptr_ty(ctx, 3);

  // Decode the per-tensor fields of the descriptor (base ptr, shape, stride).
  // Tile dimensions are owned by this filler and computed below from
  // shapePerCTA / warpsPerCTA, so we ignore the decoder's blockShape.
  auto [srcPtr, tensorShape, tensorStride, _decodedBlockShape] =
      decodeTDMDescriptorFull(
          rewriter, loc, group0, group1,
          group2.has_value()
              ? std::optional<ArrayRef<Value>>(group2.value().get())
              : std::nullopt,
          group3.has_value()
              ? std::optional<ArrayRef<Value>>(group3.value().get())
              : std::nullopt,
          numDims);

  // Per-warp tile shape: how many elements each active warp writes/reads in
  // one TDM instruction.  When warpsPerCTA was derived from a smaller K than
  // numWarps (warp_used_hint), each active warp covers a proportionally
  // larger tile so the total CTA coverage stays the same.
  SmallVector<int64_t> tileShape(numDims);
  for (size_t i = 0; i < numDims; ++i)
    tileShape[i] = shapePerCTA[i] / static_cast<int64_t>(warpsPerCTA[i]);

  auto kMessage = str_attr("message");
  auto kWarp = str_attr("warp");
  auto kBlock = str_attr("block");
  auto kOffset = str_attr("offset");
  auto kPartition = str_attr("partition");

  auto cgaLayout = triton::gpu::SharedLinearEncodingAttr::get(
                       ctx, sharedLayout, /*layoutAlignment=*/16)
                       .getCGALayout()
                       .getLinearLayout();

  // For an axis-aligned `warp_used_hint`, place the K identity rows of
  // the warp sublayout at `warpHint->basisBits`; otherwise the layout
  // uses the lowest log2(K) bits (canonical-prefix placement).
  ArrayRef<int32_t> warpBasisBits =
      warpHint ? ArrayRef<int32_t>(warpHint->basisBits) : ArrayRef<int32_t>{};

  auto tdmLayout = triton::gpu::getTDMLinearLayout(
      shapePerCTA, warpsPerCTA, cgaLayout, numWarps, warpBasisBits);

  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);

  // Anchor warpId at i0 = 0 of the active coset.  For canonical-prefix
  // hints (i0 == 0) and for the no-hint case this is a no-op.  For
  // shifted/strided hints this aligns the i-th active warp onto the
  // i-th identity row of the warp sublayout.
  Value warpIdShifted = warpId;
  if (warpHint && warpHint->i0 != 0)
    warpIdShifted = b.xor_(warpId, b.i32_val(warpHint->i0));

  auto warpOffset = applyLinearLayout(
      loc, rewriter, tdmLayout,
      {{kMessage, b.i32_val(0)}, {kWarp, warpIdShifted}, {kBlock, ctaId}});

  // Extract per-dimension offsets and update input offsets
  SmallVector<Value> globalOffset(numDims);
  for (size_t i = 0; i < numDims; ++i) {
    globalOffset[i] = warpOffset[i].second;
    offset[i] = b.add(offset[i], globalOffset[i]);
  }

  Value baseOffset = b.i32_val(0);
  for (size_t i = 0; i < numDims; ++i) {
    Value dimOffset = b.mul(offset[i], tensorStride[i]);
    baseOffset = b.add(baseOffset, dimOffset);
  }
  srcPtr = b.gep(globalPtrTy, elementType, srcPtr, baseOffset);

  auto tdmToShared = tdmLayout.invertAndCompose(sharedLayout);
  auto sharedOffsets = applyLinearLayout(
      loc, rewriter, tdmToShared,
      {{kMessage, b.i32_val(0)}, {kWarp, warpIdShifted}, {kBlock, ctaId}});

  // Extract the offset and partition index from the result
  Value dstOffset = b.i32_val(0);
  Value partitionIdx = b.i32_val(0);
  bool isPartitioned = tdmToShared.hasOutDim(kPartition);
  for (auto &[name, val] : sharedOffsets) {
    if (name == kOffset) {
      dstOffset = val;
    } else if (name == kPartition) {
      partitionIdx = val;
    }
  }

  // Select the correct base pointer for partitioned tensors
  Value dstPtr = dstPtrs[0];
  if (isPartitioned) {
    assert(dstPtrs.size() > 1 &&
           "Partitioned tensors must have multiple bases");
    // Create a vector of base pointers for dynamic indexing
    auto ptrTy = dstPtrs[0].getType();
    auto vecTy = VectorType::get({static_cast<int64_t>(dstPtrs.size())}, ptrTy);
    Value basesVec = b.undef(vecTy);
    for (size_t i = 0; i < dstPtrs.size(); ++i) {
      basesVec = b.insert_element(basesVec, dstPtrs[i], b.i32_val(i));
    }
    // Use vector extract to select the correct base pointer
    dstPtr = b.extract_element(basesVec, partitionIdx);
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

  // TDM store does not support padding in general. However, if the padding
  // interval equals the innermost dimension, we can support it by:
  // 1. Adjusting fastest tile_dim0 to include padding (tile_dim0 + padding).
  // Note that in triton numDims-1 is the fastest dim whereas dim0 is the
  // fastest in the TDM descriptor.
  // 2. Setting tensor_dim0 (fastest) to min(tensor_dim0, original_tile_dim0).
  // We have to do this after adjusting the tensor shape based on the offset and
  // cga offset.
  // This leverages HW OOB checking to skip padding elements while
  // preserving the original tensor shape if smaller. Note that the pre
  // conditions are checked in the verifier already
  int64_t encodedTileDim0 = tileShape[numDims - 1];
  if (isStore && padInterval > 0 && padAmount > 0) {
    int64_t originalTileDim0 = tileShape[numDims - 1];
    encodedTileDim0 = originalTileDim0 + padAmount;

    // Adjust tensor dimension to be min(tensor_dim0, original_tile_dim0)
    Value origTileVal = b.i32_val(originalTileDim0);
    Value cmp = b.icmp_ult(tensorShape[numDims - 1], origTileVal);
    tensorShape[numDims - 1] =
        b.select(cmp, tensorShape[numDims - 1], origTileVal);
  }

  // Update group0 with addresses
  Value globalAddr = b.ptrtoint(i64_ty, srcPtr);
  Value ldsAddr = b.ptrtoint(i32_ty, dstPtr);

  // Predicate off redundant warps reported by the layout's free-variable
  // mask.  Bits of warpId where the warp sublayout has an all-zero row
  // are "free" — toggling them must not change the participating piece.
  // For axis-aligned `warp_used_hint`, those are exactly the warp-index
  // bit positions NOT in `warpHint->basisBits`.  The active-warp test is
  // `((warpId ^ i0) & warpFreeMask) == 0`.  Without a hint, the free bits
  // are the high bits above log2(K), `i0 == 0`, and the test reduces to
  // `(warpId & warpFreeMask) == 0`.
  {
    auto freeMasks = tdmLayout.getFreeVariableMasks();
    int32_t warpFreeMask = freeMasks.lookup(kWarp);
    if (warpFreeMask != 0) {
      Value isActive = b.icmp_eq(b.and_(warpIdShifted, b.i32_val(warpFreeMask)),
                                 b.i32_val(0));
      Value layoutPred = b.select(isActive, b.i32_val(1), b.i32_val(0));
      pred = b.and_(pred, layoutPred);
    }
  }
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
    group1[3] = b.lshr(tensorShape[numDims - 2], b.i32_val(16));
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

  // Encode per-warp tile dimensions.  Bit-field layout matches the chart in
  // createTDMDescriptor: tile_dim0 in group1[3]<31:16>, tile_dim1 in
  // group1[4]<15:0>, tile_dim2 in group1[4]<31:16>, tile_dim3 in
  // group2[3]<31:16>, tile_dim4 in group3[2]<31:16>.  The lower halves of
  // group1[3], group2[3], and group3[2] are tensor-shape bits already set
  // above (or left as zero by createTDMDescriptor when unused).
  group1[3] = b.or_(group1[3], b.i32_val(encodedTileDim0 << 16));
  if (numDims >= 2)
    group1[4] = b.i32_val(tileShape[numDims - 2] & 0xFFFF);
  if (numDims >= 3)
    group1[4] = b.or_(group1[4], b.i32_val(tileShape[numDims - 3] << 16));
  if (numDims >= 4 && group2.has_value())
    group2.value().get()[3] =
        b.or_(group2.value().get()[3], b.i32_val(tileShape[numDims - 4] << 16));
  if (numDims == 5 && group3.has_value())
    group3.value().get()[2] =
        b.or_(group3.value().get()[2], b.i32_val(tileShape[numDims - 5] << 16));

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

// Fill TDM descriptor for gather/scatter operations (2D only).
// Gather reads from non-contiguous rows in global memory to LDS.
// Scatter writes from LDS to non-contiguous rows in global memory.
void fillTDMDescriptorForGatherScatter(
    RewriterBase &rewriter, Location loc,
    const LLVMTypeConverter *typeConverter, Type elementType,
    SmallVector<int64_t> blockShape, unsigned padInterval, unsigned padAmount,
    SmallVector<Value> &group0, SmallVector<Value> &group1,
    SmallVector<Value> &group2, SmallVector<Value> &group3, Value ldsRowOffset,
    Value globalColOffset, Value ldsPtr, Value pred, Value barrierPtr,
    const triton::LinearLayout &cgaLayout, Value ctaId,
    ArrayRef<Value> rowIndices, bool use32BitIndices, bool isGather) {
  assert(!rowIndices.empty() && "Gather/scatter requires row indices.");

  auto ctx = rewriter.getContext();
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  Type globalPtrTy = ptr_ty(ctx, 1);
  Type sharedPtrTy = ptr_ty(ctx, 3);

  // Decode descriptor to get tensor info
  auto [globalPtr, tensorShape, tensorStride, decodedBlockShape] =
      decodeTDMDescriptorFull(rewriter, loc, group0, group1, group2, group3,
                              /*numDims=*/2);

  // Apply CTA column offset to the base pointer.
  // Row positions are specified by rowIndices, so only column offset applies.
  auto kBlock = str_attr("block");
  auto cgaOffsets =
      applyLinearLayout(loc, rewriter, cgaLayout, {{kBlock, ctaId}});
  Value cgaColOffset = b.mul(cgaOffsets[1].second, tensorStride[1]);
  globalPtr = b.gep(globalPtrTy, elementType, globalPtr, cgaColOffset);

  // For scatter, only apply column offset to global address
  // Row positions are specified by rowIndices
  Value colOffset = b.mul(globalColOffset, tensorStride[1]);
  globalPtr = b.gep(globalPtrTy, elementType, globalPtr, colOffset);

  // Calculate LDS offset based on row offset only (column always starts at 0)
  Value ldsOffset = b.mul(ldsRowOffset, b.i32_val(blockShape[1]));

  // Apply padding if needed
  if (padInterval > 0 && padAmount > 0) {
    Value iVal = b.i32_val(log2(padInterval));
    Value pVal = b.i32_val(log2(padAmount));
    Value padOffset = b.shl(i32_ty, b.ashr(ldsOffset, iVal), pVal);
    ldsOffset = b.add(ldsOffset, padOffset);
  }
  ldsPtr = b.gep(sharedPtrTy, elementType, ldsPtr, ldsOffset);

  // Adjust column tensor shape for OOB handling - subtract column offset to
  // get remaining elements.
  tensorShape[1] = b.smax(b.i32_val(0), b.sub(tensorShape[1], globalColOffset));

  // For scatter with padding (store-from-LDS): clamp tensor_dim0 to the
  // original column width so OOB checking drops padding elements before they
  // reach global memory.  We do this before encoding group1 so the clamped
  // value flows through naturally (matching the fillTDMDescriptor pattern).
  // tile_dim0 widening is handled later in the tile_dim0 fixup block.
  if (!isGather && padInterval > 0 && padAmount > 0) {
    Value originalColWidth = b.i32_val(blockShape.back());
    Value cmp = b.icmp_ult(tensorShape[1], originalColWidth);
    tensorShape[1] = b.select(cmp, tensorShape[1], originalColWidth);
  }

  // Update group0 with addresses and enable gather/scatter mode
  Value globalAddr = b.ptrtoint(i64_ty, globalPtr);
  Value ldsAddr = b.ptrtoint(i32_ty, ldsPtr);

  // Set gather/scatter bits: bit 31 = enable, bit 30 = 32-bit indices
  Value predWithGatherScatter = b.or_(pred, b.i32_val(1 << 31));
  if (use32BitIndices) {
    predWithGatherScatter = b.or_(predWithGatherScatter, b.i32_val(1 << 30));
  }

  group0[0] = predWithGatherScatter;
  group0[1] = ldsAddr;
  group0[2] = b.trunc(i32_ty, globalAddr);

  // group0[3]: preserve type bits, set global_addr upper 25 bits
  Value globalAddrHigh = b.trunc(i32_ty, b.lshr(globalAddr, b.i64_val(32)));
  globalAddrHigh = b.and_(globalAddrHigh, b.i32_val(0x01FFFFFF));
  Value typeBits = b.and_(group0[3], b.i32_val(0xC0000000));
  group0[3] = b.or_(typeBits, globalAddrHigh);

  // Update group1 with adjusted tensor shapes for proper OOB handling
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

  // Encode tile_dim0 for gather/scatter as the full undivided column width:
  // gather/scatter is row-indexed across all warps, so each TDM instruction
  // covers the full row.  createTDMDescriptor leaves the upper 16 bits of
  // group1[3] zero, so a plain OR is sufficient.
  if (blockShape.size() >= 2) {
    int64_t tileDim0 = blockShape.back();

    // For scatter with padding: widen tile_dim0 to include padding so the
    // hardware's LDS read stride matches the padded row width.
    if (!isGather && padInterval > 0 && padAmount > 0)
      tileDim0 += padAmount;

    group1[3] = b.or_(group1[3], b.i32_val(tileDim0 << 16));
  }

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

// Compute how many elements each partition buffer advances between consecutive
// TDM instruction slices, accounting for padding if present.
//
// For a partitioned layout that splits one piece into multiple TDM
// instructions, each instruction writes a "slice" of data into each partition
// buffer.  We need to know the padded size of that slice so the next
// instruction can offset its LDS pointer correctly.
static int64_t computePerPartitionSliceStride(
    ArrayRef<int64_t> blockShape, unsigned partitionDim,
    int64_t sliceExtentPerPartition, unsigned numPartitions,
    triton::gpu::PartitionedSharedEncodingAttr partitionedEnc) {
  if (auto paddedEnc = triton::gpu::getPaddedEncoding(partitionedEnc)) {
    SmallVector<int64_t> perPartitionShape(blockShape.begin(),
                                           blockShape.end());
    perPartitionShape[partitionDim] = sliceExtentPerPartition;

    int64_t stride = paddedEnc.getPaddedSize(perPartitionShape);
    // getPaddedSize subtracts trailing padding, but that padding still occupies
    // space between slices, so add it back.
    int64_t unpaddedSize = product(perPartitionShape);
    for (auto [interval, padding] :
         llvm::zip_equal(paddedEnc.getIntervals(), paddedEnc.getPaddings())) {
      if (unpaddedSize % interval == 0)
        stride += padding;
    }
    return stride;
  }
  // No padding: product of all dims except partitionDim, times slice extent.
  int64_t stride = sliceExtentPerPartition;
  for (size_t d = 0; d < blockShape.size(); ++d) {
    if (static_cast<unsigned>(d) != partitionDim)
      stride *= blockShape[d];
  }
  return stride;
}

// Emit a single TDM intrinsic (load or store) for the given block shape.
// This handles both the 2D (d2 intrinsic) and >2D (full intrinsic) cases.
static void emitTDMIntrinsic(
    RewriterBase &rewriter, Location loc,
    const LLVMTypeConverter *typeConverter, ArrayRef<Value> desc,
    size_t numDims, Type elementType, SmallVector<int64_t> effectiveBlockShape,
    unsigned padInterval, unsigned padAmount, SmallVector<Value> globalOffset,
    ArrayRef<Value> instrDstPtrs, Value pred, Value multicastMask,
    Value barrier, const triton::LinearLayout &instrSharedLayout, Value ctaId,
    bool isLoad, ArrayRef<unsigned> warpsPerCTA, int numWarps,
    const std::optional<WarpHintInfo> &warpHint = std::nullopt) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto v8i32Ty = VectorType::get(8, rewriter.getI32Type());
  Value group4Zero = LLVM::ZeroOp::create(rewriter, loc, v8i32Ty);

  if (numDims > 2) {
    auto group0Vec = SmallVector<Value>(desc.begin(), desc.begin() + 4);
    auto group1Vec = SmallVector<Value>(desc.begin() + 4, desc.begin() + 12);
    auto group2Vec = SmallVector<Value>(desc.begin() + 12, desc.begin() + 16);
    auto group3Vec = SmallVector<Value>(desc.begin() + 16, desc.end());

    fillTDMDescriptor(rewriter, loc, typeConverter, elementType,
                      effectiveBlockShape, padInterval, padAmount, group0Vec,
                      group1Vec, std::ref(group2Vec), std::ref(group3Vec),
                      globalOffset, instrDstPtrs, pred, multicastMask, barrier,
                      instrSharedLayout, ctaId, !isLoad, warpsPerCTA, numWarps,
                      warpHint);

    auto group0 = packLLVector(loc, group0Vec, rewriter);
    auto group1 = packLLVector(loc, group1Vec, rewriter);
    auto group2 = packLLVector(loc, group2Vec, rewriter);
    auto group3 = packLLVector(loc, group3Vec, rewriter);

    const char *intrinsicName = isLoad ? "llvm.amdgcn.tensor.load.to.lds"
                                       : "llvm.amdgcn.tensor.store.from.lds";
    LLVM::createLLVMIntrinsicCallOp(
        rewriter, loc, intrinsicName, {},
        {group0, group1, group2, group3, group4Zero, b.i32_val(0)});
  } else {
    auto group0Vec = SmallVector<Value>(desc.begin(), desc.begin() + 4);
    auto group1Vec = SmallVector<Value>(desc.begin() + 4, desc.end());

    fillTDMDescriptor(
        rewriter, loc, typeConverter, elementType, effectiveBlockShape,
        padInterval, padAmount, group0Vec, group1Vec, std::nullopt,
        std::nullopt, globalOffset, instrDstPtrs, pred, multicastMask, barrier,
        instrSharedLayout, ctaId, !isLoad, warpsPerCTA, numWarps, warpHint);

    auto group0 = packLLVector(loc, group0Vec, rewriter);
    auto group1 = packLLVector(loc, group1Vec, rewriter);
    auto v4i32Ty = VectorType::get(4, rewriter.getI32Type());
    Value group2Zero = LLVM::ZeroOp::create(rewriter, loc, v4i32Ty);
    Value group3Zero = LLVM::ZeroOp::create(rewriter, loc, v4i32Ty);

    const char *intrinsicName = isLoad ? "llvm.amdgcn.tensor.load.to.lds"
                                       : "llvm.amdgcn.tensor.store.from.lds";
    LLVM::createLLVMIntrinsicCallOp(
        rewriter, loc, intrinsicName, {},
        {group0, group1, group2Zero, group3Zero, group4Zero, b.i32_val(0)});
  }
}

// Emit TDM load/store, potentially split into multiple instructions for
// partitioned shared memory.
//
// When partitionInfo is set, the warp distribution is adjusted so each wave's
// tile fits within a single LDS partition.  If there aren't enough warps to
// cover all logical pieces in one instruction, the operation is split into
// multiple sequential TDM instructions, each handling a contiguous slice of
// the tensor along partitionDim.
void emitTDMLoadStore(RewriterBase &rewriter, Location loc,
                      const LLVMTypeConverter *typeConverter,
                      ArrayRef<Value> desc, ArrayRef<int64_t> blockShape,
                      int numWarps, unsigned padInterval, unsigned padAmount,
                      ArrayRef<Value> offset, ArrayRef<Value> dstPtrs,
                      Value pred, Value multicastMask, Type elementType,
                      Value barrierPtr, bool isLoad,
                      const triton::LinearLayout &sharedLayout,
                      Attribute encoding, Value ctaId,
                      std::optional<uint32_t> warpUsedHint) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  size_t numDims = blockShape.size();
  assert(numDims <= 5);

  auto partitionedEnc = dyn_cast<PartitionedSharedEncodingAttr>(encoding);

  // When warp_used_hint is present, derive the distribution from the
  // active-warp count K instead of numWarps.  The verifier guarantees
  // the hint is an axis-aligned coset with K a power of two, and for
  // partitioned encodings that warpsPerCTA[partitionDim] >=
  // numLogicalPieces so the copy fits in a single instruction.
  // Multi-instruction slicing is not supported in the hinted path.
  // Inactive warps are turned into hardware no-ops via
  // fillTDMDescriptor's free-variable-mask predication, anchored at i0
  // by an XOR before the active-warp test.
  std::optional<WarpHintInfo> warpHint;
  int effectiveWarps = numWarps;
  if (warpUsedHint) {
    warpHint = extractWarpHintInfo(*warpUsedHint, numWarps);
    effectiveWarps = static_cast<int>(warpHint->K);
  }

  SmallVector<unsigned> warpsPerCTA;
  unsigned numTDMInstructions = 1;
  std::tie(warpsPerCTA, numTDMInstructions) =
      distributeTDMWarpsAlignToPartition(blockShape, effectiveWarps, encoding);
  assert((!warpUsedHint || numTDMInstructions == 1) &&
         "verifier should guarantee single-instruction emission for the "
         "hinted path");

  // Fast path: single instruction covers the entire block.
  if (numTDMInstructions == 1) {
    emitTDMIntrinsic(rewriter, loc, typeConverter, desc, numDims, elementType,
                     to_vector(blockShape), padInterval, padAmount,
                     to_vector(offset), dstPtrs, pred, multicastMask,
                     barrierPtr, sharedLayout, ctaId, isLoad, warpsPerCTA,
                     numWarps, warpHint);
    return;
  }

  // --- Multi-instruction path ---
  //
  // The tensor is split into `numTDMInstructions` slices along partitionDim.
  // Each slice covers `warpsAlongPartition` pieces (one piece per warp along
  // that dim).  Each piece has `pieceSize` elements along partitionDim.
  //
  // For each slice we:
  //   1. Advance the global offset along partitionDim
  //   2. Advance each partition's LDS pointer by the padded stride
  //   3. Build a LinearLayout for the slice (fewer groups)
  //   4. Emit one TDM intrinsic
  unsigned partitionDim = partitionedEnc.getPartitionDim();
  unsigned numLogicalPieces = partitionedEnc.getNumLogicalPieces();
  int64_t pieceSize = blockShape[partitionDim] / numLogicalPieces;
  unsigned warpsAlongPartition = warpsPerCTA[partitionDim];
  int64_t sliceExtent = static_cast<int64_t>(warpsAlongPartition) * pieceSize;

  unsigned numPartitions = partitionedEnc.getNumPartitions();
  unsigned numGroupsInSlice = warpsAlongPartition / numPartitions;

  // Build the shared layout for one slice.
  SmallVector<int64_t> sliceShape(blockShape.begin(), blockShape.end());
  sliceShape[partitionDim] = sliceExtent;
  auto *ctx = partitionedEnc.getContext();
  auto sliceEncoding = triton::gpu::PartitionedSharedEncodingAttr::get(
      ctx, numPartitions, numGroupsInSlice, partitionedEnc.getPartitionDim(),
      partitionedEnc.getPartitionLayout());
  triton::LinearLayout sliceLayout =
      triton::gpu::isPaddedEncoding(sliceEncoding)
          ? triton::gpu::paddedLinearLayout(sliceShape, sliceEncoding)
          : triton::gpu::toLinearLayout(sliceShape, sliceEncoding);

  // Per-partition LDS stride between slices (accounts for padding).
  int64_t elementsPerSlice = computePerPartitionSliceStride(
      blockShape, partitionDim, sliceExtent / numPartitions, numPartitions,
      partitionedEnc);

  for (unsigned instrIdx = 0; instrIdx < numTDMInstructions; ++instrIdx) {
    SmallVector<Value> globalOffset(offset.begin(), offset.end());
    globalOffset[partitionDim] =
        b.add(globalOffset[partitionDim], b.i32_val(instrIdx * sliceExtent));

    SmallVector<int64_t> effectiveBlockShape(blockShape.begin(),
                                             blockShape.end());
    effectiveBlockShape[partitionDim] = sliceExtent;

    // Only the last instruction signals the barrier.
    Value barrier = (instrIdx == numTDMInstructions - 1) ? barrierPtr : Value();

    // Advance each partition buffer's pointer by the padded slice stride.
    SmallVector<Value> instrDstPtrs;
    Value elemOffset = b.i32_val(instrIdx * elementsPerSlice);
    for (size_t i = 0; i < dstPtrs.size(); ++i) {
      instrDstPtrs.push_back(b.gep(ptr_ty(rewriter.getContext(), 3),
                                   elementType, dstPtrs[i], elemOffset));
    }

    emitTDMIntrinsic(rewriter, loc, typeConverter, desc, numDims, elementType,
                     effectiveBlockShape, padInterval, padAmount, globalOffset,
                     instrDstPtrs, pred, multicastMask, barrier, sliceLayout,
                     ctaId, isLoad, warpsPerCTA, numWarps);
  }
}

// Emit a TDM gather or scatter operation for non-contiguous row access.
size_t getTDMGatherScatterInstrinsicCount(size_t numIndices,
                                          bool use32BitIndices) {
  if (numIndices == 0)
    return 0;

  // Determine max indices per instruction based on index size
  size_t maxIndicesPerInstr = use32BitIndices ? 8 : 16;

  return llvm::divideCeil(numIndices, maxIndicesPerInstr);
}

void emitTDMGatherScatter(RewriterBase &rewriter, Location loc,
                          const LLVMTypeConverter *typeConverter,
                          ArrayRef<Value> desc, ArrayRef<int64_t> blockShape,
                          unsigned padInterval, unsigned padAmount,
                          Value ldsPtr, Value pred, Type elementType,
                          Value barrierPtr,
                          const triton::LinearLayout &cgaLayout, Value ctaId,
                          ArrayRef<Value> rowIndices, Value colOffset,
                          bool isGather, int numWarps,
                          RankedTensorType indicesType) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  assert(!rowIndices.empty() && "Gather/scatter requires row indices");
  assert(colOffset && "Gather/scatter requires column offset");

  bool use32BitIndices =
      indicesType.getElementType().getIntOrFloatBitWidth() == 32;
  size_t maxIndicesPerInstr = use32BitIndices ? 8 : 16;

  // Use LinearLayout to determine:
  // 1. Which registers are broadcasted — remove duplicates
  // 2. Which warps are redundant — zero the pred to make instruction a no-op
  // 3. Per-batch LDS row offset — via applyLinearLayout per batch
  // This analysis is direction-agnostic: the index layout determines which warp
  // owns which rows in LDS, regardless of whether data flows to LDS (gather) or
  // from LDS (scatter).
  auto indexLL = triton::gpu::toLinearLayout(indicesType);
  assert(indexLL.getNumOutDims() == 1 &&
         "Gather/scatter index layout must have exactly one output dimension");
  auto freeVarMasks = indexLL.getFreeVariableMasks();

  auto kRegister = rewriter.getStringAttr("register");
  auto kLane = rewriter.getStringAttr("lane");
  auto kWarp = rewriter.getStringAttr("warp");

  // Remove broadcasted (duplicated) register entries. After this, indexLL
  // has a compact register dimension and effectiveRowIndices contains only
  // unique index values.
  SmallVector<Value> effectiveRowIndices(rowIndices.begin(), rowIndices.end());
  auto removeBcast = actionRemoveBroadcastedRegs(indexLL);
  if (!removeBcast.isIdentity()) {
    indexLL = removeBcast.apply(indexLL);
    effectiveRowIndices = removeBcast.apply(
        SmallVector<Value>(rowIndices.begin(), rowIndices.end()));
  }

  Value warpId = getLaneAndWarpId(rewriter, loc).second;

  // If any warp bits are free, those warps hold redundant copies.
  // Zero the pred so the instruction becomes a no-op.
  int32_t warpFreeMask = freeVarMasks.lookup(kWarp);
  if (warpFreeMask != 0) {
    Value isActive =
        b.icmp_eq(b.and_(warpId, b.i32_val(warpFreeMask)), b.i32_val(0));
    pred = b.select(isActive, pred, b.i32_val(0));
  }

  // The index encoding may cover fewer warps than the CTA actually has.
  // applyLinearLayout wraps extra warp IDs via modular arithmetic,
  // causing silent duplication. Predicate off explicitly.
  int numLayoutWarps = indexLL.getInDimSize(kWarp);
  if (numLayoutWarps < numWarps) {
    Value inRange = b.icmp_ult(warpId, b.i32_val(numLayoutWarps));
    pred = b.select(inRange, pred, b.i32_val(0));
  }

  size_t contigIndiceCount = indexLL.getNumConsecutiveInOut();
  maxIndicesPerInstr = std::min(maxIndicesPerInstr, contigIndiceCount);

  // Precompute LDS row offset for each instruction batch via
  // applyLinearLayout with the actual register index and warp ID.
  SmallVector<Value> batchLdsOffsets;
  auto kBlock = rewriter.getStringAttr("block");
  for (size_t startIdx = 0; startIdx < effectiveRowIndices.size();
       startIdx += maxIndicesPerInstr) {
    auto offsets = applyLinearLayout(loc, rewriter, indexLL,
                                     {{kRegister, b.i32_val(startIdx)},
                                      {kLane, b.i32_val(0)},
                                      {kWarp, warpId},
                                      {kBlock, b.i32_val(0)}});
    batchLdsOffsets.push_back(offsets[0].second);
  }

  size_t numIndicesPerWarp = effectiveRowIndices.size();
  size_t numInstructions = batchLdsOffsets.size();

  // Get the descriptor groups (gather/scatter uses 2D format: 12 dwords)
  auto group0Vec = SmallVector<Value>(desc.begin(), desc.begin() + 4);
  auto group1Vec = SmallVector<Value>(desc.begin() + 4, desc.end());

  // For TDM gather/scatter, we need group2 and group3 for indices
  SmallVector<Value> group2Vec(4, b.i32_val(0));
  SmallVector<Value> group3Vec(4, b.i32_val(0));

  // Issue multiple TDM instructions if needed
  for (size_t instrIdx = 0; instrIdx < numInstructions; ++instrIdx) {
    size_t startIdx = instrIdx * maxIndicesPerInstr;
    size_t endIdx = std::min(startIdx + maxIndicesPerInstr, numIndicesPerWarp);

    // Get the subset of indices for this batch
    SmallVector<Value> batchIndices(effectiveRowIndices.begin() + startIdx,
                                    effectiveRowIndices.begin() + endIdx);

    // Make copies of the descriptor groups for this iteration
    auto g0 = group0Vec;
    auto g1 = group1Vec;
    auto g2 = group2Vec;
    auto g3 = group3Vec;

    Value ldsRowOffset = batchLdsOffsets[instrIdx];

    fillTDMDescriptorForGatherScatter(
        rewriter, loc, typeConverter, elementType, to_vector(blockShape),
        padInterval, padAmount, g0, g1, g2, g3, ldsRowOffset, colOffset, ldsPtr,
        pred, barrierPtr, cgaLayout, ctaId, batchIndices, use32BitIndices,
        isGather);

    // Pack and emit the instruction
    auto group0 = packLLVector(loc, g0, rewriter);
    auto group1 = packLLVector(loc, g1, rewriter);
    auto group2 = packLLVector(loc, g2, rewriter);
    auto group3 = packLLVector(loc, g3, rewriter);

    auto v8i32Ty = VectorType::get(8, rewriter.getI32Type());
    Value group4Zero = LLVM::ZeroOp::create(rewriter, loc, v8i32Ty);

    const char *intrinsicName = isGather ? "llvm.amdgcn.tensor.load.to.lds"
                                         : "llvm.amdgcn.tensor.store.from.lds";
    LLVM::createLLVMIntrinsicCallOp(
        rewriter, loc, intrinsicName, {},
        {group0, group1, group2, group3, group4Zero, b.i32_val(0)});
  }
}

// ---------------------------------------------------------------------------
// Implicit op-merging: analysis + merged emit.

namespace {

// Predicate form of AsyncTDMCopyGlobalToLocalOp::validateWarpUsedHint,
// used to validate the *union* of member hints during merge analysis
// (the verifier checks each member individually, but not the union).
// Delegates to the verifier so there is exactly one source of truth for
// the axis-aligned-coset rule -- otherwise an algebraic fix in one
// place can drift out of sync with the other.
bool isAxisAlignedCoset(uint32_t hint, int numWarps) {
  return !triton::amdgpu::AsyncTDMCopyGlobalToLocalOp::validateWarpUsedHint(
      hint, static_cast<int64_t>(numWarps));
}

using TDMCopyGlobalToLocalOp = triton::amdgpu::AsyncTDMCopyGlobalToLocalOp;

// Two TDM copies pass the v1 mergeability filter iff they have the same
// destination MemDescType (same shared-memory encoding + same
// shapePerCTA, enforced via MLIR's structural attribute equality) AND
// distinct SSA destination values.  The "distinct value" rule is a
// conservative stand-in for full non-overlap analysis and is sufficient
// for the GEMM use case where each operand has its own shared buffer.
//
// The same-encoding requirement is intentionally strict for v1: the
// merged-form lowering only emits per-wave `select` chains for the
// `tile_dim*` and per-buffer addresses.  Relaxing it (mixed encodings,
// mixed pad bits, mixed partition arity, mixed shapePerCTA) would
// require additional `s_cselect_b32` chains for `dstPtrs` arity,
// `group1[0]` pad bits, and the LDS-offset computation, but no IR or
// attribute changes.
bool canMergeWith(TDMCopyGlobalToLocalOp first,
                  TDMCopyGlobalToLocalOp candidate) {
  if (first.getResult().getType() != candidate.getResult().getType())
    return false;
  if (first.getResult() == candidate.getResult())
    return false;
  return true;
}

void emitMergeGroup(MutableArrayRef<Operation *> run, int numWarps,
                    DenseMap<Operation *, TDMMergeGroupInfo> &result) {
  // Greedy pack: from the front of `run`, find the longest power-of-two
  // prefix whose member hints (a) stay pairwise disjoint, (b) have the
  // same active-warp count K, (c) all match canMergeWith with the first
  // member, and (d) the running union stays a verifier-legal
  // axis-aligned coset.
  while (run.size() >= 2) {
    auto firstOp = cast<TDMCopyGlobalToLocalOp>(run.front());
    uint32_t unionHint =
        static_cast<uint32_t>(firstOp.getWarpUsedHintAttr().getInt());
    unsigned firstK = llvm::popcount(unionHint);
    size_t n = 1;
    for (size_t i = 1; i < run.size(); ++i) {
      auto op = cast<TDMCopyGlobalToLocalOp>(run[i]);
      auto hint = static_cast<uint32_t>(op.getWarpUsedHintAttr().getInt());
      if (llvm::popcount(hint) != firstK)
        break;
      if (unionHint & hint)
        break;
      if (!canMergeWith(firstOp, op))
        break;
      uint32_t newUnion = unionHint | hint;
      if (!isAxisAlignedCoset(newUnion, numWarps))
        break;
      unionHint = newUnion;
      n = i + 1;
    }
    // Round n down to the largest power of two.
    size_t p2 = size_t{1} << llvm::Log2_64(static_cast<uint64_t>(n));
    if (p2 >= 2) {
      // The running `unionHint` is the OR over the n-prefix; when the
      // greedy run is itself power-of-two-sized (the common case) it is
      // already the OR over the p2-prefix and we can reuse it directly.
      // Otherwise we rebuild from the first p2 members.
      uint32_t finalUnion = unionHint;
      if (n != p2) {
        finalUnion = 0;
        for (size_t i = 0; i < p2; ++i) {
          auto op = cast<TDMCopyGlobalToLocalOp>(run[i]);
          finalUnion |=
              static_cast<uint32_t>(op.getWarpUsedHintAttr().getInt());
        }
      }
      TDMMergeGroupInfo info;
      info.members.assign(run.begin(), run.begin() + p2);
      info.unionHint = finalUnion;
      info.unionInfo = extractWarpHintInfo(finalUnion, numWarps);
      LLVM_DEBUG({
        llvm::dbgs() << "[tdm-merge] group of " << p2 << " ops, unionHint=0x"
                     << llvm::Twine::utohexstr(finalUnion) << "\n";
        for (auto *op : info.members)
          llvm::dbgs() << "  " << *op << "\n";
      });
      for (auto *op : info.members)
        result[op] = info;
      run = run.drop_front(p2);
    } else {
      run = run.drop_front(1);
    }
  }
}

} // namespace

llvm::DenseMap<Operation *, TDMMergeGroupInfo>
computeTDMMergeGroups(ModuleOp mod) {
  llvm::DenseMap<Operation *, TDMMergeGroupInfo> result;

  // Walk every block in the module that contains at least one TDM
  // copy.  Within each such block we scan ops in program order,
  // accumulating a running list of "candidate" TDM copies that satisfy
  // the v1 mergeability filter (hint set, no mbarrier, pairwise
  // compatible).  Pure (memory-effect-free) ops between candidates are
  // allowed to thread through; any side-effecting op (or an
  // unhinted/mbarrier'd TDM copy) closes the current run.
  //
  // We deliberately avoid calling `lookupNumWarps` on blocks that hold
  // no TDM op: that helper fatal-errors when the surrounding module has
  // no `ttg.num-warps`, and many empty / non-TritonGPU modules in lit
  // tests fall into that category.
  llvm::SmallSetVector<Block *, 8> blocks;
  mod->walk(
      [&](TDMCopyGlobalToLocalOp tdm) { blocks.insert(tdm->getBlock()); });
  for (Block *block : blocks) {
    int numWarps = triton::gpu::lookupNumWarps(block->getParentOp());
    SmallVector<Operation *> candidates;
    auto flush = [&]() {
      if (candidates.size() >= 2)
        emitMergeGroup(candidates, numWarps, result);
      candidates.clear();
    };

    for (Operation &op : *block) {
      if (auto tdm = dyn_cast<TDMCopyGlobalToLocalOp>(&op)) {
        // Ops with mbarrier are not eligible.  A hint must be present:
        // no hint means there is nothing to merge with.
        if (tdm.getWarpUsedHintAttr() && !tdm.getBarrier()) {
          candidates.push_back(&op);
          continue;
        }
        // A hint-less or mbarrier-carrying TDM copy still has memory
        // side effects, so it breaks the run for the candidates that
        // came before.
        flush();
        continue;
      }
      // Pure ops (arith/index math producing offsets, etc.) thread
      // through.  Any side-effecting op breaks the run; this includes
      // async_wait, barrier, and mbarrier ops.
      if (isMemoryEffectFree(&op))
        continue;
      flush();
    }
    flush();
  }

  return result;
}

void emitTDMLoadStoreMerged(
    RewriterBase &rewriter, Location loc,
    const LLVMTypeConverter *typeConverter,
    ArrayRef<SmallVector<Value>> descPerMember, ArrayRef<int64_t> blockShape,
    int numWarps, unsigned padInterval, unsigned padAmount,
    ArrayRef<SmallVector<Value>> offsetPerMember,
    ArrayRef<SmallVector<Value>> dstPtrsPerMember,
    ArrayRef<Value> predPerMember, Value multicastMask, Type elementType,
    bool isLoad, const triton::LinearLayout &sharedLayout, Attribute encoding,
    Value ctaId, const TDMMergeGroupInfo &groupInfo) {
  size_t N = groupInfo.members.size();
  assert(N >= 2 && llvm::isPowerOf2_64(N) && "merge group size invariant");
  assert(descPerMember.size() == N && offsetPerMember.size() == N &&
         dstPtrsPerMember.size() == N && predPerMember.size() == N);

  // Pull each member's hint straight from the op attribute; the merge
  // analysis guarantees every member carries a verifier-legal hint and
  // no member carries an mbarrier.  Asserting both here keeps the
  // analysis-vs-emit contract local and lets us drop the parallel
  // hint/barrier ArrayRef parameters.
  SmallVector<uint32_t, 4> hintPerMember(N);
  for (size_t i = 0; i < N; ++i) {
    auto memberOp =
        cast<triton::amdgpu::AsyncTDMCopyGlobalToLocalOp>(groupInfo.members[i]);
    auto hintAttr = memberOp.getWarpUsedHintAttr();
    assert(hintAttr && "merge member must carry a warp_used_hint");
    assert(!memberOp.getBarrier() && "merge member must not carry an mbarrier");
    hintPerMember[i] = static_cast<uint32_t>(hintAttr.getInt());
  }

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  size_t numDims = blockShape.size();
  // numDims is bounded by the inner fillTDMDescriptor (1..5).

  // Each member's descriptor is computed for that member's own K_i =
  // K_union / N warps; the merged emission then `s_cselect_b32`s
  // between member descriptors per wave.  Picking warpsPerCTA from
  // K_member (not K_union) keeps `tileShape = blockShape /
  // warpsPerCTA` consistent with each member's `basisBits` (whose
  // popcount is log2(K_member)).  Mergeability requires the same K
  // across members, so K_member is uniform.
  assert(groupInfo.unionInfo.K % N == 0 &&
         "K_union must be divisible by group size");
  unsigned kMember = groupInfo.unionInfo.K / static_cast<unsigned>(N);
  auto [warpsPerCTA, numTDMInstructions] = distributeTDMWarpsAlignToPartition(
      blockShape, static_cast<int>(kMember), encoding);
  assert(numTDMInstructions == 1 &&
         "verifier guarantees single-instruction emission for hinted ops");
  (void)numTDMInstructions;

  // Decode every member's hint exactly once and keep them around for
  // selector-bit mapping AND per-member descriptor filling.
  SmallVector<WarpHintInfo, 4> infoPerMember;
  infoPerMember.reserve(N);
  for (size_t i = 0; i < N; ++i)
    infoPerMember.push_back(extractWarpHintInfo(hintPerMember[i], numWarps));

  // Compute the per-wave member selector index from the union basis.
  // The union's basis bits are the bit positions in `unionInfo.basisBits`;
  // the *member-specific* sub-basis for a single member of size K_i =
  // K/N occupies log2(K_i) of those bits, leaving log2(N) bits as
  // "selector bits" — exactly the bits where the i0_i offsets differ
  // from i0_union.  We extract those bits from (warpId ^ i0_union) and
  // pack them into a single 0..N-1 index, with the bit ordering matching
  // the order in which members appear in `groupInfo.members`.
  auto [_laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  Value warpIdShifted = warpId;
  if (groupInfo.unionInfo.i0 != 0)
    warpIdShifted = b.xor_(warpId, b.i32_val(groupInfo.unionInfo.i0));

  // Identify selector bits: union basis bits that are NOT in the first
  // member's basis.  Because every member has the same K_i and the
  // union is axis-aligned, the first member's basisBits are a subset of
  // the union's basisBits; the remaining union bits are the selector.
  const WarpHintInfo &firstInfo = infoPerMember.front();
  llvm::SmallDenseSet<int32_t> firstBasis(firstInfo.basisBits.begin(),
                                          firstInfo.basisBits.end());
  SmallVector<int32_t, 5> selectorBits;
  for (int32_t bit : groupInfo.unionInfo.basisBits)
    if (!firstBasis.count(bit))
      selectorBits.push_back(bit);
  assert(selectorBits.size() == llvm::Log2_32(static_cast<unsigned>(N)) &&
         "selector bits must enumerate the log2(N) per-wave choices");

  // Map from raw selector value (binary expansion over selectorBits) to
  // the matching member index, by looking at how each member's i0
  // projects onto the selector bits.
  SmallVector<unsigned> selValToMember(N, ~0u);
  for (size_t i = 0; i < N; ++i) {
    uint32_t delta = infoPerMember[i].i0 ^ groupInfo.unionInfo.i0;
    unsigned selVal = 0;
    for (size_t bi = 0; bi < selectorBits.size(); ++bi) {
      if ((delta >> selectorBits[bi]) & 1u)
        selVal |= 1u << bi;
    }
    assert(selVal < N && selValToMember[selVal] == ~0u &&
           "members must map injectively onto selector values");
    selValToMember[selVal] = static_cast<unsigned>(i);
  }

  // Build per-member packed descriptor slot vectors by running
  // fillTDMDescriptor once per member with the member's own hint and
  // operands, then `select` slot-by-slot keyed on the selector index.
  SmallVector<SmallVector<Value>> g0PerMember(N);
  SmallVector<SmallVector<Value>> g1PerMember(N);
  SmallVector<SmallVector<Value>> g2PerMember(N);
  SmallVector<SmallVector<Value>> g3PerMember(N);
  for (size_t i = 0; i < N; ++i) {
    ArrayRef<Value> di = descPerMember[i];
    SmallVector<Value> g0(di.begin(), di.begin() + 4);
    SmallVector<Value> g1(di.begin() + 4,
                          numDims > 2 ? di.begin() + 12 : di.end());
    SmallVector<Value> g2;
    SmallVector<Value> g3;
    if (numDims > 2) {
      g2.assign(di.begin() + 12, di.begin() + 16);
      g3.assign(di.begin() + 16, di.end());
    }
    fillTDMDescriptor(
        rewriter, loc, typeConverter, elementType,
        SmallVector<int64_t>(blockShape.begin(), blockShape.end()), padInterval,
        padAmount, g0, g1,
        numDims > 2 ? std::optional<std::reference_wrapper<SmallVector<Value>>>(
                          std::ref(g2))
                    : std::nullopt,
        numDims > 2 ? std::optional<std::reference_wrapper<SmallVector<Value>>>(
                          std::ref(g3))
                    : std::nullopt,
        SmallVector<Value>(offsetPerMember[i].begin(),
                           offsetPerMember[i].end()),
        dstPtrsPerMember[i], predPerMember[i], multicastMask,
        /*barrierPtr=*/Value(), sharedLayout, ctaId, /*isStore=*/!isLoad,
        warpsPerCTA, numWarps, infoPerMember[i]);
    g0PerMember[i] = std::move(g0);
    g1PerMember[i] = std::move(g1);
    g2PerMember[i] = std::move(g2);
    g3PerMember[i] = std::move(g3);
  }

  // Per-wave `select` chain over selector value.  The selector value is
  // a packed integer in [0, N); we materialize it once and then chain
  // `icmp eq` + `select` per slot.  The compiler lowers these to
  // s_cmp_eq + s_cselect_b32 because the selector is uniform across the
  // wave.
  Value selectorVal = b.i32_val(0);
  for (size_t bi = 0; bi < selectorBits.size(); ++bi) {
    Value bit = b.and_(b.lshr(warpIdShifted, b.i32_val(selectorBits[bi])),
                       b.i32_val(1));
    selectorVal = b.or_(selectorVal, b.shl(bit, b.i32_val(bi)));
  }

  // Pre-materialize `selectorVal == s` for every selector value so the
  // per-slot `select` chain reuses N-1 cmp values across all 12 (2D)
  // or 20 (3D-5D) descriptor slots, instead of rebuilding them
  // O(num_slots) times and relying on downstream CSE to fold.
  SmallVector<Value, 4> selectorEq(N);
  for (size_t s = 0; s + 1 < N; ++s)
    selectorEq[s] = b.icmp_eq(selectorVal, b.i32_val(s));

  auto selectSlot = [&](ArrayRef<SmallVector<Value>> perMember,
                        size_t slotIdx) -> Value {
    // Start from the last member's slot and chain `select(sel == m,
    // perMember[m], acc)` walking backwards.  Iterating from N-1 down
    // produces a right-leaning chain that the compiler turns into N-1
    // s_cselect_b32 ops.
    Value acc = perMember[selValToMember[N - 1]][slotIdx];
    for (size_t s = N - 1; s-- > 0;)
      acc = b.select(selectorEq[s], perMember[selValToMember[s]][slotIdx], acc);
    return acc;
  };

  SmallVector<Value> g0Merged(g0PerMember[0].size());
  for (size_t k = 0; k < g0Merged.size(); ++k)
    g0Merged[k] = selectSlot(g0PerMember, k);
  SmallVector<Value> g1Merged(g1PerMember[0].size());
  for (size_t k = 0; k < g1Merged.size(); ++k)
    g1Merged[k] = selectSlot(g1PerMember, k);
  SmallVector<Value> g2Merged;
  SmallVector<Value> g3Merged;
  if (numDims > 2) {
    g2Merged.resize(g2PerMember[0].size());
    for (size_t k = 0; k < g2Merged.size(); ++k)
      g2Merged[k] = selectSlot(g2PerMember, k);
    g3Merged.resize(g3PerMember[0].size());
    for (size_t k = 0; k < g3Merged.size(); ++k)
      g3Merged[k] = selectSlot(g3PerMember, k);
  }

  auto v8i32Ty = VectorType::get(8, rewriter.getI32Type());
  auto v4i32Ty = VectorType::get(4, rewriter.getI32Type());
  Value group4Zero = LLVM::ZeroOp::create(rewriter, loc, v8i32Ty);
  const char *intrinsicName = isLoad ? "llvm.amdgcn.tensor.load.to.lds"
                                     : "llvm.amdgcn.tensor.store.from.lds";

  if (numDims > 2) {
    auto group0 = packLLVector(loc, g0Merged, rewriter);
    auto group1 = packLLVector(loc, g1Merged, rewriter);
    auto group2 = packLLVector(loc, g2Merged, rewriter);
    auto group3 = packLLVector(loc, g3Merged, rewriter);
    LLVM::createLLVMIntrinsicCallOp(
        rewriter, loc, intrinsicName, {},
        {group0, group1, group2, group3, group4Zero, b.i32_val(0)});
  } else {
    auto group0 = packLLVector(loc, g0Merged, rewriter);
    auto group1 = packLLVector(loc, g1Merged, rewriter);
    Value group2Zero = LLVM::ZeroOp::create(rewriter, loc, v4i32Ty);
    Value group3Zero = LLVM::ZeroOp::create(rewriter, loc, v4i32Ty);
    LLVM::createLLVMIntrinsicCallOp(
        rewriter, loc, intrinsicName, {},
        {group0, group1, group2Zero, group3Zero, group4Zero, b.i32_val(0)});
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

  constexpr int cacheScope = 8; // (8) = L2 scope
  const int hintValue = cacheScope | static_cast<int>(isSpeculative);
  IntegerAttr hint = rewriter.getI32IntegerAttr(hintValue);

  // Iterate over each register and emit a prefetch intrinsic
  SmallVector<Value> offsets(ll.getInDimSize(kRegister));
  for (int reg = 0; reg < ll.getInDimSize(kRegister); reg++) {
    auto regIndices =
        ll.apply({{kRegister, reg}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});

    // XOR the base indices with the register specific indices
    SmallVector<Value> indices;
    for (auto [base, regIdx] : llvm::zip(baseIndices, regIndices)) {
      assert(base.first == regIdx.first);
      Value combined = b.xor_(base.second, b.i32_val(regIdx.second));
      indices.emplace_back(combined);
    }

    // Compute the local offset from tile ptr for this prefetch based on the
    // computed indices
    Value localOffset = dot64(indices, scaledStride);
    Value prefetchPtr = b.gep(globalPtrTy, elementType, tilePtr, localOffset);

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

    ROCDL::GlobalPrefetchOp::create(rewriter, loc, prefetchPtr, hint, {}, {},
                                    {});

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
