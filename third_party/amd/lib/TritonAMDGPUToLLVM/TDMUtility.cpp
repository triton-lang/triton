#include "TDMUtility.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"
#include <numeric>
#include <optional>

// Include shared C-compatible TDM utilities
#include "../../backend/include/TDMCommon.h"

namespace mlir::LLVM::AMD {

namespace {

Value vecGet(TritonLLVMOpBuilder &b, Value vec, int idx) {
  return b.extract_element(vec, b.i32_val(idx));
}

Value vecSet(TritonLLVMOpBuilder &b, Value vec, int idx, Value val) {
  return b.insert_element(vec, val, b.i32_val(idx));
}

// Helper to decode a value spanning two 32-bit words
Value decode48BitValue(RewriterBase &rewriter, TritonLLVMOpBuilder &b,
                       Value group, int startIdx) {
  Value low = b.lshr(vecGet(b, group, startIdx), b.i32_val(16));
  Value high = b.shl(vecGet(b, group, startIdx + 1), b.i32_val(16));
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

// Shared layout analysis for TDM gather/scatter, used by both
// getTDMGatherScatterInstrinsicCount (wait-count pass) and
// emitTDMGatherScatter (lowering) so the instruction-count logic
// cannot get out of sync.
//
// Uses the LinearLayout of the index tensor to determine:
// 1. Which registers are broadcasted — remove duplicates
// 2. Which warps are redundant (freeVarMasks)
// 3. The effective number of indices per warp and max indices per instruction
// This analysis is direction-agnostic: the index layout determines which warp
// owns which rows in LDS, regardless of whether data flows to LDS (gather) or
// from LDS (scatter).
struct GatherScatterLayoutAnalysis {
  triton::LinearLayout indexLL; // post-broadcast-removal
  ColumnAction removeBcastAction;
  size_t maxIndicesPerInstr;
  size_t effectiveRegCount;
  size_t numInstructions;
  llvm::MapVector<StringAttr, int32_t> freeVarMasks; // from original layout
};

GatherScatterLayoutAnalysis
analyzeGatherScatterLayout(RankedTensorType indicesType) {
  assert(indicesType.getEncoding());

  bool use32BitIndices =
      indicesType.getElementType().getIntOrFloatBitWidth() == 32;
  size_t maxIndicesPerInstr = use32BitIndices ? 8 : 16;

  auto indexLL = triton::gpu::toLinearLayout(indicesType);
  assert(indexLL.getNumOutDims() == 1 &&
         "Gather/scatter index layout must have exactly one output dimension");
  auto freeVarMasks = indexLL.getFreeVariableMasks();

  // Remove broadcasted (duplicated) register entries so indexLL has a compact
  // register dimension containing only unique index values.
  auto removeBcastAction = actionRemoveBroadcastedRegs(indexLL);
  if (!removeBcastAction.isIdentity())
    indexLL = removeBcastAction.apply(indexLL);

  size_t contigIndiceCount = indexLL.getNumConsecutiveInOut();
  maxIndicesPerInstr = std::min(maxIndicesPerInstr, contigIndiceCount);

  auto kRegister = StringAttr::get(indicesType.getContext(), "register");
  size_t effectiveRegCount = indexLL.getInDimSize(kRegister);
  size_t numInstructions =
      effectiveRegCount == 0
          ? 0
          : llvm::divideCeil(effectiveRegCount, maxIndicesPerInstr);

  return {std::move(indexLL), std::move(removeBcastAction),
          maxIndicesPerInstr, effectiveRegCount,
          numInstructions,    std::move(freeVarMasks)};
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

SmallVector<Value> unpackTDMDescriptor(RewriterBase &rewriter, Location loc,
                                       Value descStruct) {
  auto scalars = unpackLLElements(loc, descStruct, rewriter);
  assert((scalars.size() == 12 || scalars.size() == 20) &&
         "TDM descriptor must be 12 (2D) or 20 (3D-5D) i32 scalars");
  SmallVector<Value> groups;
  groups.push_back(packLLVector(
      loc, SmallVector<Value>(scalars.begin(), scalars.begin() + 4), rewriter));
  groups.push_back(packLLVector(
      loc, SmallVector<Value>(scalars.begin() + 4, scalars.begin() + 12),
      rewriter));
  if (scalars.size() == 20) {
    groups.push_back(packLLVector(
        loc, SmallVector<Value>(scalars.begin() + 12, scalars.begin() + 16),
        rewriter));
    groups.push_back(packLLVector(
        loc, SmallVector<Value>(scalars.begin() + 16, scalars.begin() + 20),
        rewriter));
  }
  return groups;
}

SmallVector<Value> scalarizeTDMDescriptor(RewriterBase &rewriter, Location loc,
                                          ArrayRef<Value> vectors) {
  assert((vectors.size() == 2 || vectors.size() == 4) &&
         "TDM descriptor must be 2 (2D) or 4 (3D-5D) vector groups");
  SmallVector<Value> scalars;
  for (Value vec : vectors) {
    auto unpacked = unpackLLVector(loc, vec, rewriter);
    scalars.append(unpacked.begin(), unpacked.end());
  }
  return scalars;
}

void updateTensorDescriptor(RewriterBase &rewriter, Location loc,
                            Type elementType, ArrayRef<int64_t> blockShape,
                            Value &group0, Value &group1,
                            ArrayRef<Value> addOffsets,
                            ArrayRef<Value> setBounds, Value dest, Value pred,
                            Value barrier) {
  size_t numDims = blockShape.size();
  assert(numDims == 2 && "updateTensorDescriptor currently supports 2D");

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value v16 = b.i32_val(16);

  // ---- add_offsets: bump global_addr ----
  if (!addOffsets.empty()) {
    auto elementBitWidth = elementType.getIntOrFloatBitWidth();
    Value elemSize = b.i64_val(elementBitWidth / 8);

    // Decode current 48-bit global_addr from group0[2:3] (valid bit masked).
    Value addrLo = vecGet(b, group0, 2);
    Value addrHi = b.and_(vecGet(b, group0, 3), b.i32_val(0x7FFFFFFF));
    Value addr = b.or_(b.zext(i64_ty, addrLo),
                       b.shl(b.zext(i64_ty, addrHi), b.i64_val(32)));

    // Byte delta:
    //   addOffsets[1] (innermost, stride 1) +
    //   addOffsets[0] * tensor_dim0_stride (from group1[5]),
    // all scaled by element size.  Promote to i64 before the multiply so
    // addOffsets[0] * stride0 doesn't overflow i32 for large tensors.
    // Offsets are signed (advancing backward is allowed) so sext; stride is
    // unsigned so zext.
    Value stride0 = vecGet(b, group1, 5);
    Value off0_64 = b.sext(i64_ty, addOffsets[0]);
    Value off1_64 = b.sext(i64_ty, addOffsets[1]);
    Value stride0_64 = b.zext(i64_ty, stride0);
    Value deltaElem = b.add(off1_64, b.mul(off0_64, stride0_64));
    Value byteDelta = b.mul(deltaElem, elemSize);
    addr = b.add(addr, byteDelta);

    // Re-pack, restoring the valid bit in group0[3] bit 31.
    Value newLo = b.trunc(i32_ty, addr);
    Value newHi = b.trunc(i32_ty, b.lshr(addr, b.i64_val(32)));
    newHi = b.or_(newHi, b.i32_val(1 << 31));
    group0 = vecSet(b, group0, 2, newLo);
    group0 = vecSet(b, group0, 3, newHi);
  }

  // ---- set_bounds: absolute rewrite of tensor_dim ----
  // tensor_dim_inner (setBounds[1] for 2D) spans
  //   group1[1] hi-16 | group1[2] lo-16
  // tensor_dim_outer (setBounds[0]) spans
  //   group1[2] hi-16 | group1[3] lo-16
  if (!setBounds.empty()) {
    auto stampDim = [&](int loDword, int hiDword, Value newDim) {
      // loDword: keep lo-16, replace hi-16 with (newDim & 0xFFFF) << 16
      Value loHi = b.shl(b.and_(newDim, b.i32_val(0xFFFF)), v16);
      Value g_lo = b.or_(
          b.and_(vecGet(b, group1, loDword), b.i32_val(0x0000FFFF)), loHi);
      group1 = vecSet(b, group1, loDword, g_lo);
      // hiDword: keep hi-16, replace lo-16 with (newDim >> 16) & 0xFFFF
      Value hiLo = b.and_(b.lshr(newDim, v16), b.i32_val(0xFFFF));
      Value g_hi = b.or_(
          b.and_(vecGet(b, group1, hiDword), b.i32_val(0xFFFF0000)), hiLo);
      group1 = vecSet(b, group1, hiDword, g_hi);
    };
    stampDim(/*loDword=*/1, /*hiDword=*/2, setBounds[1]); // inner
    stampDim(/*loDword=*/2, /*hiDword=*/3, setBounds[0]); // outer
  }

  // ---- dest: rewrite lds_addr in group0[1] ----
  if (dest) {
    Value ldsAddr = b.ptrtoint(i32_ty, dest);
    group0 = vecSet(b, group0, 1, ldsAddr);
  }

  // ---- pred: rewrite group0[0] ----
  if (pred) {
    // Note: this clobbers any mode bits stored in group0[0] (gather/scatter
    // mode bit in bit 31, index-size bit in bit 30).  TODO: Gather/scatter
    // mutation is not yet supported here; a future revision should preserve
    // them.
    group0 = vecSet(b, group0, 0, pred);
  }

  // ---- barrier: enable bit (group1[0] bit 18) + addr (group1[1] lo-16) ----
  if (barrier) {
    Value g1_0 = vecGet(b, group1, 0);
    Value g1_1 = vecGet(b, group1, 1);
    g1_0 = b.or_(g1_0, b.shl(b.i32_val(1), b.i32_val(18)));
    g1_1 = b.or_(b.and_(g1_1, b.i32_val(0xFFFF0000)),
                 b.and_(b.lshr(b.ptrtoint(i32_ty, barrier), b.i32_val(3)),
                        b.i32_val(0x0000FFFF)));
    group1 = vecSet(b, group1, 0, g1_0);
    group1 = vecSet(b, group1, 1, g1_1);
  }
}

// Decode a full TDM descriptor for 1D-5D tensors.  Returns (base,
// tensorShape[], tensorStride[]).  `groups` is 2 (1D-2D) or 4 (3D-5D) entries.
// Block shape is intentionally NOT decoded: every TDM lowering path rewrites
// tile_dim* against its own warp distribution, so reading them back here
// would only generate dead IR.
std::tuple<Value, SmallVector<Value>, SmallVector<Value>>
decodeTDMDescriptorFull(RewriterBase &rewriter, Location loc,
                        ArrayRef<Value> groups, size_t numDims) {
  auto ctx = rewriter.getContext();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type globalPtrTy = ptr_ty(ctx, 1);

  // Decode base address from group0
  Value globalAddrLow = vecGet(b, groups[0], 2);
  Value globalAddrHigh = b.and_(vecGet(b, groups[0], 3), b.i32_val(0x7FFFFFFF));
  globalAddrLow = b.zext(i64_ty, globalAddrLow);
  globalAddrHigh = b.shl(b.zext(i64_ty, globalAddrHigh), b.i64_val(32));
  Value globalAddr = b.or_(globalAddrLow, globalAddrHigh);
  Value srcPtr = b.inttoptr(globalPtrTy, globalAddr);

  SmallVector<Value> tensorShape(numDims);
  SmallVector<Value> tensorStride(numDims);

  // Helper: combine a 32-bit low part and a 16-bit high part into an i64.
  auto combine48 = [&](Value lo32, Value hi16InDword) -> Value {
    Value hi16 = b.and_(hi16InDword, b.i32_val(0xFFFF));
    return b.or_(b.zext(i64_ty, lo32),
                 b.shl(b.zext(i64_ty, hi16), b.i64_val(32)));
  };

  // Decode dimensions from the end (inner dimensions first)
  tensorShape[numDims - 1] = decode48BitValue(rewriter, b, groups[1], 1);

  if (numDims >= 2) {
    tensorShape[numDims - 2] = decode48BitValue(rewriter, b, groups[1], 2);

    // tensor_dim0_stride: group1[5][0:32] | group1[6][0:16] (48 bits)
    tensorStride[numDims - 2] =
        combine48(vecGet(b, groups[1], 5), vecGet(b, groups[1], 6));

    // tensor_dim1_stride: group1[6][16:32] | group1[7][0:32] (48 bits)
    if (numDims >= 3) {
      Value stride1Low = b.and_(b.lshr(vecGet(b, groups[1], 6), b.i32_val(16)),
                                b.i32_val(0xFFFF));
      Value stride1High = vecGet(b, groups[1], 7);
      tensorStride[numDims - 3] =
          b.or_(b.zext(i64_ty, stride1Low),
                b.shl(b.zext(i64_ty, stride1High), b.i64_val(16)));
    }
  }

  // tensor_dim2_stride: group2[2][0:32] | group2[3][0:16] (48 bits)
  if (numDims >= 4) {
    tensorStride[numDims - 4] =
        combine48(vecGet(b, groups[2], 2), vecGet(b, groups[2], 3));
  }

  // tensor_dim3_stride: group3[0][0:32] | group3[1][0:16] (48 bits)
  if (numDims == 5) {
    tensorStride[numDims - 5] =
        combine48(vecGet(b, groups[3], 0), vecGet(b, groups[3], 1));
  }

  // The innermost dimension always has stride 1
  tensorStride[numDims - 1] = b.i64_val(1);

  // 3rd dimension from group2 if present
  if (numDims >= 3) {
    tensorShape[numDims - 3] = vecGet(b, groups[2], 0);
  }

  // 4th dimension from group2/group3 if present
  if (numDims >= 4) {
    tensorShape[numDims - 4] = vecGet(b, groups[2], 1);
  }

  // 5th dimension from group3 if present
  if (numDims == 5) {
    // tensor_dim4 is encoded across group3[1] and group3[2]
    Value tensorDim4Low = b.and_(b.lshr(vecGet(b, groups[3], 1), b.i32_val(16)),
                                 b.i32_val(0xFFFF));
    Value tensorDim4High = b.and_(vecGet(b, groups[3], 2), b.i32_val(0xFFFF));
    tensorShape[0] = b.or_(tensorDim4Low, b.shl(tensorDim4High, b.i32_val(16)));
  }

  return {srcPtr, tensorShape, tensorStride};
}

SmallVector<Value> createTDMDescriptor(RewriterBase &rewriter, Location loc,
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

  // Strides come in as i64; the descriptor's stride slots are 48 bits wide.
  // Separate the low-32 and high-16 bits of the stride.
  SmallVector<Value> tensorStrideLo32(numDims);
  SmallVector<Value> tensorStrideHi16(numDims);
  for (size_t i = 0; i < numDims; ++i) {
    Value s = tensorStride[i];
    assert(s.getType() == i64_ty && "Expected TDM stride to be i64.");
    tensorStrideLo32[i] = b.trunc(i32_ty, s);
    tensorStrideHi16[i] =
        b.and_(b.trunc(i32_ty, b.lshr(s, b.i64_val(32))), b.i32_val(0xFFFF));
  }

  // This base descriptor records only tensor metadata.  Per-op hardware fields
  // (pred, LDS address, barrier, tile_dim*) are materialized by
  // fillTDMDescriptor / fillTDMDescriptorForGatherScatter, where the
  // destination, predicate, and warp distribution are known.

  // group0 (128 bits / 4 dwords) effective bit encoding:
  // [1:0]:     pred (to be filled later)
  // [30]:      Scatter/gather index size (0=16-bit, 1=32-bit)
  // [31]:      Scatter/gather enable (0=disabled, 1=enabled)
  // [63:32]:   lds address (to be filled later)
  // [120:64]:  global address
  // [127:126]: type - currently always set to 0x2
  // NOTE: Currently only scatter is implemented; gather (load) is TODO.
  auto v4i32Ty = VectorType::get(4, i32_ty);
  auto v8i32Ty = VectorType::get(8, i32_ty);
  Value group0 = LLVM::ZeroOp::create(rewriter, loc, v4i32Ty);
  Value globalAddr = b.ptrtoint(i64_ty, srcPtr);
  group0 = vecSet(b, group0, 2, b.trunc(i32_ty, globalAddr));
  Value g0_3 = b.trunc(i32_ty, b.lshr(globalAddr, v32));
  g0_3 = b.or_(g0_3, b.i32_val(1 << 31));
  group0 = vecSet(b, group0, 3, g0_3);

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
  Value group1 = LLVM::ZeroOp::create(rewriter, loc, v8i32Ty);
  int32_t dataSize = log2(elementSizeInBytes);
  unsigned dwordSize = 32;
  auto padIntervalBits = padInterval * elementBitWidth;
  assert(padIntervalBits % dwordSize == 0 &&
         "padInterval must be a multiple of dwordSize(32bit)");
  auto padIntervalInDwords = padIntervalBits / dwordSize;
  auto padAmountInDwords = padAmount * elementBitWidth / dwordSize;
  Value g1_0 = b.i32_val(dataSize << 16);
  if (padIntervalInDwords > 0 && padAmountInDwords > 0) {
    assert(llvm::isPowerOf2_32(padIntervalInDwords));
    int32_t log2PadIntervalDwords = log2(padIntervalInDwords);
    assert(log2PadIntervalDwords <= 8 && "padInterval too large");
    g1_0 = b.or_(g1_0, b.i32_val(1 << 20));
    g1_0 = b.or_(g1_0, b.i32_val((log2PadIntervalDwords - 1) << 22));
    g1_0 = b.or_(g1_0, b.i32_val((padAmountInDwords - 1) << 25));
  }
  group1 = vecSet(b, group1, 0, g1_0);
  // Encode 32-bit tensor shapes
  group1 = vecSet(b, group1, 1, b.shl(tensorShape[numDims - 1], v16));
  Value g1_2 = b.lshr(tensorShape[numDims - 1], v16);

  Value g1_3 = b.i32_val(0);
  if (numDims >= 2) {
    g1_2 = b.or_(g1_2, b.shl(tensorShape[numDims - 2], v16));
    g1_3 = b.lshr(tensorShape[numDims - 2], v16);
  }
  group1 = vecSet(b, group1, 2, g1_2);

  // tile_dim0/1/2 (group1[3]<31:16>, group1[4]) are intentionally left as
  // zero here and filled in by fillTDMDescriptor / fillTDMDescriptorForGather
  // Scatter, which know the per-op warp distribution.
  group1 = vecSet(b, group1, 3, g1_3);

  // Handle strides (each slot is 48 bits: low-32 + high-16).
  if (numDims >= 2) {
    group1 = vecSet(b, group1, 5, tensorStrideLo32[numDims - 2]);
    Value g1_6 = tensorStrideHi16[numDims - 2];
    if (numDims >= 3) {
      g1_6 = b.or_(
          g1_6,
          b.shl(b.and_(tensorStrideLo32[numDims - 3], b.i32_val(0xFFFF)), v16));
      Value stride1Hi32 =
          b.or_(b.lshr(tensorStrideLo32[numDims - 3], v16),
                b.shl(tensorStrideHi16[numDims - 3], b.i32_val(16)));
      group1 = vecSet(b, group1, 7, stride1Hi32);
    }
    group1 = vecSet(b, group1, 6, g1_6);
  }

  if (numDims <= 2) {
    return {group0, group1};
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
  Value group2 = LLVM::ZeroOp::create(rewriter, loc, v4i32Ty);
  if (numDims >= 3) {
    // tensor_dim2 (3rd dimension from the end)
    group2 = vecSet(b, group2, 0, tensorShape[numDims - 3]);

    // tensor_dim3 (4th dimension from the end)
    if (numDims >= 4) {
      group2 = vecSet(b, group2, 1, tensorShape[numDims - 4]);
      // tensor_dim2_stride: group2[2][0:32] | group2[3][0:16] (48 bits)
      group2 = vecSet(b, group2, 2, tensorStrideLo32[numDims - 4]);
      group2 = vecSet(b, group2, 3, tensorStrideHi16[numDims - 4]);
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
  Value group3 = LLVM::ZeroOp::create(rewriter, loc, v4i32Ty);
  if (numDims == 5) {
    // tensor_dim3_stride (4th dimension from the end) — 48 bits split across
    // group3[0] (low 32) and the lower 16 bits of group3[1] (high 16).
    group3 = vecSet(b, group3, 0, tensorStrideLo32[numDims - 5]);
    // group3[1] = stride_hi16 | (tensor_dim4_lo16 << 16)
    Value g3_1 = b.or_(tensorStrideHi16[numDims - 5],
                       b.shl(tensorShape[numDims - 5], v16));
    group3 = vecSet(b, group3, 1, g3_1);
    // Lower 16 of group3[2] is the high half of tensor_dim4; upper 16
    // (tile_dim4) is filled later by the per-op descriptor filler.
    group3 = vecSet(b, group3, 2, b.lshr(tensorShape[numDims - 5], v16));
  }

  return {group0, group1, group2, group3};
}

// Fill a TDM descriptor for regular load/store (1D-5D).  With `warpUsedHint`
// (axis-aligned, see TritonAMDGPUOps.td), tile_dim* are re-encoded against
// K-based `warpsPerCTA` and `warpId` is XOR-anchored by `lsb(hint)`; without
// a hint, basis defaults to {0..log2K-1}.
void fillTDMDescriptor(RewriterBase &rewriter, Location loc,
                       const LLVMTypeConverter *typeConverter, Type elementType,
                       SmallVector<int64_t> shapePerCTA, int numWarps,
                       unsigned padInterval, unsigned padAmount,
                       MutableArrayRef<Value> groups, SmallVector<Value> offset,
                       ArrayRef<Value> dstPtrs, Value pred, Value multicastMask,
                       Value barrierPtr,
                       const triton::LinearLayout &sharedLayout, Value ctaId,
                       bool isStore, ArrayRef<unsigned> warpsPerCTA,
                       std::optional<uint32_t> warpUsedHint) {
  size_t numDims = offset.size();
  assert(numDims >= 1 && numDims <= 5 && "TDM supports 1D to 5D tensors.");
  assert(!dstPtrs.empty() && "dstPtrs cannot be empty");
  assert(warpsPerCTA.size() == numDims &&
         "warpsPerCTA must have one entry per tensor dim");
  assert((numDims <= 2 ? groups.size() == 2 : groups.size() == 4) &&
         "groups must hold 2 (1D-2D) or 4 (3D-5D) vector entries");

  auto ctx = rewriter.getContext();
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  Type globalPtrTy = ptr_ty(ctx, 1);
  Type sharedPtrTy = ptr_ty(ctx, 3);

  // Tile dimensions are owned by this filler (computed below from
  // shapePerCTA / warpsPerCTA), so the decoder skips them.
  auto [srcPtr, tensorShape, tensorStride] =
      decodeTDMDescriptorFull(rewriter, loc, groups, numDims);

  // Per-warp tile shape; smaller-K hints scale this up so CTA coverage holds.
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

  auto tdmLayout = triton::gpu::getTDMLinearLayout(
      shapePerCTA, warpsPerCTA, cgaLayout, numWarps, warpUsedHint);

  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);

  // XOR warpId by i0 = lsb(hint) so the smallest active warp gets tile
  // offset 0; the rest follow via the warp sublayout.
  Value warpIdShifted = warpId;
  if (warpUsedHint) {
    uint32_t i0 = llvm::countr_zero(*warpUsedHint);
    if (i0 != 0)
      warpIdShifted = b.xor_(warpId, b.i32_val(i0));
  }

  auto warpOffset = applyLinearLayout(
      loc, rewriter, tdmLayout,
      {{kMessage, b.i32_val(0)}, {kWarp, warpIdShifted}, {kBlock, ctaId}});

  // Extract per-dimension offsets and update input offsets
  SmallVector<Value> globalOffset(numDims);
  for (size_t i = 0; i < numDims; ++i) {
    globalOffset[i] = warpOffset[i].second;
    offset[i] = b.add(offset[i], globalOffset[i]);
  }

  Value baseOffset = b.i64_val(0);
  for (size_t i = 0; i < numDims; ++i) {
    Value dimOffset = b.mul(b.zext(i64_ty, offset[i]), tensorStride[i]);
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

  // Predicate off redundant warps: `((warpId ^ i0) & warpFreeMask) == 0`.
  // `warpFreeMask` covers positions NOT in basisBits (or, without a hint,
  // bits above log2(K)).
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
  Value &group0 = groups[0];
  Value &group1 = groups[1];
  group0 = vecSet(b, group0, 0, pred);
  group0 = vecSet(b, group0, 1, ldsAddr);
  group0 = vecSet(b, group0, 2, b.trunc(i32_ty, globalAddr));
  Value g0_3 = b.and_(vecGet(b, group0, 3), b.i32_val(1 << 31));
  g0_3 = b.or_(g0_3, b.trunc(i32_ty, b.lshr(globalAddr, b.i64_val(32))));
  group0 = vecSet(b, group0, 3, g0_3);

  // Update group1 with tensor shapes
  Value g1_0 = vecGet(b, group1, 0);
  if (multicastMask)
    g1_0 = b.or_(g1_0, multicastMask);
  group1 = vecSet(b, group1, 1, b.shl(tensorShape[numDims - 1], b.i32_val(16)));
  Value g1_2 = b.lshr(tensorShape[numDims - 1], b.i32_val(16));
  Value g1_3 = vecGet(b, group1, 3);

  if (numDims >= 2) {
    g1_2 = b.or_(g1_2, b.shl(tensorShape[numDims - 2], b.i32_val(16)));
    g1_3 = b.and_(g1_3, b.i32_val(0xFFFF << 16));
    g1_3 = b.or_(g1_3, b.lshr(tensorShape[numDims - 2], b.i32_val(16)));
  }
  group1 = vecSet(b, group1, 2, g1_2);

  // Configure barrier
  Value g1_1 = vecGet(b, group1, 1);
  if (barrierPtr) {
    g1_0 = b.or_(g1_0, b.shl(b.i32_val(1), b.i32_val(18)));
    g1_1 =
        b.or_(g1_1, b.and_(b.lshr(b.ptrtoint(i32_ty, barrierPtr), b.i32_val(3)),
                           b.i32_val(0x00FFFF)));
  } else {
    g1_0 = b.and_(g1_0, b.i32_val(0xFFFBFFFF));
  }
  group1 = vecSet(b, group1, 0, g1_0);
  group1 = vecSet(b, group1, 1, g1_1);

  // Re-encode tile_dim0..4 against the caller's `warpsPerCTA`
  // (createTDMDescriptor assumed all numWarps).  Bit positions match the
  // chart in createTDMDescriptor:
  //   tile_dim0 -> group1[3]<31:16>, tile_dim1/2 -> group1[4]<15:0>/<31:16>,
  //   tile_dim3 -> group2[3]<31:16>, tile_dim4 -> group3[2]<31:16>
  g1_3 = b.and_(g1_3, b.i32_val(0xFFFF));
  g1_3 = b.or_(g1_3, b.i32_val(encodedTileDim0 << 16));
  group1 = vecSet(b, group1, 3, g1_3);
  if (numDims >= 2) {
    Value g1_4 = b.i32_val(tileShape[numDims - 2] & 0xFFFF);
    if (numDims >= 3)
      g1_4 = b.or_(g1_4, b.i32_val(tileShape[numDims - 3] << 16));
    group1 = vecSet(b, group1, 4, g1_4);
  }
  if (numDims >= 4) {
    Value g2_3 = b.and_(vecGet(b, groups[2], 3), b.i32_val(0xFFFF));
    g2_3 = b.or_(g2_3, b.i32_val(tileShape[numDims - 4] << 16));
    groups[2] = vecSet(b, groups[2], 3, g2_3);
  }
  if (numDims == 5) {
    Value g3_2 = b.and_(vecGet(b, groups[3], 2), b.i32_val(0xFFFF));
    g3_2 = b.or_(g3_2, b.i32_val(tileShape[numDims - 5] << 16));
    groups[3] = vecSet(b, groups[3], 2, g3_2);
  }

  // Update group2/group3 for higher dimensions
  if (numDims >= 3)
    groups[2] = vecSet(b, groups[2], 0, tensorShape[numDims - 3]);
  if (numDims >= 4)
    groups[2] = vecSet(b, groups[2], 1, tensorShape[numDims - 4]);
  if (numDims == 5) {
    Value g3_1 = b.and_(vecGet(b, groups[3], 1), b.i32_val(0xFFFF));
    g3_1 = b.or_(g3_1, b.shl(tensorShape[0], b.i32_val(16)));
    groups[3] = vecSet(b, groups[3], 1, g3_1);
    Value g3_2 = b.and_(vecGet(b, groups[3], 2), b.i32_val(0xFFFF << 16));
    g3_2 = b.or_(g3_2, b.lshr(tensorShape[0], b.i32_val(16)));
    groups[3] = vecSet(b, groups[3], 2, g3_2);
  }
}

// Fill TDM descriptor for gather/scatter operations (2D only).
// Gather reads from non-contiguous rows in global memory to LDS.
// Scatter writes from LDS to non-contiguous rows in global memory.
void fillTDMDescriptorForGatherScatter(
    RewriterBase &rewriter, Location loc,
    const LLVMTypeConverter *typeConverter, Type elementType,
    SmallVector<int64_t> blockShape, unsigned padInterval, unsigned padAmount,
    Value &group0, Value &group1, Value &group2, Value &group3,
    Value ldsRowOffset, Value globalColOffset, Value ldsPtr, Value pred,
    Value barrierPtr, const triton::LinearLayout &cgaLayout, Value ctaId,
    ArrayRef<Value> rowIndices, bool use32BitIndices, bool isGather) {
  assert(!rowIndices.empty() && "Gather/scatter requires row indices.");

  auto ctx = rewriter.getContext();
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  Type globalPtrTy = ptr_ty(ctx, 1);
  Type sharedPtrTy = ptr_ty(ctx, 3);

  // Decode descriptor to get tensor info (only group0/1 used for 2D).
  Value descGroups[2] = {group0, group1};
  auto [globalPtr, tensorShape, tensorStride] =
      decodeTDMDescriptorFull(rewriter, loc, descGroups, /*numDims=*/2);

  // Apply CTA column offset to the base pointer.
  // Row positions are specified by rowIndices, so only column offset applies.
  auto kBlock = str_attr("block");
  auto cgaOffsets =
      applyLinearLayout(loc, rewriter, cgaLayout, {{kBlock, ctaId}});
  // tensorStride is i64 (48-bit slots); zext the i32 offsets before
  // multiplying so we don't truncate to 32 bits.
  Value cgaColOffset =
      b.mul(b.zext(i64_ty, cgaOffsets[1].second), tensorStride[1]);
  globalPtr = b.gep(globalPtrTy, elementType, globalPtr, cgaColOffset);

  // For scatter, only apply column offset to global address
  // Row positions are specified by rowIndices
  Value colOffset = b.mul(b.zext(i64_ty, globalColOffset), tensorStride[1]);
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

  group0 = vecSet(b, group0, 0, predWithGatherScatter);
  group0 = vecSet(b, group0, 1, ldsAddr);
  group0 = vecSet(b, group0, 2, b.trunc(i32_ty, globalAddr));

  // group0[3]: preserve type bits, set global_addr upper 25 bits
  Value globalAddrHigh = b.trunc(i32_ty, b.lshr(globalAddr, b.i64_val(32)));
  globalAddrHigh = b.and_(globalAddrHigh, b.i32_val(0x01FFFFFF));
  Value typeBits = b.and_(vecGet(b, group0, 3), b.i32_val(0xC0000000));
  group0 = vecSet(b, group0, 3, b.or_(typeBits, globalAddrHigh));

  // Update group1 with adjusted tensor shapes for proper OOB handling
  group1 = vecSet(b, group1, 1, b.shl(tensorShape[1], b.i32_val(16)));
  Value g1_2 = b.lshr(tensorShape[1], b.i32_val(16));
  g1_2 = b.or_(g1_2, b.shl(tensorShape[0], b.i32_val(16)));
  group1 = vecSet(b, group1, 2, g1_2);
  Value g1_3 = b.and_(vecGet(b, group1, 3), b.i32_val(0xFFFF << 16));
  g1_3 = b.or_(g1_3, b.lshr(tensorShape[0], b.i32_val(16)));
  group1 = vecSet(b, group1, 3, g1_3);

  // Configure barrier
  Value g1_0 = vecGet(b, group1, 0);
  Value g1_1 = vecGet(b, group1, 1);
  if (barrierPtr) {
    g1_0 = b.or_(g1_0, b.shl(b.i32_val(1), b.i32_val(18)));
    g1_1 =
        b.or_(g1_1, b.and_(b.lshr(b.ptrtoint(i32_ty, barrierPtr), b.i32_val(3)),
                           b.i32_val(0x00FFFF)));
  } else {
    g1_0 = b.and_(g1_0, b.i32_val(0xFFFBFFFF));
  }
  group1 = vecSet(b, group1, 0, g1_0);
  group1 = vecSet(b, group1, 1, g1_1);

  // Set tile_dim1 (number of valid indices) in lower 16 bits of group1[4]
  size_t numIndices = rowIndices.size();
  Value g1_4 = b.and_(vecGet(b, group1, 4), b.i32_val(0xFFFF0000));
  g1_4 = b.or_(g1_4, b.i32_val(numIndices & 0xFFFF));
  group1 = vecSet(b, group1, 4, g1_4);

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

    Value g1_3_gs = b.and_(vecGet(b, group1, 3), b.i32_val(0xFFFF));
    g1_3_gs = b.or_(g1_3_gs, b.i32_val(tileDim0 << 16));
    group1 = vecSet(b, group1, 3, g1_3_gs);
  }

  // Fill group2 and group3 with row indices
  if (use32BitIndices) {
    // 32-bit indices: 4 in group2, 4 in group3
    for (size_t i = 0; i < 4 && i < numIndices; ++i) {
      group2 = vecSet(b, group2, i, rowIndices[i]);
    }
    for (size_t i = 4; i < 8 && i < numIndices; ++i) {
      group3 = vecSet(b, group3, i - 4, rowIndices[i]);
    }
  } else {
    // 16-bit indices: pack 2 per dword
    // Indices are i16, so zero-extend to i32 before packing
    auto packIndices = [&](Value &group, size_t baseIdx) {
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
        group = vecSet(b, group, i, dword);
      }
    };
    packIndices(group2, 0);
    packIndices(group3, 8);
  }
}

namespace {

// Compute how many elements each partition buffer advances between consecutive
// TDM instruction slices, accounting for padding if present.
//
// For a partitioned layout that splits one piece into multiple TDM
// instructions, each instruction writes a "slice" of data into each partition
// buffer.  We need to know the padded size of that slice so the next
// instruction can offset its LDS pointer correctly.
int64_t computePerPartitionSliceStride(
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
void emitTDMIntrinsic(RewriterBase &rewriter, Location loc,
                      const LLVMTypeConverter *typeConverter,
                      ArrayRef<Value> desc, size_t numDims, Type elementType,
                      SmallVector<int64_t> effectiveBlockShape, int numWarps,
                      unsigned padInterval, unsigned padAmount,
                      SmallVector<Value> globalOffset,
                      ArrayRef<Value> instrDstPtrs, Value pred,
                      Value multicastMask, Value barrier,
                      const triton::LinearLayout &instrSharedLayout,
                      Value ctaId, bool isLoad, ArrayRef<unsigned> warpsPerCTA,
                      int32_t auxBits,
                      std::optional<uint32_t> warpUsedHint = std::nullopt) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto v4i32Ty = VectorType::get(4, rewriter.getI32Type());
  auto v8i32Ty = VectorType::get(8, rewriter.getI32Type());

  SmallVector<Value, 4> groups(desc.begin(),
                               desc.begin() + (numDims > 2 ? 4 : 2));
  fillTDMDescriptor(rewriter, loc, typeConverter, elementType,
                    effectiveBlockShape, numWarps, padInterval, padAmount,
                    groups, globalOffset, instrDstPtrs, pred, multicastMask,
                    barrier, instrSharedLayout, ctaId, !isLoad, warpsPerCTA,
                    warpUsedHint);

  // Pad to 4 vector groups (intrinsic always takes 4 group operands).
  while (groups.size() < 4)
    groups.push_back(LLVM::ZeroOp::create(rewriter, loc, v4i32Ty));
  groups.push_back(LLVM::ZeroOp::create(rewriter, loc, v8i32Ty)); // group4
  groups.push_back(b.i32_val(auxBits));
  const char *intrinsicName = isLoad ? "llvm.amdgcn.tensor.load.to.lds"
                                     : "llvm.amdgcn.tensor.store.from.lds";
  LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsicName, {}, groups);
}

} // namespace

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
                      Attribute encoding, Value ctaId, int32_t auxBits,
                      std::optional<uint32_t> warpUsedHint) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  size_t numDims = blockShape.size();
  assert(numDims <= 5);

  auto partitionedEnc = dyn_cast<PartitionedSharedEncodingAttr>(encoding);

  // With a hint, derive the distribution from K = popcount(hint) instead
  // of numWarps; verifier guarantees single-instruction emission (incl.
  // partitioned encodings).  Inactive warps become HW no-ops via
  // fillTDMDescriptor's free-variable-mask predication (XOR-anchored at i0).
  int effectiveWarps = warpUsedHint ? llvm::popcount(*warpUsedHint) : numWarps;

  auto [warpsPerCTA, numTDMInstructions] =
      distributeTDMWarpsAlignToPartition(blockShape, effectiveWarps, encoding);
  assert((!warpUsedHint || numTDMInstructions == 1) &&
         "verifier should guarantee single-instruction emission for the "
         "hinted path");

  // Fast path: single instruction covers the entire block.
  if (numTDMInstructions == 1) {
    emitTDMIntrinsic(rewriter, loc, typeConverter, desc, numDims, elementType,
                     to_vector(blockShape), numWarps, padInterval, padAmount,
                     to_vector(offset), dstPtrs, pred, multicastMask,
                     barrierPtr, sharedLayout, ctaId, isLoad, warpsPerCTA,
                     auxBits, warpUsedHint);
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
                     effectiveBlockShape, numWarps, padInterval, padAmount,
                     globalOffset, instrDstPtrs, pred, multicastMask, barrier,
                     sliceLayout, ctaId, isLoad, warpsPerCTA, auxBits);
  }
}

size_t getTDMGatherScatterInstrinsicCount(RankedTensorType indicesType) {
  return analyzeGatherScatterLayout(indicesType).numInstructions;
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

  auto analysis = analyzeGatherScatterLayout(indicesType);
  auto &indexLL = analysis.indexLL;
  size_t maxIndicesPerInstr = analysis.maxIndicesPerInstr;

  auto kRegister = rewriter.getStringAttr("register");
  auto kLane = rewriter.getStringAttr("lane");
  auto kWarp = rewriter.getStringAttr("warp");

  // Apply broadcast removal to the actual row indices.
  SmallVector<Value> effectiveRowIndices(rowIndices.begin(), rowIndices.end());
  if (!analysis.removeBcastAction.isIdentity()) {
    effectiveRowIndices = analysis.removeBcastAction.apply(
        SmallVector<Value>(rowIndices.begin(), rowIndices.end()));
  }

  Value warpId = getLaneAndWarpId(rewriter, loc).second;

  // If any warp bits are free, those warps hold redundant copies.
  // Zero the pred so the instruction becomes a no-op.
  int32_t warpFreeMask = analysis.freeVarMasks.lookup(kWarp);
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
  assert(batchLdsOffsets.size() == analysis.numInstructions);

  size_t numIndicesPerWarp = effectiveRowIndices.size();

  // Get the descriptor groups (gather/scatter uses 2D format — desc has 2
  // vector entries: group0 = <4 x i32>, group1 = <8 x i32>)
  Value group0In = desc[0];
  Value group1In = desc[1];

  // For TDM gather/scatter, we need group2 and group3 for indices
  auto v4i32Ty = VectorType::get(4, rewriter.getI32Type());
  Value group2Zero = LLVM::ZeroOp::create(rewriter, loc, v4i32Ty);
  Value group3Zero = LLVM::ZeroOp::create(rewriter, loc, v4i32Ty);

  // Issue multiple TDM instructions if needed
  for (size_t instrIdx = 0; instrIdx < analysis.numInstructions; ++instrIdx) {
    size_t startIdx = instrIdx * maxIndicesPerInstr;
    size_t endIdx = std::min(startIdx + maxIndicesPerInstr, numIndicesPerWarp);

    // Get the subset of indices for this batch
    SmallVector<Value> batchIndices(effectiveRowIndices.begin() + startIdx,
                                    effectiveRowIndices.begin() + endIdx);

    // Make copies of the descriptor groups for this iteration
    Value g0 = group0In;
    Value g1 = group1In;
    Value g2 = group2Zero;
    Value g3 = group3Zero;

    Value ldsRowOffset = batchLdsOffsets[instrIdx];

    fillTDMDescriptorForGatherScatter(
        rewriter, loc, typeConverter, elementType, to_vector(blockShape),
        padInterval, padAmount, g0, g1, g2, g3, ldsRowOffset, colOffset, ldsPtr,
        pred, barrierPtr, cgaLayout, ctaId, batchIndices, use32BitIndices,
        isGather);

    auto v8i32Ty = VectorType::get(8, rewriter.getI32Type());
    Value group4Zero = LLVM::ZeroOp::create(rewriter, loc, v8i32Ty);

    const char *intrinsicName = isGather ? "llvm.amdgcn.tensor.load.to.lds"
                                         : "llvm.amdgcn.tensor.store.from.lds";
    LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsicName, {},
                                    {g0, g1, g2, g3, group4Zero, b.i32_val(0)});
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

  // Decode TDM descriptor to get the base pointer, shape, and strides.
  // desc has 2 (2D) or 4 (3D-5D) vector entries.
  auto [basePtr, tensorShape, tensorStride] =
      mlir::LLVM::AMD::decodeTDMDescriptorFull(rewriter, loc, desc, numDims);

  auto dot64 = [&](ArrayRef<Value> indices, ArrayRef<Value> strides) {
    Value ret = b.i64_val(0);
    for (auto [index, stride] : llvm::zip(indices, strides)) {
      assert(stride.getType() == i64_ty && "Expected TDM stride to be i64.");
      ret = b.add(ret, b.mul(b.zext(i64_ty, index), stride));
    }
    return ret;
  };

  // Apply the passed offsets to the base pointer.
  Value tileOffset = dot64(offset, tensorStride);
  auto tilePtr = b.gep(globalPtrTy, elementType, basePtr, tileOffset);

  // Calculate the total tensor size for bounds checking.
  Value linearTensorSize =
      b.mul(b.zext(i64_ty, tensorShape[0]), tensorStride[0]);

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
  scaledStride.back() = b.i64_val(elemPerPrefetch);

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
