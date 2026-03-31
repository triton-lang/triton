#include "Utility.h"

#include "amd/lib/TritonAMDGPUTransforms/Utility.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/DescriptorMemoryLayouts.h"
#include "triton/Tools/LayoutUtils.h"

#include <limits>

namespace tt = triton;
namespace ttg = triton::gpu;

namespace deduceMin {
int deduceMinCountInBlock(Block &block,
                          const std::function<int(Operation *)> &countFunc);

// Returns the minimum found when accumulating countFunc(op) between begin and
// end (inclusive)
int deduceMinCountBetweeOps(Operation *beginOp, Operation *endOp,
                            const std::function<int(Operation *)> &countFunc) {
  assert(beginOp && endOp);
  assert(beginOp == endOp || beginOp->isBeforeInBlock(endOp));
  int count = 0;
  for (auto op = beginOp; op != endOp; op = op->getNextNode()) {
    if (auto ifOp = llvm::dyn_cast<scf::IfOp>(op)) {
      if (ifOp.getElseRegion().empty())
        continue;

      assert(!ifOp.getThenRegion().empty() && !ifOp.getElseRegion().empty());
      auto minThen =
          deduceMinCountInBlock(ifOp.getThenRegion().front(), countFunc);
      auto minElse =
          deduceMinCountInBlock(ifOp.getElseRegion().front(), countFunc);
      count += std::min(minThen, minElse);
    } else if (auto forOp = llvm::dyn_cast<scf::ForOp>(op)) {
      if (std::optional<APInt> tripCount = forOp.getStaticTripCount()) {
        uint64_t tcVal = 0;
        if (forOp.getUnsignedCmp() && tripCount->ugt(0))
          tcVal = tripCount->getZExtValue();
        else if (!forOp.getUnsignedCmp() && tripCount->sgt(0))
          tcVal = tripCount->getSExtValue();
        if (tcVal > 0)
          count += tcVal * deduceMinCountInBlock(*forOp.getBody(), countFunc);
      }
    } else {
      count += countFunc(op);
    }
  }
  return count;
}

// Returns the minimum found when accumulating countFunc(op) for all paths
// between the block's start and end op
int deduceMinCountInBlock(Block &block,
                          const std::function<int(Operation *)> &countFunc) {
  if (block.empty())
    return 0;
  return deduceMinCountBetweeOps(&block.front(), &block.back(), countFunc);
}
} // namespace deduceMin

int deduceMinCountOnDefChain(Value defValue, Operation *consumerOp,
                             const std::function<int(Operation *)> &countFunc,
                             int pathSum, int foundMin) {
  using namespace deduceMin;
  // If the value is not defined in the same region as the consumer we need to
  // peel the parent region of consumer until we arrive at value's region
  while (consumerOp->getParentRegion() != defValue.getParentRegion()) {
    pathSum += deduceMin::deduceMinCountBetweeOps(
        &consumerOp->getBlock()->front(), consumerOp, countFunc);
    consumerOp = consumerOp->getParentOp();
  }

  // Break recursion if we arrive at the producer updating the path based on the
  // ops between producer and consumer
  if (Operation *defOp = defValue.getDefiningOp()) {
    pathSum +=
        deduceMinCountBetweeOps(defOp->getNextNode(), consumerOp, countFunc);
    foundMin = std::min(foundMin, pathSum);
    return foundMin;
  }
  // If value is a loop carried argument (BlockArgument) we need to look at
  // initial arguments of the loop and the previous iteration
  if (auto arg = mlir::dyn_cast<BlockArgument>(defValue)) {
    Block *block = arg.getOwner();
    auto forOp = dyn_cast<scf::ForOp>(block->getParentOp());

    // Failed to track, return 0 conservatively.
    if (!forOp || forOp.getBody()->empty()) {
      return 0;
    }

    Operation *firstOpInLoop = &*forOp.getBody()->begin();
    pathSum += deduceMinCountBetweeOps(firstOpInLoop, consumerOp, countFunc);

    // Break recursion early if we exceed previous min
    if (pathSum >= foundMin)
      return foundMin;

    Value incomingVal = forOp.getInitArgs()[arg.getArgNumber() - 1];
    int countLoopInit = deduceMinCountOnDefChain(incomingVal, forOp, countFunc,
                                                 pathSum, foundMin);

    Operation *yieldOp = block->getTerminator();
    Value prevVal = yieldOp->getOperand(arg.getArgNumber() - 1);
    int countPreviousIter = deduceMinCountOnDefChain(
        prevVal, yieldOp, countFunc, pathSum, foundMin);

    return std::min(std::min(countLoopInit, countPreviousIter), foundMin);
  }

  // Unsupported value, return 0 conservatively.
  return 0;
}

int deduceMinCountOnDefChain(Value defValue, Operation *consumerOp,
                             llvm::function_ref<int(Operation *)> countFunc) {
  return deduceMinCountOnDefChain(defValue, consumerOp, countFunc, 0,
                                  std::numeric_limits<int>::max());
}

// On GFX9, lanes in a warp have to write contiguously to shared memory which
// means we can only add padding at warp boundaries. With 64 lanes, this means:
// - Padding intervals must be multiples of 256 bytes for 4-byte loads.
// - Padding intervals must be multiples of 1024 bytes for 16-byte loads.
// To avoid bank conflicts when reading tensors in MFMA layout, we stagger
// continuous rows (non contig dimension) by adding padding that shifts their
// start addresses to different shared memory banks.
// take Mx64xbf16, k contiguous, kWidth=8, for example: (rX stands for row X)
// padding here is set to 16 elements (32 bytes) to avoid bank conflicts
// we can pack r0,r4,r8,r12,r16,r20,r24,r28 to compose a contiguous tile
// r0[0:8), r0[8:16),
//                   r1[0:8), r1[8:16),
//                                     r2[0:8), r2[8:16),
//                                                       r3[0:8), r3[8:16),
// r4[0:8), r4[8:16),
//                   r5[0:8), r5[8:16),
//                                     r6[0:8), r6[8:16),
//                                                       r7[0:8), r7[8:16),
// r8[0:8), r8[8:16),
// when composing padded layout, we first assemble the rows that are continuous.
// in LDS, the rows are arranged as below
//  r0,  r4, r8, r12, r16, r20, r24, r28
// pad,  r1, r5,  r9, r13, r17, r21, r25
// r29, pad, r2,  r6, r10, r14, r18, r22
// r26, r30, pad, r3 ....
static ttg::PaddedSharedEncodingAttr composePaddedLayoutForAsyncCopyCDNA4(
    ttg::DotOperandEncodingAttr dotOpEnc, ttg::TensorOrMemDesc srcTy,
    ArrayRef<unsigned> sharedOrder, bool useAsyncCopy, unsigned warpSize) {
  auto *ctx = srcTy.getContext();

  // NYI: padded layouts for tt.load/local_write which is more flexible
  if (!useAsyncCopy) {
    return {};
  }

  auto mfmaEnc = dyn_cast<ttg::AMDMfmaEncodingAttr>(dotOpEnc.getParent());
  if (!mfmaEnc) {
    return {};
  }

  auto shape = srcTy.getShape();
  int rank = shape.size();

  if (rank != 2) {
    return {};
  }

  unsigned bitWidth = getIntOrFloatOrPtrBitWidth(srcTy.getElementType());
  unsigned elemByteWidth = std::max(bitWidth / 8u, 1u);

  if (!llvm::is_contained({1, 2}, elemByteWidth)) {
    return {};
  }

  auto operandIdx = dotOpEnc.getOpIdx();
  auto kWidth = dotOpEnc.getKWidth();
  int kDimIndex = operandIdx == 0 ? 1 : 0;
  bool isKContig = sharedOrder[0] == kDimIndex;
  auto mfmaNonKDim = mfmaEnc.getInstrShape()[operandIdx];
  auto kDim = shape[kDimIndex];
  auto nonKDim = shape[(kDimIndex + 1) % 2];

  // NYI: padding for scales
  if (operandIdx >= 2) {
    return {};
  }

  if (!llvm::is_contained({16, 32}, mfmaNonKDim)) {
    return {};
  }

  if (!llvm::is_contained({4, 8, 16}, kWidth)) {
    return {};
  }

  unsigned kWidthBytes = kWidth * elemByteWidth;
  // TODO: if the actual vecSize is smaller than 16 bytes we can do better by
  // using smaller padding intervals
  unsigned vecSize = 16 / elemByteWidth;
  unsigned elemsPer8Bytes = 8 / elemByteWidth;

  // Determine row(contig) size
  unsigned contigDim = isKContig ? kDim : nonKDim;
  unsigned nonContigDim = isKContig ? nonKDim : kDim;

  // padding to avoid bank conflict
  // The bank conflict pattern depends on the ds_load instruction:
  // 1) ds_read_b128: Uses 64 banks and lanes are grouped into 4 pairs:
  //  Group1: 0-3, 12-15, 20-23, 24-27
  //  Group2: 4-7, 8-11, 16-19, 28-31
  // The upper half of the lanes follow the same pattern.
  // 2) ds_read_b64_tr (and ds_read_b64): Uses 64 banks and lanes are split into
  // 2 groups which access LDS one after another.
  // 3) Others: Use only 32 banks and lanes are split into groups each loading
  // 32 banks.

  bool useDsReadB128 = isKContig && kWidthBytes == 16;
  bool useDsReadB64Tr = !isKContig && kWidthBytes >= 8;

  // Note for isKContig && kWidthBytes == 8 we do use ds_read2_b64 which uses 32
  // banks
  unsigned numberOfBanks = (useDsReadB128 || useDsReadB64Tr) ? 64 : 32;
  unsigned bytesPerBank = 4;
  unsigned elemPerBankRow = (numberOfBanks * bytesPerBank) / elemByteWidth;

  unsigned padding = 0;
  if (useDsReadB128) {
    padding = mfmaNonKDim == 16 ? (kWidth * 2) : kWidth;
  } else if (useDsReadB64Tr) {
    padding = mfmaNonKDim == 16 ? 16 : 32;
  } else {
    padding = elemsPer8Bytes;
  }

  unsigned contigLanes = contigDim / vecSize;
  unsigned wrap = std::min(contigDim, elemPerBankRow) / padding;
  // wrap == 0 means padding > contigDim, which is not a valid configuration
  if (wrap == 0) {
    return {};
  }

  // The staggering of rows only works if we have enough (wrap) rows to stagger.
  // If we have less rows we get bank conflicts. For each pow2 too small we will
  // get 2 times more conflicts.
  unsigned requiredRows = warpSize / contigLanes * wrap;
  unsigned xWayConflicts =
      (nonContigDim >= requiredRows)
          ? 1
          : (llvm::Log2_32(requiredRows / nonContigDim) + 1);
  // Heuristic, for ds_read_b128 we do not tolerate any conflicts but for
  // ds_read_b64(_tr) we tolerate 2-way because swizzling will produce the same
  // number of conflicts.
  if ((useDsReadB128 && xWayConflicts > 1) || xWayConflicts > 2) {
    return {};
  }

  if (xWayConflicts > 1) {
    // We need to adjust the warp to allow for bank conflicts and to produce a
    // valid layout
    wrap /= (1 << (xWayConflicts - 1));
    if (wrap == 0) {
      return {};
    }
  }

  // Use 16 rows wrap if block large enough
  bool useBestWrap = false;
  unsigned bestWrap = 16;
  if (nonContigDim >= warpSize / contigLanes * bestWrap && bestWrap > wrap) {
    useBestWrap = true;
    wrap = bestWrap;
  }

  // We create linear bases mapping from [contigDim, nonContigDim] -> offset,
  std::vector<std::vector<int>> bases;

  // Keep contigSize numbers of elements contiguous in shared memory
  for (int elemLog2 = 0; elemLog2 < llvm::Log2_32(contigDim); elemLog2++)
    bases.push_back({1 << elemLog2, 0});

  // Add rows strided which has the same start offset
  unsigned paddingInterval = warpSize * vecSize;
  unsigned requiredNumBases = llvm::Log2_32(paddingInterval);
  int rowBase = 0;
  for (rowBase = llvm::Log2_32(wrap); bases.size() < requiredNumBases;
       rowBase++)
    bases.push_back({0, 1 << rowBase});

  // Add rows [0, wrap]
  for (int rowLog2 = 0; rowLog2 < llvm::Log2_32(wrap); rowLog2++)
    bases.push_back({0, 1 << rowLog2});

  // Add remaining rows
  for (; rowBase < llvm::Log2_32(nonContigDim); rowBase++)
    bases.push_back({0, 1 << rowBase});

  // Fixup: One ds_read_tr loads 8 bytes so kWidthBytes > 8 will load strided.
  // To account for this we swap rows which are accessed by the rows 16-31
  if (useDsReadB64Tr && mfmaNonKDim == 16 && kWidthBytes == 16) {
    // lane groups wrap at 16 bytes, so we have to exchange
    // rows representing 16 and 8 bytes to avoid bank conflict
    unsigned baseIdxGroup0 = 0;
    unsigned baseIdxGroup1 = 0;
    for (unsigned i = 0; i < bases.size(); i++) {
      if (bases[i][1] == 16 / elemByteWidth)
        baseIdxGroup1 = i;
      if (bases[i][1] == 8 / elemByteWidth)
        baseIdxGroup0 = i;
    }
    assert(baseIdxGroup0 != 0 && baseIdxGroup1 != 0);
    std::swap(bases[baseIdxGroup0], bases[baseIdxGroup1]);
  }

  // Fixup for KContig and mfma32 when reordered rows can not fit in 64banks
  if (useDsReadB128 && mfmaNonKDim == 32 && useBestWrap &&
      kDim < (256 / elemByteWidth)) {
    bool useWideLayout = kWidth == (16 / elemByteWidth);

    // For narrow layouts we need to shift every 16th row to the other half of
    // shared memory banks to read from all banks. For the wide layout we need
    // to ensure every 16th rows start at the same bank so lane groups access
    // different banks. This is done by swapping the bases representing offset
    // 256 (64banks) for wide layouts or 128 (32banks) for narrow layouts with
    // the base of the "16th" row which is after log2(contigDim) bases.
    int offsetBytes = useWideLayout ? 256 : 128;
    int offsetIndex = llvm::Log2_32(offsetBytes);
    int row16Index = llvm::Log2_32(contigDim);
    assert(row16Index < bases.size());
    assert(offsetIndex < bases.size());
    std::swap(bases[offsetIndex], bases[row16Index]);
  }

  // Swap bases to match srcTy dimension order
  if ((isKContig && kDimIndex == 1) || (!isKContig && kDimIndex == 0)) {
    for (auto &p : bases)
      std::swap(p[0], p[1]);
  }

  auto cgaLayout = ttg::getCGALayout(srcTy.getEncoding());
  triton::LinearLayout linearComponent(
      {
          {StringAttr::get(ctx, "offset"), bases},
      },
      triton::standardOutDimNames(ctx, rank));
  linearComponent = triton::gpu::combineCtaCgaWithShape(
      linearComponent, cgaLayout, srcTy.getShape());

  return ttg::PaddedSharedEncodingAttr::get(ctx, {{paddingInterval, padding}},
                                            std::move(linearComponent));
}

// LDS padding strategy for TDM (descriptor) loads.
//
// Currently only invoked for gfx1250 TDM loads (via
// getSharedEncIfAllUsersAreDotEncPadded in LowerLoops.cpp when useTDM is
// true).
//
// Padding is chosen per-dtype and per-access-path to minimize bank conflicts
// while avoiding unnecessary LDS waste.  The two load paths have different
// access patterns and therefore different optimal padding values.
//
//   Transposed (ds_load_tr*):
//     Used when K is contiguous in shared memory but the dot instruction
//     needs the non-K dimension contiguous in registers. The instruction
//     cooperatively loads a fixed sub-tile across shuffle groups. In each
//     execution cycle, two shuffle groups (16 lanes total) access a combined
//     row of 2 × (instBitWidth/elemBits) elements. To avoid bank conflicts,
//     the padding must equal this combined row width so that each successive
//     LDS row lands on a completely disjoint set of banks.
//
//       16-bit (fp16/bf16): ds_load_tr16_b128 → 2 × 128/16 = 16 elems → pad 16
//        8-bit (fp8/i8):    ds_load_tr8_b64   → 2 ×  64/8  = 16 elems → pad 16
//       32-bit (f32):       no transposed instruction; falls back to
//                           non-transposed ds_load_b* where each thread
//                           loads sequentially.
//
//   Non-transposed (ds_load_b*):
//     Used when the shared memory layout already matches what the dot
//     instruction expects. Each thread issues a vector load of consecutive
//     elements.  Padding ensures the LDS row stride (in dwords) avoids
//     periodic bank aliasing across the 16 nonK-positions per cycle.
//
//     For dword-or-wider elements (f32+): pad = min(vecWidth, 128/elemBits).
//     The load width in dwords equals vecWidth, so pad = vecWidth gives
//     gcd(stride_dwords, 64) = vecWidth — optimal bank separation.
//       32-bit kWidth=4: min(4, 4) = 4 elems (MFMA 16x16x4)
//       32-bit kWidth=2: min(2, 4) = 2 elems (MFMA 32x32x2)
//
//     For sub-dword elements (fp16/fp8): pad = 128/elemBits (= 4 dwords).
//     Dual-address loads (e.g. ds_load_2addr_b64) need the full 4-dword
//     stride separation to avoid cross-address bank conflicts.
//       16-bit (fp16): 128/16 =  8 elems
//        8-bit (fp8):  128/8  = 16 elems
//
// Note on 4-bit types (i4): two i4 elements are packed into one i8 in LDS,
// so from a bank-conflict perspective 4-bit behaves identically to 8-bit
// in both transposed and non-transposed paths below.

static triton::gpu::PaddedSharedEncodingAttr
composePaddedLayoutWMMA(int opIdx, unsigned vecWidth,
                        ttg::TensorOrMemDesc srcTy, ArrayRef<unsigned> order,
                        const triton::AMD::TargetInfo &targetInfo) {
  auto shape = srcTy.getShape();
  auto CGALayout = ttg::getCGALayout(srcTy.getEncoding());
  auto blockShapePerCTA =
      triton::gpu::getShapePerCTA(CGALayout.getCTASplitNum(), shape);
  int innerDimLength = blockShapePerCTA[order[0]];
  bool loadTransposed = (order[0] != (1 - opIdx));

  // Fallback: assume padding to match widest load width
  unsigned typeWidthInBit = srcTy.getElementType().getIntOrFloatBitWidth();
  unsigned padAmount = 128 / typeWidthInBit;
  if (loadTransposed) {
    // Transposed path: pad by twice the elements-per-lane of the transposed
    // instruction.  Two shuffle groups execute per cycle, each reading
    // instBitWidth/elemBits elements from the same row set.  Padding by
    // 2× ensures the stride (in dwords) is an odd multiple of the combined
    // row-access width, distributing all 16 lanes' bank accesses across
    // disjoint banks and eliminating conflicts for tile widths >= 32.
    if (auto ldsParams = targetInfo.queryLDSTransLoadParams(typeWidthInBit)) {
      padAmount = 2 * ldsParams->instBitWidth / typeWidthInBit;
    }
  } else {
    // Non-transposed path: each cycle 16 lanes at distinct nonK rows load
    // vecWidth consecutive K elements.  Padding shifts the row stride so
    // that gcd(stride_dwords, 64) is small enough for all lanes' bank
    // sets to be disjoint.
    //
    // For dword-or-wider elements (f32+): pad = min(vecWidth, 128/elemBits).
    // vecWidth elements = vecWidth dwords, giving gcd(stride_dwords, 64) =
    // vecWidth for power-of-2 BLOCK_K.  This is optimal:
    //   MFMA 16x16x4 f32 (kWidth=4): pad=4 → conflict-free
    //   MFMA 32x32x2 f32 (kWidth=2): pad=2 → conflict-free
    //
    // For sub-dword elements (fp16/fp8): keep pad = 128/elemBits (4 dwords).
    // On architectures with dual-address LDS loads (e.g. gfx1250
    // ds_load_2addr_b64 for fp8), two addresses are served simultaneously,
    // requiring the full 4-dword stride separation to avoid cross-address
    // bank conflicts.
    if (typeWidthInBit >= 32)
      padAmount = std::min(vecWidth, padAmount);
  }

  if (padAmount == 0 || padAmount >= static_cast<unsigned>(innerDimLength))
    return {};

  // When innerDimLength doesn't span all LDS banks, widen the padding
  // interval to the bank-wrap boundary so padding is inserted every N
  // rows instead of every row, at the point where the bank pattern
  // would repeat.
  constexpr unsigned ldsNumBanks = 64;
  constexpr unsigned ldsBankWidthInBytes = 4;
  unsigned elemBytes = typeWidthInBit / 8;
  unsigned bankWrapInterval = ldsNumBanks * ldsBankWidthInBytes / elemBytes;
  unsigned padInterval =
      std::max(static_cast<unsigned>(innerDimLength), bankWrapInterval);
  auto *context = srcTy.getContext();
  return triton::gpu::PaddedSharedEncodingAttr::get(
      context, {{padInterval, padAmount}}, order, shape, CGALayout);
}

ttg::SharedEncodingTrait getEncodingFromDescriptor(Operation *op,
                                                   RankedTensorType tensorType,
                                                   Value desc) {
  auto descBlockType = cast<tt::TensorDescType>(desc.getType()).getBlockType();
  auto encoding = cast<ttg::SharedEncodingTrait>(descBlockType.getEncoding());
  if (!encoding) {
    emitError(op->getLoc()) << "Missing encoding on the tensor descriptor";
    return {};
  }
  return ttg::updateEncodingForShape(op, encoding, tensorType);
}

ttg::PaddedSharedEncodingAttr
composePaddedLayout(const tt::AMD::TargetInfo &targetInfo, int opIdx,
                    unsigned vecWidth, ttg::TensorOrMemDesc srcTy,
                    ArrayRef<unsigned> sharedOrder,
                    ttg::DotOperandEncodingAttr dotOpEnc, bool useAsyncCopy) {
  if (targetInfo.getISAFamily() == triton::AMD::ISAFamily::CDNA4) {
    if (!dotOpEnc)
      return {};
    return composePaddedLayoutForAsyncCopyCDNA4(
        dotOpEnc, srcTy, sharedOrder, useAsyncCopy, targetInfo.getWarpSize());
  }

  if (targetInfo.getISAFamily() == triton::AMD::ISAFamily::GFX1250) {
    if (!srcTy.getElementType().isIntOrFloat())
      return {};
    return composePaddedLayoutWMMA(opIdx, vecWidth, srcTy, sharedOrder,
                                   targetInfo);
  }

  return {};
}
