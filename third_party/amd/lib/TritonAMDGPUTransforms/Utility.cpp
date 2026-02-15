#include "Utility.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/Triton/IR/Utility.h"
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
ttg::PaddedSharedEncodingAttr composePaddedLayoutForAsyncCopyCDNA4(
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

  // NYI: dtypes != 16bit
  if (elemByteWidth != 2) {
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

  if (!llvm::is_contained({4, 8}, kWidth)) {
    return {};
  }

  // Determine row(contig) size
  unsigned contigDim = isKContig ? kDim : nonKDim;
  unsigned nonContigDim = isKContig ? nonKDim : kDim;

  // padding to avoid bank conflict
  // For ds_read_b128. Lanes access LDS in 4 pairs of 16 lanes. we have 64 banks
  // and each lane loads 4 banks. These lane groups are:
  //  1: 0-3, 12-15, 20-23, 24-27
  //  2: 4-7, 8-11, 16-19, 28-31
  // The upper half of the lanes follow the same pattern.
  // For ds_read_b64, it splits conseuctive lanes into 2 groups which access LDS
  // one after another
  unsigned padding = 0;
  if (isKContig) {
    padding = mfmaNonKDim == 16 ? (kWidth * 2) : kWidth;
  } else {
    padding = mfmaNonKDim == 16 ? 16 : 32;
  }
  constexpr unsigned vecSize = 8; // in favor of dwordX4
  unsigned contigLanes = contigDim / vecSize;
  unsigned wrap = std::min(contigDim, 128u) / padding;
  unsigned requiredDim = warpSize / contigLanes * wrap;
  if (nonContigDim < requiredDim) {
    return {};
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

  // Keep contigSize numbers of elments contiguous in shared memory
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

  // Fixup for nonKContig and mfma16
  if (!isKContig && mfmaNonKDim == 16) {
    unsigned row4 = 0;
    unsigned row8 = 0;
    for (unsigned i = 0; i < bases.size(); i++) {
      if (bases[i][1] == 8)
        row8 = i;
      if (bases[i][1] == 4)
        row4 = i;
    }
    assert(row4 != 0 && row8 != 0);
    // lane groups wrap at row8, so we have to exchange
    // row4 and row8 to avoid bank conflict
    std::swap(bases[row4], bases[row8]);
  }

  // Fixup for KContig and mfma32 when reordered rows can not fit in 64banks
  if (isKContig && mfmaNonKDim == 32 && useBestWrap && kDim < 128) {
    bool useWideLayout = kWidth == 8;

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

ttg::PaddedSharedEncodingAttr
composePaddedLayout(const tt::AMD::TargetInfo &targetInfo,
                    ttg::DotOperandEncodingAttr dotOpEnc,
                    ttg::TensorOrMemDesc srcTy, ArrayRef<unsigned> sharedOrder,
                    bool useAsyncCopy) {
  if (useAsyncCopy &&
      targetInfo.getISAFamily() == triton::AMD::ISAFamily::CDNA4) {
    unsigned warpSize = targetInfo.getWarpSize();
    return composePaddedLayoutForAsyncCopyCDNA4(dotOpEnc, srcTy, sharedOrder,
                                                useAsyncCopy, warpSize);
  }
  return {};
}
