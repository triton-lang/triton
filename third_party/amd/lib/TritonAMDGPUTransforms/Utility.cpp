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
// start addresses to different shared memory banks. Generally it's enough to
// pad 16 continous rows (see exception below for mfma32 kContig). Therefore, we
// implement a linear mapping from logical tensor elements to shared memory
// offsets that:
// - Strides 16 consecutive rows by 1024 bytes in shared memory.
// - Fills "holes" by rows which are a multiple of 16
// For example, if each row is 256 bytes, four rows are required to fill the
// hole. The resulting reordering of rows in logical order is:
//   [r0, r16, r32, r48, r1, row17, row33, row49, row2, row18, ...]
// Corresponding byte offsets for these rows are:
//   [0,  256, 512, 768, 1024, ...]
// This approach naturally generalizes to other row sizes. For example, with
// 128-byte rows:
//   Logical row order: [r0, r16, r32, r48, r64, r80, r96, r112, r1, r17, ...]
//   Byte offsets:      [0,  128, 256, 384, ...,                 1024, ...]
// Since padding is applied in groups of 16 rows, the total data size for this
// layout must be at least 16 KB (16 * 1024 bytes).
ttg::PaddedSharedEncodingAttr composePaddedLayoutForAsyncCopyCDNA4(
    ttg::DotOperandEncodingAttr dotOpEnc, ttg::TensorOrMemDesc srcTy,
    ArrayRef<unsigned> sharedOrder, bool useAsyncCopy) {
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
  auto loadBytes = shape[0] * shape[1] * elemByteWidth;
  if (loadBytes < 16384) {
    return {};
  }

  // NYI: dtypes != 16bit
  if (elemByteWidth != 2) {
    return {};
  }

  // NYI: requires different stride factor since we stride by 16 rows
  if (std::min(shape[0], shape[1]) < 16) {
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

  // Clamp contigSize to 1024 bytes to have space for at least 16 rows per sub
  // tile (16KB) and simply repeat the tile to the full tensor size.
  contigDim = std::min(1024 / elemByteWidth, contigDim);

  // We create linear bases mapping from [contigDim, nonContigDim] -> offset,
  // representing the row reordering as described above
  std::vector<std::vector<int>> bases;
  // Keep contigSize numbers of elments contiguous in shared memory
  for (int elemLog2 = 0; elemLog2 < llvm::Log2_32(contigDim); elemLog2++)
    bases.push_back({1 << elemLog2, 0});

  // Add strided rows (by 16) to pad to 1024bytes
  auto requiredNumBases = llvm::Log2_32(1024U / elemByteWidth);
  for (int rowBase = llvm::Log2_32(16); bases.size() < requiredNumBases;
       rowBase++)
    bases.push_back({0, 1 << rowBase});

  // Add rows 1..16 afterwards to complete the tile
  for (int rowLog2 = 0; rowLog2 < llvm::Log2_32(16); rowLog2++)
    bases.push_back({0, 1 << rowLog2});

  // Compute required padding (in bytes) to avoid conflicts when accessing rows
  unsigned paddingBytes = 0;

  // To compute the required amount of padding to avoid bank conflicts we look
  // at the number of contiguous bytes loaded for a single row this directly
  // gives us the padding we require. Note for contigBytesPerLane == 16 we use a
  // different mfma layout (wide) compared to contigBytesPerLane == 8 (narrow)
  int contigBytesPerLane = kWidth * elemByteWidth;
  bool useWideLayout = contigBytesPerLane == 16;
  if (isKContig) {
    // For wide layouts we will use ds_read_b128. Lanes access LDS
    // (bank conflicts) in 4 pairs of 16 lanes since we have 64 banks and each
    // lane loads 4 banks. These (lane)groups are:
    //  1: 0-3, 12-15, 20-23, 24-27
    //  2: 4-7, 8-11, 16-19, 28-31
    // The upper half of the lanes follow the same pattern.
    // For narrow layouts we will use ds_read_b64 which splits conseuctive
    // lanes into 2 groups which access LDS one after another

    if (mfmaNonKDim == 16) {
      // For wide layouts lane groups read 32 contiguous bytes
      // For narrow layouts lane groups load 8 contiguous bytes
      paddingBytes = useWideLayout ? 32 : 8;
    }

    if (mfmaNonKDim == 32) {
      // For mfma32 32 lanes read 32 continuous rows. So for narrow layouts we
      // read 8 contiguous bytes and for wide layouts 16 bytes.
      paddingBytes = useWideLayout ? 16 : 8;

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
  } else {
    if (mfmaNonKDim == 16) {
      // For mfma16 lane groups read 32 contiguous bytes
      paddingBytes = 32;
      if (useWideLayout) {
        // For for the wide layout lane groups wrap at row 8 so we have to
        // exchange row4 and row8 to avoid conflicts (last two bases)
        std::swap(bases[bases.size() - 1], bases[bases.size() - 2]);
      }
    } else if (mfmaNonKDim == 32) {
      // For mfma32 lane groups read 64 contiguous bytes
      paddingBytes = 64;
    }
  }

  assert(paddingBytes != 0);

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

  unsigned paddingInterval = 1024 / elemByteWidth;
  unsigned paddingInElems = paddingBytes / elemByteWidth;
  return ttg::PaddedSharedEncodingAttr::get(
      ctx, {{paddingInterval, paddingInElems}}, std::move(linearComponent));
}

ttg::PaddedSharedEncodingAttr
composePaddedLayout(const tt::AMD::TargetInfo &targetInfo,
                    ttg::DotOperandEncodingAttr dotOpEnc,
                    ttg::TensorOrMemDesc srcTy, ArrayRef<unsigned> sharedOrder,
                    bool useAsyncCopy) {
  if (useAsyncCopy &&
      targetInfo.getISAFamily() == triton::AMD::ISAFamily::CDNA4) {
    return composePaddedLayoutForAsyncCopyCDNA4(dotOpEnc, srcTy, sharedOrder,
                                                useAsyncCopy);
  }
  return {};
}
