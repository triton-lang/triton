#include "TritonAMDGPUTransforms/Utility.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Tools/LayoutUtils.h"
// TODO: Not all of these are used. Remove redundant ones.
#include "TritonAMDGPUTransforms/MfmaGroup.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "TritonAMDGPUTransforms/WmmaGroup.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/DecomposeScaledBlocked.h"
#include "triton/Dialect/TritonGPU/Transforms/LayoutPropagationUtility.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/TypeSwitch.h"

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

  auto ctaLayout = ttg::getCTALayout(srcTy.getEncoding());
  triton::LinearLayout linearComponent(
      {
          {StringAttr::get(ctx, "offset"), bases},
      },
      triton::standardOutDimNames(ctx, rank));
  linearComponent = triton::gpu::combineCtaCgaWithShape(
      linearComponent, ctaLayout, srcTy.getShape());

  unsigned paddingInterval = 1024 / elemByteWidth;
  unsigned paddingInElems = paddingBytes / elemByteWidth;
  return ttg::PaddedSharedEncodingAttr::get(
      ctx, {{paddingInterval, paddingInElems}}, linearComponent);
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

// Chooses a proper MFMA instruction that can used to compute the given dot op.
// If enforcedNonKDim is not zero, it will be used to overwrite the default
// logic to choose a MFMA with matching M/N dim.
FailureOr<MfmaIntrinsic>
chooseMfmaInstruction(Location loc, int mfmaVersion, RankedTensorType cType,
                      Type aElemType, Type bElemType, int inputKSize,
                      int enforcedNonKDim, bool withScale, bool allowXF32) {
  // number of matrix elements along k dim per one MFMA instruction
  unsigned kDim = 0;

  auto resShape = cType.getShape();
  auto rank = resShape.size();
  auto M = resShape[rank - 2];
  auto N = resShape[rank - 1];

  unsigned mDim = 0;
  unsigned nDim = 0;
  if (enforcedNonKDim != 0) {
    mDim = nDim = enforcedNonKDim;
  } else {
    int minSize = std::min(M, N);
    if (minSize >= 32) {
      // On CNDA2-4, if the element type is f64, we use 16x16 intrinsic as
      // there's no 32x32 intrinsic.
      mDim = nDim = 32;
      if (aElemType.isF64() || bElemType.isF64()) {
        mDim = nDim = 16;
      }
    } else if (minSize >= 16) {
      mDim = nDim = 16;
    } else if (minSize >= 4) {
      if (M >= 64) {
        mDim = 64;
        nDim = 4;
      } else if (N >= 64) {
        mDim = 4;
        nDim = 64;
      }
    }
  }

  FailureOr<MfmaIntrinsic> maybeMfmaIntrinsic =
      MfmaIntrinsic::selectFor(loc, mfmaVersion, mDim, nDim, inputKSize,
                               aElemType, bElemType, withScale, allowXF32);

  // Fallback to FMA if the M/N dim is not supported by MFMA.
  if (failed(maybeMfmaIntrinsic)) {
    mlir::emitRemark(loc) << "Unable to select MFMA intrinsic for the request: "
                          << "version=" << mfmaVersion << ", result-shape=("
                          << M << "x" << N << "), selected-tiles=(" << mDim
                          << "x" << nDim << "), inputKSize=" << inputKSize
                          << ", aElemType=" << aElemType
                          << ", bElemType=" << bElemType
                          << ", withScale=" << (withScale ? "true" : "false")
                          << ", allowXF32=" << (allowXF32 ? "true" : "false")
                          << (enforcedNonKDim != 0
                                  ? (llvm::Twine(", enforcedNonKDim=") +
                                     llvm::Twine(enforcedNonKDim))
                                        .str()
                                  : "");
    return failure();
  }

  kDim = maybeMfmaIntrinsic->kDim;
  assert(kDim != 0);
  assert(enforcedNonKDim != 0 || (M % mDim == 0 && N % nDim == 0));
  // If inputKSize % kDim != 0 (including the case where inputKSize < kDim),
  // this layout will introduce data duplication.
  if (inputKSize % kDim != 0) {
    mlir::emitRemark(loc)
        << "Unable to select MFMA intrinsic '" << maybeMfmaIntrinsic->name
        << "' as MFMA intrinsic k-dimension size kDim=" << kDim
        << ", which is not a multiple of tile k-dimension size inputKSize="
        << inputKSize
        << ". Using this intrinsic would introduce data duplication.";
    return failure();
  }
  return maybeMfmaIntrinsic;
}

FailureOr<MfmaIntrinsic> chooseMfmaInstruction(tt::DotOp dot, int mfmaVersion,
                                               int nonKDim, bool withScale) {
  RankedTensorType aType = dot.getA().getType();
  bool allowXF32 =
      dot.getInputPrecision() == InputPrecision::TF32 && mfmaVersion == 3;
  return chooseMfmaInstruction(
      dot.getLoc(), mfmaVersion, dot.getC().getType(), aType.getElementType(),
      dot.getB().getType().getElementType(), aType.getShape().back(), nonKDim,
      withScale, allowXF32);
}

FailureOr<MfmaIntrinsic> chooseMfmaInstruction(tt::DotScaledOp dot,
                                               int mfmaVersion, int nonKDim) {
  using ::mlir::LLVM::AMD::scaleDotElemTypeToMLIRType;

  auto ctx = dot.getContext();
  int64_t inputKDim = dot.getA().getType().getShape().back();
  if (dot.getAElemType() == ScaleDotElemType::E2M1 && dot.getLhsKPack()) {
    // Since two fp4 are packed into int8, to get the correct K dim size, we
    // need to multiply it by 2.
    inputKDim *= 2;
  }
  Type aElemType = scaleDotElemTypeToMLIRType(ctx, dot.getAElemType());
  Type bElemType = scaleDotElemTypeToMLIRType(ctx, dot.getBElemType());
  return chooseMfmaInstruction(dot.getLoc(), mfmaVersion, dot.getC().getType(),
                               aElemType, bElemType, inputKDim, nonKDim,
                               /*withScale=*/true, /*allowXF32=*/false);
}

FailureOr<MfmaIntrinsic> chooseMfmaInstruction(tt::DotScaledOp dot,
                                               int mfmaVersion, int nonKDim,
                                               bool useFp16) {
  // For scaled dot, we handle it with fp16 or bf16 emulation for now.
  Builder b(dot.getContext());
  Type elemType = useFp16 ? b.getF16Type() : b.getBF16Type();
  return chooseMfmaInstruction(dot.getLoc(), mfmaVersion, dot.getC().getType(),
                               elemType, elemType,
                               dot.getA().getType().getShape().back(), nonKDim,
                               /*withScale=*/false, /*allowXF32=*/false);
}
