#include "TritonAMDGPUTransforms/Passes.h"
#include "amd/lib/TritonAMDGPUToLLVM/AsyncUtility.h"
#include "amd/lib/TritonAMDGPUToLLVM/TargetInfo.h"
#include "amd/lib/TritonAMDGPUTransforms/PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/Support/Debug.h"
#include <variant>

#define DEBUG_TYPE "tritonamdgpu-pipeline-lower-loops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

//===----------------------------------------------------------------------===//
// This file will conditionally allocate lds memory, create local/async load
// operations, and create schedule for these operations. After lowerLoops,
// schedule will be passed to expandLoops and eventually to PipelineExpander.
//===----------------------------------------------------------------------===//

using mlir::triton::AMD::AttrBypassLDS;

namespace mlir {
struct StreamCopyChainOps {
  tt::LoadOp loadOp;
  ttg::MemDescIndexOp subviewOp;
  ttg::LocalStoreOp localStoreOp;
  ttg::LocalLoadOp maybeLocalLoadOp;
};

struct AsyncCopyChainOps {
  ttg::AsyncCopyGlobalToLocalOp copyOp;
  ttg::AsyncCommitGroupOp commitOp;
  ttg::AsyncWaitOp waitOp;
  ttg::LocalLoadOp maybeLocalLoadOp;
};

using StreamOpVariant = std::variant<StreamCopyChainOps, AsyncCopyChainOps>;
using LoadToStreamOpMap = llvm::MapVector<Operation *, StreamOpVariant>;

bool canBeConvertedToAsyncLoad(unsigned numBuffers, tt::LoadOp loadOp,
                               ttg::SharedEncodingTrait sharedEnc,
                               tt::ModuleAxisInfoAnalysis &axisInfoAnalysis,
                               const tt::AMD::TargetInfo &targetInfo);

AsyncCopyChainOps createAsyncCopy(tt::LoadOp loadOp, Value alloc,
                                  Value extractIdx) {
  OpBuilder builder(loadOp);
  Location loc = loadOp.getLoc();

  // Extract local subview from shared allocation
  auto viewLoad = triton::createSingleBufferView(builder, alloc, extractIdx)
                      .getDefiningOp<ttg::MemDescIndexOp>();

  auto copyOp = builder.create<ttg::AsyncCopyGlobalToLocalOp>(
      loc, loadOp.getPtr(), viewLoad, loadOp.getMask(), loadOp.getOther(),
      loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
  auto commitOp =
      builder.create<ttg::AsyncCommitGroupOp>(loc, copyOp->getResult(0));
  ttg::AsyncWaitOp waitOp =
      builder.create<ttg::AsyncWaitOp>(loc, commitOp->getResult(0), 0);

  auto maybeSharedLoad = tt::replaceUsesWithLocalLoad(
      builder, loadOp->getResult(0), viewLoad, waitOp);

  return {copyOp, commitOp, waitOp, maybeSharedLoad};
}

void scheduleLocalLoad(ttg::LocalLoadOp localLoadOp,
                       tt::CoarseSchedule &schedule, int stage,
                       const tt::CoarseSchedule::Cluster &cluster) {
  schedule.insert(localLoadOp, stage, cluster);
  // If its only user is a ConvertLayout, we place it into the same stage so
  // it can be folded by a later pass
  if (localLoadOp->hasOneUse()) {
    auto cvt = *localLoadOp->getUsers().begin();
    if (isa<ttg::ConvertLayoutOp>(cvt)) {
      schedule.insert(cvt, stage, cluster);
    }
  }
}

StreamCopyChainOps createStreamCopy(tt::LoadOp loadOp, Value alloc,
                                    Value extractIdx) {
  OpBuilder builder(loadOp);
  Location loc = loadOp.getLoc();

  // Extract local subview from shared allocation
  auto viewLoad = triton::createSingleBufferView(builder, alloc, extractIdx)
                      .getDefiningOp<ttg::MemDescIndexOp>();

  tt::LoadOp newLoadOp = cast<tt::LoadOp>(builder.clone(*loadOp));
  auto storeOp = builder.create<ttg::LocalStoreOp>(loc, newLoadOp, viewLoad);
  auto maybeLocalLoad =
      tt::replaceUsesWithLocalLoad(builder, loadOp->getResult(0), viewLoad);

  return {newLoadOp, viewLoad, storeOp, maybeLocalLoad};
}

// Returns the given |inputValue|'s dot user result encoding and updates |opIdx|
// and |vecSize| with which dot operand |inputValue| is fed into if possible.
ttg::AMDMfmaEncodingAttr getDotEncoding(Value inputValue, unsigned *opIdx,
                                        unsigned *vecSize) {
  if (!inputValue.hasOneUse())
    return nullptr;

  Operation *user = *inputValue.getUsers().begin();
  if (user->getNumResults() != 1 ||
      user->getBlock() != inputValue.getParentBlock())
    return nullptr;

  LDBG("getDotEncoding user: " << *user);
  if (auto dotOp = dyn_cast<tt::DotOpInterface>(user)) {
    OpOperand &use = *inputValue.getUses().begin();
    *opIdx = use.getOperandNumber();
    auto operandType = cast<RankedTensorType>(inputValue.getType());
    *vecSize = ttg::toLinearLayout(operandType).getNumConsecutiveInOut();
    auto dotType = cast<RankedTensorType>(dotOp->getResult(0).getType());
    return dyn_cast<ttg::AMDMfmaEncodingAttr>(dotType.getEncoding());
  }

  return getDotEncoding(user->getResult(0), opIdx, vecSize);
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
std::optional<ttg::PaddedSharedEncodingAttr>
composePaddedLayoutForAsyncCopyCDNA4(ttg::DotOperandEncodingAttr dotOpEnc,
                                     ttg::TensorOrMemDesc srcTy,
                                     ArrayRef<unsigned> order,
                                     ArrayRef<unsigned> sharedOrder,
                                     unsigned bitWidth, bool useAsyncCopy) {
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

  unsigned elemByteWidth = std::max(bitWidth / 8u, 1u);
  auto loadBytes = shape[0] * shape[1] * elemByteWidth;
  if (loadBytes < 16384) {
    return {};
  }

  // NYI: dtypes != 16bit
  if (elemByteWidth != 2) {
    return {};
  }

  // NYI: requires different stride factor
  if (std::min(shape[0], shape[1]) < 32) {
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

  // We clamp contigSize to 512 bytes (to reduce the number of cases handled
  // below) and simply repeat the tile to the full tensor size.
  contigDim = std::min(512U / elemByteWidth, contigDim);

  std::vector<std::vector<int>> bases;
  // Keep contigSize numbers of elments contig in shared memory
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
      // For mfma32 lane groups read 16 contiguous bytes
      paddingBytes = 16;
      // Because we load 32 rows we need to place every 16th row into the second
      // half of banks to spread loads over all banks. This is done by swapping
      // the bases representing offset 128 (32banks) and the base of the 16th
      // row which depends on the contigDim
      int row16Index = contigDim == 32 ? 5 : (contigDim == 64 ? 6 : 7);
      int upperBankIndex = llvm::Log2_32(128 / elemByteWidth);
      assert(row16Index < bases.size());
      assert(upperBankIndex < bases.size());
      std::swap(bases[upperBankIndex], bases[row16Index]);
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

std::optional<ttg::PaddedSharedEncodingAttr> composePaddedLayout(
    const tt::AMD::TargetInfo &targetInfo, ttg::DotOperandEncodingAttr dotOpEnc,
    ttg::TensorOrMemDesc srcTy, ArrayRef<unsigned> order,
    ArrayRef<unsigned> sharedOrder, unsigned bitWidth, bool useAsyncCopy) {
  if (useAsyncCopy &&
      targetInfo.getISAFamily() == triton::AMD::ISAFamily::CDNA4) {
    return composePaddedLayoutForAsyncCopyCDNA4(
        dotOpEnc, srcTy, order, sharedOrder, bitWidth, useAsyncCopy);
  }
  return {};
}

// Adapted from
// lib/Dialect/TritonGPU/Transforms/Utility.cpp::getSharedEncIfAllUsersAreDotEnc
// to support AMDMfmaEncodingAttr.
// TODO(max): figure out how to refactor to use upstream
//
// If all the transitive uses of the given value have are used by a convert to
// the same dot operand encoding, return true and get the shared encoding that
// needs to be used to be compatible with users' layouts.
std::optional<ttg::SharedEncodingTrait> getSharedEncIfAllUsersAreDotEnc(
    Operation *loadOp, tt::ModuleAxisInfoAnalysis &axisInfoAnalysis,
    const tt::AMD::TargetInfo &targetInfo, bool useAsyncCopy) {
  assert(loadOp);
  Value loadedValue = loadOp->getResult(0);
  llvm::SmallVector<ttg::SharedEncodingTrait> sharedEncs;
  for (Operation *user : loadedValue.getUsers()) {
    LDBG(" getSharedEncIfAllUsersAreDotEnc current user: " << *user);
    if (user->getNumResults() != 1)
      return std::nullopt;

    ttg::SharedEncodingTrait tempAttr;
    Value userResult = user->getResult(0);
    Type userResType = userResult.getType();
    if (auto memDesc = dyn_cast<ttg::MemDescType>(userResType)) {
      // First time we find a shared encoding in the chain, save it and try to
      // use it if it is compatible with the other users.
      tempAttr = cast<ttg::SharedEncodingTrait>(memDesc.getEncoding());
      // If the immediate user is ttg::LocalAllocOp, likely it's created in
      // TritonAMDGPUOptimizeDotOperands. We should just respect it.
      if (!getSharedEncIfAllUsersAreDotEnc(user, axisInfoAnalysis, targetInfo,
                                           useAsyncCopy)
               .has_value() &&
          !isa<ttg::LocalAllocOp>(user)) {
        return std::nullopt;
      }
      LDBG("Deduced shared encoding candidate from memDesc: " << tempAttr);
      sharedEncs.push_back(tempAttr);
    } else {
      if (!(isa<ttg::ConvertLayoutOp>(user) ||
            user->hasTrait<OpTrait::LocalLoadTrait>()))
        return std::nullopt;

      auto srcTy = cast<ttg::TensorOrMemDesc>(loadedValue.getType());
      auto ctaLayout = ttg::getCTALayout(srcTy.getEncoding());
      auto order = getOrderForMemory(srcTy);
      unsigned bitWidth = srcTy.getElementType().getIntOrFloatBitWidth();
      SmallVector<unsigned> sharedOrder;
      int rank = order.size();
      // TODO rework this when shared -> dotOperand conversions support
      // arbitrary shared memory ordering
      if (rank == 3) {
        // Move the batch dimension (dim #0) to be the last so that it will be
        // the slowest varying dimension.
        for (unsigned i = 0; i < rank; ++i)
          if (order[i] != 0)
            sharedOrder.emplace_back(order[i]);
        sharedOrder.emplace_back(0);
      } else {
        sharedOrder = order;
      }

      auto userResEnc = cast<ttg::TensorOrMemDesc>(userResType).getEncoding();
      if (auto dotOpEnc = dyn_cast<ttg::DotOperandEncodingAttr>(userResEnc)) {

        // Determine if we can use padded layouts and fallback to swizzled
        // layouts if not
        bool canUseAsyncCopy = false;
        if (useAsyncCopy && isa<tt::LoadOp>(loadOp)) {
          canUseAsyncCopy = canBeConvertedToAsyncLoad(
              2, cast<tt::LoadOp>(loadOp), {}, axisInfoAnalysis, targetInfo);
        }
        if (auto maybePadded =
                composePaddedLayout(targetInfo, dotOpEnc, srcTy, order,
                                    sharedOrder, bitWidth, canUseAsyncCopy)) {
          tempAttr = *maybePadded;
        } else {
          tempAttr = ttg::SwizzledSharedEncodingAttr::get(
              loadedValue.getContext(), dotOpEnc, srcTy.getShape(), sharedOrder,
              ctaLayout, bitWidth, /*needTrans=*/false);
        }
        LDBG("Deduced shared encoding candidate from dot layout: " << tempAttr);
        sharedEncs.push_back(tempAttr);
      } else if (auto llEnc = dyn_cast<ttg::LinearEncodingAttr>(userResEnc)) {
        // We use linear layout directly for scaled dot fp8 operands. For such
        // cases, we need to look further down the def-use chain to find the dot
        // op for the mfma layout to deduce operand index and other information.
        unsigned opIdx;
        unsigned vecSize;
        if (auto mfmaEnc = getDotEncoding(userResult, &opIdx, &vecSize)) {
          LDBG("deduced opIdx: " << opIdx << "; deduced vecSize: " << vecSize);
          tempAttr = mfmaEnc.composeSharedLayoutForOperand(
              ctaLayout, opIdx, srcTy.getShape(), order, vecSize, bitWidth,
              /*needTrans=*/false);
          LDBG("Deduced shared encoding candidate from mfma layout: "
               << tempAttr);
          sharedEncs.push_back(tempAttr);
        }
      }
    }
  }

  auto equalSharedEncIgnoreVec = [](ttg::SwizzledSharedEncodingAttr a,
                                    ttg::SwizzledSharedEncodingAttr b) {
    if (!a || !b)
      return false;
    return (a.getPerPhase() == b.getPerPhase() &&
            a.getMaxPhase() == b.getMaxPhase() &&
            a.getOrder() == b.getOrder() &&
            a.getCTALayout() == b.getCTALayout());
  };
  if (sharedEncs.empty() || !sharedEncs.front())
    return std::nullopt;
  auto maxVecSharedEnc = sharedEncs.front();

  // TODO add heuristic support for padded layouts
  for (auto sharedEnc : llvm::drop_begin(sharedEncs, 1)) {
    if (!equalSharedEncIgnoreVec(
            dyn_cast<ttg::SwizzledSharedEncodingAttr>(sharedEnc),
            dyn_cast<ttg::SwizzledSharedEncodingAttr>(maxVecSharedEnc))) {
      LDBG("Incompatible shared encodings");
      return std::nullopt;
    }
    if (cast<ttg::SwizzledSharedEncodingAttr>(sharedEnc).getVec() >
        cast<ttg::SwizzledSharedEncodingAttr>(maxVecSharedEnc).getVec()) {
      maxVecSharedEnc = sharedEnc;
    }
  }

  LDBG("Deduced shared encoding: " << maxVecSharedEnc);

  return maxVecSharedEnc;
}

bool canBeConvertedToAsyncLoad(unsigned numBuffers, tt::LoadOp loadOp,
                               ttg::SharedEncodingTrait sharedEnc,
                               tt::ModuleAxisInfoAnalysis &axisInfoAnalysis,
                               const tt::AMD::TargetInfo &targetInfo) {
  // If we have a single buffer we would require another barrier after the
  // local_reads so instead we fall back to pipeline with registers
  // Removing this check will create incorrect IR, see
  // MembarUtility.h:membarFilter
  if (numBuffers <= 1)
    return false;

  using tt::AMD::ISAFamily;
  if (sharedEnc && llvm::is_contained({ISAFamily::CDNA3, ISAFamily::CDNA4},
                                      targetInfo.getISAFamily())) {
    // Compute the final vecSize we can use for the combination of
    // sourceEncoding and sharedEncoding. We can only use AsyncCopy if the
    // target supports the requested or a smaller vecSize because we cannot
    // stride when loading directly to lds on GFX9
    auto srcTy = cast<RankedTensorType>(loadOp.getPtr().getType());
    auto regLayout = triton::gpu::toLinearLayout(srcTy);
    // It's the allocation so we trim the multibuffer dimension
    auto srcShape = srcTy.getShape();
    triton::LinearLayout sharedLayout;
    auto paddedEnc = dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(sharedEnc);
    if (paddedEnc) {
      sharedLayout = paddedEnc.getLinearComponent();
    } else {
      sharedLayout = triton::gpu::toLinearLayout(srcShape, sharedEnc);
    }
    auto regToSharedLayout = regLayout.invertAndCompose(sharedLayout);

    unsigned vecSize = regToSharedLayout.getNumConsecutiveInOut();
    if (paddedEnc) {
      vecSize = std::min(vecSize, paddedEnc.getMinInterval());
    }
    unsigned elemBitWidth = tt::getPointeeBitWidth(srcTy);

    if (fitToValidDirectToLdsVecSize(vecSize, elemBitWidth, targetInfo) == 0)
      return false;
  }

  // Checks whether the global pointer's contiguity and mask alignment allows
  // for at least 32 bit wide loads
  return triton::canBeConvertedToAsyncLoad(loadOp, axisInfoAnalysis);
}

// Convert load ops into shared memory allocation loads and apply
// multi-buffering based on the required number of buffers.
LoadToStreamOpMap
createStreamOps(const LoadToInfoMap &loadToInfo, scf::ForOp &forOp,
                const int &numBuffers, bool useAsyncCopy,
                tt::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  IRRewriter builder(forOp);
  Location loc = forOp.getLoc();
  Value minusOne = builder.create<arith::ConstantIntOp>(loc, -1, 32);
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
  Value extractIdx = minusOne;
  Value numBuffersVal =
      builder.create<arith::ConstantIntOp>(loc, numBuffers, 32);

  unsigned newOperandIndex = forOp.getBody()->getNumArguments();
  // Patch the loop to add the new loop carried dependency.
  forOp = addIterArgsToLoop(builder, forOp, {extractIdx});

  // Create one counter for the extract indices to avoid creating long
  // live range.
  extractIdx = forOp.getBody()->getArgument(newOperandIndex);

  builder.setInsertionPoint(forOp.getBody(), forOp.getBody()->begin());
  extractIdx = builder.create<arith::AddIOp>(loc, extractIdx, one);
  Value cndExt = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                               extractIdx, numBuffersVal);
  extractIdx = builder.create<arith::SelectOp>(loc, cndExt, extractIdx, zero);

  // Patch the yield with the updated counter.
  appendToForOpYield(forOp, {extractIdx});

  LoadToStreamOpMap loadToStreamOp;
  for (auto &[l, info] : loadToInfo) {
    if (!info.sharedEncoding)
      continue;

    auto loadOp = dyn_cast<tt::LoadOp>(l);
    if (!loadOp)
      continue;

    // Create an allocation that can hold distance number of loadOp shapes.
    auto ty = cast<RankedTensorType>(loadOp->getResultTypes()[0]);
    Value alloc = triton::createAlloc(forOp, ty, loadOp->getLoc(),
                                      info.sharedEncoding, numBuffers);
    assert(alloc && "Failed to create alloc for the async load.");
    auto arch = getAMDArch(loadOp->getParentOfType<ModuleOp>());
    triton::AMD::TargetInfo targetInfo(arch ? arch->str() : "");

    // Replace the old load with multi-buffered loads
    if (useAsyncCopy &&
        canBeConvertedToAsyncLoad(numBuffers, loadOp, info.sharedEncoding,
                                  axisInfoAnalysis, targetInfo)) {
      loadToStreamOp[loadOp] = createAsyncCopy(loadOp, alloc, extractIdx);
    } else {
      loadToStreamOp[loadOp] = createStreamCopy(loadOp, alloc, extractIdx);
    }
  }

  return loadToStreamOp;
}

static void dumpSchedule(tt::CoarseSchedule &schedule, llvm::StringRef msg) {
  LLVM_DEBUG({
    llvm::dbgs() << "\n";
    LDBG(msg);
    schedule.dump();
  });
};

namespace SingleDotSchedule {
using namespace mlir::SingleDotSchedule;
using ClusterMap = DenseMap<tt::CoarseSchedule::ClusterHash, int>;

ClusterMap createClusterMap(tt::CoarseSchedule &schedule) {
  DenseMap<tt::CoarseSchedule::ClusterHash, int> clusterMap;
  for (auto &[op, stageAndCluster] : schedule.opToStageAndCluster) {
    auto [stage, cluster] = stageAndCluster;
    tt::CoarseSchedule::ClusterHash clusterHash =
        tt::CoarseSchedule::hashCluster(cluster);
    clusterMap[clusterHash] = *cluster;
  }

  return clusterMap;
}

// Remap global and compute clusters to the right place
void remapClusters(tt::CoarseSchedule &schedule, ClusterMap clusterMap,
                   Clusters &clusters) {
  for (auto &[op, stageAndCluster] : schedule.opToStageAndCluster) {
    auto [stage, cluster] = stageAndCluster;
    tt::CoarseSchedule::ClusterHash clusterHash =
        tt::CoarseSchedule::hashCluster(stageAndCluster.second);
    int oldClusterId = clusterMap[clusterHash];
    if (oldClusterId == 0) {
      stageAndCluster.second = clusters[SCHED_GLOBAL_LOAD];
    } else {
      assert(oldClusterId == 1);
      stageAndCluster.second = clusters[SCHED_COMPUTE];
    }
  }
}

// Init Schedule Config based on settings and loop characteristics.
// Create clusters in order of ops in loop. This can interleave ops
// from different stages in the same cluster to achieve better backend
// scheduling.
//   WARNING: Changing the order of schedule.clusters.newAtBack() calls
//            can cause invalid schedules to be produced.
LogicalResult initSchedule(int maxDist, Stages &stages, int numStages,
                           int &numBuffers, bool useAsyncCopy, bool waitAtTail,
                           Clusters &clusters, tt::CoarseSchedule &schedule) {
  LDBG("Init SingleDotSchedule");
  int lastStage = numStages - 1;
  stages[SCHED_GLOBAL_LOAD] = 0;
  stages[SCHED_LOCAL_STORE] = 0;
  stages[SCHED_LOCAL_LOAD] = lastStage;
  stages[SCHED_COMPUTE] = lastStage;
  stages[SCHED_ASYNC_WAIT] = stages[SCHED_LOCAL_LOAD];

  bool pairedGlobalLoadLocalStore = stages[SCHED_LOCAL_STORE] == 0;
  stages[SCHED_LOCAL_STORE] += maxDist;
  if (waitAtTail) {
    stages[SCHED_ASYNC_WAIT] = std::max(0, stages[SCHED_LOCAL_LOAD] - 1);
  }

  LDBG(
      "Stage schedule:" << "  GLOBAL_LOAD stage = " << stages[SCHED_GLOBAL_LOAD]
                        << ", LOCAL_STORE stage = " << stages[SCHED_LOCAL_STORE]
                        << ", LOCAL_LOAD stage = " << stages[SCHED_LOCAL_LOAD]
                        << ", COMPUTE stage = " << stages[SCHED_COMPUTE]
                        << ", ASYNC_WAIT stage = " << stages[SCHED_ASYNC_WAIT]
                        << "; total = " << numStages);

  if (stages[SCHED_LOCAL_STORE] >= numStages ||
      stages[SCHED_LOCAL_STORE] > stages[SCHED_LOCAL_LOAD]) {
    LDBG("Invalid stage schedule");
    return failure();
  }

  // Calculate the number of buffers needed for each load.
  // TODO: Use the precise number of buffers needed by the particular load.
  numBuffers =
      std::max(1, stages[SCHED_LOCAL_LOAD] - stages[SCHED_LOCAL_STORE]);
  // If we use AsyncCopy we need one more buffer since we are not using a
  // register buffer
  if (useAsyncCopy) {
    numBuffers += 1;
  }

  LDBG("deduced max shared memory buffer number = " << numBuffers);

  // We place async wait as the first cluster because we want to have it being
  // the first in the main loop after pipelining.
  // In case we use async_copy with pingpong, we need to place async_wait at
  // the end of the previous iteration, so it can guarantee the correct
  // dependency when warp0 and warp1 are pipelined.
  int asyncWaitCluster = waitAtTail ? 4 : 0;
  // If tt.load and ttg.local_store are in the same stage
  //   spread them apart to allow overlap with compute
  // else
  //   Initiate ttg.local_store before tt.load
  int globalLoadCluster = 1;
  int localStoreCluster = 3;
  if (!pairedGlobalLoadLocalStore) {
    globalLoadCluster = 3;
    localStoreCluster = 2;
  }

  // If ttg.local_load and ttg.local_store are in the same stage
  //   spread them apart to allow overlap with compute
  // else if they share the buffer
  //   ttg.local_load must come first
  // else
  //   schedule ttg.local_load in the middle
  int localLoadCluster = globalLoadCluster;
  if (stages[SCHED_LOCAL_LOAD] == stages[SCHED_LOCAL_STORE]) {
    localLoadCluster = std::max(3, localStoreCluster + 1);
  } else if (numBuffers == 1 && localLoadCluster >= localStoreCluster) {
    // For 1 buffer, ttg.local_load must occur before ttg.local_store
    localLoadCluster = localStoreCluster - 1;
  }

  // Schedule compute with ttg.local_load if paired
  // otherwise, schedule in the middle
  int computeCluster = 2;
  if (stages[SCHED_LOCAL_LOAD] == stages[SCHED_COMPUTE]) {
    computeCluster = localLoadCluster;
  }

  // Create a hash map to associate cluster hash in old schedule with its
  // clusterID
  ClusterMap clusterMap = createClusterMap(schedule);

  // Make assignments
  Clusters clusterVec;
  schedule.clusters.clear();
  std::generate(clusterVec.begin(), clusterVec.end(),
                [&]() { return schedule.clusters.newAtBack(); });

  clusters[SCHED_GLOBAL_LOAD] = clusterVec[globalLoadCluster];
  clusters[SCHED_LOCAL_STORE] = clusterVec[localStoreCluster];
  clusters[SCHED_LOCAL_LOAD] = clusterVec[localLoadCluster];
  clusters[SCHED_COMPUTE] = clusterVec[computeCluster];
  clusters[SCHED_ASYNC_WAIT] = clusterVec[asyncWaitCluster];

  remapClusters(schedule, clusterMap, clusters);

  LDBG("Cluster schedule:" << "  GLOBAL_LOAD cluster = " << globalLoadCluster
                           << ", LOCAL_STORE cluster = " << localStoreCluster
                           << ", LOCAL_LOAD cluster = " << localLoadCluster
                           << ", COMPUTE cluster = " << computeCluster
                           << ", ASYNC_WAIT cluster = " << asyncWaitCluster
                           << "; total = " << SCHED_SIZE);

  return success();
}

void scheduleAsyncCopy(const AsyncCopyChainOps &asyncOps, tt::LoadOp loadOp,
                       tt::CoarseSchedule &schedule, const Stages &stages,
                       const Clusters &clusters) {
  auto [copyOp, commitOp, waitOp, maybeLocalLoadOp] = asyncOps;
  auto [loadStage, loadCluster] = schedule[loadOp];
  schedule.insert(copyOp, loadStage, loadCluster);
  // Place ttg.async_commit_group op following AsyncCopyGlobalToLocal so the
  // later UpdateAsyncWaitCount pass can deduce better waitcnts
  schedule.insert(commitOp, loadStage, loadCluster);
  // If the LocalLoads are scheduled to a later stage than AsyncCopy we need to
  // place the AsyncCopy prefetches after the AsyncWaits which create a barrier
  // to ensure all warps are finished reading the shared buffer we will write
  // into. This is done by scheduling AsyncWait as the first cluster.
  // If AsyncCopy and LocalLoads are in the same stage we do not assign a
  // schdule so they are placed before the LocalLoads
  if (loadStage != stages[SCHED_LOCAL_LOAD])
    schedule.insert(waitOp, stages[SCHED_ASYNC_WAIT],
                    clusters[SCHED_ASYNC_WAIT]);

  if (maybeLocalLoadOp && stages[SCHED_LOCAL_LOAD] != stages[SCHED_COMPUTE]) {
    scheduleLocalLoad(maybeLocalLoadOp, schedule, stages[SCHED_LOCAL_LOAD],
                      clusters[SCHED_LOCAL_LOAD]);
  }
}

void scheduleStreamCopy(const StreamCopyChainOps &streamOps,
                        tt::LoadOp oldLoadOp, tt::CoarseSchedule &schedule,
                        const Stages &stages, const Clusters &clusters) {
  auto [newLoadOp, subviewOp, localStoreOp, maybeLocalLoadOp] = streamOps;
  auto [loadStage, loadCluster] = schedule[oldLoadOp];

  schedule.insert(newLoadOp, loadStage, loadCluster);
  schedule.insert(subviewOp, stages[SCHED_LOCAL_STORE],
                  clusters[SCHED_LOCAL_STORE]);
  schedule.insert(localStoreOp, stages[SCHED_LOCAL_STORE],
                  clusters[SCHED_LOCAL_STORE]);
  if (maybeLocalLoadOp && stages[SCHED_LOCAL_LOAD] != stages[SCHED_COMPUTE]) {
    scheduleLocalLoad(maybeLocalLoadOp, schedule, stages[SCHED_LOCAL_LOAD],
                      clusters[SCHED_LOCAL_LOAD]);
  }
}

void scheduleStreamOps(const LoadToStreamOpMap &loadToStreamOp,
                       tt::CoarseSchedule &schedule, const Stages &stages,
                       const Clusters &clusters) {
  for (auto [l, streamOps] : loadToStreamOp) {
    auto loadOp = dyn_cast<tt::LoadOp>(l);
    if (!loadOp)
      continue;

    if (auto asyncOps = std::get_if<AsyncCopyChainOps>(&streamOps)) {
      scheduleAsyncCopy(*asyncOps, loadOp, schedule, stages, clusters);
    } else if (auto sOps = std::get_if<StreamCopyChainOps>(&streamOps)) {
      scheduleStreamCopy(*sOps, loadOp, schedule, stages, clusters);
    }
  }
}

void updateSchedule(scf::ForOp &forOp, const LoadToInfoMap &loadToInfo,
                    tt::CoarseSchedule &schedule,
                    triton::AMD::ModuleAxisInfoAnalysis &axisInfoAnalysis,
                    int numStages, bool useAsyncCopy, bool waitAtTail) {
  LDBG("SingleDotSchedule::updateSchedule");
  Stages stages;
  Clusters clusters;

  int maxDist = 0;
  for (auto &[l, info] : loadToInfo) {
    maxDist = std::max(maxDist, info.distToUse);
  }

  int numBuffers = 1;
  if (failed(initSchedule(maxDist, stages, numStages, numBuffers, useAsyncCopy,
                          waitAtTail, clusters, schedule)))
    return;

  // Convert the loads into shared memory allocations and loads from them.
  auto loadToStreamOps = createStreamOps(loadToInfo, forOp, numBuffers,
                                         useAsyncCopy, axisInfoAnalysis);

  scheduleStreamOps(loadToStreamOps, schedule, stages, clusters);
  dumpSchedule(schedule, "Coarse schedule stream ops:");

  scheduleDependencies(forOp, schedule);
  dumpSchedule(schedule, "Coarse schedule with dependencies:");
  ttg::scheduleDistanceOneDependencies(forOp, schedule);
  dumpSchedule(schedule, "Coarse schedule with dist 1:");
  tt::CoarseSchedule::Cluster computeCluster = clusters[SCHED_COMPUTE];
  ttg::scheduleRemainingToLastStage(forOp, schedule, computeCluster);
  dumpSchedule(schedule, "Final coarse schedule:");
}
} // namespace SingleDotSchedule

namespace ChainedDotSchedule {
using namespace mlir::ChainedDotSchedule;

void scheduleAsyncCopy(const AsyncCopyChainOps &asyncOps, tt::LoadOp loadOp,
                       tt::CoarseSchedule &schedule,
                       const ChainedDotClusters &clusters) {
  auto [loadStage, loadCluster] = schedule[loadOp];
  auto [copyOp, commitOp, waitOp, maybeLocalLoadOp] = asyncOps;

  schedule.insert(copyOp, loadStage, loadCluster);
  // Place ttg.async_commit_group op following AsyncCopyGlobalToLocal so the
  // later UpdateAsyncWaitCount pass can deduce better waitcnts
  schedule.insert(commitOp, loadStage, loadCluster);

  if (loadStage == STAGE_GLOBAL_LOAD_1) {
    schedule.insert(waitOp, STAGE_LOCAL_LOAD_1, clusters[CLUSTER_ASYNC_WAIT_1]);
    if (maybeLocalLoadOp)
      scheduleLocalLoad(maybeLocalLoadOp, schedule, STAGE_LOCAL_LOAD_1,
                        clusters[CLUSTER_LOCAL_LOAD_1]);
  } else {
    schedule.insert(waitOp, STAGE_LOCAL_LOAD_2, clusters[CLUSTER_ASYNC_WAIT_2]);
    if (maybeLocalLoadOp)
      scheduleLocalLoad(maybeLocalLoadOp, schedule, STAGE_LOCAL_LOAD_2,
                        clusters[CLUSTER_LOCAL_LOAD_2]);
  }
}

void scheduleStreamCopy(const StreamCopyChainOps &streamOps, tt::LoadOp loadOp,
                        tt::CoarseSchedule &schedule,
                        const ChainedDotClusters &clusters) {
  auto [loadStage, loadCluster] = schedule[loadOp];
  auto [copyOp, subviewOp, localStoreOp, maybeLocalLoadOp] = streamOps;
  schedule.insert(copyOp, loadStage, loadCluster);

  if (loadStage == STAGE_GLOBAL_LOAD_1) {
    schedule.insert(subviewOp, STAGE_LOCAL_WRITE_1,
                    clusters[CLUSTER_LOCAL_WRITE_1]);
    schedule.insert(localStoreOp, STAGE_LOCAL_WRITE_1,
                    clusters[CLUSTER_LOCAL_WRITE_1]);

    if (maybeLocalLoadOp)
      schedule.insert(maybeLocalLoadOp, STAGE_LOCAL_LOAD_1,
                      clusters[CLUSTER_LOCAL_LOAD_1]);
  } else {
    schedule.insert(subviewOp, STAGE_LOCAL_WRITE_2,
                    clusters[CLUSTER_LOCAL_WRITE_2]);
    schedule.insert(localStoreOp, STAGE_LOCAL_WRITE_2,
                    clusters[CLUSTER_LOCAL_WRITE_2]);
    if (maybeLocalLoadOp)
      schedule.insert(maybeLocalLoadOp, STAGE_LOCAL_LOAD_2,
                      clusters[CLUSTER_LOCAL_LOAD_2]);
  }

  if (maybeLocalLoadOp) {
    if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(
            *maybeLocalLoadOp->getUsers().begin())) {
      auto [localLoadStage, localLoadCluster] = schedule[maybeLocalLoadOp];
      schedule.insert(cvt, localLoadStage, localLoadCluster);
    }
  }
}

void scheduleStreamOps(const LoadToStreamOpMap &loadToStreamOp,
                       tt::CoarseSchedule &schedule,
                       const ChainedDotClusters &clusters) {
  for (auto [l, streamOps] : loadToStreamOp) {
    auto loadOp = dyn_cast<tt::LoadOp>(l);
    if (!loadOp)
      continue;

    if (auto asyncOps = std::get_if<AsyncCopyChainOps>(&streamOps)) {
      scheduleAsyncCopy(*asyncOps, loadOp, schedule, clusters);
    } else if (auto sOps = std::get_if<StreamCopyChainOps>(&streamOps)) {
      scheduleStreamCopy(*sOps, loadOp, schedule, clusters);
    }
  }
}

void updateSchedule(scf::ForOp &forOp, const LoadToInfoMap &loadToInfo,
                    tt::CoarseSchedule &schedule,
                    triton::AMD::ModuleAxisInfoAnalysis &axisInfoAnalysis,
                    bool useAsyncCopy) {
  LDBG("ChainedDotSchedule::updateSchedule");
  ChainedDotClusters clusters;
  int cnt = clusters.size() - schedule.clusters.size();
  for (int i = 0; i < cnt; i++) {
    schedule.clusters.newAtBack();
  }
  auto it = schedule.clusters.begin();
  for (int i = 0; i < clusters.size(); i++, it++) {
    clusters[i] = it;
  }

  // Convert the loads into shared memory allocations and loads from them.
  // TODO support different numBuffers
  int numBuffers = useAsyncCopy ? 2 : 1;
  auto loadToStreamOps = createStreamOps(loadToInfo, forOp, numBuffers,
                                         useAsyncCopy, axisInfoAnalysis);
  scheduleStreamOps(loadToStreamOps, schedule, clusters);

  for (auto [l, _] : loadToInfo) {
    schedule.erase(l);
    l->erase();
  }

  scheduleDependencies(forOp, schedule);
  dumpSchedule(schedule, "Coarse schedule with dependencies:");

  triton::gpu::scheduleDistanceOneDependencies(forOp, schedule);
  dumpSchedule(schedule, "Coarse schedule with dist 1:");

  tt::CoarseSchedule::Cluster lastCluster = clusters.back();
  triton::gpu::scheduleRemainingToLastStage(forOp, schedule, lastCluster);
  dumpSchedule(schedule, "Final coarse schedule:");
}
} // namespace ChainedDotSchedule

void lowerLoop(scf::ForOp forOp,
               triton::AMD::ModuleAxisInfoAnalysis &axisInfoAnalysis,
               bool useAsyncCopy, bool usePingpong) {
  tt::CoarseSchedule schedule;
  if (failed(schedule.deSerialize(forOp, /*normalizeClusterId=*/false))) {
    return;
  }

  dumpSchedule(schedule, "[lowerLoops]deserialized schedule:");

  int numStages = schedule.getNumStages();

  // i.e., we can still disable `waitAtTail` by explicitly disabling
  // pingpong, which is the only use case of this scheduling variant.
  bool waitAtTail = usePingpong && (numStages == 3) && useAsyncCopy;

  llvm::MapVector<Operation *, std::pair<int, Operation *>> loadOpToIndLevel =
      getIndirectLevel(axisInfoAnalysis, forOp, numStages);

  auto arch = getAMDArch(forOp->getParentOfType<ModuleOp>());
  triton::AMD::TargetInfo targetInfo(arch ? arch->str() : "");

  LoadToInfoMap loadToInfo;
  for (const auto &[load, info] : loadOpToIndLevel) {
    auto [distance, use] = info;
    if (load->hasAttrOfType<BoolAttr>(AttrBypassLDS)) {
      load->removeAttr(AttrBypassLDS);
      loadToInfo[load] = {nullptr, distance, use};
    } else {
      LDBG("Deduce shared encoding for: " << *load);
      auto sharedEncoding =
          getSharedEncIfAllUsersAreDotEnc(load, axisInfoAnalysis, targetInfo,
                                          useAsyncCopy)
              .value_or(nullptr);
      loadToInfo[load] = {sharedEncoding, distance, use};
      LDBG("Populate loadInfo with shared encoding: " << sharedEncoding);
    }
  }

  if (succeeded(mlir::ChainedDotSchedule::checkPreconditions(forOp, numStages,
                                                             loadToInfo))) {
    ChainedDotSchedule::updateSchedule(forOp, loadToInfo, schedule,
                                       axisInfoAnalysis, useAsyncCopy);
  } else {
    SingleDotSchedule::updateSchedule(forOp, loadToInfo, schedule,
                                      axisInfoAnalysis, numStages, useAsyncCopy,
                                      waitAtTail);
  }

  dumpSchedule(schedule, "[lowerLoops]updated schedule:");

  schedule.serialize(forOp);
}

void lowerLoops(ModuleOp moduleOp, bool useAsyncCopy, bool usePingpong) {
  triton::AMD::ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);
  SmallVector<scf::ForOp> loops;
  moduleOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
  if (loops.empty())
    return;
  for (auto forOp : loops) {
    lowerLoop(forOp, axisInfoAnalysis, useAsyncCopy, usePingpong);
  }
}

} // namespace mlir
