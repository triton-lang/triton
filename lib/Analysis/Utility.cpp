#include "triton/Analysis/Utility.h"

#include <deque>

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include "triton/Tools/Sys/GetEnv.hpp"

namespace mlir {

using namespace triton;
using namespace triton::gpu;

SmallVector<unsigned> ReduceOpHelper::getOrderWithAxisAtBeginning() {
  auto order = toLinearEncoding(srcTy).getOrder();
  auto it = std::find(order.begin(), order.end(), axis);
  // delete the axis from order
  order.erase(it);
  // insert axis at the beginning of order
  order.insert(order.begin(), axis);
  return order;
}

// Thread offset is the thread index offset of two adjacent threads on the
// reduction axis within the warp.
unsigned ReduceOpHelper::getThreadOffsetOnReductionAxis() {
  auto *ctx = srcEncoding.getContext();
  auto linearLayout = toLinearLayout(srcTy);
  auto kLane = mlir::StringAttr::get(ctx, "lane");
  const auto &bases = linearLayout.getBases();
  const auto &lanes = bases.find(kLane)->second;
  auto offset = 1;
  for (const auto &lane : lanes) {
    if (lane[axis] != 0)
      break;
    offset *= 2;
  }
  return offset;
}

// Cases where distributed shared memory is not required in ConvertLayout:
// (1) numCTAs == 1
// (2) numCTAs > 1 but srcCTALayout == dstCTALayout
// TODO: Case with SliceLayout as srcLayout and numCTAs > 1 is to be implemented
// in the future
bool shouldUseDistSmem(Attribute srcLayout, Attribute dstLayout) {
  unsigned numCTAs = getNumCTAs(srcLayout);
  assert(numCTAs == getNumCTAs(dstLayout) &&
         "Invalid layout conversion: the numbers of CTAs of src and dst "
         "layouts are different");

  // Case (1): Never use dsmem when numCTAs == 1
  if (numCTAs == 1)
    return false;

  // Case where CTAsPerCGA of srcLayout in the sliced dim is not 1 is not
  // implemented yet
  if (auto sliceLayout = mlir::dyn_cast<SliceEncodingAttr>(srcLayout)) {
    auto dim = sliceLayout.getDim();
    auto CTAsPerCGA = getCTAsPerCGA(sliceLayout.getParent());
    if (CTAsPerCGA[dim] != 1)
      llvm::report_fatal_error("Layout conversion to be implemented");
  }

  // Case where CTAsPerCGA of dstLayout in the sliced dim is not 1 is supported
  if (auto sliceLayout = mlir::dyn_cast<SliceEncodingAttr>(dstLayout)) {
    auto dim = sliceLayout.getDim();
    auto CTAsPerCGA = getCTAsPerCGA(sliceLayout.getParent());
    if (CTAsPerCGA[dim] != 1)
      return true;
  }

  // The above two branches make sure that it is legal to call getCTALayout of
  // srcLayout and dstLayout

  // Case (2): Do not use dsmem when srcCTALayout == dstCTALayout
  auto srcCTALayout = getCTALayout(srcLayout);
  auto dstCTALayout = getCTALayout(dstLayout);
  if (srcCTALayout == dstCTALayout)
    return false;

  // Dsmem access is required when srcCTALayout != dstCTALayout
  return true;
}

unsigned ReduceOpHelper::getInterWarpSizeWithUniqueData() {
  return getWarpsPerCTA(srcEncoding, srcShape)[axis];
}

unsigned ReduceOpHelper::getIntraWarpSizeWithUniqueData() {
  return getThreadsPerWarp(srcEncoding, srcShape)[axis];
}

bool ReduceOpHelper::isWarpSynchronous() {
  return getWarpsPerCTA(srcEncoding, srcShape)[axis] == 1;
}

SmallVector<unsigned> ReduceOpHelper::getScratchRepShape() {
  SmallVector<unsigned> smemShape;
  // This case doesn't need inter-warp communication
  if (isWarpSynchronous())
    return {0, 0};

  smemShape = convertType<unsigned>(srcShape);
  smemShape[axis] = getInterWarpSizeWithUniqueData();

  return smemShape;
}

unsigned ReduceOpHelper::getScratchSizeInBytes() {
  auto smemShape = getScratchRepShape();
  auto elems = product<unsigned>(smemShape);

  unsigned bytesPerElem = 0;
  for (const auto &ty : srcElementTypes) {
    bytesPerElem += ceil<unsigned>(ty.getIntOrFloatBitWidth(), 8);
  }
  return bytesPerElem * elems;
}

bool ReduceOpHelper::isReduceWithinCTA() {
  // TODO: Support reduce across CTAS
  // Layout optimization passes such as PlanCTAPass and
  // RemoveLayoutConversionPass should avoid cross-CTA reduction
  return getCTASplitNum(srcEncoding)[axis] == 1;
}

bool ReduceOpHelper::isAssociative() {
  auto dtype = srcElementTypes[0];
  if (!type::isFloat(dtype))
    return true;
  size_t reduce_size = srcShape[axis];
  if (reduce_size <= 2)
    return true;
  bool hasNoAssociativeOp = false;
  op.walk([&](Operation *nestedOp) -> WalkResult {
    if (isa<arith::AddFOp, arith::MulFOp>(nestedOp)) {
      // Only when the data type is float point and reduce size greater than 2,
      // and has addf or mulf op, we though it's a non-associative reduce.
      hasNoAssociativeOp = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return !hasNoAssociativeOp;
}

unsigned ScanLoweringHelper::getAxisNumElementsPerThread() {
  return getEncoding().getContigPerThread()[getAxis()];
}

unsigned ScanLoweringHelper::getNonAxisNumElementsPerThread() {
  auto contigPerThread = getEncoding().getContigPerThread();
  contigPerThread[getAxis()] = 1;
  return product<unsigned>(contigPerThread);
}

Region &ScanLoweringHelper::getCombineOp() { return scanOp.getCombineOp(); }

unsigned ScanLoweringHelper::getAxisNumThreadsPerWarpWithUniqueData() {
  return getEncoding().getThreadsPerWarp()[getAxis()];
}

unsigned ScanLoweringHelper::getNonAxisNumThreadsPerWarp() {
  auto nThreads = product(getEncoding().getThreadsPerWarp());
  return nThreads / getAxisNumThreadsPerWarpWithUniqueData();
}

// Return the flat numbers of threads computing independent scan results.
unsigned ScanLoweringHelper::getNonAxisNumThreadsPerCTA() {
  auto nWarps = product(getEncoding().getWarpsPerCTA());
  return (nWarps / getAxisNumWarpsWithUniqueData()) *
         getNonAxisNumThreadsPerWarp();
}

unsigned ScanLoweringHelper::getAxisNumWarpsWithUniqueData() {
  return getEncoding().getWarpsPerCTA()[getAxis()];
}

unsigned ScanLoweringHelper::getAxisNumBlocks() {
  auto contigPerThread = getEncoding().getContigPerThread();
  auto threadsPerWarp = getEncoding().getThreadsPerWarp();
  auto warpsPerCTA = getEncoding().getWarpsPerCTA();
  unsigned axis = getAxis();
  return ceil<unsigned>(
      getShape()[axis],
      (contigPerThread[axis] * threadsPerWarp[axis] * warpsPerCTA[axis]));
}

unsigned ScanLoweringHelper::getNonAxisNumBlocks() {
  auto contigPerThread = getEncoding().getContigPerThread();
  auto threadsPerWarp = getEncoding().getThreadsPerWarp();
  auto warpsPerCTA = getEncoding().getWarpsPerCTA();
  auto rank = contigPerThread.size();
  unsigned axis = getAxis();
  unsigned numBlocks = 1;
  for (unsigned i = 0; i < rank; i++) {
    if (i == axis)
      continue;
    numBlocks *=
        ceil<unsigned>(getShape()[i], (contigPerThread[i] * threadsPerWarp[i] *
                                       warpsPerCTA[i]));
  }
  return numBlocks;
}

bool ScanLoweringHelper::isSupported() {
  // TODO: Support the following cases:
  // 1. Scan on non-blocking encodings
  if (!isa<BlockedEncodingAttr>(legacyEncoding))
    return false;
  return true;
}

unsigned ScanLoweringHelper::getScratchSizeInElems() {
  unsigned numWarps = product(getEncoding().getWarpsPerCTA());
  unsigned numNonAxisElementsPerWarp =
      getNonAxisNumThreadsPerWarp() * getNonAxisNumElementsPerThread();
  unsigned numElements = numWarps * numNonAxisElementsPerWarp *
                         getAxisNumBlocks() * getNonAxisNumBlocks();
  return numElements;
}

unsigned ScanLoweringHelper::getScratchSizeInBytes() {
  // Lowering will fail later if the layout is not supported.
  if (!isSupported())
    return 0;

  unsigned axisNumWarps = getAxisNumWarpsWithUniqueData();
  if (axisNumWarps == 1)
    return 0;
  unsigned elementSizeInBytes = 0;
  for (const auto &ty : srcElementTypes) {
    elementSizeInBytes += ceil<unsigned>(ty.getIntOrFloatBitWidth(), 8);
  }
  return elementSizeInBytes * getScratchSizeInElems();
}

DecomposedWarpConversion
getWarpLayoutConvertDecomposition(RankedTensorType srcTy,
                                  RankedTensorType dstTy) {
  // Two layouts, ll_src and ll_dst, representing the same tensor can be
  // viewed as surjections of GF(2) vector spaces:
  //
  //            ll_src: H_src -> M   and   ll_dst: H_dst -> M,
  //
  // where each is represented by a 'subpermutation' matrix, i.e., a permutation
  // matrix with zero columns possibly inserted. A layout conversion can be
  // viewed as a map P': H_src -> H_dst which factors ll_src = ll_dst \circ P'.
  //
  // For a conversion not needing data movement between different warps, we
  // choose the following representation, where P is a permutation matrix and
  // K_1 and K_2 are (possibly trivial) spaces meant to ensure equally sized
  // lane and register dimensions between layouts:
  //                                  P
  //     H_src -> H_src \oplus K_1 -------> H_dst \oplus K_2 -> H_dst.
  //
  // As a permutation, P can be viewed as a product of cycles permuting lane and
  // register index bits. Any such permutation can be expressed as a composition
  //
  //                    P = P_mixed \circ P_lane \circ P_reg,
  //
  // where P_mixed is a product of disjoint transpositions (r_i l_j) between
  // lane and register bits and where P_lane and P_reg are permutations purely
  // involving lane bits and register bits, respectively. Such a representation
  // is not unique, and we choose the factorization method which slices out
  // subsequences of consecutive lane bits from cycles involving both bit types.
  // Further explanation of this method is below.
  //
  // The decomposition is performed in two stages. First, we compute the
  // permutation matrix `P` by using `invertAndCompose` to generate a skeleton
  // and then fill in any zero columns. Second, we walk the cycles of `P` to
  // factor out mixed transpositions to build `mixedTranspositions`, `pReg`, and
  // `pLane`.

  // We remove any broadcasting in the register dimensions of the layouts before
  // forming the permutation `P` as the components of the decomposition directly
  // inform the number of emitted instructions, and leaving broadcasting in
  // would unnecessarily inflate the count.
  auto srcLayout = toLinearLayout(srcTy);
  auto dstLayout = toLinearLayout(dstTy);
  auto removeBroadcastSrc = actionRemoveBroadcastedRegs(srcLayout);
  auto removeBroadcastDst = actionRemoveBroadcastedRegs(dstLayout);
  srcLayout = removeBroadcastSrc.apply(srcLayout);
  dstLayout = removeBroadcastDst.apply(dstLayout);

  // We want to describe the conversion from `srcLayout` to `dstLayout` as a
  // permutation. Since this requires that each input dimension have the same
  // size in each of the layouts, we first pad the lane and register dimensions
  // with zero vectors if needed.
  auto *ctx = srcTy.getContext();
  StringAttr kReg = StringAttr::get(ctx, "register");
  StringAttr kLane = StringAttr::get(ctx, "lane");

  // Determine the target sizes of the register and lane dimensions for padding.
  int nSrcRegBases = srcLayout.getInDimSizeLog2(kReg);
  int nDstRegBases = dstLayout.getInDimSizeLog2(kReg);
  int nSrcLaneBases = srcLayout.getInDimSizeLog2(kLane);
  int nDstLaneBases = dstLayout.getInDimSizeLog2(kLane);
  int nRegBases = std::max(nSrcRegBases, nDstRegBases);
  int nLaneBases = std::max(nSrcLaneBases, nDstLaneBases);
  // Restrict attention to the input dimensions which matter.
  SmallVector<StringAttr> inDimNames{kReg, kLane};
  auto outDimNames = llvm::to_vector(srcLayout.getOutDimNames());
  auto S = srcLayout.sublayout(inDimNames, outDimNames);
  auto T = dstLayout.sublayout(inDimNames, outDimNames);
  // Conditionally pad.
  if (nSrcRegBases != nDstRegBases || nSrcLaneBases != nDstLaneBases) {
    auto padWithZeros = [&](const LinearLayout &ll) {
      auto newBases = ll.getBases();
      auto padDim = [&](StringAttr dim, int dimSize) {
        auto &dimBases = newBases[dim];
        dimBases.reserve(dimSize);
        for (int i = ll.getInDimSizeLog2(dim); i < dimSize; ++i)
          dimBases.emplace_back(outDimNames.size(), 0);
      };
      padDim(kReg, nRegBases);
      padDim(kLane, nLaneBases);
      // Surjectivity is not expected in general since we do not consider
      // the 'warp' and 'block' dimensions of the original layouts.
      return LinearLayout(std::move(newBases), ll.getOutDims(),
                          /*requireSurjective=*/false);
    };
    S = padWithZeros(S);
    T = padWithZeros(T);
  }

  // Flatten outs for ease of building `P`, and reorder outs as flattening
  // depends on output dimension order.
  if (outDimNames != llvm::to_vector(T.getOutDimNames()))
    T = T.transposeOuts(outDimNames);
  S = S.flattenOuts();
  T = T.flattenOuts();
  
  // We compute T^transpose \circ S, which serves as a skeleton for `P`, then
  // fill in zero columns, prioritizing producing fixed points. As we only need
  // the basis vectors of `P`, we never actually produce the LinearLayout.
  auto pBases = S.invertAndCompose(T).getBases();

  // Find the common and uncommon zeros of S and T
  SmallVector<std::pair<int32_t, int32_t>> srcFreeZeros;
  SmallVector<std::pair<int32_t, int32_t>> dstFreeZeros;
  for (auto [dimIdx, dim] : llvm::enumerate(inDimNames)) {
    for (int inIdx = 0; inIdx < S.getInDimSizeLog2(dim); ++inIdx) {
      int sVal = S.getBasis(dim, inIdx)[0];
      int tVal = T.getBasis(dim, inIdx)[0];
      if (sVal == 0 && tVal == 0) {
        pBases[dim][inIdx][dimIdx] = 1 << inIdx;
      } else if (sVal == 0) {
        srcFreeZeros.emplace_back(dimIdx, inIdx);
      } else if (tVal == 0) {
        dstFreeZeros.emplace_back(dimIdx, inIdx);
      }
    }
  }
  // Fill in non-fixed-point zero vectors
  for (auto [srcZeroLoc, dstZeroLoc] : llvm::zip(srcFreeZeros, dstFreeZeros)) {
    auto [srcDimIdx, srcIdx] = srcZeroLoc;
    auto [dstDimIdx, dstIdx] = dstZeroLoc;
    auto inDim = inDimNames[srcDimIdx];
    pBases[inDim][srcIdx][dstDimIdx] = 1 << dstIdx;
  }

  // We walk the cycles of `P` to build the bases for `pReg` and `pLane` while
  // factoring out mixed transpositions from cycles that include both register
  // and lane basis vectors. `pReg` and `pLane` themselves only have one input
  // and output dimension each.
  LinearLayout::BasesT pRegBases, pLaneBases;
  auto &regBases = pRegBases[kReg];
  auto &laneBases = pLaneBases[kLane];
  regBases.resize(nRegBases, {0});
  laneBases.resize(nLaneBases, {0});
  SmallVector<std::pair<int, int>> mixedTranspositions;

  llvm::BitVector visited(nRegBases + nLaneBases, false);
  auto flatIdx = [&](StringAttr dim, int32_t index) {
    return (dim == kReg) ? index : nRegBases + index;
  };

  for (auto dim : inDimNames) {
    int inDimSize = S.getInDimSizeLog2(dim);
    for (int i = 0; i < inDimSize; ++i) {
      if (visited.test(flatIdx(dim, i)))
        continue;

      // Start a new cycle, tracking the entry basis vector and the 'current'
      // one as we walk the cycle.
      StringAttr entryDim = dim;
      int32_t entryIdx = i;
      StringAttr currDim = entryDim;
      int32_t currIdx = entryIdx;

      // We slice out subsequences of consecutive lane basis vectors appearing
      // in mixed cycles by factoring out transpositions (r_i l_j) as in
      //
      // (.. r_m l_j .. l_k r_i ..) = (r_i l_j) * (.. r_m r_i ..)(l_j .. l_k).
      //
      // The permutations are applied right-to-left, and the block `l_j .. l_k`
      // indicates a contiguous subsequence of lane basis vectors. Note that the
      // transposition does not commute with the other two cycles.
      //
      // The following variables are used to track the start and end points of
      // such subsequences.
      int32_t /*r_m*/ regStartIdx = -1;
      int32_t /*l_j*/ laneStartIdx = -1;
      int32_t /*l_k*/ laneEndIdx = -1;
      int32_t /*r_i*/ regEndIdx = -1;

      do {
        // Determine the next basis vector in the current cycle.
        visited.set(flatIdx(currDim, currIdx));
        auto nextVec = pBases.lookup(currDim)[currIdx];
        StringAttr nextDim;
        int32_t nextIdx;
        for (auto [nextDimIdx, nextVal] : llvm::enumerate(nextVec)) {
          if (nextVal != 0) {
            nextDim = inDimNames[nextDimIdx];
            nextIdx = llvm::Log2_32(nextVal);
          }
        }
        // Set a `pReg` or `pLane` vector, or mark an r->l or l->r transition.
        if (currDim == kReg && nextDim == kReg) {
          regBases[currIdx][0] = 1 << nextIdx;
        } else if (currDim == kLane && nextDim == kLane) {
          laneBases[currIdx][0] = 1 << nextIdx;
        } else if (currDim == kReg && nextDim == kLane) {
          regStartIdx = currIdx;
          laneStartIdx = nextIdx;
        } else {
          regEndIdx = nextIdx;
          laneEndIdx = currIdx;
        }
        // If a subsequence of the form (.. r_m l_j .. l_k r_i ..) has been
        // found, perform the prescribed factorization.
        if (regEndIdx >= 0) {
          // Assign r_m to map to r_i as in (.. r_m r_i ..).
          regBases[regStartIdx][0] = 1 << regEndIdx;
          // Assign l_k to map to l_j as in (l_j .. l_k).
          laneBases[laneEndIdx][0] = 1 << laneStartIdx;
          // Record (r_i l_j) as a factor.
          mixedTranspositions.emplace_back(regEndIdx, laneStartIdx);
          // Reset the auxiliary variables.
          regStartIdx = laneStartIdx = laneEndIdx = regEndIdx = -1;
        }

        currDim = nextDim;
        currIdx = nextIdx;
      } while (flatIdx(currDim, currIdx) != flatIdx(entryDim, entryIdx));
    }
  }
  assert(visited.all() && "Cycle walk incomplete");

  auto pReg = LinearLayout(std::move(pRegBases), {{kReg, 1 << nRegBases}},
                           /*requireSurjective=*/true);
  auto pLane = LinearLayout(std::move(pLaneBases), {{kLane, 1 << nLaneBases}},
                            /*requireSurjective=*/true);
  return {std::move(pReg), std::move(pLane), std::move(mixedTranspositions)};
}

SmallVector<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>>
getReshapeDecomposition(ArrayRef<int64_t> srcShape,
                        ArrayRef<int64_t> dstShape) {
  SmallVector<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>> ret;

  if (srcShape.empty()) {
    assert(dstShape.empty());
    return ret;
  }
  ret.push_back({});

  int srcIdx = 0;
  int dstIdx = 0;
  int srcNElems = 1;
  int dstNElems = 1;
  while (srcIdx < srcShape.size() || dstIdx < dstShape.size()) {
    if (srcNElems < dstNElems || //
        (srcIdx < srcShape.size() && srcNElems == 1) ||
        (srcIdx < srcShape.size() && srcShape[srcIdx] == 1)) {
      assert(srcIdx < srcShape.size());
      srcNElems *= srcShape[srcIdx];
      ret.back().first.push_back(srcIdx);
      srcIdx++;
    } else if (dstNElems < srcNElems ||
               (dstIdx < dstShape.size() && dstShape[dstIdx] == 1)) {
      assert(dstIdx < dstShape.size());
      dstNElems *= dstShape[dstIdx];
      ret.back().second.push_back(dstIdx);
      dstIdx++;
    } else {
      ret.push_back({});
      srcNElems = 1;
      dstNElems = 1;
    }
  }
  return ret;
}

unsigned ScanLoweringHelper::getAxisElementStride() {
  auto order = getOrder();
  unsigned stride = 1;
  for (unsigned dim : order) {
    if (dim == getAxis())
      return stride;
    stride *= getEncoding().getContigPerThread()[dim];
  }
  llvm_unreachable("Axis not found in order");
}

unsigned ScanLoweringHelper::getAxisThreadStride() {
  auto encoding = getEncoding();
  auto kThread = StringAttr::get(encoding.getContext(), "lane");
  // OOOGHHH This is nasty. We should implement this lowering via LLs natively
  // to avoid this
  auto threadsPerWarp = encoding.basesPerDim(kThread, /*skipBroadcast=*/false);
  auto order = getOrder();
  unsigned stride = 1;
  for (unsigned dim : order) {
    if (dim == getAxis())
      return stride;
    stride *= threadsPerWarp[dim];
  }
  llvm_unreachable("Axis not found in order");
}

unsigned ScanLoweringHelper::getAxisBlockStride() {
  auto order = getOrder();
  unsigned stride = 1;
  auto contigPerThread = getEncoding().getContigPerThread();
  auto threadsPerWarp = getEncoding().getThreadsPerWarp();
  auto warpsPerCTA = getEncoding().getWarpsPerCTA();
  for (unsigned dim : order) {
    if (dim == getAxis())
      return stride;
    stride *= ceil<unsigned int>(getShape()[dim], contigPerThread[dim] *
                                                      threadsPerWarp[dim] *
                                                      warpsPerCTA[dim]);
  }
  llvm_unreachable("Axis not found in order");
}

GatherLoweringHelper::GatherLoweringHelper(triton::GatherOp gatherOp)
    : gatherOp(gatherOp) {}

unsigned GatherLoweringHelper::getScratchSizeInBytes() {
  // If the gather is warp-local, no scratch space is needed.
  if (isWarpLocal())
    return 0;

  // Otherwise, performing the gather will require scratch space to communicate
  // the source tensor across threads. For now, assume the whole source tensor
  // is written back to shared memory.
  RankedTensorType srcType = gatherOp.getSrc().getType();
  return product(srcType.getShape()) *
         ceil<unsigned>(srcType.getElementTypeBitWidth(), 8);
}

bool GatherLoweringHelper::isWarpLocal() {
  // The gather is warp-local if for each column along the gather axis in the
  // source and index tensors, all the elements are owned by the same warp.
  RankedTensorType srcType = gatherOp.getSrc().getType();
  RankedTensorType idxType = gatherOp.getIndices().getType();
  LinearLayout srcLayout = toLinearLayout(srcType);
  LinearLayout idxLayout = toLinearLayout(idxType);

  Builder b(gatherOp.getContext());
  StringAttr kBlock = b.getStringAttr("block");
  StringAttr kWarp = b.getStringAttr("warp");
  StringAttr kLane = b.getStringAttr("lane");
  StringAttr kGatherDim =
      b.getStringAttr("dim" + std::to_string(gatherOp.getAxis()));

  // The tensor layouts must be distributed layouts, where the basis matrix is a
  // subpermutation matrix (permutation matrix plus zeros for broadcasting).
  // FIXME(jeff): Check this invariant somehow.
  //
  // We want to know if all elements of a column along the gather axis are
  // mapped to the same set of warps, which means the gather can be performed
  // entirely within the warp. We need to query
  //
  //   srcLayout.invert().sublayoutIsZero({kGatherDim}, {kBlock, kWarp})
  //
  // But due to broadcasting, the matrix might not be invertible. But since the
  // matrix is a permutation matrix (checked below), we can instead query
  //
  //   srcLayout.sublayoutIsZero({kBlock, kWarp}, {kGatherDim})
  //
  // Which implies that changing the warp will not change the gather dimension.
  // And since there is no swizzling, this applies to all warps.
  if (!srcLayout.sublayoutIsZero({kBlock, kWarp}, kGatherDim) ||
      !idxLayout.sublayoutIsZero({kBlock, kWarp}, kGatherDim))
    return false;

  SmallVector<StringAttr> otherDims;
  for (unsigned dim = 0, rank = srcType.getRank(); dim < rank; ++dim) {
    if (dim != gatherOp.getAxis()) {
      otherDims.push_back(b.getStringAttr("dim" + Twine(dim)));
    }
  }

  // If the gather axis `dimN` is invariant to the warp, but the `(block, warp)`
  // mapping to all other dimensions must be the same for both layouts. If so,
  // then the warp that owns a particular index element also owns all the source
  // elements it could index into.
  if (srcLayout.sublayout({kBlock, kWarp}, otherDims) !=
      idxLayout.sublayout({kBlock, kWarp}, otherDims))
    return false;

  // The two constraints above ensure that data-movement to perform the gather
  // operation are contained within a warp. The subsequent constraints simplify
  // codegen.

  // Require that for any given gather column, the threads mapped to the column
  // in the index and source tensors are the same. This means we don't need to
  // xor shuffle across threads before emitting index shuffles; we push warp
  // shuffling to layout conversions.
  return srcLayout.sublayout(kLane, otherDims) ==
         idxLayout.sublayout(kLane, otherDims);
}

unsigned getNumScratchElements(ArrayRef<unsigned> shape) {
  if (shape.empty())
    return 0;
  return product<unsigned>(shape);
}

bool supportMMA(triton::DotOp op, int version) {
  // Refer to mma section for the data type supported by Volta and Hopper
  // Tensor Core in
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-884-f16
  auto aElemTy = op.getA().getType().getElementType();
  auto bElemTy = op.getB().getType().getElementType();
  if (version == 5) {
    if (triton::tools::getBoolEnv("DISABLE_MMA_V5"))
      return false;
    auto retType = op.getType();
    auto retShapePerCTA = getShapePerCTA(retType);
    auto rank = retShapePerCTA.size();
    int numWarps = lookupNumWarps(op);
    if (aElemTy.isInteger() || bElemTy.isInteger() ||
        retType.getElementType().isInteger())
      return false;
    if (op.getType().getRank() != 2)
      return false;
    if (numWarps != 4 && numWarps != 8) {
      // Currently only support numWarps 4 or 8 for TMEM load and store.
      return false;
    }
    if (!(retShapePerCTA[rank - 2] % 64 == 0 &&
          retShapePerCTA[rank - 1] % 8 == 0))
      return false;
    return true;
  }
  if (version == 3) {
    if (triton::tools::getBoolEnv("DISABLE_MMA_V3"))
      return false;
    auto retType = op.getType();
    RankedTensorType typeA = op.getA().getType();
    int k = typeA.getShape().back();
    // If k size is smaller than the native mma size, we cannot use MMA.
    if (k < 256 / aElemTy.getIntOrFloatBitWidth())
      return false;
    auto retShapePerCTA = getShapePerCTA(retType);
    auto rank = retShapePerCTA.size();
    int numWarps = lookupNumWarps(op);
    // TODO(Keren): for now, fallback to MMAv2 if handling batch matmul.
    if (rank == 3)
      return false;
    if (!(numWarps % 4 == 0 && retShapePerCTA[rank - 2] % 64 == 0 &&
          retShapePerCTA[rank - 1] % 8 == 0 &&
          (llvm::isa<Float8E5M2Type, Float8E4M3FNType>(aElemTy) ||
           aElemTy.isInteger(8) || aElemTy.isF16() || aElemTy.isBF16() ||
           aElemTy.isF32()))) {
      return false;
    }
    // We cannot use MMA_V3 if we need to accumulate in F32 within the MMA op.
    if (op.getMaxNumImpreciseAcc() < 32 &&
        (llvm::isa<Float8E5M2Type, Float8E4M3FNType>(aElemTy)) &&
        cast<RankedTensorType>(op.getType()).getElementType().isF32()) {
      return false;
    }
  }
  if (aElemTy.isF32() && bElemTy.isF32()) {
    return op.getInputPrecision() == InputPrecision::TF32 && version >= 2;
  }
  return supportMMA(op.getA(), version) && supportMMA(op.getB(), version);
}

bool supportMMA(Value value, int version) {
  // Tell whether a DotOp support MMA by the operand type(either $a or $b).
  // We cannot get both the operand types(in TypeConverter), here we assume the
  // types of both the operands are identical here.
  assert((version == 1 || version == 2 || version == 3) &&
         "Unexpected MMA layout version found");
  auto elemTy =
      cast<triton::gpu::TensorOrMemDesc>(value.getType()).getElementType();
  // FP8 is not natively supported on all mma versions but it can always be
  // promoted to fp16 therefore we can always support it.
  bool isFP8 = llvm::isa<Float8E5M2Type, Float8E4M3FNType, Float8E5M2FNUZType,
                         Float8E4M3FNUZType>(elemTy);
  return isFP8 || elemTy.isF16() || elemTy.isBF16() ||
         ((elemTy.isF32() || elemTy.isF64()) && version >= 2) ||
         (elemTy.isInteger(8) && version >= 2);
}

// For MMAV3 dotOperand layout matches mma operand for f16 and bf16 cases.
bool matchMmaV3AndDotOperandLayout(RankedTensorType srcTy,
                                   RankedTensorType dstTy) {
  auto mmaLayout = dyn_cast<NvidiaMmaEncodingAttr>(srcTy.getEncoding());
  auto dotOperandLayout = dyn_cast<DotOperandEncodingAttr>(dstTy.getEncoding());
  if (!mmaLayout || !dotOperandLayout) {
    return false;
  }
  int elementTypeSize = srcTy.getElementType().getIntOrFloatBitWidth();
  auto parentTy = srcTy.cloneWithEncoding(dotOperandLayout.getParent());
  auto ans = mmaLayout.getVersionMajor() == 3 &&
             dotOperandLayout.getOpIdx() == 0 &&
             mmaLayout.getWarpsPerCTA()[1] == 1 &&
             !cvtNeedsSharedMemory(parentTy, srcTy) && elementTypeSize == 8 &&
             dotOperandLayout.getKWidth() == 32 / elementTypeSize;
  return ans;
}

bool matchMFMAAndDotOperandShuffleCase(RankedTensorType srcTy,
                                       RankedTensorType dstTy) {
  auto mfmaLayout = dyn_cast<AMDMfmaEncodingAttr>(srcTy.getEncoding());
  auto dotOperandLayout = dyn_cast<DotOperandEncodingAttr>(dstTy.getEncoding());
  if (!mfmaLayout || !dotOperandLayout)
    return false;

  // Currently supporting 32x32 and 16x16 FP8 MFMA -> dot operand case
  return dotOperandLayout.getParent() == mfmaLayout &&
         dotOperandLayout.getOpIdx() == 0 && mfmaLayout.getIsTransposed() &&
         dotOperandLayout.getKWidth() == 8 &&
         ((mfmaLayout.getMDim() == 16 && mfmaLayout.getNDim() == 16) ||
          (mfmaLayout.getMDim() == 32 && mfmaLayout.getNDim() == 32)) &&
         triton::type::isFloat8(srcTy.getElementType()) &&
         triton::type::isFloat8(dstTy.getElementType()) &&
         mfmaLayout.getWarpsPerCTA()[1] == 1;
}

// We get the smallest submap of srcTy^{-1} * dstTy that is not the identity
// under the common dimensions. The idea here is that if we have a
// transformation that's the identity on kBlock, we don't need to use
// distributed shared memory. If it's also the identity on kWarp, we can
// transfer via warp-shuffles, and if it's the identity on kLane just have to
// reorder the registers.
LinearLayout minimalCvtLayout(Type srcTy_, Type dstTy_) {
  auto srcTy = cast<triton::gpu::TensorOrMemDesc>(srcTy_);
  auto dstTy = cast<triton::gpu::TensorOrMemDesc>(dstTy_);
  LinearLayout srcLayout = toLinearLayout(srcTy);
  LinearLayout dstLayout = toLinearLayout(dstTy);
  auto sDims = to_vector(srcLayout.getInDimNames());
  auto dDims = to_vector(dstLayout.getInDimNames());
  SmallVector<StringAttr> dims;
  for (int i = 0; i < std::min(sDims.size(), dDims.size()); ++i) {
    auto srcDim = sDims[sDims.size() - i - 1];
    auto dstDim = dDims[dDims.size() - i - 1];
    if (srcDim != dstDim) {
      break;
    }
    dims.push_back(srcDim);
  }

  auto comp = dstLayout.invertAndCompose(srcLayout);
  // We try to quotient by the slowers moving subspace first
  for (auto dim : dims) {
    auto quotient = comp.quotient(dim);
    if (!quotient.has_value()) {
      break;
    }
    comp = *quotient;
  }
  return comp;
}

bool cvtReordersRegisters(RankedTensorType srcTy, RankedTensorType dstTy) {
  auto layout = minimalCvtLayout(srcTy, dstTy);
  MLIRContext *ctx = srcTy.getContext();
  auto kRegister = StringAttr::get(ctx, "register");
  auto outDims = to_vector(layout.getOutDimNames());
  return outDims.empty() || ArrayRef(outDims) == ArrayRef({kRegister});
}

bool cvtNeedsWarpShuffle(RankedTensorType srcTy, RankedTensorType dstTy) {
  auto layout = minimalCvtLayout(srcTy, dstTy);
  MLIRContext *ctx = srcTy.getContext();
  auto kRegister = StringAttr::get(ctx, "register");
  auto kLane = StringAttr::get(ctx, "lane");
  if (to_vector(layout.getOutDimNames()) ==
      SmallVector<StringAttr, 2>{kRegister, kLane}) {
    auto factors = getWarpLayoutConvertDecomposition(srcTy, dstTy);
    return (factors.mixedTranspositions.size() < 2);
  }
  return false;
}

bool cvtNeedsSharedMemory(RankedTensorType srcTy, RankedTensorType dstTy) {
  // TODO(jlebar): Remove these special cases `isMfmaToDotShortcut` once
  // they're fully subsumed by the linear-layout checks.
  return !cvtReordersRegisters(srcTy, dstTy) &&
         !cvtNeedsWarpShuffle(srcTy, dstTy) &&
         !matchMmaV3AndDotOperandLayout(srcTy, dstTy) &&
         // to be removed when generalized warp shuffle conversions
         // are ready:
         !matchMFMAAndDotOperandShuffleCase(srcTy, dstTy);
}

bool atomicNeedsSharedMemory(Value value) {
  auto type = value.getType();
  if (isa<RankedTensorType>(type) || value.use_empty())
    return false;
  return true;
}

namespace {

/// A data structure similar to SetVector but maintains
/// a deque instead of a vector to allow for efficient
/// push_back and pop_front operations.
/// Using SetVector doesn't suffice our needs because
/// it only pushes and pops from the back.
/// For example, if we have a queue like this:
/// 0->4 1->2->3
///    ^--------
/// where 3 depends on 4, once we pop 3, we found
/// 4 is not ready, so we check 2 and push 3 back
/// to the queue.
struct DFSSubgraphState {
  DFSSubgraphState() : set(), deque() {}
  DenseSet<Operation *> set;
  std::deque<Operation *> deque;

  bool push_back(Operation *op) {
    if (set.insert(op).second) {
      deque.push_back(op);
      return true;
    }
    return false;
  }

  Operation *pop_front() {
    Operation *op = deque.front();
    deque.pop_front();
    set.erase(op);
    return op;
  }

  bool empty() { return deque.empty(); }
};

/// DFS post-order implementation that maintains a global count to work across
/// multiple invocations, to help implement topological sort on multi-root DAGs.
/// We traverse all operations but only record the ones that appear in
/// `toSort` for the final result.
struct DFSState {
  DFSState(const SetVector<Operation *> &set) : toSort(set), seen() {}
  const SetVector<Operation *> &toSort;
  SmallVector<Operation *, 16> topologicalCounts;
  DenseSet<Operation *> seen;

  /// We mark each op as ready if all its operands and parents ops are seen. If
  /// an op is ready, we add it to the queue. Otherwise, we keep adding its
  /// operands to the ancestors set.
  /// We always want an op to be scheduled after all its parents to handle
  /// correctly cases with scf operations.
  void addToReadyQueue(Operation *op, DFSSubgraphState &subGraph,
                       SmallVector<Operation *, 4> &readyQueue) {
    bool ready = true;
    for (Value operand : op->getOperands()) {
      auto def = operand.getDefiningOp();
      if (def && !seen.count(def)) {
        subGraph.push_back(def);
        ready = false;
      }
    }
    Operation *parent = op->getParentOp();
    while (parent) {
      if (!seen.count(parent)) {
        subGraph.push_back(parent);
        ready = false;
      }
      parent = parent->getParentOp();
    }
    if (ready)
      readyQueue.push_back(op);
  }
};

void dfsPostorder(Operation *root, DFSState *state) {
  DFSSubgraphState subGraph;
  subGraph.push_back(root);
  SmallVector<Operation *> ops;
  while (!subGraph.empty()) {
    // Nodes in the ready queue are ready to be processed.
    // Meaning that either their operands are all seen or they have null
    // operands.
    SmallVector<Operation *, 4> readyQueue;
    auto *current = subGraph.pop_front();
    state->addToReadyQueue(current, subGraph, readyQueue);
    while (!readyQueue.empty()) {
      Operation *current = readyQueue.pop_back_val();
      if (!state->seen.insert(current).second)
        continue;
      ops.push_back(current);
      for (Value result : current->getResults()) {
        for (Operation *op : result.getUsers())
          state->addToReadyQueue(op, subGraph, readyQueue);
      }
      for (Region &region : current->getRegions()) {
        for (Operation &op : region.getOps())
          state->addToReadyQueue(&op, subGraph, readyQueue);
      }
    }
  }

  for (Operation *op : llvm::reverse(ops)) {
    if (state->toSort.count(op) > 0)
      state->topologicalCounts.push_back(op);
  }
}

} // namespace

SetVector<Operation *>
multiRootTopologicalSort(const SetVector<Operation *> &toSort) {
  if (toSort.empty()) {
    return toSort;
  }

  // Run from each root with global count and `seen` set.
  DFSState state(toSort);
  for (auto *s : toSort) {
    assert(toSort.count(s) == 1 && "NYI: multi-sets not supported");
    dfsPostorder(s, &state);
  }

  // Reorder and return.
  SetVector<Operation *> res;
  for (auto it = state.topologicalCounts.rbegin(),
            eit = state.topologicalCounts.rend();
       it != eit; ++it) {
    res.insert(*it);
  }
  return res;
}

SetVector<Operation *> multiRootGetSlice(Operation *op,
                                         TransitiveFilter backwardFilter,
                                         TransitiveFilter forwardFilter) {
  SetVector<Operation *> slice;
  slice.insert(op);

  unsigned currentIndex = 0;
  SetVector<Operation *> backwardSlice;
  SetVector<Operation *> forwardSlice;
  while (currentIndex != slice.size()) {
    auto *currentOp = (slice)[currentIndex];
    // Compute and insert the backwardSlice starting from currentOp.
    backwardSlice.clear();
    BackwardSliceOptions opt;
    opt.omitBlockArguments = true;
    opt.filter = backwardFilter;
    (void)getBackwardSlice(currentOp, &backwardSlice, opt);
    slice.insert(backwardSlice.begin(), backwardSlice.end());

    // Compute and insert the forwardSlice starting from currentOp.
    forwardSlice.clear();
    getForwardSlice(currentOp, &forwardSlice, forwardFilter);
    slice.insert(forwardSlice.begin(), forwardSlice.end());
    ++currentIndex;
  }
  return multiRootTopologicalSort(slice);
}

namespace {
// Copied from TestDeadCodeAnalysis.cpp, because some dead code analysis
// interacts with constant propagation, but SparseConstantPropagation
// doesn't seem to be sufficient.
class ConstantAnalysis : public DataFlowAnalysis {
public:
  using DataFlowAnalysis::DataFlowAnalysis;

  LogicalResult initialize(Operation *top) override {
    WalkResult result = top->walk([&](Operation *op) {
      ProgramPoint programPoint(op);
      if (failed(visit(&programPoint)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    return success(!result.wasInterrupted());
  }

  LogicalResult visit(ProgramPoint *point) override {
    Operation *op = point->getOperation();
    Attribute value;
    if (matchPattern(op, m_Constant(&value))) {
      auto *constant = getOrCreate<dataflow::Lattice<dataflow::ConstantValue>>(
          op->getResult(0));
      propagateIfChanged(constant, constant->join(dataflow::ConstantValue(
                                       value, op->getDialect())));
      return success();
    }
    // Dead code analysis requires every operands has initialized ConstantValue
    // state before it is visited.
    // https://github.com/llvm/llvm-project/blob/2ec1aba2b69faa1de5f71832a48e25aa3b5d5314/mlir/lib/Analysis/DataFlow/DeadCodeAnalysis.cpp#L322
    // That's why we need to set all operands to unknown constants.
    setAllToUnknownConstants(op->getResults());
    for (Region &region : op->getRegions()) {
      for (Block &block : region.getBlocks())
        setAllToUnknownConstants(block.getArguments());
    }
    return success();
  }

private:
  /// Set all given values as not constants.
  void setAllToUnknownConstants(ValueRange values) {
    dataflow::ConstantValue unknownConstant(nullptr, nullptr);
    for (Value value : values) {
      auto *constant =
          getOrCreate<dataflow::Lattice<dataflow::ConstantValue>>(value);
      propagateIfChanged(constant, constant->join(unknownConstant));
    }
  }
};
} // namespace

std::unique_ptr<DataFlowSolver> createDataFlowSolver() {
  auto solver = std::make_unique<DataFlowSolver>();
  solver->load<dataflow::DeadCodeAnalysis>();
  solver->load<ConstantAnalysis>();
  return solver;
}

} // namespace mlir
