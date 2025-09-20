#include "triton/Tools/GenericSwizzling.h"

#include "third_party/f2reduce/f2reduce.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "generic-swizzling"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

#if defined(_MSC_VER) && !defined(__clang__)
// from https://gist.github.com/pps83/3210a2f980fd02bb2ba2e5a1fc4a2ef0
#include <intrin.h>

static int __builtin_ctzll(unsigned long long x) {
  unsigned long r;
  _BitScanForward64(&r, x);
  return static_cast<int>(r);
}

#endif

void printBasis(const llvm::SmallVector<int32_t> &basis,
                const std::string &name) {
  llvm::errs() << name << ": ";
  for (int32_t b : basis)
    llvm::errs() << b << " ";
  llvm::errs() << "\n";
}

using namespace mlir;
using namespace mlir::triton;

namespace {

// Goes from bases of the form [[1], [2], [4], [8]] to [1, 2, 4, 8]
SmallVector<int32_t> flatten(const LinearLayout &ll, StringAttr dim) {
  assert(ll.getNumOutDims() == 1);
  auto outDim = *ll.getOutDimNames().begin();
  SmallVector<int32_t> vec;
  for (int i = 0; i < ll.getInDimSizeLog2(dim); ++i)
    vec.push_back(ll.getBasis(dim, i, outDim));
  return vec;
};

SmallVector<int32_t> removeZeros(ArrayRef<int32_t> vec) {
  SmallVector<int32_t> result;
  for (int32_t r : vec) {
    if (r != 0) {
      result.push_back(r);
    }
  }
  return result;
}

// [1, 2, 4, 8] -> [[1], [2], [4], [8]]
std::vector<std::vector<int32_t>> unflatten(ArrayRef<int32_t> basis) {
  std::vector<std::vector<int32_t>> unflattened;
  for (int32_t b : basis)
    unflattened.push_back({b});
  return unflattened;
}

// Compute the nullspace basis of `vectors`
SmallVector<int32_t> nullspaceBasis(ArrayRef<int32_t> vectors, int32_t dim) {
  // Solve A^T x = 0, where A is the matrix of vectors
  // To do this, we form a matrix where each vector is a row
  const int32_t nRows = vectors.size();
  auto mat = std::make_unique<uint64_t[]>(nRows);
  for (int i = 0; i < nRows; ++i)
    mat[i] = static_cast<uint64_t>(vectors[i]);
  f2reduce::inplace_rref_strided(mat.get(), /*rows=*/nRows, /*cols=*/dim,
                                 /*stride=*/1);

  llvm::SmallDenseSet<int32_t> pivotCols;
  for (int32_t r = 0; r < nRows; ++r)
    if (mat[r])
      pivotCols.insert(__builtin_ctzll(mat[r]));

  SmallVector<int32_t> basis;
  for (int32_t freeCol = 0; freeCol < dim; ++freeCol) {
    if (!pivotCols.contains(freeCol)) {
      uint64_t vec = 1ull << freeCol;
      for (int32_t r = 0; r < nRows; ++r)
        if (mat[r] & (1ull << freeCol)) {
          const int32_t pivot = __builtin_ctzll(mat[r]);
          vec ^= (1ull << pivot);
        }
      basis.push_back(static_cast<int32_t>(vec));
    }
  }
  return basis;
}

// Find the smallest tile that we can read and write to smem
// without sacrificing vectorisation and split it into its own
// `reps` dimension
LinearLayout buildReps(MLIRContext *ctx, const LinearLayout &src,
                       const LinearLayout &dst, const LinearLayout &smem) {
  auto kVec = StringAttr::get(ctx, "vector");
  auto kBank = StringAttr::get(ctx, "bank");
  auto kSegment = StringAttr::get(ctx, "segment");
  auto kReps = StringAttr::get(ctx, "reps");
  auto kReg = StringAttr::get(ctx, "register");
  // A basis is a rep if:
  // 1) It is in registers in both src and dst
  // 2) It is in the segment of smem (i.e., is not part of just one
  //    load/store)
  SetVector<int32_t> srcRegs(llvm::from_range_t{}, flatten(src, kReg));
  SetVector<int32_t> dstRegs(llvm::from_range_t{}, flatten(dst, kReg));
  SetVector<int32_t> smemSegment(llvm::from_range_t{}, flatten(smem, kSegment));
  SetVector<int32_t> segment;
  SetVector<int32_t> reps;
  for (auto s : smemSegment) {
    if (srcRegs.contains(s) && dstRegs.contains(s)) {
      reps.insert(s);
    } else {
      segment.insert(s);
    }
  }

  auto smemReps = LinearLayout({{kVec, smem.getBases().lookup(kVec)},
                                {kBank, smem.getBases().lookup(kBank)},
                                {kSegment, unflatten(to_vector(segment))},
                                {kReps, unflatten(to_vector(reps))}},
                               smem.getOutDims(),
                               /*requireSurjective=*/true);
  return smemReps;
}

SmallVector<int32_t> computeSegment(const SmallVector<int32_t> &bankSrc,
                                    const SmallVector<int32_t> &bankDst,
                                    int32_t dim, int32_t lenSegment) {
  llvm::SmallDenseSet<int32_t> setSrc(bankSrc.begin(), bankSrc.end());
  llvm::SmallDenseSet<int32_t> setDst(bankDst.begin(), bankDst.end());
  // Remove the 0 as it's not a basis
  setSrc.erase(0);
  setDst.erase(0);

  SmallVector<int32_t> segment;
  for (int32_t b = 0; b < dim; ++b)
    if (!setSrc.contains(1 << b) && !setDst.contains(1 << b))
      segment.push_back(1 << b);
  if (segment.size() >= lenSegment) {
    segment.resize(lenSegment);
    return segment;
  }

  // A and B are the difference sets
  SmallVector<int32_t> A, B;
  for (int32_t v : setSrc)
    if (!setDst.contains(v))
      A.push_back(v);
  for (int32_t v : setDst)
    if (!setSrc.contains(v))
      B.push_back(v);
  if (A.size() > B.size()) {
    std::swap(A, B);
  }
  llvm::sort(A);
  llvm::sort(B);
  // A is the smaller set now
  auto logBankConflicts = std::min<int32_t>(
      std::max<int32_t>(0, lenSegment - A.size() - segment.size()), A.size());
  // Conflict-free
  for (int i = logBankConflicts; i < A.size(); ++i)
    segment.push_back(A[i] ^ B[i]);
  // Write conflicts
  segment.append(A.begin(), A.begin() + logBankConflicts);
  // Read conflicts
  segment.append(B.begin(), B.begin() + logBankConflicts);

  if (segment.size() > lenSegment)
    segment.resize(lenSegment);
  return segment;
}

SmallVector<int32_t> complementBasis(ArrayRef<int32_t> basis, int32_t dim) {
  const int32_t nRows = basis.size();
  auto mat = std::make_unique<uint64_t[]>(nRows);
  for (int r = 0; r < nRows; ++r)
    mat[r] = static_cast<uint64_t>(basis[r]);

  f2reduce::inplace_rref_strided(mat.get(), /*rows=*/nRows,
                                 /*cols=*/dim, /*stride=*/1);

  llvm::SmallDenseSet<int32_t> pivotCols;
  for (int r = 0; r < nRows; ++r) {
    if (mat[r]) {
      pivotCols.insert(__builtin_ctzll(mat[r])); // leading-1 position
    }
  }

  SmallVector<int32_t> comp;
  for (int i = 0; i < dim; ++i)
    if (!pivotCols.contains(i))
      comp.push_back(1 << i);

  return comp;
}
} // namespace

namespace mlir::triton::gpu {

SmallVector<int32_t> intersectionBasis(ArrayRef<int32_t> b1,
                                       ArrayRef<int32_t> b2, int32_t dim) {
  // If needed to be generic, this can be done computing
  // nullspaceBasis(concat(nullspaceBasis(b1), nullspaceBasis(b2)))
  // but doing this returns the bases in an arbitrary order!
  auto isPowerOf2 = [](int32_t x) { return llvm::isPowerOf2_32(x); };
  bool powerOf2 = llvm::all_of(b1, isPowerOf2) && llvm::all_of(b2, isPowerOf2);
  if (powerOf2) {
    SmallVector<int32_t> result;
    // Heuristic: We choose to retain the order relative to b1
    SetVector<int32_t> set2(b2.begin(), b2.end());
    for (int32_t b : b1) {
      if (b != 0 && set2.contains(b)) {
        result.push_back(b);
      }
    }
    return result;
  } else {
    auto ns1 = nullspaceBasis(b1, dim);
    auto ns2 = nullspaceBasis(b2, dim);
    auto joint = llvm::to_vector(llvm::concat<int32_t>(ns1, ns2));
    return nullspaceBasis(joint, dim);
  }
}

std::pair<int, int> bankConflicts(ArrayRef<int32_t> tileSrc,
                                  ArrayRef<int32_t> tileDst,
                                  const LinearLayout &smem) {
  auto *ctx = smem.getOutDimNames().begin()->getContext();
  auto smemFlat = smem.flattenOuts();
  auto inDim = *smem.getInDimNames().begin();
  // Look at the intersection between the segment bases and the tile bases
  // We don't need to intersect with the bases that covert the bank (as in
  // the first 32 / bitwidth bases) because if we hit any of those broadcasting
  // will avoid the bank conflict
  auto segment = StringAttr::get(ctx, "segment");
  auto segmentBases = flatten(smemFlat, segment);

  int32_t rank = smem.getTotalOutDimSizeLog2();
  // compute conflicts
  int write = 1 << intersectionBasis(segmentBases, tileSrc, rank).size();
  int read = 1 << intersectionBasis(segmentBases, tileDst, rank).size();
  return {read - 1, write - 1};
}

std::pair<int, int> bankConflictsLdSt(const LinearLayout &src,
                                      const LinearLayout &dst,
                                      const LinearLayout &smem,
                                      int32_t bitwidth) {
  auto srcFlat = src.flattenOuts();
  auto dstFlat = dst.flattenOuts();
  auto *ctx = smem.getOutDimNames().begin()->getContext();
  auto S = [ctx](StringRef str) { return StringAttr::get(ctx, str); };
  auto kVec = S("vector");
  auto srcLane = flatten(srcFlat, S("lane"));
  auto dstLane = flatten(dstFlat, S("lane"));
  auto log2Vec =
      llvm::Log2_32(std::max(smem.getInDimSize(kVec) * bitwidth / 32, 1));
  srcLane.resize(srcLane.size() - log2Vec);
  dstLane.resize(dstLane.size() - log2Vec);
  return bankConflicts(srcLane, dstLane, smem);
}

int bankConflictsMemDesc(const LinearLayout &reg, const LinearLayout &smem,
                         int32_t bitwidth) {
  auto *ctx = smem.getInDimNames().begin()->getContext();
  auto S = [ctx](StringRef str) { return StringAttr::get(ctx, str); };

  assert(smem.hasInDim(S("offset")) && "shared layout must have an offset dim");
  assert(reg.hasInDim(S("register")) &&
         "register layout must have a register dim");
  auto regNoBroadcast = actionRemoveBroadcastedRegs(reg).apply(reg);
  auto regToShared = regNoBroadcast.invertAndCompose(smem);
  auto [elemsPerVec, permutation] =
      largestVectorisation(ctx, regToShared, bitwidth);
  regNoBroadcast = permutation.apply(regNoBroadcast);

  int32_t vecSize = elemsPerVec;
  int32_t bankSize =
      std::min(32 * 32 / (vecSize * bitwidth), smem.getTotalInDimSize());
  int32_t segmentSize = smem.getTotalInDimSize() / (bankSize * vecSize);
  SmallVector<std::pair<StringAttr, int32_t>> newInDims = {
      {S("vector"), vecSize},
      {S("bank"), bankSize},
      {S("segment"), segmentSize},
  };
  auto smemReshaped = smem.reshapeIns(newInDims);
  return bankConflictsLdSt(regNoBroadcast, regNoBroadcast, smemReshaped,
                           bitwidth)
      .first;
}

std::optional<SmallVector<int32_t>> optimalSwizzlingTile(
    const LinearLayout &a, const LinearLayout &b, int32_t nRegA, int32_t nRegB,
    ArrayRef<int32_t> laneIdTileA, ArrayRef<int32_t> laneIdTileB) {
  // For now se just implement the .v4 variants for all the instructions
  // We could generalise this in the future
  assert(nRegA + laneIdTileA.size() == nRegB + laneIdTileB.size());
  // normalise nRegA >= nRegB
  if (nRegA < nRegB) {
    return optimalSwizzlingTile(b, a, nRegB, nRegA, laneIdTileB, laneIdTileA);
  }
  assert(nRegA >= nRegB);

  auto *ctx = a.getInDimNames().begin()->getContext();
  auto kReg = StringAttr::get(ctx, "register");
  auto kLane = StringAttr::get(ctx, "lane");
  auto dim = a.getTotalOutDimSizeLog2();
  // map from b to a
  LinearLayout cvt = b.invertAndCompose(a);

  // The contiguous tile of ld.shared.b32.v4 for a packed element of size
  // bitwidth is composed of 128/bitwidth register elements
  // The contiguous tile of ldmatrix.v4 for a packed element of size bitwidth
  // is composed of 32/bitwidth register elements and the bases 0, 1st as given
  // by the laneAddr
  // The contiguous tile of ldmatrix.v4.trans for a packed element of size 16
  // is composed of the bases 2, 3, 4th as given by the laneAddr

  // Note that for register elements, we can choose any register basis we want,
  // but the lane bases are fixed

  // In this function, we compute a tile (set of bases) such that it matches
  // the tiles of A and B

  auto regA = flatten(a, kReg);
  auto regB = flatten(b, kReg);
  auto laneA = flatten(a, kLane);
  auto laneB = flatten(b, kLane);

  // Compute the number of registers that start the tile
  SmallVector<int32_t> vbasis = intersectionBasis(regA, regB, dim);
  // We need to have at least nRegB vectorisation
  if (vbasis.size() < nRegB) {
    return std::nullopt;
  }
  vbasis.resize(nRegB);

  auto index = [](ArrayRef<int32_t> lane, ArrayRef<int32_t> laneIdTile) {
    SmallVector<int32_t> ret;
    for (auto id : laneIdTile) {
      ret.push_back(lane[id]);
    }
    return ret;
  };
  auto laneTileA = index(laneA, laneIdTileA);
  auto laneTileB = index(laneB, laneIdTileB);

  // We need the tiles to be contiguous
  auto isZero = [](int32_t b) { return b == 0; };
  if (llvm::any_of(laneTileA, isZero) || llvm::any_of(laneTileB, isZero)) {
    return std::nullopt;
  }
  // The first lanes must map to registers in A
  for (int i = 0; i < nRegA - nRegB; ++i) {
    if (cvt.getBasis(kLane, laneIdTileB[i], kReg) == 0) {
      return std::nullopt;
    }
  }
  // The rest of the lanes must map to each other
  for (auto [idxA, idxB] :
       llvm::zip(laneIdTileA, laneIdTileB.take_back(laneIdTileA.size()))) {
    if (cvt.getBasis(kLane, idxB, kLane) != (1 << idxA)) {
      return std::nullopt;
    }
  }
  vbasis.append(laneTileB.begin(), laneTileB.end());
  return vbasis;
}

LinearLayout
optimalSwizzling(const LinearLayout &src, const LinearLayout &dst,
                 int32_t bitwidth, ArrayRef<int32_t> vbasis,
                 ArrayRef<int32_t> tileSrc, ArrayRef<int32_t> tileDst,
                 ArrayRef<std::pair<StringAttr, int32_t>> outDims) {
  // We work on the flattened tensors as the tensor dimensions are not relevant
  assert(src.getNumOutDims() == 1 && dst.getNumOutDims() == 1 &&
         "src and dst must have a single output dimension");

  const int32_t dim = src.getTotalOutDimSizeLog2();
  auto *ctx = src.getInDimNames().begin()->getContext();
  auto kReg = StringAttr::get(ctx, "register");

  auto regsNotZero = [kReg](const LinearLayout &ll) {
    return llvm::all_of(
        ll.getBases().lookup(kReg),
        [](const std::vector<int32_t> &basis) { return basis[0] != 0; });
  };
  assert(
      regsNotZero(src) &&
      "Remove register broadcasting from src. See actionRemoveBroadcastedRegs");
  assert(
      regsNotZero(dst) &&
      "Remove register broadcasting from dst. See actionRemoveBroadcastedRegs");

  llvm::SmallVector<int32_t> bankSrc;
  bankSrc.append(vbasis.begin(), vbasis.end());
  bankSrc.append(tileSrc.begin(), tileSrc.end());
  llvm::SmallVector<int32_t> bankDst;
  bankDst.append(vbasis.begin(), vbasis.end());
  bankDst.append(tileDst.begin(), tileDst.end());

  // Bits in a bank segment: 32 banks x 32 bits
  constexpr int32_t bankBits = 32 * 32;
  // Bases needed to cover a whole bank segment
  const int32_t lenBbasis = std::min<int32_t>(
      llvm::Log2_32(bankBits / ((1 << vbasis.size()) * bitwidth)),
      dim - vbasis.size());
  // Bases to cover all the tensor
  const int32_t lenSbasis = dim - lenBbasis - vbasis.size();

  auto sbasis = computeSegment(bankSrc, bankDst, dim, lenSbasis);

  // The bank is the complement of the union of the vector and the start of the
  // segments
  SmallVector<int32_t> unionBasis;
  unionBasis.append(vbasis.begin(), vbasis.end());
  unionBasis.append(sbasis.begin(), sbasis.end());
  SmallVector<int32_t> bbasis = complementBasis(unionBasis, dim);

  assert(bbasis.size() == lenBbasis + (lenSbasis - sbasis.size()) &&
         "bbasis size mismatch");

  // Build the 1D result layout
  StringAttr vecAttr = StringAttr::get(ctx, "vector");
  StringAttr bankAttr = StringAttr::get(ctx, "bank");
  StringAttr segAttr = StringAttr::get(ctx, "segment");

  // src has just 1 outDim
  LinearLayout basis1D({{vecAttr, unflatten(vbasis)},
                        {bankAttr, unflatten(bbasis)},
                        {segAttr, unflatten(sbasis)}},
                       src.getOutDims(), /*requireSurjective=*/true);
  basis1D = buildReps(ctx, src, dst, basis1D);

  return basis1D.reshapeOuts(outDims);
}

SmallVector<int32_t> getVbasis(const LinearLayout &src, const LinearLayout &dst,
                               int32_t bitwidth) {
  auto *ctx = src.getInDimNames().begin()->getContext();
  auto kReg = StringAttr::get(ctx, "register");
  auto regSrc = flatten(src, kReg);
  auto regDst = flatten(dst, kReg);
  auto dim = src.getTotalOutDimSizeLog2();
  // Observation: We always get optimal throughput regardless of the basis
  // we put in the first bank (first log2(32/bitwidth) bases in vbasis.
  // Proof:
  //   1. The elements of vbasis are not swizzled so the access pattern
  //      will be homogeneous for all the emitted instructions
  //   2. If we put a basis that is in registers for src and
  //      in lanes for dst, it will vectorize when writing and
  //      broadcast when reading
  //   3. If one basis is in the lane for src and in the register for dst,
  //      we still get full throughput as all the writes are done in parallel
  //      https://forums.developer.nvidia.com/t/bytes-in-shared-memory/49416/2?utm_source=chatgpt.com
  //   4. If one basis is in the register and other is in the warp, different
  //      warps will be executed independently so nothing to do here
  // Because of this, we can:
  // 1. Put whatever we want on the first log2(32/bitwidth) bases in vbasis
  // 2. Put vectorise into b32.x2 or b32.4 as long as the intersection between
  //    if the intersection between the registers of the part we are
  //    vectorising and the lanes of the other is not empty.

  SmallVector<int32_t> vbasis = intersectionBasis(regSrc, regDst, dim);
  // Restrict the vectorisation to the maximum we can use
  auto maxVecBases = llvm::Log2_32(128 / bitwidth);
  if (vbasis.size() > maxVecBases) {
    vbasis.resize(maxVecBases);
  }
  int log2VecSrc = vbasis.size();
  int log2VecDst = vbasis.size();
  int basesPerBank = bitwidth > 32 ? 0 : llvm::Log2_32(32 / bitwidth);

  // Count the number of bases that can be vectorised after filling up bank0
  // without creating bank conflicts

  auto kLane = StringAttr::get(ctx, "lane");
  llvm::SmallDenseSet<int32_t> laneSrcSet(llvm::from_range_t{},
                                          flatten(src, kLane));
  llvm::SmallDenseSet<int32_t> laneDstSet(llvm::from_range_t{},
                                          flatten(dst, kLane));

  // Find other bases that are in the register that are in the lane of the other
  // layout This would account for more vectorisation on the register side and
  // will not affect bank conflicts on the other side
  auto intersectAfterBank0 =
      [basesPerBank, bitwidth](ArrayRef<int32_t> regs, ArrayRef<int32_t> vbasis,
                               const llvm::SmallDenseSet<int32_t> &otherLane) {
        SmallVector<int32_t> vec;
        // Handle bitwidth > 32
        auto maxVecLog2 = llvm::Log2_32(128 / std::max(bitwidth, 32));
        for (auto r : regs) {
          if (llvm::is_contained(vbasis, r)) {
            continue;
          }
          if (otherLane.contains(r)) {
            vec.push_back(r);
          }
          if (vec.size() == maxVecLog2) {
            break;
          }
        }
        return vec;
      };

  // Whether we will vectorise src or dst
  bool vecSrc = true;
  if ((1 << vbasis.size()) * bitwidth < 32) {
    // Extend vbasis following the register orderings to avoid implicit
    // reorderings and pick the layout that offers the best extra
    // vectorisation opportunities.
    auto isPrefix = [](ArrayRef<int32_t> prefix,
                       ArrayRef<int32_t> regs) -> bool {
      assert(prefix.size() <= regs.size());
      for (size_t i = 0; i < prefix.size(); ++i) {
        if (prefix[i] != regs[i]) {
          return false;
        }
      }
      return true;
    };
    bool srcPrefix = isPrefix(vbasis, regSrc);
    bool dstPrefix = isPrefix(vbasis, regDst);

    auto movePrefixToFront = [&vbasis](ArrayRef<int32_t> regs) {
      SmallVector<int32_t> reordered;
      reordered.append(vbasis.begin(), vbasis.end());
      for (int32_t r : regs) {
        if (!llvm::is_contained(vbasis, r)) {
          reordered.push_back(r);
        }
      }
      return reordered;
    };

    // If neither is a prefix we will have to emit PRMTs
    // We move the basis to the front and continue
    if (!vbasis.empty() && !srcPrefix && !dstPrefix) {
      regSrc = movePrefixToFront(regSrc);
      regDst = movePrefixToFront(regDst);
      srcPrefix = isPrefix(vbasis, regSrc);
      dstPrefix = isPrefix(vbasis, regDst);
      assert(srcPrefix && dstPrefix && "src and dst must be prefixes");
    }
    auto fillVbasis = [basesPerBank](ArrayRef<int32_t> regs,
                                     ArrayRef<int32_t> vbasis) {
      SmallVector<int32_t> ret;
      ret.append(vbasis.begin(), vbasis.end());
      for (int i = vbasis.size(); i < basesPerBank && i < regs.size(); ++i) {
        ret.push_back(regs[i]);
      }
      return ret;
    };

    // Choose the one that intersects more with the lane of the other after
    // filling up bank0. This avoids creating bank conflicts with the
    // non-vectorised part.
    if (srcPrefix && dstPrefix) {
      auto vecSrc =
          intersectAfterBank0(regSrc, fillVbasis(regSrc, vbasis), laneDstSet);
      auto vecDst =
          intersectAfterBank0(regDst, fillVbasis(regDst, vbasis), laneSrcSet);
      // Choose to go on with the one with the most vectorisation
      srcPrefix = vecSrc >= vecDst;
      dstPrefix = vecDst > vecSrc;
    }
    assert(srcPrefix != dstPrefix &&
           "src and dst must have different prefixes");
    // Fill up vbasis up to basesPerBank
    vecSrc = srcPrefix;
    vbasis = fillVbasis(vecSrc ? regSrc : regDst, vbasis);
  }

  // Append the extra vectorisation bases
  vbasis.append(vecSrc ? intersectAfterBank0(regSrc, vbasis, laneDstSet)
                       : intersectAfterBank0(regDst, vbasis, laneSrcSet));
  return vbasis;
}

LinearLayout optimalSwizzlingLdSt(const LinearLayout &src,
                                  const LinearLayout &dst, int32_t bitwidth) {
  auto *ctx = src.getInDimNames().begin()->getContext();
  auto kReg = StringAttr::get(ctx, "register");
  auto kLane = StringAttr::get(ctx, "lane");
  auto srcFlat = src.flattenOuts();
  auto dstFlat = dst.flattenOuts();
  auto regSrc = flatten(srcFlat, kReg);
  auto regDst = flatten(dstFlat, kReg);
  auto laneSrc = flatten(srcFlat, kLane);
  auto laneDst = flatten(dstFlat, kLane);

  // Compute the vectorisation basis
  auto vbasis = getVbasis(srcFlat, dstFlat, bitwidth);

  // Create the tile for ld.shared and st.shared
  auto tileSrc = laneSrc;
  auto tileDst = laneDst;

  // Remove the last 1 or 2 bases if we do load/store of 64 or 128 bits
  auto getVectorSize = [](ArrayRef<int32_t> regs, ArrayRef<int32_t> vbasis) {
    for (auto [i, v] : llvm::enumerate(vbasis)) {
      if (i < regs.size() && regs[i] == v) {
        i++;
      } else {
        return i;
      }
    }
    return vbasis.size();
  };
  auto vectorSizeSrc = getVectorSize(regSrc, vbasis);
  if ((1 << vectorSizeSrc) * bitwidth > 32) {
    auto vn = llvm::Log2_32((1 << vectorSizeSrc) * bitwidth / 32);
    tileSrc = to_vector(ArrayRef(tileSrc).drop_back(vn));
  }
  auto vectorSizeDst = getVectorSize(regDst, vbasis);
  if ((1 << vectorSizeDst) * bitwidth > 32) {
    auto vn = llvm::Log2_32((1 << vectorSizeDst) * bitwidth / 32);
    tileDst = to_vector(ArrayRef(tileDst).drop_back(vn));
  }

  // Remove the bases that we already used in the vectorisation
  auto removeVbasis = [](ArrayRef<int32_t> lane, ArrayRef<int32_t> vbasis) {
    SmallVector<int32_t> ret;
    for (auto b : lane) {
      if (!llvm::is_contained(vbasis, b)) {
        ret.push_back(b);
      }
    }
    return ret;
  };
  tileSrc = removeVbasis(tileSrc, vbasis);
  tileDst = removeVbasis(tileDst, vbasis);

  return optimalSwizzling(srcFlat, dstFlat, bitwidth, vbasis, tileSrc, tileDst,
                          src.getOutDims());
}

std::pair<LinearLayout, std::pair<int32_t, int32_t>>
optimalSwizzling(const LinearLayout &src, const LinearLayout &dst,
                 ArrayRef<LocalMemOpTile> srcTiles,
                 ArrayRef<LocalMemOpTile> dstTiles, int32_t bitwidth) {
  assert(bitwidth <= 128 && "bitwidth must be <= 128");
  auto srcFlat = src.flattenOuts();
  auto dstFlat = dst.flattenOuts();
  // Number of total bases needed to cover the necessary contiguous tile
  // We assume using ld.shared.b32.v4 in the case of ld/st ops
  const auto totalBases = llvm::Log2_32(128 / bitwidth);

  auto *ctx = src.getInDimNames().begin()->getContext();
  auto kReg = StringAttr::get(ctx, "register");

  // Find the pairs of instructions that we can use to lower this converet
  SmallVector<std::tuple<std::pair<int32_t, int32_t>, SmallVector<int32_t>>>
      instr;
  for (const auto &[idxSrc, instrSrc] : llvm::enumerate(srcTiles)) {
    auto logRegSrc = totalBases - instrSrc.laneContig.size();
    for (const auto &[idxDst, instrDst] : llvm::enumerate(dstTiles)) {
      auto logRegDst = totalBases - instrDst.laneContig.size();
      auto maybeTile =
          optimalSwizzlingTile(srcFlat, dstFlat, logRegSrc, logRegDst,
                               instrSrc.laneContig, instrDst.laneContig);
      if (maybeTile.has_value()) {
        instr.push_back({{idxSrc, idxDst}, std::move(*maybeTile)});
      }
    }
  }
  auto getTile =
      [](const LocalMemOpTile &instr, ArrayRef<int32_t> regs,
         ArrayRef<int32_t> lane,
         ArrayRef<int32_t> vbasis) -> std::optional<SmallVector<int32_t>> {
    // pick the first 3 - laneAddr.size() registers that are not in vbasis
    SmallVector<int32_t> tile;
    auto regNeeded = 3 - instr.laneAddr.size();
    assert(regNeeded >= 0 && "laneAddr.size() must be <= 3");
    for (int32_t r : regs) {
      if (regNeeded == 0) {
        break;
      }
      if (!llvm::is_contained(vbasis, r)) {
        tile.push_back(r);
        regNeeded--;
      }
    }
    // Not enough registers to fill in the tile
    if (regNeeded > 0) {
      return std::nullopt;
    }
    for (auto i : instr.laneAddr) {
      tile.push_back(lane[i]);
    }
    return tile;
  };

  auto kLane = StringAttr::get(ctx, "lane");
  auto regSrc = flatten(srcFlat, kReg);
  auto regDst = flatten(dstFlat, kReg);
  auto laneSrc = flatten(srcFlat, kLane);
  auto laneDst = flatten(dstFlat, kLane);

  // Get the associated src/dst tiles for each instruction if they exist
  SmallVector<std::tuple<std::pair<int32_t, int32_t>, SmallVector<int32_t>,
                         SmallVector<int32_t>, SmallVector<int32_t>>>
      tiles;
  for (auto [instrs, vbasis] : instr) {
    auto maybeTileSrc =
        getTile(srcTiles[instrs.first], regSrc, laneSrc, vbasis);
    auto maybeTileDst =
        getTile(dstTiles[instrs.second], regDst, laneDst, vbasis);
    if (!maybeTileSrc.has_value() || !maybeTileDst.has_value()) {
      continue;
    }
    tiles.push_back({instrs, std::move(vbasis), std::move(*maybeTileSrc),
                     std::move(*maybeTileDst)});
  }

  if (tiles.empty()) {
    // We lower to an ld / st, but can't use LDS128/STS128
    auto smem = optimalSwizzlingLdSt(src, dst, bitwidth);
    return {smem, {0, 0}};
  } else {
    // We choose the pair of instructions that minimises the total bank
    // conflicts
    SmallVector<std::tuple<int, LinearLayout, std::pair<int32_t, int32_t>>>
        smems;
    for (auto [instrs, vbasis, tileSrc, tileDst] : tiles) {
      auto smem = optimalSwizzling(srcFlat, dstFlat, bitwidth, vbasis, tileSrc,
                                   tileDst, src.getOutDims());
      auto [read, write] = bankConflicts(tileSrc, tileDst, smem);
      smems.push_back({read + write, smem, {instrs.first, instrs.second}});
    }
    // Current heuristic: Minimise total bank conflicts
    // We break ties looking at the number of rounds we do to move the data
    auto kReps = StringAttr::get(ctx, "reps");
    auto it = llvm::min_element(smems, [kReps](const auto &a, const auto &b) {
      return std::get<0>(a) < std::get<0>(b) ||
             (std::get<0>(a) == std::get<0>(b) &&
              std::get<1>(a).getInDimSize(kReps) >
                  std::get<1>(b).getInDimSize(kReps));
    });
    return {std::get<1>(*it), std::get<2>(*it)};
  }
}

} // namespace mlir::triton::gpu
