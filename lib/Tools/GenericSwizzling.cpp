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

std::pair<int, int> logBankConflicts(ArrayRef<int32_t> tileSrc,
                                     ArrayRef<int32_t> tileDst,
                                     const LinearLayout &smem,
                                     int32_t bitwidth) {
  auto *ctx = smem.getOutDimNames().begin()->getContext();
  auto smemFlat = smem.flattenOuts();
  auto inDim = *smem.getInDimNames().begin();
  // Take all the bases in the first bank (32 bits)
  auto smemBases =
      flatten(smemFlat.flattenIns(), *smemFlat.getInDimNames().begin());
  auto nBankZero = llvm::Log2_32(std::max<int32_t>(1, 32 / bitwidth));
  if (smemBases.size() >= nBankZero) {
    smemBases.resize(nBankZero);
  }
  // And segments
  auto segment = StringAttr::get(ctx, "segment");
  auto segmentBases = flatten(smemFlat, segment);
  auto bankZero =
      llvm::to_vector(llvm::concat<int32_t>(smemBases, segmentBases));

  int32_t rank = smem.getTotalOutDimSizeLog2();
  // compute conflicts
  int write = intersectionBasis(bankZero, tileSrc, rank).size();
  int read = intersectionBasis(bankZero, tileDst, rank).size();
  return {read, write};
}

std::pair<int, int> logBankConflictsLdSt(const LinearLayout &src,
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
  return logBankConflicts(srcLane, dstLane, smem, bitwidth);
}

int logBankConflictsMemDesc(const LinearLayout &reg, const LinearLayout &smem,
                            int32_t bitwidth) {
  auto *ctx = smem.getInDimNames().begin()->getContext();
  auto S = [ctx](StringRef str) { return StringAttr::get(ctx, str); };

  assert(smem.hasInDim(S("offset")) && "shared layout must have an offset dim");
  assert(reg.hasInDim(S("register")) &&
         "register layout must have a register dim");

  int32_t vecSize = reg.invertAndCompose(smem).getNumConsecutiveInOut();
  int32_t bankSize =
      std::min(32 * 32 / (vecSize * bitwidth), smem.getTotalInDimSize());
  int32_t segmentSize = smem.getTotalInDimSize() / (bankSize * vecSize);
  SmallVector<std::pair<StringAttr, int32_t>> newInDims = {
      {S("vector"), vecSize},
      {S("bank"), bankSize},
      {S("segment"), segmentSize},
  };
  auto smemReshaped = smem.reshapeIns(newInDims);
  return logBankConflictsLdSt(reg, reg, smemReshaped, bitwidth).first;
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
  auto dim = src.getTotalOutDimSizeLog2();
  SmallVector<int32_t> vbasis = intersectionBasis(regSrc, regDst, dim);
  // Restrict the vectorisation to the maximum we can use
  auto maxVecBases = llvm::Log2_32(128 / bitwidth);
  if (vbasis.size() > maxVecBases) {
    vbasis.resize(maxVecBases);
  }
  // We fill-up vbasis until it has 32 bits as best we can
  std::optional<bool> srcFillsBank = std::nullopt;
  if ((1 << vbasis.size()) * bitwidth < 32) {
    auto basesPerBank = llvm::Log2_32(32 / bitwidth);
    auto kWarp = StringAttr::get(ctx, "warp");
    auto warpSrc = removeZeros(flatten(srcFlat, kWarp));
    auto warpDst = removeZeros(flatten(dstFlat, kWarp));
    auto removeVec = [&vbasis](ArrayRef<int32_t> vec) {
      SmallVector<int32_t> result;
      for (int32_t r : vec) {
        if (!llvm::is_contained(vbasis, r)) {
          result.push_back(r);
        }
      }
      return result;
    };
    auto regSrcWarp = intersectionBasis(removeVec(regSrc), warpDst, dim);
    auto regDstWarp = intersectionBasis(removeVec(regDst), warpSrc, dim);
    // Maximise vectorisation in the load or the store without creating
    // conflicts
    SmallVector<int32_t> largest;
    if (regSrcWarp.size() == regDstWarp.size() && regSrcWarp.size() > 0) {
      // We choose the one with the lowest basis in the hope that it will
      // avoid PRMTs. The comparison of the mins will be strict as the sets
      // removeVec(regSrc) and removeVec(regDst) don't intersect
      if (*llvm::min_element(regSrcWarp) < *llvm::min_element(regDstWarp)) {
        largest = regSrcWarp;
        srcFillsBank = true;
      } else {
        largest = regDstWarp;
        srcFillsBank = false;
      }
    } else {
      srcFillsBank = regSrcWarp.size() > regDstWarp.size();
      largest = srcFillsBank.value() ? regSrcWarp : regDstWarp;
    }
    vbasis.append(largest.begin(), largest.end());

    if (vbasis.size() < basesPerBank) {
      // Pad the vectorisation to 32 bits with warp bases
      auto warpSrcWarp = intersectionBasis(warpSrc, warpDst, dim);
      vbasis.append(warpSrcWarp.begin(), warpSrcWarp.end());
    }

    int i = 0;
    while (vbasis.size() < basesPerBank &&
           (i < warpSrc.size() || i < warpDst.size())) {
      // If we have not filled up a whole bank, we add more warp bases
      // until we have 32 bits. They will at least avoid bank conflicts in one
      // direction
      if (i < warpSrc.size() && !llvm::is_contained(vbasis, warpSrc[i])) {
        vbasis.push_back(warpSrc[i]);
      }
      if (vbasis.size() < basesPerBank && i < warpDst.size() &&
          !llvm::is_contained(vbasis, warpDst[i])) {
        vbasis.push_back(warpDst[i]);
      }
      ++i;
    }

    // Trim to basesPerBank if we have added more
    // The idea here is that implementing asymmetric vectorisation without bank
    // conflicts is a bit tricky. Basically, in this case, you need to use the
    // vectorisation base in the swizzling pattern. As such, you would not be
    // able to vectorise all the `ld.shared` instructions that you emit, but
    // just about half of them (the ones that are not swizzled). We don't
    // implement this yet
    if (vbasis.size() > basesPerBank) {
      vbasis.resize(basesPerBank);
    }
  }
  auto log2Vec = llvm::Log2_32(
      std::max<int32_t>(1, ((1 << vbasis.size()) * bitwidth) / 32));
  auto tileSrc = to_vector(ArrayRef(laneSrc).drop_back(log2Vec));
  auto tileDst = to_vector(ArrayRef(laneDst).drop_back(log2Vec));
  auto smem = optimalSwizzling(srcFlat, dstFlat, bitwidth, vbasis, tileSrc,
                               tileDst, src.getOutDims());

  // We might be able to vectorise a bit more the load or the store
  // This may happen when there is broadcasting
  // e.g for fp32
  // src = {reg = [], lane = [1, 2, 4, 8, 16], warp = [32]}
  // dst = {reg = [8, 32], lane = [0, 0, 1, 2, 4], warp = [16]}
  if (log2Vec < 2) {
    auto smemFlat = smem.flattenOuts();
    // For every bank line, find if it is in regSrc or regDst
    // and if so, store the index in the vector
    SmallVector<size_t> idxBanksInRegSrc;
    SmallVector<size_t> idxBanksInRegDst;
    auto kBank = StringAttr::get(ctx, "bank");
    const auto &banks = flatten(smemFlat, kBank);
    for (auto [i, r] : llvm::enumerate(banks)) {
      if (llvm::is_contained(regSrc, r)) {
        idxBanksInRegSrc.push_back(i);
      }
      if (llvm::is_contained(regDst, r)) {
        idxBanksInRegDst.push_back(i);
      }
    }

    // Choose src/dst if we used them to fill the bank
    // Otherwise choose the max vectorisation
    SmallVector<size_t> bBasisOrder;
    if (srcFillsBank.has_value() && srcFillsBank.value()) {
      bBasisOrder = std::move(idxBanksInRegSrc);
    } else if (srcFillsBank.has_value() && !srcFillsBank.value()) {
      bBasisOrder = std::move(idxBanksInRegDst);
    } else {
      bBasisOrder = idxBanksInRegSrc.size() > idxBanksInRegDst.size()
                        ? std::move(idxBanksInRegSrc)
                        : std::move(idxBanksInRegDst);
    }
    for (int i = 0; i < banks.size(); ++i) {
      if (!llvm::is_contained(bBasisOrder, i)) {
        bBasisOrder.push_back(i);
      }
    }
    smem = ColumnAction(bBasisOrder, kBank, smem.getInDimSizeLog2(kBank))
               .apply(smem);
  }

  return smem;
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
      auto [read, write] = logBankConflicts(tileSrc, tileDst, smem, bitwidth);
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
