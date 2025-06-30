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
  SetVector<int32_t> reps;
  for (int32_t b : srcRegs) {
    if (dstRegs.contains(b) && smemSegment.contains(b)) {
      reps.insert(b);
    }
  }
  // Split segment into segment and reps
  SetVector<int32_t> segment;
  for (int32_t b : flatten(smem, kSegment)) {
    if (!reps.contains(b)) {
      segment.insert(b);
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

std::pair<int, int> logBankConflicts(const LinearLayout &src,
                                     const LinearLayout &dst,
                                     const LinearLayout &smem,
                                     int32_t bitwidth) {
  // build vector + segment basis
  auto srcFlat = src.flattenOuts();
  auto dstFlat = dst.flattenOuts();
  auto smemFlat = smem.flattenOuts();
  auto *ctx = smem.getOutDimNames().begin()->getContext();
  auto S = [ctx](StringRef str) { return StringAttr::get(ctx, str); };
  auto vecBasis = flatten(smemFlat, S("vector"));
  auto segBasis = flatten(smemFlat, S("segment"));
  auto bank0 = llvm::to_vector(llvm::concat<int32_t>(vecBasis, segBasis));
  auto bitsPerThread = smem.getInDimSize(S("vector")) * bitwidth;
  if (bitsPerThread < 32) {
    // In this case, the first few bank bases are also used to
    // cover the 0th bank so we need to add them.
    unsigned basesMissing = llvm::Log2_32(32 / bitsPerThread);
    unsigned nBankBases = smemFlat.getInDimSizeLog2(S("bank"));
    for (int i = 0; i < std::min(basesMissing, nBankBases); ++i) {
      bank0.push_back(smemFlat.getBasis(S("bank"), i)[0]);
    }
  }
  auto srcLane = flatten(srcFlat, S("lane"));
  auto dstLane = flatten(dstFlat, S("lane"));
  if (bitsPerThread > 32) {
    // The transaction is split into 2 or 4 transactions. Each of them
    // taking 16 or 8 lanes. We just look at those lanes when computing
    // bank conflicts.
    auto logWavefronts = llvm::Log2_32(bitsPerThread / 32);
    srcLane.resize(srcLane.size() - logWavefronts);
    dstLane.resize(dstLane.size() - logWavefronts);
  }
  int32_t rank = smem.getTotalOutDimSizeLog2();
  // compute conflicts
  int read = intersectionBasis(bank0, dstLane, rank).size();
  int write = intersectionBasis(bank0, srcLane, rank).size();
  return {read, write};
}

LinearLayout optimalSwizzling(const LinearLayout &src, const LinearLayout &dst,
                              int32_t bitwidth) {
  assert(llvm::equal(src.getInDimNames(), dst.getInDimNames()) &&
         "src and dst must have identical in dims");
  assert(llvm::equal(src.getOutDims(), dst.getOutDims()) &&
         "src and dst must have identical out dims and shape");
  assert((bitwidth > 0) && llvm::isPowerOf2_32(bitwidth) &&
         "bitwidth must be a power of two");

  const int32_t dim = src.getTotalOutDimSizeLog2();
  assert(src.getNumInDims() != 0);
  auto *ctx = src.getInDimNames().begin()->getContext();
  auto kReg = StringAttr::get(ctx, "register");
  auto kLane = StringAttr::get(ctx, "lane");

  // We work on the flattened tensors as the tensor dimensions are not relevant
  const LinearLayout srcFlat = src.flattenOuts();
  const LinearLayout dstFlat = dst.flattenOuts();
  auto regsNotZero = [kReg](const LinearLayout &ll) {
    return llvm::all_of(
        ll.getBases().lookup(kReg),
        [](const std::vector<int32_t> &basis) { return basis[0] != 0; });
  };
  assert(
      regsNotZero(srcFlat) &&
      "Remove register broadcasting from src. See actionRemoveBroadcastedRegs");
  assert(
      regsNotZero(dstFlat) &&
      "Remove register broadcasting from dst. See actionRemoveBroadcastedRegs");

  auto regSrc = flatten(srcFlat, kReg);
  auto regDst = flatten(dstFlat, kReg);
  auto laneSrc = flatten(srcFlat, kLane);
  auto laneDst = flatten(dstFlat, kLane);

  // Compute the vectorisation we can use
  SmallVector<int32_t> vbasis = intersectionBasis(regSrc, regDst, dim);
  // Restrict the vectorisation to the maximum we can use
  auto maxVecBases = llvm::Log2_32(128 / bitwidth);
  if (vbasis.size() > maxVecBases) {
    vbasis.resize(maxVecBases);
  }

  // Bits in a bank segment: 32 banks x 32 bits
  constexpr int32_t bankBits = 32 * 32;
  // Bases needed to cover a whole bank segment
  const int32_t lenBbasis = std::min<int32_t>(
      llvm::Log2_32(bankBits / ((1 << vbasis.size()) * bitwidth)),
      dim - vbasis.size());
  // Bases to cover all the tensor
  const int32_t lenSbasis = dim - lenBbasis - vbasis.size();

  auto bankSrc = llvm::to_vector(llvm::concat<int32_t>(vbasis, laneSrc));
  auto bankDst = llvm::to_vector(llvm::concat<int32_t>(vbasis, laneDst));

  // Whether we'll use b32.v1 / b32.v2 / b32.v4
  auto b32Vec =
      llvm::Log2_32(std::max<int32_t>((1 << vbasis.size()) * bitwidth / 32, 1));
  // Drop the last vec bases of the banks
  bankSrc.resize(bankSrc.size() - b32Vec);
  bankDst.resize(bankDst.size() - b32Vec);

  auto sbasis = computeSegment(bankSrc, bankDst, dim, lenSbasis);

  // The bank is the complement of the union of the vector and the start of the
  // segments
  auto unionBasis = llvm::to_vector(llvm::concat<int32_t>(vbasis, sbasis));
  SmallVector<int32_t> bbasis = complementBasis(unionBasis, dim);
  // We might be able to vectorise a bit more the load or the store
  // This may happen when there is broadcasting
  // e.g for fp32
  // src = {reg = [], lane = [1, 2, 4, 8, 16], warp = [32]}
  // dst = {reg = [8, 32], lane = [0, 0, 1, 2, 4], warp = [16]}
  if (b32Vec < 2) {
    // For every bank line, find if it is in regSrc or regDst
    // and if so, store the index in the vector
    SmallVector<int32_t> banksInRegSrc;
    SmallVector<int32_t> banksInRegDst;
    for (auto r : bbasis) {
      if (llvm::is_contained(regSrc, r)) {
        banksInRegSrc.push_back(r);
      }
      if (llvm::is_contained(regDst, r)) {
        banksInRegDst.push_back(r);
      }
    }
    // Choose the max vectorisation. Bias towards dst for no very good reason
    // Move the vectorisation elements to the front
    auto newBbasis = banksInRegSrc.size() > banksInRegDst.size()
                         ? std::move(banksInRegSrc)
                         : std::move(banksInRegDst);
    SmallVector<int32_t> others;
    for (auto b : bbasis) {
      if (!llvm::is_contained(newBbasis, b)) {
        others.push_back(b);
      }
    }
    newBbasis.append(others.begin(), others.end());
    bbasis = std::move(newBbasis);
  }

  assert(bbasis.size() == lenBbasis + (lenSbasis - sbasis.size()) &&
         "bbasis size mismatch");

  // Build the 1D result layout
  StringAttr vecAttr = StringAttr::get(ctx, "vector");
  StringAttr bankAttr = StringAttr::get(ctx, "bank");
  StringAttr segAttr = StringAttr::get(ctx, "segment");

  LinearLayout basis1D({{vecAttr, unflatten(vbasis)},
                        {bankAttr, unflatten(bbasis)},
                        {segAttr, unflatten(sbasis)}},
                       srcFlat.getOutDims(), /*requireSurjective=*/true);
  basis1D = buildReps(ctx, srcFlat, dstFlat, basis1D);

  return basis1D.reshapeOuts(src.getOutDims());
}

} // namespace mlir::triton::gpu
