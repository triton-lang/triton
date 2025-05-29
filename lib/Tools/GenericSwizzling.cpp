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
  auto outDimNames = to_vector(ll.getOutDimNames());
  assert(outDimNames.size() == 1);
  SmallVector<int32_t> vec;
  for (int i = 0; i < ll.getInDimSizeLog2(dim); ++i)
    vec.push_back(ll.getBasis(dim, i, outDimNames[0]));
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
SmallVector<int32_t> nullspaceBasis(ArrayRef<int32_t> vectors, int32_t rank) {
  // Solve A^T x = 0, where A is the matrix of vectors
  // To do this, we form a matrix where each vector is a row
  const int32_t nRows = vectors.size();
  auto mat = std::make_unique<uint64_t[]>(nRows);
  for (int i = 0; i < nRows; ++i)
    mat[i] = static_cast<uint64_t>(vectors[i]);
  f2reduce::inplace_rref_strided(mat.get(), /*rows=*/nRows, /*cols=*/rank,
                                 /*stride=*/1);

  // Collect pivot columns.
  llvm::SmallDenseSet<int32_t> pivotCols;
  for (int32_t r = 0; r < nRows; ++r)
    if (mat[r])
      pivotCols.insert(__builtin_ctzll(mat[r]));

  // Free columns are range(rank) - pivotCols.
  SmallVector<int32_t> basis;
  for (int32_t freeCol = 0; freeCol < rank; ++freeCol) {
    if (!pivotCols.contains(freeCol)) {
      uint64_t vec = 1ull << freeCol; // start with e_free
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
  // 2) It is not swizzled
  // 3) It is not vectorised
  // After a moment's reflection, it should be clear that the
  // set of basis that can be reps is exactly the set of elements
  // that are in src[kReg] \cup dst[kReg], but are not in smem[kVec].
  // This is because the swizzled elements are xors of elements that
  // are in the src[kLane] \cup dst[kLane], which is disjoint to this set
  SetVector<int32_t> srcRegs(llvm::from_range_t{}, flatten(src, kReg));
  SetVector<int32_t> dstRegs(llvm::from_range_t{}, flatten(dst, kReg));
  SetVector<int32_t> smemVecs(llvm::from_range_t{}, flatten(smem, kVec));
  SetVector<int32_t> reps;
  for (int32_t b : srcRegs) {
    if (dstRegs.contains(b) && !smemVecs.contains(b)) {
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
  // if (reps.size() == 0) {
  //   return smemReps;
  // }
  //// We now transpose the out dims so that reps are the last ones
  //// In other words, we want the vec+bank+segment to be contiguous in smem
  // auto repsMask = llvm::accumulate(reps, ~0u, [](uint32_t a, uint32_t b) {
  // return a & b; }); auto rank = smemReps.getTotalOutDimSizeLog2();
  //// Reshape to out dims = 2x2x...x2
  // SmallVector<std::pair<StringAttr, int32_t>> splitDims;
  // for (int32_t i = 0; i < rank; ++i) {
  //   splitDims.push_back({StringAttr::get(ctx, std::to_string(i)), 2});
  // }
  // auto smemReps2 = smemReps.reshapeOuts(splitDims);

  // SmallVector<StringAttr> front;
  // SmallVector<StringAttr> back;
  // for (int32_t i = 0; i < rank; ++i) {
  //   if (repsMask & (1 << i)) {
  //     back.push_back(StringAttr::get(ctx, std::to_string(i)));
  //   } else {
  //     front.push_back(StringAttr::get(ctx, std::to_string(i)));
  //   }
  // }
  // auto transDims = llvm::to_vector(llvm::concat<StringAttr>(front, back));
  // smemReps2 = smemReps2.transposeOuts(transDims);
  // auto ret = smemReps2.reshapeOuts(smemReps.getOutDims());
  //// we now have all the reps to be the trailing bases of the smem
  // return ret;
}

SmallVector<int32_t> computeSegment(const SmallVector<int32_t> &bankSrc,
                                    const SmallVector<int32_t> &bankDst,
                                    int32_t rank, int32_t lenSegment) {
  llvm::SmallDenseSet<int32_t> setSrc(bankSrc.begin(), bankSrc.end());
  llvm::SmallDenseSet<int32_t> setDst(bankDst.begin(), bankDst.end());
  // Remove the 0 as it's not a basis
  setSrc.erase(0);
  setDst.erase(0);

  SmallVector<int32_t> segment;
  for (int32_t b = 0; b < rank; ++b)
    if (!setSrc.contains(1 << b) && !setDst.contains(1 << b))
      segment.push_back(1 << b); // free variables
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

SmallVector<int32_t> complementBasis(ArrayRef<int32_t> basis, int32_t rank) {
  const int32_t nRows = basis.size();
  // Build an nRows × rank matrix: each row is one of your 'basis' vectors.
  auto mat = std::make_unique<uint64_t[]>(nRows);
  for (int r = 0; r < nRows; ++r)
    mat[r] = static_cast<uint64_t>(basis[r]);

  // RREF that, with 'rows = nRows' and 'cols = rank'.
  f2reduce::inplace_rref_strided(mat.get(), /*rows=*/nRows,
                                 /*cols=*/rank, /*stride=*/1);

  // Collect which coordinate-columns have pivots.
  llvm::SmallDenseSet<int32_t> pivotCols;
  for (int r = 0; r < nRows; ++r) {
    if (mat[r]) {
      pivotCols.insert(__builtin_ctzll(mat[r])); // leading-1 position
    }
  }

  // Now any coordinate i ∈ [0,rank) that *isn’t* in pivotCols is free,
  // so the unit-vector (1<<i) belongs in your complement.
  SmallVector<int32_t> comp;
  for (int i = 0; i < rank; ++i)
    if (!pivotCols.contains(i))
      comp.push_back(1 << i);

  return comp;
}
} // anonymous namespace

namespace mlir::triton::gpu {

SmallVector<int32_t> intersectionBasis(ArrayRef<int32_t> b1,
                                       ArrayRef<int32_t> b2, int32_t rank) {
  auto ns1 = nullspaceBasis(b1, rank);
  auto ns2 = nullspaceBasis(b2, rank);
  auto joint = llvm::to_vector(llvm::concat<int32_t>(ns1, ns2));
  auto result = nullspaceBasis(joint, rank);
  return result;
}

LinearLayout optimalSwizzling(const LinearLayout &src, const LinearLayout &dst,
                              int32_t bitwidth) {
  assert(llvm::equal(src.getInDimNames(), dst.getInDimNames()) &&
         "src and dst must have identical in dims");
  assert(llvm::equal(src.getOutDims(), dst.getOutDims()) &&
         "src and dst must have identical out dims and shape");
  assert((bitwidth > 0) && llvm::isPowerOf2_32(bitwidth) &&
         "bitwidth must be a power of two");
  assert(bitwidth <= 32 && "NYI: support for larger bitwidths");

  const int32_t rank = src.getTotalOutDimSizeLog2();
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
  const StringAttr out1D = *srcFlat.getOutDimNames().begin(); // single dim

  auto regSrc = flatten(srcFlat, kReg);
  auto regDst = flatten(dstFlat, kReg);
  auto laneSrc = flatten(srcFlat, kLane);
  auto laneDst = flatten(dstFlat, kLane);

  // Compute the vectorisation we can use
  SmallVector<int32_t> vbasis = intersectionBasis(regSrc, regDst, rank);
  // Restrict the vectorisation to the maximum we can use
  auto maxVecBases = llvm::Log2_32(128 / bitwidth);
  if (vbasis.size() > maxVecBases) {
    vbasis.resize(maxVecBases);
  }

  // Bits in a bank segment: 32 banks x 32 bits
  constexpr int32_t bankBits = 32 * 32;
  // Bases needed to cover a whole bank segment
  // FIXME: Handle small converts
  assert(bankBits >= (1 << vbasis.size()) * bitwidth && "cvt too small");
  const int32_t lenBbasis =
      llvm::Log2_32(bankBits / ((1 << vbasis.size()) * bitwidth));
  // Bases to cover all the tensor
  const int32_t lenSbasis = rank - lenBbasis - vbasis.size();

  auto bankSrc = llvm::to_vector(llvm::concat<int32_t>(vbasis, laneSrc));
  auto bankDst = llvm::to_vector(llvm::concat<int32_t>(vbasis, laneDst));
  auto sbasis = computeSegment(bankSrc, bankDst, rank, lenSbasis);

  // The bank is the complement of the union of the vector and the start of the
  // segments
  auto unionBasis = llvm::to_vector(llvm::concat<int32_t>(vbasis, sbasis));
  SmallVector<int32_t> bbasis = complementBasis(unionBasis, rank);
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
