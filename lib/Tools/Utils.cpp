#include "triton/Tools/Utils.h"

#include "third_party/f2reduce/f2reduce.h"
#include "triton/Tools/LinearLayout.h"

namespace mlir::triton {

bool hasRegisterInDims(const LinearLayout &layout) {
  static const auto expectedInDims =
      SmallVector<std::string>({"register", "lane", "warp", "block"});
  auto inDimNamesStr = llvm::to_vector(llvm::map_range(
      layout.getInDimNames(), [](StringAttr attr) { return attr.str(); }));
  return inDimNamesStr == expectedInDims;
}

bool hasCanonicalOutDims(const LinearLayout &layout) {
  for (auto [i, dim] : llvm::enumerate(layout.getOutDimNames())) {
    if (dim.str() != ("dim" + llvm::Twine(i)).str()) {
      return false;
    }
  }
  return true;
}

bool hasPowerOfTwoBases(const LinearLayout &layout) {
  for (const auto &dimBases : llvm::make_second_range(layout.getBases())) {
    for (const auto &basis : dimBases) {
      // The basis broadcasts if it's all zeros
      bool isBroadcast = true;
      for (int32_t val : basis) {
        if (val != 0) {
          if (!isBroadcast) {
            return false;
          }
          isBroadcast = false;
          if ((val & (val - 1)) != 0) { // val is not a power of two
            return false;
          }
        }
      }
    }
  }
  return true;
}

bool isDistributedLayout(const LinearLayout &layout) {
  return hasRegisterInDims(layout) && hasCanonicalOutDims(layout) &&
         hasPowerOfTwoBases(layout);
}

bool hasSharedMemoryInDims(const LinearLayout &layout) {
  static const auto expectedInDims =
      SmallVector<std::string>({"offset", "iteration"});
  auto inDimNamesStr = llvm::to_vector(llvm::map_range(
      layout.getInDimNames(), [](StringAttr attr) { return attr.str(); }));
  return inDimNamesStr == expectedInDims;
}

bool isSharedMemoryLayout(const LinearLayout &layout) {
  return hasSharedMemoryInDims(layout) && hasCanonicalOutDims(layout);
}

namespace {
std::unique_ptr<uint64_t[]> concatMatrices(const LinearLayout &A,
                                           const LinearLayout &B) {
  assert(A.getTotalOutDimSizeLog2() == B.getTotalOutDimSizeLog2() &&
         "Matrices must have the same number of output dimensions");
  int numRows = A.getTotalOutDimSizeLog2();
  int numCols = A.getTotalInDimSizeLog2() + B.getTotalInDimSizeLog2();
  assert(numCols <= 64 && "Can't handle huge layouts");

  auto concat = A.getMatrix();
  auto Bmat = B.getMatrix();
  auto numColsA = A.getTotalInDimSizeLog2();
  for (int r = 0; r < numRows; r++) {
    concat[r] |= Bmat[r] << numColsA;
  }
  return concat;
}

//  TODO Document
LinearLayout solve(const LinearLayout &A, const LinearLayout &B) {
  // A and B may not be surjective, but they have the same image
  // We assert this in the pivot computation
  int numRows = A.getTotalOutDimSizeLog2();
  int numColsA = A.getTotalInDimSizeLog2();
  int numColsB = B.getTotalInDimSizeLog2();
  int numCols = numColsA + numColsB;
  std::unique_ptr<uint64_t[]> combinedMat = concatMatrices(A, B);
  f2reduce::inplace_rref_strided(combinedMat.get(), numRows, numCols,
                                 /*stride=*/1);
  // Compute the pivot columns
  // Since A and B have the same image, each row will either have a pivot
  // or will be all zeros
  SmallVector<int32_t> pivotCols;
  for (int r = 0; r < numRows; r++) {
    auto row = combinedMat[r];
    if (row == 0) {
      continue;
    }
    int c = __builtin_ctzll(combinedMat[r]);
    assert(c < numColsA && "Bug in the algorithm. Pivot column is in B");
    assert(pivotCols.empty() ||
           pivotCols.back() < c && "Pivot columns are not in increasing order");
    pivotCols.push_back(c);
  }

  // Extract A^{-1}B and complete the mapping using zeros for the unnecessary
  // fromDims
  std::unique_ptr<uint64_t[]> retMat(new uint64_t[numColsA]());
  int j = 0;
  for (int r = 0; r < numColsA; r++) {
    auto isPivot = j < numRows && pivotCols[j] == r;
    retMat[r] = isPivot ? combinedMat[j++] >> numColsA : 0;
  }

  // We need names for the in/out dim of the flattened layout we're going to
  // read off from `m`.  These could be anything, doesn't matter.
  StringAttr inDim1D = *A.getInDimNames().begin();
  StringAttr outDim1D = *A.getOutDimNames().begin();

  // Read off the new bases.  These are for a flattened 1D -> 1D
  LinearLayout::BasesT retBases;
  auto &bs = retBases[inDim1D];
  for (int c = 0; c < numColsA; c++) {
    int32_t basis = 0;
    for (int r = 0; r < numRows; r++) {
      basis |= (retMat[r] >> c & 1) << r;
    }
    bs.push_back({basis});
  }

  LinearLayout retFlattened(std::move(retBases),
                            {{outDim1D, A.getTotalInDimSize()}},
                            /*requireSurjective=*/false);

  SmallVector<std::pair<StringAttr, int32_t>> retInDims;
  SmallVector<std::pair<StringAttr, int32_t>> retOutDims;
  for (StringAttr dim : B.getInDimNames()) {
    retInDims.push_back({dim, B.getInDimSize(dim)});
  }
  for (StringAttr dim : A.getInDimNames()) {
    retOutDims.push_back({dim, A.getInDimSize(dim)});
  }
  return retFlattened.reshapeIns(retInDims).reshapeOuts(retOutDims);
}
} // anonymous namespace

// Computes `from^{-1} ⋅ to`
// See the docs in the header file for more information
LinearLayout convertToFrom(const LinearLayout &to, const LinearLayout &from) {
  // We could transpose outer, but it's better explicit than implicit.
  assert(llvm::to_vector(to.getOutDimNames()) ==
         llvm::to_vector(from.getOutDimNames()));
  // We are computing from^{-1} ⋅ to, so from must be surjective so that
  // it has a left inverse.
  assert(from.isSurjective());

  if (isDistributedLayout(to) && isDistributedLayout(from)) {
    // Recall that the invariant for distributed layouts (see
    // isDistributedLayout):
    // - Each basis vector has either zero or one non-zero values
    // - If it has one, it is a power of two
    // - It is surjective, i.e., it is a wide matrix
    // In other words, it forms a permutation matrix with (perhaps) some
    // zero columns (broadcasting).
    auto inDims = llvm::to_vector(to.getInDimNames());
    auto outDims = llvm::to_vector(to.getOutDimNames());
    // All but the register dimensions must be the same size (can be relaxed)
    for (int i = 1; i < inDims.size(); i++) {
      assert(from.getInDimSize(inDims[i]) == to.getInDimSize(inDims[i]));
    }
    // Heuristic for broadcasting
    // Imagine we have two layouts with `warps = [[0, 0],  [0, 0]]`
    // (broadcasting) on both layouts. We could map any warp to any warp in the
    // conversion. Now, we want to map them as the identity map, to mark that
    // nothing needs to be done there. `solve` would map all the warps to zero
    // in this case. The heuristic here is as follows:
    // - If a dimension is the same for both layouts, we want to map it as the
    // identity
    //   Equivalently, we don't add it to the conversion
    // - Otherwise, we just do the solve (i.e. map all the equivalent elements
    // to the
    //   same input element) to take advantage of broadcasting in shared memory
    //   and avoid saving repeated elements in shared memory
    const auto &toBases = to.getBases();
    const auto &fromBases = from.getBases();
    SmallVector<StringAttr> nonIdentityDims;
    LinearLayout toSublayout = LinearLayout::empty();
    LinearLayout fromSublayout = LinearLayout::empty();
    for (auto dim : inDims) {
      // Sublayout the dimensions that are not identity
      auto toDim = to.sublayout(dim, outDims);
      auto fromDim = from.sublayout(dim, outDims);
      if (toDim == fromDim) {
        toSublayout *= toDim;
        fromSublayout *= fromDim;
      }
    }
    if (toSublayout == LinearLayout::empty()) {
      assert(fromSublayout == LinearLayout::empty());
      return LinearLayout::empty();
    }
    // At this point, we have two  layouts with the same in/out dimensions and
    // not necessarily surjective, but with the same image. We just perform a
    // solve and return the result
    return solve(fromSublayout, toSublayout);
  } else {
    // TODO(Lezcano): Make the assertion below work. For that we need to
    // uniformise the shmem layouts defined in ConvertLayoutOpToLLVM.cpp and
    // LinearLayoutConversions.cpp

    // bool regToShmem = isDistributedLayout(from) && isSharedMemoryLayout(to);
    // bool shmemToReg = isSharedMemoryLayout(from) && isDistributedLayout(to);
    // assert(regToShmem || shmemToReg);
    return solve(from, to);
  }
}
} // namespace mlir::triton
