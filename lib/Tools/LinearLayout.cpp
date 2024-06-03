#include "triton/Tools/LinearLayout.h"

#include <cstdint>
#include <vector>

#include "mlir/IR/BuiltinAttributes.h"
#include "third_party/f2reduce/f2reduce.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::triton {

namespace {
using BasesT = LinearLayout::BasesT;
using llvm::Twine;

BasesT makeBasesMap(
    ArrayRef<std::pair<StringAttr, std::vector<std::vector<int32_t>>>> bases) {
  BasesT ret;
  for (const auto &[inDim, inDimBases] : bases) {
    ret[inDim] = inDimBases;
  }
  return ret;
}

std::string stringifyBases(const BasesT &bases,
                           ArrayRef<StringAttr> outDimNames) {
  std::string ret;

  if (bases.empty())
    return "(empty layout)\n";

  // TODO: Add spaces for alignment.
  for (const auto &[inDim, inDimBases] : bases) {
    if (inDimBases.empty()) {
      ret += " - " + inDim.str() + " is a size 1 dimension\n";
      continue;
    }

    ret += " - " +
           join(llvm::seq(inDimBases.size()), "\n   ",
                [&, &inDim = inDim, &inDimBases = inDimBases](int i) {
                  return inDim.str() + "=" + std::to_string(1 << i) + " -> (" +
                         join(inDimBases[i], ", ") + ")";
                }) +
           "\n";
  }
  ret += "where out dims are: [" +
         join(outDimNames, ", ", [](StringAttr s) { return s.str(); }) + "]\n";
  return ret;
}

BasesT validateBases(BasesT bases, ArrayRef<StringAttr> outDimNames) {
  if (bases.empty())
    return bases;

  for (const auto &[inDim, inDimBases] : bases) {
    for (const auto &basis : inDimBases) {
      if (llvm::any_of(basis, [](int32_t b) { return b < 0; })) {
        llvm::report_fatal_error(
            "Invalid bases passed to LinearLayout.  Expected all basis "
            "values to be non-negative, but found a negative value for "
            "in dimension '" +
            Twine(inDim) + "'.  Full list of bases:\n" +
            stringifyBases(bases, outDimNames));
      }
    }
  }

  // Check that the bases all have length equal to outDimNames.size().
  for (const auto &[inDim, inDimBases] : bases) {
    for (const auto &basis : inDimBases) {
      if (basis.size() != outDimNames.size()) {
        llvm::report_fatal_error(
            "Invalid bases passed to LinearLayout.  Expect all bases to have "
            "the same size, equal to outDimNames.size() (" +
            Twine(outDimNames.size()) +
            ").  But this failed for in dimension '" + Twine(inDim) +
            "'.  Full list of bases:\n" + stringifyBases(bases, outDimNames));
      }
    }
  }

  return bases;
}

// Dump the matrix to stderr in a human-readable format for debugging.
void dumpMatrix(uint64_t *m, int numRows, int numCols) {
  assert(numCols <= 64);
  for (int r = 0; r < numRows; r++) {
    llvm::errs() << "0b";
    for (int c = 0; c < numCols; c++) {
      llvm::errs() << ((m[r] & (1 << c)) != 0 ? "1" : "0");
    }
    llvm::errs() << "\n";
  }
}

// Build a matrix of size sum(outDimSizeLog2) x sum(inDimSizeLog2) representing
// the bases of the given layout.  This can then be used by f2reduce.
//
// This function is called from the constructor of LinearLayout, so be careful
// not to use any functions that create LLs in here.
std::unique_ptr<uint64_t[]> getMatrix(const LinearLayout &layout) {
  int numRows = layout.getTotalOutDimSizeLog2();
  int numCols = layout.getTotalInDimSizeLog2();

  // Don't handle giant LLs.  This makes some things easier; for example, each
  // row can be a single uint64_t.
  assert(numCols <= 64 && "LinearLayout too large");
  assert(numRows <= 64 && "LinearLayout too large");

  // Suppose we have a layout specified by the following values.
  //
  //   L(0,1) = (0b01, 0b1)
  //   L(0,2) = (0b10, 0b0)
  //   L(1,0) = (0b10, 0b0)
  //   L(2,0) = (0b11, 0b0)
  //
  // We will create one column per entry above.  The max bit width of the
  // codomain is (2,1), so our matrix will have 2+1=3 rows.  The final matrix
  // will be
  //
  //  | L(0,1)[0] L(0,2)[0] L(1,0)[0] L(2,0)[0] |   | 0b1001 |
  //  |    ↓         ↓         ↓         ↓      |   | 0b0111 |
  //  | L(0,1)[1] L(0,2)[1] L(1,0)[1] L(2,0)[1] | = | 0b1000 |
  //  |    ↓         ↓         ↓         ↓      |
  //
  // Note `new uint64_t[n]()` is zero-initialized, but `new uint64_t[n]` is not.
  std::unique_ptr<uint64_t[]> m(new uint64_t[numRows]());
  int r = 0;
  for (StringAttr outDim : layout.getOutDimNames()) {
    int c = 0;
    for (StringAttr inDim : layout.getInDimNames()) {
      for (int i = 0; i < layout.getInDimSizeLog2(inDim); i++) {
        uint64_t basis = layout.getBasis(inDim, i, outDim);
        for (int j = 0; j < layout.getOutDimSizeLog2(outDim); j++) {
          m[r + j] |= ((basis >> j) & 1) << c;
        }
        c++;
      }
    }
    r += layout.getOutDimSizeLog2(outDim);
  }

  return m;
}

// Compute the rank of the matrix formed by taking the bases for the given
// outDim as columns.  In other words, finds the number of linearly-independent
// bases for this output dimension.
int getMatrixRank(std::unique_ptr<uint64_t[]> m, int numRows, int numCols) {
  // stride is specified in number of 64-bit words per row, and we pack our
  // matrix so that there's only one uint64_t per row.
  assert(numCols <= 64);
  f2reduce::inplace_rref_strided(m.get(), numRows, numCols, /*stride=*/1);

  // The rank of the reduced matrix is simply the number of nonzero rows.
  int rank = 0;
  for (int i = 0; i < numRows; i++) {
    if (m[i] != 0)
      rank++;
  }
  return rank;
}

template <typename T, typename U>
void assertDimsEqualIgnoringOrder(T &&a, U &&b) {
  llvm::DenseSet<StringAttr> as(a.begin(), a.end());
  llvm::DenseSet<StringAttr> bs(b.begin(), b.end());
  if (as != bs) {
    llvm::report_fatal_error("Dimensions must match, ignoring order, but they "
                             "don't.  Got dims: [" +
                             Twine(triton::join(a, ", ")) + "] and [" +
                             triton::join(b, ", ") + "]");
  }
}

} // anonymous namespace

LinearLayout::LinearLayout(BasesT bases, ArrayRef<StringAttr> outDimNames,
                           bool requireSurjective /*=true*/)
    : bases(validateBases(std::move(bases), outDimNames)),
      outDimNames(outDimNames.begin(), outDimNames.end()) {
  // Determine whether the this layout is surjective, i.e. that every `out`
  // coordinate can be reached by some `in` coordinate.
  //
  // It's prohibitively slow to calculate this naively, but thankfully, this is
  // equivalent to checking that the number of linearly-independent bases is
  // equal to sum(getOutDimSizeLog2).  This can be computed by finding the rank
  // of the matrix whose columns are those bases.  We can compute the rank of
  // our matrix using Gaussian elimination, which runs in O(n^3) for an n x n
  // matrix.  Our matrix size is sum(inDimSizeLog2) x sum(outDimSizeLog2), so
  // this should be plenty fast.
  this->surjective =
      getMatrixRank(getMatrix(*this), /*numRows=*/getTotalOutDimSizeLog2(),
                    /*numCols=*/getTotalInDimSizeLog2()) ==
      getTotalOutDimSizeLog2();

  if (requireSurjective && !surjective) {
    llvm::report_fatal_error("Layout is expected to be surjective, i.e. every "
                             "`out` coordinate can be reached by some `in` "
                             "coordinate, but was not:\n" +
                             Twine(toString()));
  }
}

LinearLayout::LinearLayout(
    ArrayRef<std::pair<StringAttr, std::vector<std::vector<int32_t>>>> bases,
    ArrayRef<StringAttr> outDimNames, bool requireSurjective /*=true*/)
    : LinearLayout(makeBasesMap(bases), outDimNames, requireSurjective) {}

/*static*/ LinearLayout LinearLayout::identity1D(int32_t size,
                                                 StringAttr inDimName,
                                                 StringAttr outDimName) {
  if (size == 0)
    return LinearLayout::empty();

  assert(llvm::isPowerOf2_32(size));
  std::vector<std::vector<int32_t>> powersOf2;
  for (int32_t i = 1; i < size; i *= 2) {
    powersOf2.emplace_back().push_back(i);
  }
  return LinearLayout({{inDimName, std::move(powersOf2)}}, {outDimName});
}

/*static*/ LinearLayout LinearLayout::zeros1D(int32_t size,
                                              StringAttr inDimName,
                                              StringAttr outDimName) {
  if (size == 0)
    return LinearLayout::empty();

  assert(llvm::isPowerOf2_32(size));
  std::vector<std::vector<int32_t>> zeros;
  for (int i = 0; i < llvm::Log2_32(size); i++) {
    zeros.emplace_back().push_back(0);
  }
  return LinearLayout({{inDimName, zeros}}, {outDimName});
}

int32_t LinearLayout::getOutDimIndex(StringAttr outDim) const {
  // Sadly SetVector doesn't provide an O(1) way to do this.
  for (int i = 0; i < outDimNames.size(); ++i) {
    if (outDimNames[i] == outDim) {
      return i;
    }
  }
  llvm::report_fatal_error("outDim " + Twine(outDim) + " is not in layout\n" +
                           toString());
}

int32_t LinearLayout::getInDimSizeLog2(StringAttr inDim) const {
  auto it = bases.find(inDim);
  assert(it != bases.end());
  return it->second.size();
}

int32_t LinearLayout::getTotalInDimSizeLog2() const {
  return std::accumulate(getInDimNames().begin(), getInDimNames().end(), 0,
                         [&](int32_t acc, StringAttr inDim) {
                           return acc + getInDimSizeLog2(inDim);
                         });
}

int32_t LinearLayout::getOutDimSizeLog2(StringAttr outDim) const {
  // TODO(jlebar): Cache this?
  int32_t outDimIdx = getOutDimIndex(outDim);
  int32_t max = 0;
  for (const auto &[inDim, inDimBases] : bases) {
    for (const auto &basis : inDimBases) {
      max = std::max(max, basis[outDimIdx]);
    }
  }
  return max == 0 ? 0 : llvm::Log2_32(max) + 1;
}

int32_t LinearLayout::getTotalOutDimSizeLog2() const {
  return std::accumulate(getOutDimNames().begin(), getOutDimNames().end(), 0,
                         [&](int32_t acc, StringAttr outDim) {
                           return acc + getOutDimSizeLog2(outDim);
                         });
}

int32_t LinearLayout::getNumConsecutiveInOut() const {
  if (bases.empty() || getNumOutDims() == 0)
    return 1;

  // Count how many of the initial bases for the first in-dim are
  // (2^i, 0, ..., 0).
  const auto &firstInDimBases = bases.begin()->second;
  int consec = 0;
  for (; consec < firstInDimBases.size(); consec++) {
    const auto &basis = firstInDimBases[consec];
    if (basis[0] != (1 << consec) ||
        !std::all_of(basis.begin() + 1, basis.end(),
                     [](int32_t x) { return x == 0; })) {
      break;
    }
  }

  // `or` together all other bases' first out-dim.
  int32_t otherBits = 0;
  for (const auto &[inDim, inDimBases] : bases) {
    for (int i = 0; i < inDimBases.size(); i++) {
      if (inDim != bases.begin()->first || i >= consec) {
        otherBits |= inDimBases[i][0];
      }
    }
  }
  int32_t trailingZeros = otherBits != 0 ? __builtin_ctz(otherBits) : 31;

  return 1 << std::min(consec, trailingZeros);
}

LinearLayout LinearLayout::transposeIns(ArrayRef<StringAttr> newInDims) const {
  assertDimsEqualIgnoringOrder(newInDims, getInDimNames());

  BasesT newBases;
  for (const auto &inDim : newInDims) {
    newBases[inDim] = bases.find(inDim)->second;
  }
  return LinearLayout(std::move(newBases), outDimNames.getArrayRef(),
                      surjective);
}

LinearLayout
LinearLayout::transposeOuts(ArrayRef<StringAttr> newOutDims) const {
  assertDimsEqualIgnoringOrder(newOutDims, getOutDimNames());

  std::vector<int32_t> permutation;
  for (const auto &outDim : newOutDims) {
    permutation.push_back(getOutDimIndex(outDim));
  }

  BasesT newBases;
  for (const auto &[inDim, inDimBases] : bases) {
    auto &newInDimBases = newBases[inDim];
    for (const auto &basis : inDimBases) {
      std::vector<int32_t> newBasis;
      for (int32_t i : permutation) {
        newBasis.push_back(basis[i]);
      }
      newInDimBases.push_back(std::move(newBasis));
    }
  }
  return LinearLayout(std::move(newBases), newOutDims, surjective);
}

LinearLayout LinearLayout::reshapeIns(
    ArrayRef<std::pair<StringAttr, int32_t>> newInDims) const {
  assert(llvm::all_of(newInDims, [&](auto &inDim) {
    return llvm::isPowerOf2_32(inDim.second);
  }));
  assert(getTotalInDimSize() == std::accumulate(newInDims.begin(),
                                                newInDims.end(), 1,
                                                [&](int32_t acc, auto &inDim) {
                                                  return acc * inDim.second;
                                                }));

  // First flatten into a single in-dimension.  Then split it up according to
  // `newInDims`.
  SmallVector<std::vector<int32_t>> flatBases;
  for (const auto &[inDim, inDimBases] : bases) {
    for (const auto &basis : inDimBases) {
      flatBases.push_back(basis);
    }
  }

  BasesT newBases;
  int i = 0;
  for (const auto &[inDim, inDimSize] : newInDims) {
    auto &newInDimBases = newBases[inDim];
    for (int j = 0; j < llvm::Log2_32(inDimSize); j++) {
      newInDimBases.push_back(flatBases[i++]);
    }
  }
  return LinearLayout(std::move(newBases), outDimNames.getArrayRef(),
                      surjective);
}

LinearLayout LinearLayout::reshapeOuts(
    ArrayRef<std::pair<StringAttr, int32_t>> newOutDims) const {
  assert(llvm::all_of(newOutDims, [&](auto &outDim) {
    return llvm::isPowerOf2_32(outDim.second);
  }));
  assert(getTotalOutDimSize() ==
         std::accumulate(
             newOutDims.begin(), newOutDims.end(), 1,
             [&](int32_t acc, auto &outDim) { return acc * outDim.second; }));

  SmallVector<int32_t> shifts;
  shifts.push_back(0);
  for (StringAttr outDim : getOutDimNames()) {
    shifts.push_back(shifts.back() + getOutDimSizeLog2(outDim));
  }

  // Flatten into a single out-dimension.  Then split it up according to
  // `newOutDims`.
  llvm::MapVector<StringAttr, std::vector<int32_t>> flatBases;
  for (const auto &[inDim, inDimBases] : bases) {
    auto &flatInBases = flatBases[inDim];
    for (const auto &basis : inDimBases) {
      int b = 0;
      for (int i = 0; i < basis.size(); i++) {
        b += basis[i] << shifts[i];
      }
      flatInBases.push_back(b);
    }
  }

  BasesT newBases;
  for (const auto &[inDim, flatInBases] : flatBases) {
    std::vector<std::vector<int32_t>> &newInDimBases = newBases[inDim];
    for (int32_t b : flatInBases) {
      std::vector<int32_t> multiDimBasis;
      for (int32_t newSize : llvm::make_second_range(newOutDims)) {
        multiDimBasis.push_back(b % newSize);
        b /= newSize;
      }
      newInDimBases.push_back(std::move(multiDimBasis));
    }
  }

  return LinearLayout(std::move(newBases),
                      llvm::to_vector(llvm::make_first_range(newOutDims)),
                      surjective);
}

LinearLayout operator*(LinearLayout inner, LinearLayout outer) {
  // Check that elements common to both outerDimsRange and innerDimsRange appear
  // in the same relative order.
  auto checkCommonDims = [&](auto outerDimsRange, auto innerDimsRange) {
    llvm::DenseSet<StringAttr> outerDims(outerDimsRange.begin(),
                                         outerDimsRange.end());
    llvm::DenseSet<StringAttr> innerDims(innerDimsRange.begin(),
                                         innerDimsRange.end());

    std::vector<StringAttr> outerCommonDims;
    for (StringAttr dim : outerDimsRange) {
      if (innerDims.contains(dim)) {
        outerCommonDims.push_back(dim);
      }
    }

    std::vector<StringAttr> innerCommonDims;
    for (StringAttr dim : innerDimsRange) {
      if (outerDims.contains(dim)) {
        innerCommonDims.push_back(dim);
      }
    }

    if (outerCommonDims != innerCommonDims) {
      llvm::report_fatal_error(
          "Cannot multiply layouts.  All in/out dimensions common to both "
          "layouts must appear in the same relative order, but they "
          "don't.\nOuter:\n" +
          Twine(outer.toString()) + "\nInner:\n" + inner.toString());
    }
  };

  // Check that dims common to outer and inner have the same relative order.
  checkCommonDims(outer.getInDimNames(), inner.getInDimNames());
  checkCommonDims(outer.getOutDimNames(), inner.getOutDimNames());

  // Get the sizeLog2 of all input and output dimensions we're going to
  // consider, in order.  `inner` is more minor, so its dimensions come first.
  llvm::MapVector<StringAttr, int32_t> inDimSizes;
  llvm::SetVector<StringAttr> outDimNames;
  for (const auto &layout : {inner, outer}) {
    for (StringAttr inDim : layout.getInDimNames()) {
      inDimSizes[inDim] += layout.getInDimSizeLog2(inDim);
    }
    for (StringAttr outDim : layout.getOutDimNames()) {
      outDimNames.insert(outDim);
    }
  }
  BasesT allBases;
  for (auto [inDimName, inDimSize] : inDimSizes) {
    std::vector<std::vector<int32_t>> &inDimBases = allBases[inDimName];

    // Fill with zeros.
    inDimBases = std::vector<std::vector<int32_t>>(
        inDimSize, std::vector<int32_t>(outDimNames.size(), 0));

    for (auto [outDimIdx, outDimName] : llvm::enumerate(outDimNames)) {
      if (inner.hasInDim(inDimName) && inner.hasOutDim(outDimName)) {
        for (int i = 0; i < inner.getInDimSizeLog2(inDimName); i++) {
          inDimBases[i][outDimIdx] = inner.getBasis(inDimName, i, outDimName);
        }
      }
      if (outer.hasInDim(inDimName) && outer.hasOutDim(outDimName)) {
        int offset =
            inner.hasInDim(inDimName) ? inner.getInDimSizeLog2(inDimName) : 0;
        int shift = inner.hasOutDim(outDimName)
                        ? inner.getOutDimSizeLog2(outDimName)
                        : 0;
        for (int i = 0; i < outer.getInDimSizeLog2(inDimName); i++) {
          inDimBases[offset + i][outDimIdx] =
              outer.getBasis(inDimName, i, outDimName) << shift;
        }
      }
    }
  }

  return LinearLayout(std::move(allBases), outDimNames.getArrayRef(),
                      inner.isSurjective() && outer.isSurjective());
}

SmallVector<std::pair<StringAttr, int32_t>>
LinearLayout::apply(ArrayRef<std::pair<StringAttr, int32_t>> ins) const {
  assertDimsEqualIgnoringOrder(llvm::make_first_range(ins), getInDimNames());

  SmallVector<std::pair<StringAttr, int32_t>> ret;
  for (StringAttr outDim : getOutDimNames()) {
    int32_t outVal = 0;
    for (auto &[inDim, val] : ins) {
      for (int i = 0; i < getInDimSizeLog2(inDim); i++) {
        if (val & (1 << i))
          outVal ^= getBasis(inDim, i, outDim);
      }
    }
    ret.push_back({outDim, outVal});
  }
  return ret;
}

LinearLayout LinearLayout::compose(const LinearLayout &outer) const {
  assertDimsEqualIgnoringOrder(getOutDimNames(), outer.getInDimNames());
  for (StringAttr outDim : getOutDimNames()) {
    assert(getOutDimSize(outDim) <= outer.getInDimSize(outDim));
  }

  BasesT newBases;
  for (const auto &[inDim, inDimBases] : bases) {
    auto &newInDimBases = newBases[inDim];
    for (const auto &basis : inDimBases) {
      SmallVector<std::pair<StringAttr, int32_t>> bases;
      for (auto [outDim, b] : llvm::zip(getOutDimNames(), basis)) {
        bases.push_back({outDim, b});
      }
      auto newBases = outer.apply(bases);
      auto newBasesRange = llvm::make_second_range(newBases);
      newInDimBases.push_back(
          std::vector<int32_t>(newBasesRange.begin(), newBasesRange.end()));
    }
  }

  bool compositionIsSurjective =
      isSurjective() && outer.isSurjective() &&
      llvm::all_of(getOutDimNames(), [&](StringAttr outDim) {
        return getOutDimSize(outDim) == outer.getInDimSize(outDim);
      });
  return LinearLayout(std::move(newBases), outer.getOutDimNames(),
                      compositionIsSurjective);
}

LinearLayout LinearLayout::invertAndCompose(const LinearLayout &outer) const {
  assertDimsEqualIgnoringOrder(getOutDimNames(), outer.getOutDimNames());
  for (StringAttr outDim : getOutDimNames()) {
    assert(getOutDimSize(outDim) <= outer.getOutDimSize(outDim));
  }
  assert(outer.isSurjective());

  int numRowsThis = this->getTotalOutDimSizeLog2();
  int numRowsOuter = outer.getTotalOutDimSizeLog2();
  int numColsThis = this->getTotalInDimSizeLog2();
  int numColsOuter = outer.getTotalInDimSizeLog2();

  std::unique_ptr<uint64_t[]> matThis = getMatrix(*this);

  // Increase the number of rows in matOuter's storage, because we may add some
  // rows to it.
  std::unique_ptr<uint64_t[]> matOuter = [&] {
    std::unique_ptr<uint64_t[]> mat =
        getMatrix(outer.transposeOuts(this->getOutDimNames()));
    std::unique_ptr<uint64_t[]> expanded = std::unique_ptr<uint64_t[]>(
        new uint64_t[numRowsOuter + numColsOuter]());
    std::memcpy(expanded.get(), mat.get(), numRowsOuter * sizeof(uint64_t));
    return expanded;
  }();

  // Check if `o` is injective.  Because it's surjective, it's sufficient to
  // check whether any columns of `o` are 0.  If it's not injective, we'll add
  // rows to `o` until it is.
  uint64_t colBits = 0;
  for (int c = 0; c < numRowsOuter; c++) {
    colBits |= matOuter[c];
  }
  bool outerWasInjective = colBits == (1 << numColsOuter) - 1;
  for (int c = 0; c < numColsOuter; c++) {
    if ((colBits & (1 << c)) == 0) {
      matOuter[numRowsOuter++] = (1 << c);
    }
  }

  // Concatenate `matOuter` and `matThis` horizontally (i.e. `matThis`
  // is to the right of `matOuter`).
  int combinedNumRows = std::max(numRowsThis, numRowsOuter);
  int combinedNumCols = numColsThis + numColsOuter;
  assert(combinedNumCols <= 64 && "Can't handle huge layouts");

  std::unique_ptr<uint64_t[]> m(new uint64_t[combinedNumRows]());
  for (int r = 0; r < numRowsOuter; r++) {
    m[r] = matOuter[r];
  }
  for (int r = 0; r < numRowsThis; r++) {
    m[r] |= matThis[r] << numColsOuter;
  }

  // Perform Gaussian elimination on `m`.  Because `outer` was modified to be
  // bijective, the first half of the matrix should be the identity matrix.  The
  // remaining half are the bases for the combined transformation.
  //
  // `stride` is specified in number of 64-bit words per row, and we pack our
  // matrix so that there's only one uint64_t per row.
  f2reduce::inplace_rref_strided(m.get(), combinedNumRows, combinedNumCols,
                                 /*stride=*/1);

  // Check that the first half of the matrix is indeed the identity.
  for (int r = 0; r < std::min(numRowsOuter, numColsOuter); r++) {
    for (int c = 0; c < std::min(numColsOuter, numRowsOuter); c++) {
      if (((m[r] >> c) & 1) != (r == c ? 1 : 0)) {
        llvm::report_fatal_error("First half of the matrix was not the "
                                 "identity, bug in invertAndCompose");
      }
    }
  }

  // We need names for the in/out dim of the flattened layout we're going to
  // read off from `m`.  These could be anything, doesn't matter.
  StringAttr inDim1D = *getInDimNames().begin();
  StringAttr outDim1D = *getOutDimNames().begin();

  // Read off the new bases.  These are for a flattened 1D -> 1D transformation
  // from `this`'s in-dims to `outer`'s in-dims.
  BasesT newBases;
  auto &bs = newBases[inDim1D];
  for (int c = 0; c < numColsThis; c++) {
    int32_t basis = 0;
    for (int r = 0; r < numRowsOuter; r++) {
      basis |= (m[r] >> (numColsOuter + c) & 1) << r;
    }
    bs.push_back({basis});
  }

  LinearLayout flatComposed(std::move(newBases), outDim1D,
                            isSurjective() && outerWasInjective);

  SmallVector<std::pair<StringAttr, int32_t>> retInDims;
  SmallVector<std::pair<StringAttr, int32_t>> retOutDims;
  for (StringAttr dim : getInDimNames()) {
    retInDims.push_back({dim, getInDimSize(dim)});
  }

  // The out-dim sizes are not simply outer.getInDimSize().  If the resulting
  // layout is not surjective, it could be that the new layout is unable to map
  // the high-order bits of the last out-dimension.  For example, if `outer` has
  // in bases [1, 2, 0], then outer's in-dim size is 2^3=8, but the composition
  // will have out-dim size 2^2=4, because there's no input to `outer` that maps
  // to 8.
  int remainingOutDims = flatComposed.getTotalOutDimSize();
  for (StringAttr dim : outer.getInDimNames()) {
    int size = std::min(remainingOutDims, outer.getInDimSize(dim));
    retOutDims.push_back({dim, size});
    remainingOutDims /= size;
  }
  return flatComposed.reshapeIns(retInDims).reshapeOuts(retOutDims);
}

bool operator==(LinearLayout lhs, LinearLayout rhs) {
  // llvm::MapVector doesn't have an operator== :(.
  if (lhs.getOutDimNames() != rhs.getOutDimNames())
    return false;
  if (lhs.bases.size() != rhs.bases.size())
    return false;
  for (auto it1 = lhs.bases.begin(), it2 = rhs.bases.begin();
       it1 != lhs.bases.end(); ++it1, ++it2) {
    if (*it1 != *it2)
      return false;
  }
  return true;
}

std::string LinearLayout::toString() const {
  return stringifyBases(bases, getOutDimNames());
}

} // namespace mlir::triton
