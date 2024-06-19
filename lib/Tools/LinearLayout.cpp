#include "triton/Tools/LinearLayout.h"

#include <cstdint>
#include <vector>

#include "mlir/IR/BuiltinAttributes.h"
#include "third_party/f2reduce/f2reduce.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "linear_layout"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

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

// Get a matrix for `layout` with its codomain expanded so it's injective, i.e.
// each input element maps to a unique output element.  We do this by finding
// columns that are equal to 0 and adding a new row with a 1 in that column.
std::tuple<std::unique_ptr<uint64_t[]>, int /*numRows*/, int /*numCols*/>
getInjectiveMat(const LinearLayout &layout) {
  int numRows = layout.getTotalOutDimSizeLog2();
  int numCols = layout.getTotalInDimSizeLog2();
  std::unique_ptr<uint64_t[]> mat = getMatrix(layout);

  // Bits of mat or-reduced along the columns (so there's just one row).
  uint64_t colBits = 0;
  for (int r = 0; r < numRows; r++) {
    colBits |= mat[r];
  }
  auto expanded = std::unique_ptr<uint64_t[]>(new uint64_t[numRows + numCols]);
  std::memcpy(expanded.get(), mat.get(), numRows * sizeof(uint64_t));
  for (int c = 0; c < numCols; c++) {
    if ((colBits & (1 << c)) == 0) {
      expanded[numRows++] = (1 << c);
    }
  }
  return std::make_tuple(std::move(expanded), numRows, numCols);
}

// Compute the rank of the matrix formed by taking the bases for the given
// outDim as columns.  In other words, finds the number of linearly-independent
// bases for this output dimension.
int getMatrixRank(std::unique_ptr<uint64_t[]> m, int numRows, int numCols) {
  // f2reduce underflows if the number of cols is 0, return the rank early in
  // this case.
  if (numCols == 0) {
    return 0;
  }
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

void eraseEmptyInOutDims(BasesT &bases,
                         llvm::MapVector<StringAttr, int32_t> &outDims) {
  // Erase empty out-dims.
  SmallVector<int> emptyOutDims;
  for (auto [i, outDim] : llvm::enumerate(
           llvm::to_vector_of<StringAttr>(llvm::make_first_range(outDims)))) {
    if (outDims[outDim] == 1) {
      emptyOutDims.push_back(i);
      outDims.erase(outDim);
    }
  }
  if (outDims.empty()) {
    bases.clear();
    return;
  }

  for (auto &[inDim, inDimBases] : bases) {
    for (auto &basis : inDimBases) {
      // Erase the basis elements corresponding to the empty out-dims.
      for (int i : llvm::reverse(emptyOutDims)) {
        basis.erase(basis.begin() + i);
      }
    }
  }

  // Erase empty in-dims.
  // TODO: This needs a test-case.
  for (StringAttr inDim :
       llvm::to_vector_of<StringAttr>(llvm::make_first_range(bases))) {
    if (bases[inDim].empty()) {
      bases.erase(inDim);
    }
  }
}

} // anonymous namespace

/*static*/ std::optional<LinearLayout>
LinearLayout::tryCreate(BasesT bases,
                        ArrayRef<std::pair<StringAttr, int32_t>> outDims,
                        bool requireSurjective) {
  LinearLayout ll(std::move(bases), std::move(outDims), NoCheckInvariants{});
  std::optional<std::string> error = ll.checkInvariants(requireSurjective);
  if (error) {
    return std::nullopt;
  }
  return ll;
}

LinearLayout::LinearLayout(BasesT bases,
                           ArrayRef<std::pair<StringAttr, int32_t>> outDims,
                           NoCheckInvariants)
    : bases(std::move(bases)) {
  for (auto [outDim, size] : outDims) {
    this->outDims[outDim] = size;
  }
}

LinearLayout::LinearLayout(BasesT bases, ArrayRef<StringAttr> outDimNames)
    : bases(std::move(bases)) {
  // Infer out-dim sizes.
  for (StringAttr outDim : outDimNames) {
    outDims[outDim] = 1;
  }
  for (const auto &[inDim, inDimBases] : this->bases) {
    for (const auto &basis : inDimBases) {
      for (int i = 0; i < basis.size(); i++) {
        int32_t &size = outDims[outDimNames[i]];
        size = std::max<int32_t>(size, llvm::NextPowerOf2(basis[i]));
      }
    }
  }

  std::optional<std::string> error =
      checkInvariants(/*requireSurjective=*/true);
  if (error.has_value()) {
    llvm::report_fatal_error(StringRef(*error));
  }
}

LinearLayout::LinearLayout(BasesT bases,
                           ArrayRef<std::pair<StringAttr, int32_t>> outDims,
                           bool requireSurjective)
    : LinearLayout(std::move(bases), std::move(outDims), NoCheckInvariants{}) {
  std::optional<std::string> error = checkInvariants(requireSurjective);
  if (error.has_value()) {
    llvm::report_fatal_error(StringRef(*error));
  }
}

std::optional<std::string>
LinearLayout::checkInvariants(bool requireSurjective) {
  LDBG("checkInvariants: " << toString());
  // Check that basis values are non-negative.
  for (const auto &[inDim, inDimBases] : bases) {
    for (const auto &basis : inDimBases) {
      if (llvm::any_of(basis, [](int32_t b) { return b < 0; })) {
        return "Invalid bases passed to LinearLayout.  Expected all basis "
               "values to be non-negative, but found a negative value for "
               "in dimension '" +
               inDim.str() + "'.  Full list of bases:" + toString() + "\n";
      }
    }
  }

  // Check that the bases all have length equal to outDimNames.size().
  for (const auto &[inDim, inDimBases] : bases) {
    for (const auto &basis : inDimBases) {
      if (basis.size() != outDims.size()) {
        return "Invalid bases passed to LinearLayout.  Expect all bases to "
               "have the same size, equal to outDimNames.size() (" +
               std::to_string(outDims.size()) +
               ").  But this failed for in dimension '" + inDim.str() +
               "'.  Full list of bases:" + toString() + "\n";
      }
    }
  }

  // Check that the out-dim sizes are powers of 2.
  for (const auto &[outDim, size] : outDims) {
    if (!llvm::isPowerOf2_32(size)) {
      return "Invalid out-dim size " + std::to_string(size) + " for out-dim '" +
             outDim.str() + "'.  Out-dim sizes must be powers of 2.\n";
    }
  }

  // Check that the bases are smaller than the out-dim sizes.
  SmallVector<StringAttr> outDimNames = llvm::to_vector(getOutDimNames());
  for (const auto &[inDim, inDimBases] : this->bases) {
    for (const auto &basis : inDimBases) {
      for (int i = 0; i < basis.size(); i++) {
        if (basis[i] >= outDims[outDimNames[i]]) {
          return "Invalid basis " + std::to_string(basis[i]) + " for in-dim '" +
                 inDim.str() + "' and out-dim '" + outDimNames[i].str() +
                 "'.  Basis must be less than the out-dim size.\n";
        }
      }
    }
  }

  // Determine whether the this layout is surjective, i.e. that every `out`
  // coordinate can be reached by some `in` coordinate.
  //
  // It's prohibitively slow to calculate this naively, but thankfully, this
  // is equivalent to checking that the number of linearly-independent bases
  // is equal to sum(getOutDimSizeLog2).  This can be computed by finding
  // the rank of the matrix whose columns are those bases.  We can compute
  // the rank of our matrix using Gaussian elimination, which runs in O(n^3)
  // for an n x n matrix.  Our matrix size is sum(inDimSizeLog2) x
  // sum(outDimSizeLog2), so this should be plenty fast.
  this->surjective =
      getMatrixRank(getMatrix(*this), /*numRows=*/getTotalOutDimSizeLog2(),
                    /*numCols=*/getTotalInDimSizeLog2()) ==
      getTotalOutDimSizeLog2();

  if (requireSurjective && !surjective) {
    return "Layout is expected to be surjective, i.e. every `out` coordinate "
           "can be reached by some `in` coordinate, but was not:" +
           toString();
  }
  return std::nullopt;
}

LinearLayout::LinearLayout(
    ArrayRef<std::pair<StringAttr, std::vector<std::vector<int32_t>>>> bases,
    ArrayRef<StringAttr> outDimNames)
    : LinearLayout(makeBasesMap(bases), outDimNames) {}

LinearLayout::LinearLayout(
    ArrayRef<std::pair<StringAttr, std::vector<std::vector<int32_t>>>> bases,
    ArrayRef<std::pair<StringAttr, int32_t>> outDims, bool requireSurjective)
    : LinearLayout(makeBasesMap(bases), outDims, requireSurjective) {}

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
  int i = 0;
  for (auto [name, _] : outDims) {
    if (name == outDim) {
      return i;
    }
    i++;
  }
  llvm::report_fatal_error("outDim " + Twine(outDim) + " is not in layout" +
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
  auto it = outDims.find(outDim);
  assert(it != outDims.end());
  return llvm::Log2_32(it->second);
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
  return LinearLayout(std::move(newBases), llvm::to_vector(outDims),
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

  SmallVector<std::pair<StringAttr, int32_t>> newOutDimSizes;
  for (auto outDim : newOutDims) {
    newOutDimSizes.push_back({outDim, getOutDimSize(outDim)});
  }
  return LinearLayout(std::move(newBases), newOutDimSizes, surjective);
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

  // First flatten into a single in-dimension.  Then split it up according
  // to `newInDims`.
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
  return LinearLayout(std::move(newBases), llvm::to_vector(outDims),
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

  return LinearLayout(std::move(newBases), newOutDims, surjective);
}

LinearLayout operator*(LinearLayout inner, LinearLayout outer) {
  // Check that elements common to both outerDimsRange and innerDimsRange
  // appear in the same relative order.
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
          "don't.\nOuter:" +
          Twine(outer.toString()) + "\nInner:" + inner.toString());
    }
  };

  // Check that dims common to outer and inner have the same relative order.
  checkCommonDims(outer.getInDimNames(), inner.getInDimNames());
  checkCommonDims(outer.getOutDimNames(), inner.getOutDimNames());

  // Get the sizeLog2 of all input and output dimensions we're going to
  // consider, in order.  `inner` is more minor, so its dimensions come
  // first.
  llvm::MapVector<StringAttr, int32_t> inDimSizesLog2;
  llvm::MapVector<StringAttr, int32_t> outDimSizesLog2;
  for (const auto &layout : {inner, outer}) {
    for (StringAttr inDim : layout.getInDimNames()) {
      inDimSizesLog2[inDim] += layout.getInDimSizeLog2(inDim);
    }
    for (StringAttr outDim : layout.getOutDimNames()) {
      outDimSizesLog2[outDim] += layout.getOutDimSizeLog2(outDim);
    }
  }

  BasesT allBases;
  for (auto [inDimName, inDimSizeLog2] : inDimSizesLog2) {
    std::vector<std::vector<int32_t>> &inDimBases = allBases[inDimName];

    // Fill with zeros.
    inDimBases = std::vector<std::vector<int32_t>>(
        inDimSizeLog2, std::vector<int32_t>(outDimSizesLog2.size(), 0));

    for (auto [outDimIdx, outDimNameAndSize] :
         llvm::enumerate(outDimSizesLog2)) {
      auto [outDimName, outDimSize] = outDimNameAndSize;
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

  llvm::SmallVector<std::pair<StringAttr, int32_t>> outDimSizes;
  for (auto [outDim, sizeLog2] : outDimSizesLog2) {
    outDimSizes.push_back({outDim, 1 << sizeLog2});
  }
  return LinearLayout(std::move(allBases), outDimSizes,
                      inner.isSurjective() && outer.isSurjective());
}

std::optional<LinearLayout>
LinearLayout::divideRight(const LinearLayout &divisor) {
  // Strip off the top N bases for each input dimension of divisor.  This
  // gives a candidate quotient.  Then check if quotient * divisor equals
  // `this`.
  BasesT newBases = bases;
  for (StringAttr inDim : divisor.getInDimNames()) {
    if (getInDimSizeLog2(inDim) < divisor.getInDimSizeLog2(inDim)) {
      return std::nullopt;
    }
    auto &newInDimBases = newBases[inDim];
    newInDimBases.resize(newInDimBases.size() -
                         divisor.getInDimSizeLog2(inDim));
  }

  llvm::MapVector<StringAttr, int32_t> newOutDims = outDims;
  for (const auto [outDim, outDimSize] : divisor.outDims) {
    if (newOutDims[outDim] < outDimSize) {
      return std::nullopt;
    }
    newOutDims[outDim] /= outDimSize;
  }

  eraseEmptyInOutDims(newBases, newOutDims);

  LDBG("this->divideRight(divisor)=candidate_quotient");
  LDBG("this:" << *this);
  LDBG("divisor:" << divisor);
  LDBG("newOutDims: " << triton::join(newOutDims, ", ", [](auto &p) {
         return p.first.str() + "=" + std::to_string(p.second);
       }));
  std::optional<LinearLayout> candidateQuotient = LinearLayout::tryCreate(
      std::move(newBases), std::move(newOutDims).takeVector(),
      /*requireSurjective=*/false);
  if (!candidateQuotient.has_value()) {
    LDBG("candidate quotient failed invariant checks");
    return std::nullopt;
  }
  LDBG("candidate_quotient:" << candidateQuotient);

  if (*candidateQuotient * divisor == *this) {
    return *candidateQuotient;
  }
  return std::nullopt;
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
  return LinearLayout(std::move(newBases), llvm::to_vector(outer.outDims),
                      compositionIsSurjective);
}

LinearLayout LinearLayout::invertAndCompose(const LinearLayout &outer) const {
  assertDimsEqualIgnoringOrder(getOutDimNames(), outer.getOutDimNames());
  for (StringAttr outDim : getOutDimNames()) {
    assert(getOutDimSize(outDim) <= outer.getOutDimSize(outDim));
  }
  assert(outer.isSurjective());

  // Make both `this` and `outer` injective.  We need to do this on the
  // `outer` layout because we can't invert a non-injective function.  We
  // choose to do so on the `this` layout as well.  The rest of the comment
  // explains why we make that choice.
  //
  // Recall from the header that C = A.invertAndCompose(B) just means that
  //  A(x) = B(C(x)).
  //
  // Sometimes we may have a choice of multiple values for a particular
  // C(x). For example, if A(1) = B(0) = B(1) = 0, then C(1) can be either 0
  // or 1.
  //
  // We want to choose C such that C(x) != 0 where possible.  For example,
  // suppose we are transfering from registers to registers and we have the
  // following layouts.
  //
  //   A(thread=1, block=0) = 1
  //   A(thread=2, block=0) = 2
  //   A(thread=0, block=1) = 0
  //
  //   B(thread=1, block=0) = 2
  //   B(thread=2, block=0) = 1
  //   B(thread=0, block=1) = 0
  //
  // Notice that A and B both have the same data in each of their two
  // blocks. So if we want to transfer from A to B, we don't need to cross
  // blocks, which is expensive.  We want A.invertAndCompose(B) to reflect
  // that choice.
  //
  // Let A' be A with the last line changed to "=4", and similarly for B'.
  // When transfering from A' to B', we can't cross blocks even if we wanted
  // to, because the two blocks now have different data.  But also, any
  // mapping of thread+block from A' to B' is also valid for mapping from A
  // to B.
  //
  // Thus making A and B injective encodes our desire not to cross blocks,
  // or more generally our desire that C(x) != 0 where possible.
  auto [matThis, numRowsThis, numColsThis] = getInjectiveMat(*this);
  auto [matOuter, numRowsOuter, numColsOuter] = getInjectiveMat(
      outer.transposeOuts(llvm::to_vector(this->getOutDimNames())));

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

  // Perform Gaussian elimination on `m`.  Because `outer` was modified to
  // be bijective, the first half of the matrix should be the identity
  // matrix.  The remaining half are the bases for the combined
  // transformation.
  //
  // `stride` is specified in number of 64-bit words per row, and we pack
  // our matrix so that there's only one uint64_t per row.
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

  // Read off the new bases.  These are for a flattened 1D -> 1D
  // transformation from `this`'s in-dims to `outer`'s in-dims.
  BasesT newBases;
  auto &bs = newBases[inDim1D];
  for (int c = 0; c < numColsThis; c++) {
    int32_t basis = 0;
    for (int r = 0; r < numRowsOuter; r++) {
      basis |= (m[r] >> (numColsOuter + c) & 1) << r;
    }
    bs.push_back({basis});
  }

  LinearLayout flatComposed(std::move(newBases),
                            {{outDim1D, outer.getTotalInDimSize()}},
                            /*requireSurjective=*/false);

  SmallVector<std::pair<StringAttr, int32_t>> retInDims;
  SmallVector<std::pair<StringAttr, int32_t>> retOutDims;
  for (StringAttr dim : getInDimNames()) {
    retInDims.push_back({dim, getInDimSize(dim)});
  }
  for (StringAttr dim : outer.getInDimNames()) {
    retOutDims.push_back({dim, outer.getInDimSize(dim)});
  }
  return flatComposed.reshapeIns(retInDims).reshapeOuts(retOutDims);
}

bool operator==(LinearLayout lhs, LinearLayout rhs) {
  // llvm::MapVector doesn't have an operator== :(.
  if (llvm::to_vector(lhs.getOutDimNames()) !=
      llvm::to_vector(rhs.getOutDimNames()))
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
  // Start with a newline because we print out a bulleted list; it doesn't
  // make sense for the first line of this list to be on the same line as
  // any previous text.
  std::string ret = "\n";
  std::string outDimsStr =
      "[" +
      join(outDims, ", ",
           [](auto dimAndSize) {
             auto [outDim, size] = dimAndSize;
             return outDim.str() + " (size " + std::to_string(size) + ")";
           }) +
      "]";

  if (bases.empty()) {
    if (outDims.empty()) {
      return "\n(empty layout)";
    } else {
      return "\n(empty layout with out-dims " + outDimsStr + ")";
    }
  }

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
  ret += "where out dims are: " + outDimsStr;
  return ret;
}

} // namespace mlir::triton
