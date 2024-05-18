#include "triton/Tools/LinearLayout.h"

#include <cstdint>
#include <vector>

#include "mlir/IR/BuiltinAttributes.h"
#include "third_party/f2reduce/f2reduce.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/ADT/STLExtras.h"
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

// Compute the rank of the matrix formed by taking the bases for the given
// outDim as columns.  In other words, finds the number of linearly-independent
// bases for this output dimension.
int getMatrixRank(const LinearLayout &layout, StringAttr outDim) {
  // Suppose we have a layout specified by the following key values.
  //
  //   L(0,1) = 0b01
  //   L(0,2) = 0b10
  //   L(1,0) = 0b10
  //   L(2,0) = 0b11
  //
  // We will create one column per key value.  The max bit width of these values
  // is 2, so our matrix will have 2 rows.  The final matrix will be
  //
  //  |    ↑      ↑      ↑      ↑   |   | 0b0111 |
  //  | L(0,1) L(0,2) L(1,0) L(2,0) | = | 0b1001 |
  //  |    ↓      ↓      ↓      ↓   |
  int numRows = layout.getOutDimSizeLog2(outDim);

  int numCols = 0;
  for (StringAttr inDim : layout.getInDimNames()) {
    numCols += layout.getInDimSizeLog2(inDim);
  }

  if (numCols == 0 || numRows == 0)
    return 0;

  // Don't handle giant LLs.  This makes some things easier; for example, each
  // row can be a single uint64_t.
  assert(numCols <= 64 && "LinearLayout too large");
  assert(numRows <= 64 && "LinearLayout too large");

  // Note that `new int[n]()` is zero-initialized, whereas `new int[n]` is not.
  std::unique_ptr<uint64_t[]> m(new uint64_t[numRows]());

  // Fill in the matrix.
  int c = 0;
  for (StringAttr inDim : layout.getInDimNames()) {
    for (int i = 0; i < layout.getInDimSizeLog2(inDim); i++) {
      uint64_t basis = layout.getBasis(inDim, i, outDim);
      for (int j = 0; j < numRows; j++) {
        m[j] |= ((basis >> j) & 1) << c;
      }
      c++;
    }
  }

  // stride is specified in number of 64-bit words per row.
  f2reduce::inplace_rref_strided(m.get(), numRows, numCols, /*stride=*/1);

  // The rank of the reduced matrix is simply the number of nonzero rows.
  int rank = 0;
  for (int i = 0; i < numRows; i++) {
    if (m[i] != 0)
      rank++;
  }
  return rank;
}

// Check that the given layout is surjective, i.e. that every `out` coordinate
// can be reached by some `in` coordinate.
//
// It's sufficient to check each output dimension indepedently.  Still,
// it's prohibitively slow to calculate this naively.
//
// Thankfully, this is equivalent to checking that the number of
// linearly-independent bases for outDim d is equal to getOutDimSizeLog2(d).
// This can be computed by finding the rank of the matrix whose columns are
// those bases.  We can compute the rank of our matrix using Gaussian
// elimination, which runs in O(n^3) for an n x n matrix.  Our matrix size is
// log(product(inDimSize)) x log(outDimSize), and we do this numOutDims times,
// so this should be plenty fast overall.
void validateSurjectivity(const LinearLayout &layout) {
  for (const auto &outDim : layout.getOutDimNames()) {
    unsigned rank = getMatrixRank(layout, outDim);
    unsigned expectedRank = layout.getOutDimSizeLog2(outDim);
    if (rank != expectedRank) {
      llvm::report_fatal_error(
          "Invalid bases passed to LinearLayout.  Expected bases to be "
          "surjective, i.e. all possible output coordinates can be reached "
          "by some input coordinates.  But this failed for output dimension " +
          Twine(outDim) + ", where we got rank " + Twine(rank) +
          " instead of expected rank " + Twine(expectedRank) +
          ".  Full list of bases:\n" +
          Twine(stringifyBases(layout.getBases(), layout.getOutDimNames())));
    }
  }
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

LinearLayout::LinearLayout(BasesT bases, ArrayRef<StringAttr> outDimNames)
    : bases(validateBases(std::move(bases), outDimNames)),
      outDimNames(outDimNames.begin(), outDimNames.end()) {
  validateSurjectivity(*this);
}

LinearLayout::LinearLayout(
    ArrayRef<std::pair<StringAttr, std::vector<std::vector<int32_t>>>> bases,
    ArrayRef<StringAttr> outDimNames)
    : LinearLayout(makeBasesMap(bases), outDimNames) {}

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

LinearLayout LinearLayout::transposeIns(ArrayRef<StringAttr> newInDims) const {
  assertDimsEqualIgnoringOrder(newInDims, getInDimNames());

  BasesT newBases;
  for (const auto &inDim : newInDims) {
    newBases[inDim] = bases.find(inDim)->second;
  }
  return LinearLayout(std::move(newBases), outDimNames.getArrayRef());
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
  return LinearLayout(std::move(newBases), newOutDims);
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

  return LinearLayout(std::move(allBases), outDimNames.getArrayRef());
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
  return LinearLayout(std::move(newBases), outer.getOutDimNames());
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
