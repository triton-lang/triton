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
    ArrayRef<std::pair<StringAttr,
                       ArrayRef<std::pair<StringAttr, std::vector<int32_t>>>>>
        bases) {
  BasesT ret;
  for (const auto &[inDim, out] : bases) {
    for (const auto &[outDim, basis] : out) {
      ret[inDim][outDim] = basis;
    }
  }
  return ret;
}

std::string stringifyBases(const BasesT &bases) {
  std::string ret;
  for (const auto &[inDim, outs] : bases) {
    ret += " - " + inDim.str() + "\n";
    for (const auto &[outDim, bs] : outs) {
      ret += "   " + outDim.str() + ": [" + join(bs) + "]\n";
    }
  }
  return ret;
}

BasesT validateBases(BasesT bases) {
  if (bases.empty())
    return bases;

  for (const auto &[inDim, outs] : bases) {
    for (const auto &[outDim, bs] : outs) {
      if (llvm::any_of(bs, [](int32_t b) { return b < 0; })) {
        llvm::report_fatal_error(
            "Invalid bases passed to LinearLayout.  Expected all basis "
            "values to be non-negative, but found a negative value for "
            "in dimension '" +
            Twine(inDim) + "', out dimension '" + Twine(outDim) +
            "'.  Full list of bases:\n" + stringifyBases(bases));
      }
    }
  }

  // Check that the bases for each `in` dim all have the same length.
  for (const auto &[inDim, outs] : bases) {
    int expectedSize = outs.front().second.size();
    for (const auto &[outDim, bs] : outs) {
      if (bs.size() != expectedSize) {
        llvm::report_fatal_error(
            "Invalid bases passed to LinearLayout.  For a given `in` "
            "dimension, expected all basis lists to have the same "
            "length.  But this failed for in dimension '" +
            Twine(inDim) + "'.  Full list of bases:\n" + stringifyBases(bases));
      }
    }
  }

  // Check that every `in` dim has the same set of `out` dims, in the same
  // order.
  auto expectedOuts =
      llvm::to_vector(llvm::make_first_range(bases.front().second));
  for (const auto &[inDim, outs] : bases) {
    if (llvm::to_vector(llvm::make_first_range(outs)) != expectedOuts) {
      llvm::report_fatal_error(
          "Invalid bases passed to LinearLayout.  Expected all bases to have "
          "the same out dimensions, in the same order.  But this failed for in "
          "dimension '" +
          Twine(inDim) + "'.  Full list of bases:\n" + stringifyBases(bases));
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
      uint64_t basis = layout.getBasis(inDim, outDim, i);
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
          Twine(stringifyBases(layout.getBases())));
    }
  }
}

} // anonymous namespace

LinearLayout::LinearLayout(BasesT bases)
    : bases(validateBases(std::move(bases))) {
  validateSurjectivity(*this);
}

LinearLayout::LinearLayout(
    ArrayRef<std::pair<StringAttr,
                       ArrayRef<std::pair<StringAttr, std::vector<int32_t>>>>>
        bases)
    : LinearLayout(makeBasesMap(bases)) {}

/*static*/ LinearLayout LinearLayout::identity1D(int32_t size,
                                                 StringAttr inDimName,
                                                 StringAttr outDimName) {
  if (size == 0)
    return LinearLayout::empty();

  assert(llvm::isPowerOf2_32(size));
  std::vector<int32_t> powersOf2;
  for (int32_t i = 1; i < size; i *= 2) {
    powersOf2.push_back(i);
  }
  return LinearLayout(
      {{inDimName, std::pair(outDimName, std::move(powersOf2))}});
}

/*static*/ LinearLayout LinearLayout::zeros1D(int32_t size,
                                              StringAttr inDimName,
                                              StringAttr outDimName) {
  if (size == 0)
    return LinearLayout::empty();

  assert(llvm::isPowerOf2_32(size));
  return LinearLayout(
      {{inDimName,
        std::pair(outDimName, std::vector<int32_t>(llvm::Log2_32(size)))}});
}

bool LinearLayout::hasInDim(StringAttr inDim) const {
  return bases.find(inDim) != bases.end();
}

bool LinearLayout::hasOutDim(StringAttr outDim) const {
  if (bases.empty())
    return false;
  return bases.front().second.find(outDim) != bases.front().second.end();
}

int32_t LinearLayout::getInDimSizeLog2(StringAttr inDim) const {
  auto it = bases.find(inDim);
  assert(it != bases.end());
  return it->second.front().second.size();
}

int32_t LinearLayout::getOutDimSizeLog2(StringAttr outDim) const {
  // TODO(jlebar): Cache this?
  int32_t max = 0;
  for (const auto &[inDim, outs] : bases) {
    auto it = outs.find(outDim);
    assert(it != outs.end());
    const std::vector<int32_t> &bs = it->second;
    int32_t sizeLog2 =
        bs.empty() ? 0 : llvm::Log2_32(*llvm::max_element(bs)) + 1;
    max = std::max(max, sizeLog2);
  }
  return max;
}

LinearLayout LinearLayout::transposeIns(ArrayRef<StringAttr> newInDims) const {
  // newInDims must contain the same values as getInDims(), ignoring order.
  DenseSet<StringAttr> newInDimsSet(newInDims.begin(), newInDims.end());
  DenseSet<StringAttr> oldInDimsSet(getInDimNames().begin(),
                                    getInDimNames().end());
  assert(newInDimsSet == oldInDimsSet);

  BasesT newBases;
  for (const auto &inDim : newInDims) {
    for (const auto &outDim : getOutDimNames()) {
      newBases[inDim][outDim] = bases.find(inDim)->second.lookup(outDim);
    }
  }
  return LinearLayout(std::move(newBases));
}

LinearLayout
LinearLayout::transposeOuts(ArrayRef<StringAttr> newOutDims) const {
  // newOutDims must contain the same values as getOutDimNames(), ignoring
  // order.
  DenseSet<StringAttr> newOutDimsSet(newOutDims.begin(), newOutDims.end());
  DenseSet<StringAttr> oldOutDimsSet(getOutDimNames().begin(),
                                     getOutDimNames().end());
  assert(newOutDimsSet == oldOutDimsSet);

  BasesT newBases;
  for (const auto &[inDim, outs] : bases) {
    for (const auto &outDim : newOutDims) {
      newBases[inDim][outDim] = outs.lookup(outDim);
    }
  }
  return LinearLayout(std::move(newBases));
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

  // Get the sizeLog2 of all inner and outer dimensions we're going to consider,
  // in order.  `inner` is more minor, so its dimensions come first.
  llvm::MapVector<StringAttr, int32_t> inDimSizes;
  llvm::MapVector<StringAttr, int32_t> outDimSizes;
  for (const auto &layout : {inner, outer}) {
    for (StringAttr inDim : layout.getInDimNames()) {
      inDimSizes[inDim] += layout.getInDimSizeLog2(inDim);
    }
    for (StringAttr outDim : layout.getOutDimNames()) {
      outDimSizes[outDim] += layout.getOutDimSizeLog2(outDim);
    }
  }

  BasesT allBases;
  for (auto [inDimName, inDimSize] : inDimSizes) {
    for (auto [outDimName, outDimSize] : outDimSizes) {
      std::vector<int32_t> bases(inDimSize);
      if (inner.hasInDim(inDimName) && inner.hasOutDim(outDimName)) {
        for (int i = 0; i < inner.getInDimSizeLog2(inDimName); i++) {
          bases[i] = inner.getBasis(inDimName, outDimName, i);
        }
      }
      if (outer.hasInDim(inDimName) && outer.hasOutDim(outDimName)) {
        int offset =
            inner.hasInDim(inDimName) ? inner.getInDimSizeLog2(inDimName) : 0;
        int shift = inner.hasOutDim(outDimName)
                        ? inner.getOutDimSizeLog2(outDimName)
                        : 0;
        for (int i = 0; i < outer.getInDimSizeLog2(inDimName); i++) {
          bases[offset + i] = outer.getBasis(inDimName, outDimName, i) << shift;
        }
      }
      allBases[inDimName][outDimName] = std::move(bases);
    }
  }

  return LinearLayout(std::move(allBases));
}

SmallVector<std::pair<StringAttr, int32_t>>
LinearLayout::apply(ArrayRef<std::pair<StringAttr, int32_t>> ins) const {
  // `ins` must contain the same dimensions as getInDimNames(), modulo
  // reordering.
  auto insNames = llvm::make_first_range(ins);
  llvm::DenseSet<StringAttr> argInDims(insNames.begin(), insNames.end());
  llvm::DenseSet<StringAttr> thisInDims(getInDimNames().begin(),
                                        getInDimNames().end());
  if (argInDims != thisInDims) {
    llvm::report_fatal_error(
        Twine("Cannot apply layout.  The given input dimensions must match the "
              "layout's input dimensions, modulo reordering, but they don't.") +
        "\nGiven dims:" + triton::join(insNames, ", ") +
        "\nLayout's dims:" + triton::join(getInDimNames(), ", "));
  }

  SmallVector<std::pair<StringAttr, int32_t>> ret;
  for (StringAttr outDim : getOutDimNames()) {
    int32_t outVal = 0;
    for (auto &[inDim, val] : ins) {
      for (int i = 0; i < getInDimSizeLog2(inDim); i++) {
        if (val & (1 << i))
          outVal ^= getBasis(inDim, outDim, i);
      }
    }
    ret.push_back({outDim, outVal});
  }
  return ret;
}

bool operator==(LinearLayout lhs, LinearLayout rhs) {
  // llvm::MapVector doesn't have an operator== :(.
  if (lhs.bases.size() != rhs.bases.size())
    return false;
  for (auto it1 = lhs.bases.begin(), it2 = rhs.bases.begin();
       it1 != lhs.bases.end(); ++it1, ++it2) {
    if (it1->first != it2->first)
      return false;
    for (auto jt1 = it1->second.begin(), jt2 = it2->second.begin();
         jt1 != it1->second.end(); ++jt1, ++jt2) {
      if (jt1->first != jt2->first || jt1->second != jt2->second)
        return false;
    }
  }
  return true;
}

std::string LinearLayout::toString() const {
  // We could use stringifyBases here, but now that we know the layout is valid,
  // we can create something nicer.
  if (bases.empty())
    return "(empty layout)\n";

  // TODO: Add spaces for alignment.
  std::string ret;
  for (const auto &inDim : getInDimNames()) {
    if (getInDimSizeLog2(inDim) == 0) {
      ret += " - " + inDim.str() + " is a size 1 dimension\n";
      continue;
    }

    ret +=
        " - " +
        join(llvm::seq(getInDimSizeLog2(inDim)), "\n   ",
             [&](int i) {
               return inDim.str() + "=" + std::to_string(1 << i) + " -> (" +
                      join(getOutDimNames(), ", ",
                           [&](StringAttr outDim) {
                             return std::to_string(getBasis(inDim, outDim, i));
                           }) +
                      ")";
             }) +
        "\n";
  }
  ret += "where out dims are: [" +
         join(getOutDimNames(), ", ", [](StringAttr s) { return s.str(); }) +
         "]\n";
  return ret;
}

} // namespace mlir::triton
