#ifndef TRITON_IR_UTILITY_H_
#define TRITON_IR_UTILITY_H_

#include "triton/Dialect/Triton/IR/Dialect.h"
#include <algorithm>
#include <numeric>

namespace mlir {

template <typename T, typename U> SmallVector<T> convertType(ArrayRef<U> in) {
  SmallVector<T> out;
  for (const auto &i : in)
    out.push_back(T(i));
  return out;
}

template <typename T, typename VecU>
SmallVector<T> convertType(const VecU &in) {
  return convertType<T>(ArrayRef(in));
}

template <typename Int> Int product(llvm::ArrayRef<Int> arr) {
  return std::accumulate(arr.begin(), arr.end(), 1, std::multiplies{});
}
template <typename VecT> auto product(const VecT &vec) {
  return product(llvm::ArrayRef(vec));
}
template <typename Int> Int getNumElements(ArrayRef<Int> shape) {
  if (shape.empty()) {
    return 0;
  }
  return product(shape);
}

// TODO(jlebar): Rename to ceilOfRatio.
template <typename Int> Int ceil(Int m, Int n) { return (m + n - 1) / n; }

/// Get the highest power of 2 divisor of an integer.
template <typename T> T highestPowOf2Divisor(T n) {
  if (n == 0) {
    return (static_cast<T>(1) << (sizeof(T) * 8 - 2));
  }
  return (n & (~(n - 1)));
}

/// Get the next power of 2 for an integer (or the integer itself if it is a
/// power of 2).
template <typename T> T nextPowOf2(T n) {
  if (n == 0) {
    return 1;
  }
  n--;
  for (unsigned i = 1; i < sizeof(T) * 8; i <<= 1) {
    n |= n >> i;
  }
  return n + 1;
}

namespace triton {

// Many functions here have two overloads, fn(ArrayRef<T>) and fn(const VecT&).
// This is helpful because C++ won't both convert a vector to ArrayRef *and*
// infer the proper type T in one step.  So without the second overload, we
// would have to explicitly convert most arguments to ArrayRef at the callsite.

template <typename T, typename U>
SmallVector<T> applyPermutation(ArrayRef<T> vec, ArrayRef<U> permutation) {
  static_assert(std::is_integral_v<U>);
  assert(vec.size() == permutation.size());

  // Check that `permutation` is actually a permutation.
#ifndef NDEBUG
  SmallVector<U> sortedPerm(permutation);
  llvm::sort(sortedPerm);
  for (U i = 0; i < static_cast<U>(sortedPerm.size()); i++) {
    assert(sortedPerm[i] == i);
  }
#endif

  SmallVector<T> ret;
  ret.reserve(vec.size());
  for (const U &i : permutation) {
    ret.push_back(vec[i]);
  }
  return ret;
}

template <typename VecT, typename PermT>
auto applyPermutation(const VecT &vec, const PermT &permutation) {
  return applyPermutation(ArrayRef(vec), ArrayRef(permutation));
}

template <typename T>
[[nodiscard]] SmallVector<T> inversePermutation(ArrayRef<T> permutation) {
  // Check that `permutation` is actually a permutation.
#ifndef NDEBUG
  SmallVector<T> sortedPerm(permutation);
  llvm::sort(sortedPerm);
  for (int i = 0; i < sortedPerm.size(); ++i) {
    assert(sortedPerm[i] == i);
  }
#endif

  SmallVector<T> ret(permutation.size());
  for (int i = 0; i < permutation.size(); ++i) {
    ret[permutation[i]] = i;
  }
  return ret;
}

template <typename VecT>
[[nodiscard]] auto inversePermutation(const VecT &permutation) {
  return inversePermutation(ArrayRef(permutation));
}

template <typename T, typename U>
[[nodiscard]] SmallVector<T> gather(ArrayRef<T> elems, ArrayRef<U> indices) {
  SmallVector<T> ret;
  ret.reserve(indices.size());
  for (const U &i : indices) {
    ret.push_back(elems[i]);
  }
  return ret;
}

template <typename VecT, typename IdxT>
[[nodiscard]] auto gather(const VecT &elems, const IdxT &indices) {
  return gather(ArrayRef(elems), ArrayRef(indices));
}

// Is `vec` [0, 1, ..., n]?  Returns true on empty list.
template <typename T> bool isIota(ArrayRef<T> vec) {
  static_assert(std::is_integral_v<T>);
  for (T i = 0; i < vec.size(); ++i) {
    if (vec[i] != i) {
      return false;
    }
  }
  return true;
}

template <typename VecT> bool isIota(const VecT &vec) {
  return isIota(ArrayRef(vec));
}

// Is `vals` some permutation of the numbers 0..(vals.size()-1)?
template <typename T> bool isPermutationOfIota(ArrayRef<T> vals) {
  SmallVector<T> sorted(vals);
  llvm::sort(sorted);
  return isIota(sorted);
}

template <typename VecT> bool IsPermutationOfIota(const VecT &vec) {
  return isPermutationOfIota(ArrayRef(vec));
}

// Is `vec` [i, i+1, ..., i+n]?  Returns true on empty list.
template <typename T> bool isConsecutive(ArrayRef<T> vec) {
  static_assert(std::is_integral_v<T>);
  for (int i = 1; i < vec.size(); i++) {
    if (vec[i] != vec[i - 1] + 1) {
      return false;
    }
  }
  return true;
}

template <typename VecT> bool isConsecutive(const VecT &vec) {
  return isConsecutive(ArrayRef(vec));
}

// LLVM's STLExtras.h provides a bunch of functions that work over ranges, but
// it's missing min/max_element until
// https://github.com/llvm/llvm-project/commit/fab2bb8b makes it into Triton.
// TODO(jlebar): Remove this once we have the LLVM helpers.
template <typename R> auto min_element(R &&Range) {
  return std::min_element(llvm::adl_begin(Range), llvm::adl_end(Range));
}
template <typename R, typename Compare>
auto min_element(R &&Range, Compare &&C) {
  return std::min_element(llvm::adl_begin(Range), llvm::adl_end(Range),
                          std::forward<Compare>(C));
}
template <typename R> auto max_element(R &&Range) {
  return std::max_element(llvm::adl_begin(Range), llvm::adl_end(Range));
}
template <typename R, typename T, typename Compare>
auto max_element(R &&Range, Compare &&C) {
  return std::max_element(llvm::adl_begin(Range), llvm::adl_end(Range),
                          std::forward<Compare>(C));
}

} // namespace triton
} // namespace mlir

#endif
