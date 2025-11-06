#ifndef TRITON_IR_UTILITY_H_
#define TRITON_IR_UTILITY_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include <algorithm>
#include <cstdint>
#include <numeric>

namespace mlir {

// Bitwidth of pointers
constexpr int kPtrBitWidth = 64;

// Returns the bit width of a type, treating pointer-like types as 64-bit.
// This handles LLVM dialect pointer types.
inline int getIntOrFloatOrPtrBitWidth(Type type) {
  if (isa<LLVM::LLVMPointerType, triton::PointerType>(type))
    return kPtrBitWidth;
  return type.getIntOrFloatBitWidth();
}

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
  return std::accumulate(arr.begin(), arr.end(), 1, std::multiplies<Int>());
}
template <typename VecT> auto product(const VecT &vec) {
  return product(llvm::ArrayRef(vec));
}

// TODO(jlebar): Rename to ceilOfRatio.
template <typename Int> Int ceil(Int m, Int n) { return (m + n - 1) / n; }

/// Get the highest power of 2 divisor of an integer.
template <typename T> constexpr T highestPowOf2Divisor(T n) {
  // When n is 0 or min, return the highest power of 2. The min case is handled
  // separately to avoid underflow when T is a signed integer. Technically
  // in that case the correct divisor is -n, but this value is outside the
  // range of possible values, so we take the next best alternative.
  if (n == 0 || n == std::numeric_limits<T>::min()) {
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
  for (size_t i = 0; i < vec.size(); ++i) {
    if (vec[i] != static_cast<T>(i)) {
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

template <typename VecT> bool isPermutationOfIota(const VecT &vec) {
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

template <typename T> auto seq(T start, T end, T step) {
  auto len = ceil<T>(end - start, step);
  return llvm::map_range(llvm::seq<T>(0, len),
                         [=](T i) { return start + i * step; });
}

// Combine the current mask with the given predicate.
Value getPredMask(RewriterBase &rewriter, Type typeLike, Value currentMask,
                  Value pred);

// Get the value of the induction variable at the end of the loop.
Value getLastInductionValue(OpBuilder &b, scf::ForOp loop);

MakeTensorPtrOp getMakeTensorPtrOp(Value v);

bool isHostSideDescriptor(Value v);

bool isKernel(FunctionOpInterface funcOp);

unsigned getBitwidth(RankedTensorType ty);
} // namespace triton
} // namespace mlir

extern "C" {
enum TritonPluginResult {
  TP_SUCCESS = 0,
  TP_GENERIC_FAILURE = 1,
};
};
#define TRITON_PLUGIN_API                                                      \
  extern "C" __attribute__((visibility("default"))) TritonPluginResult

struct TritonPlugin {
  TritonPlugin() = delete;
  TritonPlugin(std::string filename) : filename(filename) {}

private:
  const std::string ENUMERATE_PASSES = "tritonEnumeratePluginPasses";
  using enumeratePassesType =
      std::function<TritonPluginResult(uint32_t *, const char **)>;
  using enumeratePassesCType = TritonPluginResult (*)(uint32_t *,
                                                      const char **);

  const std::string ADD_PASS = "tritonAddPluginPass";
  using addPassType =
      std::function<TritonPluginResult(mlir::PassManager *, const char *)>;
  using addPassCType = TritonPluginResult (*)(mlir::PassManager *,
                                              const char *);

  const std::string REGISTER_PASS = "tritonRegisterPluginPass";
  using registerPassType = std::function<TritonPluginResult(const char *)>;
  using registerPassCType = TritonPluginResult (*)(const char *);

  llvm::Error checkLibraryValid(const std::string &error) const {
    if (!library.isValid()) {
      auto msg = llvm::Twine("Failed to load plugin library: " + error + "\n");
      return llvm::createStringError(msg);
    }
    return llvm::Error::success();
  }

  llvm::Expected<intptr_t> getAddressOfSymbol(const std::string &symbol) const {
    if (auto isValid = checkLibraryValid("not loaded"))
      return isValid;
    intptr_t getDetailsFn =
        (intptr_t)library.getAddressOfSymbol(symbol.c_str());
    if (!getDetailsFn) {
      auto msg = llvm::Twine("Failed to get symbol: " + symbol + "\n");
      return llvm::createStringError(msg);
    }
    return getDetailsFn;
  }

  template <typename T, typename U>
  llvm::Expected<T> getAPI(const std::string &symbol) const {
    llvm::Expected<intptr_t> getDetailsFn = getAddressOfSymbol(symbol);
    if (auto Err = getDetailsFn.takeError()) {
      return Err;
    }
    auto func = reinterpret_cast<U>(*getDetailsFn);
    return func;
  }

public:
  llvm::Error loadPlugin() const {
    std::string error;
    library = llvm::sys::DynamicLibrary::getPermanentLibrary(filename.c_str(),
                                                             &error);
    return checkLibraryValid(error);
  }

  llvm::Expected<std::vector<const char *>> getPassHandles() const {
    auto apiOrErr =
        getAPI<enumeratePassesType, enumeratePassesCType>(ENUMERATE_PASSES);
    if (auto Err = apiOrErr.takeError())
      return Err;
    auto enumeratePluginPasses = *apiOrErr;

    std::vector<const char *> passNames;
    uint32_t passCount = 0;
    enumeratePluginPasses(&passCount, nullptr);
    if (passCount == 0)
      return passNames;

    passNames.resize(passCount);
    enumeratePluginPasses(&passCount, passNames.data());
    return passNames;
  }

  TritonPluginResult addPass(mlir::PassManager *pm,
                             const char *passHandle) const {
    auto apiOrErr = getAPI<addPassType, addPassCType>(ADD_PASS);
    if (auto Err = apiOrErr.takeError())
      return TP_GENERIC_FAILURE;
    auto addPass = *apiOrErr;
    return addPass(pm, passHandle);
  }

  TritonPluginResult registerPass(const char *passHandle) const {
    auto apiOrErr = getAPI<registerPassType, registerPassCType>(REGISTER_PASS);
    if (auto Err = apiOrErr.takeError())
      return TP_GENERIC_FAILURE;
    auto registerPass = *apiOrErr;
    return registerPass(passHandle);
  }

private:
  std::string filename = "";
  mutable llvm::sys::DynamicLibrary library;
};

#endif
