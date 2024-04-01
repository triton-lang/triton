#include <iostream>
#include <map>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <type_traits>

namespace triton {

namespace interpreter {

enum class MemSemantic { ACQUIRE_RELEASE, ACQUIRE, RELEASE, RELAXED };

enum class RMWOp { ADD, FADD, AND, OR, XOR, XCHG, MAX, MIN, UMIN, UMAX };

std::map<MemSemantic, int> mem_semantic_map = {
    {MemSemantic::ACQUIRE_RELEASE, __ATOMIC_ACQ_REL},
    {MemSemantic::ACQUIRE, __ATOMIC_ACQUIRE},
    {MemSemantic::RELEASE, __ATOMIC_RELEASE},
    {MemSemantic::RELAXED, __ATOMIC_RELAXED},
};

// Use compiler builtin atomics instead of std::atomic which requires
// each variable to be declared as atomic.
// Currently work for clang and gcc.
template <bool is_min, typename T> T atomic_cmp(T *ptr, T val, int order) {
  auto cmp = [](T old, T val) {
    if constexpr (is_min) {
      return old > val;
    } else {
      return old < val;
    }
  };
  // First load
  T old_val = __atomic_load_n(ptr, order);
  while (cmp(old_val, val)) {
    if (__atomic_compare_exchange(ptr, &old_val, &val, false, order, order)) {
      break;
    }
  }
  return old_val;
}

template <typename T> T atomic_fadd(T *ptr, T val, int order) {
  T old_val;
  T new_val;
  // First load
  // Load ptr as if uint32_t or uint64_t and then memcpy to T
  if constexpr (sizeof(T) == 4) {
    uint32_t tmp = __atomic_load_n(reinterpret_cast<uint32_t *>(ptr), order);
    std::memcpy(&old_val, &tmp, sizeof(T));
  } else if constexpr (sizeof(T) == 8) {
    uint64_t tmp = __atomic_load_n(reinterpret_cast<uint64_t *>(ptr), order);
    std::memcpy(&old_val, &tmp, sizeof(T));
  } else {
    throw std::invalid_argument("Unsupported data type");
  }
  while (true) {
    new_val = old_val + val;
    if (__atomic_compare_exchange(ptr, &old_val, &new_val, false, order,
                                  order)) {
      break;
    }
  }
  return old_val;
}

class AtomicOp {
public:
  AtomicOp(const uint64_t *ptr, size_t numel, int order)
      : ptr(ptr), numel(numel), order(order) {}

  void apply() {
    for (size_t i = 0; i < numel; ++i) {
      applyAt(reinterpret_cast<void *>(ptr[i]), i);
    }
  }

protected:
  virtual void applyAt(void *, size_t i) = 0;

  const uint64_t *ptr;
  size_t numel;
  int order;
};

class AtomicRMWOpBase : public AtomicOp {
public:
  AtomicRMWOpBase(const uint64_t *ptr, const void *val, void *ret,
                  const bool *mask, size_t numel, int order)
      : AtomicOp(ptr, numel, order), val(val), ret(ret), mask(mask) {}

protected:
  void applyAt(void *loc, size_t i) override final {
    if (mask[i]) {
      applyAtMasked(loc, i);
    }
  }

  virtual void applyAtMasked(void *loc, size_t i) = 0;

  const void *val;
  void *ret;
  const bool *mask;
};

template <typename DType, RMWOp Op, typename = void>
class AtomicRMWOp : public AtomicRMWOpBase {
public:
  using AtomicRMWOpBase::AtomicRMWOpBase;
};

template <typename DType, RMWOp Op>
class AtomicRMWOp<DType, Op, std::enable_if_t<Op == RMWOp::ADD>>
    : public AtomicRMWOpBase {
public:
  using AtomicRMWOpBase::AtomicRMWOpBase;

protected:
  void applyAtMasked(void *loc, size_t i) override {
    *(static_cast<DType *>(ret) + i) =
        __atomic_fetch_add(static_cast<DType *>(loc),
                           *(static_cast<const DType *>(val) + i), order);
  }
};

template <typename DType, RMWOp Op>
class AtomicRMWOp<DType, Op, std::enable_if_t<Op == RMWOp::FADD>>
    : public AtomicRMWOpBase {
public:
  using AtomicRMWOpBase::AtomicRMWOpBase;

protected:
  void applyAtMasked(void *loc, size_t i) override {
    *(static_cast<DType *>(ret) + i) =
        atomic_fadd(static_cast<DType *>(loc),
                    *(static_cast<const DType *>(val) + i), order);
  }
};

template <typename DType, RMWOp Op>
class AtomicRMWOp<DType, Op, std::enable_if_t<Op == RMWOp::AND>>
    : public AtomicRMWOpBase {
public:
  using AtomicRMWOpBase::AtomicRMWOpBase;

protected:
  void applyAtMasked(void *loc, size_t i) override {
    *(static_cast<DType *>(ret) + i) =
        __atomic_fetch_and(static_cast<DType *>(loc),
                           *(static_cast<const DType *>(val) + i), order);
  }
};

template <typename DType, RMWOp Op>
class AtomicRMWOp<DType, Op, std::enable_if_t<Op == RMWOp::OR>>
    : public AtomicRMWOpBase {
public:
  using AtomicRMWOpBase::AtomicRMWOpBase;

protected:
  void applyAtMasked(void *loc, size_t i) override {
    *(static_cast<DType *>(ret) + i) =
        __atomic_fetch_or(static_cast<DType *>(loc),
                          *(static_cast<const DType *>(val) + i), order);
  }
};

template <typename DType, RMWOp Op>
class AtomicRMWOp<DType, Op, std::enable_if_t<Op == RMWOp::XOR>>
    : public AtomicRMWOpBase {
public:
  using AtomicRMWOpBase::AtomicRMWOpBase;

protected:
  void applyAtMasked(void *loc, size_t i) override {
    *(static_cast<DType *>(ret) + i) =
        __atomic_fetch_xor(static_cast<DType *>(loc),
                           *(static_cast<const DType *>(val) + i), order);
  }
};

template <typename DType, RMWOp Op>
class AtomicRMWOp<DType, Op,
                  std::enable_if_t<Op == RMWOp::MAX || Op == RMWOp::UMAX>>
    : public AtomicRMWOpBase {
public:
  using AtomicRMWOpBase::AtomicRMWOpBase;

protected:
  void applyAtMasked(void *loc, size_t i) override {
    *(static_cast<DType *>(ret) + i) = atomic_cmp</*is_min=*/false>(
        static_cast<DType *>(loc), *(static_cast<const DType *>(val) + i),
        order);
  }
};

template <typename DType, RMWOp Op>
class AtomicRMWOp<DType, Op,
                  std::enable_if_t<Op == RMWOp::MIN || Op == RMWOp::UMIN>>
    : public AtomicRMWOpBase {
public:
  using AtomicRMWOpBase::AtomicRMWOpBase;

protected:
  void applyAtMasked(void *loc, size_t i) override {
    *(static_cast<DType *>(ret) + i) = atomic_cmp</*is_min=*/true>(
        static_cast<DType *>(loc), *(static_cast<const DType *>(val) + i),
        order);
  }
};

template <typename DType, RMWOp Op>
class AtomicRMWOp<DType, Op, std::enable_if_t<Op == RMWOp::XCHG>>
    : public AtomicRMWOpBase {
public:
  using AtomicRMWOpBase::AtomicRMWOpBase;

protected:
  void applyAtMasked(void *loc, size_t i) override {
    *(static_cast<DType *>(ret) + i) =
        __atomic_exchange_n(static_cast<DType *>(loc),
                            *(static_cast<const DType *>(val) + i), order);
  }
};

class AtomicCASOp : public AtomicOp {
public:
  AtomicCASOp(const uint64_t *ptr, void *expected, const void *desired,
              size_t itemsize, size_t numel, int order)
      : AtomicOp(ptr, numel, order), expected(expected), desired(desired),
        itemsize(itemsize) {}

protected:
  void applyAt(void *loc, size_t i) override {
    // Atomic operations perform bitwise comparison, so it's safe to
    // use number of bytes (itemsize) to determine the type of pointers
    if (itemsize == 1) {
      __atomic_compare_exchange(
          static_cast<uint8_t *>(loc), static_cast<uint8_t *>(expected) + i,
          static_cast<const uint8_t *>(desired) + i, false, order, order);
    } else if (itemsize == 2) {
      __atomic_compare_exchange(
          static_cast<uint16_t *>(loc), static_cast<uint16_t *>(expected) + i,
          static_cast<const uint16_t *>(desired) + i, false, order, order);
    } else if (itemsize == 4) {
      __atomic_compare_exchange(
          static_cast<uint32_t *>(loc), static_cast<uint32_t *>(expected) + i,
          static_cast<const uint32_t *>(desired) + i, false, order, order);
    } else if (itemsize == 8) {
      __atomic_compare_exchange(
          static_cast<uint64_t *>(loc), static_cast<uint64_t *>(expected) + i,
          static_cast<const uint64_t *>(desired) + i, false, order, order);
    } else {
      // The ‘__atomic’ builtins can be used with any integral scalar or pointer
      // type that is 1, 2, 4, or 8 bytes in length. 16-byte integral types are
      // also allowed if ‘__int128’ (see 128-bit Integers) is supported by the
      // architecture.
      // https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html
      throw std::invalid_argument("Invalid byte size");
    }
  }

private:
  void *expected;
  const void *desired;
  size_t itemsize;
};

template <RMWOp Op, typename... SupportedDTypes>
std::unique_ptr<AtomicOp>
makeAtomicRMWOp(pybind11::dtype dtype, const uint64_t *ptr, const void *val,
                void *ret, const bool *mask, size_t numel, int order) {
  // Iterate over all supported data types, make one that matches, and return
  std::unique_ptr<AtomicOp> atomic_op;
  auto try_make_op = [&]<typename T>() {
    if (dtype.is(pybind11::dtype::of<T>())) {
      atomic_op = std::make_unique<AtomicRMWOp<T, Op>>(ptr, val, ret, mask,
                                                       numel, order);
    }
  };

  (try_make_op.template operator()<SupportedDTypes>(), ...);
  if (!atomic_op) {
    throw std::invalid_argument("Unsupported data type");
  }
  // Make it a unique_ptr
  return atomic_op;
}

} // namespace interpreter

} // namespace triton
