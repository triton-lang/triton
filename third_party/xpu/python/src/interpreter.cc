#include <iostream>
#include <map>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <type_traits>

namespace py = pybind11;

namespace {

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

  virtual ~AtomicOp() = default;

protected:
  virtual void applyAt(void *, size_t i) = 0;

  const uint64_t *ptr;
  size_t numel;
  int order;
};

template <typename DType> class AtomicRMWOpBase : public AtomicOp {
public:
  AtomicRMWOpBase(const uint64_t *ptr, const void *val, void *ret,
                  const bool *mask, size_t numel, int order)
      : AtomicOp(ptr, numel, order), val(val), ret(ret), mask(mask) {}

protected:
  void applyAt(void *loc, size_t i) override final {
    if (mask[i]) {
      *(static_cast<DType *>(ret) + i) =
          applyAtMasked(static_cast<DType *>(loc),
                        *(static_cast<const DType *>(val) + i), order);
    }
  }

  virtual DType applyAtMasked(DType *loc, const DType value, int order) = 0;

  const void *val;
  void *ret;
  const bool *mask;
};

template <typename DType, RMWOp Op, typename = void>
class AtomicRMWOp : public AtomicRMWOpBase<DType> {
public:
  using AtomicRMWOpBase<DType>::AtomicRMWOpBase;
};

template <typename DType, RMWOp Op>
class AtomicRMWOp<DType, Op, std::enable_if_t<Op == RMWOp::ADD>>
    : public AtomicRMWOpBase<DType> {
public:
  using AtomicRMWOpBase<DType>::AtomicRMWOpBase;

protected:
  DType applyAtMasked(DType *loc, const DType value, int order) override {
    return __atomic_fetch_add(loc, value, order);
  }
};

template <typename DType, RMWOp Op>
class AtomicRMWOp<DType, Op, std::enable_if_t<Op == RMWOp::FADD>>
    : public AtomicRMWOpBase<DType> {
public:
  using AtomicRMWOpBase<DType>::AtomicRMWOpBase;

protected:
  DType applyAtMasked(DType *loc, const DType value, int order) override {
    return atomic_fadd(loc, value, order);
  }
};

template <typename DType, RMWOp Op>
class AtomicRMWOp<DType, Op, std::enable_if_t<Op == RMWOp::AND>>
    : public AtomicRMWOpBase<DType> {
public:
  using AtomicRMWOpBase<DType>::AtomicRMWOpBase;

protected:
  DType applyAtMasked(DType *loc, const DType value, int order) override {
    return __atomic_fetch_and(loc, value, order);
  }
};

template <typename DType, RMWOp Op>
class AtomicRMWOp<DType, Op, std::enable_if_t<Op == RMWOp::OR>>
    : public AtomicRMWOpBase<DType> {
public:
  using AtomicRMWOpBase<DType>::AtomicRMWOpBase;

protected:
  DType applyAtMasked(DType *loc, const DType value, int order) override {
    return __atomic_fetch_or(loc, value, order);
  }
};

template <typename DType, RMWOp Op>
class AtomicRMWOp<DType, Op, std::enable_if_t<Op == RMWOp::XOR>>
    : public AtomicRMWOpBase<DType> {
public:
  using AtomicRMWOpBase<DType>::AtomicRMWOpBase;

protected:
  DType applyAtMasked(DType *loc, const DType value, int order) override {
    return __atomic_fetch_xor(loc, value, order);
  }
};

template <typename DType, RMWOp Op>
class AtomicRMWOp<DType, Op,
                  std::enable_if_t<Op == RMWOp::MAX || Op == RMWOp::UMAX>>
    : public AtomicRMWOpBase<DType> {
public:
  using AtomicRMWOpBase<DType>::AtomicRMWOpBase;

protected:
  DType applyAtMasked(DType *loc, const DType value, int order) override {
    return atomic_cmp</*is_min=*/false>(loc, value, order);
  }
};

template <typename DType, RMWOp Op>
class AtomicRMWOp<DType, Op,
                  std::enable_if_t<Op == RMWOp::MIN || Op == RMWOp::UMIN>>
    : public AtomicRMWOpBase<DType> {
public:
  using AtomicRMWOpBase<DType>::AtomicRMWOpBase;

protected:
  DType applyAtMasked(DType *loc, const DType value, int order) override {
    return atomic_cmp</*is_min=*/true>(loc, value, order);
  }
};

template <typename DType, RMWOp Op>
class AtomicRMWOp<DType, Op, std::enable_if_t<Op == RMWOp::XCHG>>
    : public AtomicRMWOpBase<DType> {
public:
  using AtomicRMWOpBase<DType>::AtomicRMWOpBase;

protected:
  DType applyAtMasked(DType *loc, const DType value, int order) override {
    return __atomic_exchange_n(loc, value, order);
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
      uint8_t desired_val = *(static_cast<const uint8_t *>(desired) + i);
      __atomic_compare_exchange_n(static_cast<uint8_t *>(loc),
                                  static_cast<uint8_t *>(expected) + i,
                                  desired_val, false, order, order);
    } else if (itemsize == 2) {
      uint16_t desired_val = *(static_cast<const uint16_t *>(desired) + i);
      __atomic_compare_exchange_n(static_cast<uint16_t *>(loc),
                                  static_cast<uint16_t *>(expected) + i,
                                  desired_val, false, order, order);
    } else if (itemsize == 4) {
      uint32_t desired_val = *(static_cast<const uint32_t *>(desired) + i);
      __atomic_compare_exchange_n(static_cast<uint32_t *>(loc),
                                  static_cast<uint32_t *>(expected) + i,
                                  desired_val, false, order, order);
    } else if (itemsize == 8) {
      uint64_t desired_val = *(static_cast<const uint64_t *>(desired) + i);
      __atomic_compare_exchange_n(static_cast<uint64_t *>(loc),
                                  static_cast<uint64_t *>(expected) + i,
                                  desired_val, false, order, order);
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

// This is a workaround because explicit template parameter list for lambdas is
// a C++20 extension:
// auto try_make_op = [&]<typename T>() {
//   if (dtype.is(pybind11::dtype::of<T>())) {
//     atomic_op = std::make_unique<AtomicRMWOp<T, Op>>(ptr, val, ret, mask,
//                                                      numel, order);
//   }
// };
template <RMWOp Op> struct OpCreator {
  pybind11::dtype dtype;
  const uint64_t *ptr;
  const void *val;
  void *ret;
  const bool *mask;
  size_t numel;
  int order;
  std::unique_ptr<AtomicOp> &atomic_op;

  template <typename T> void create() {
    if (!atomic_op && dtype.is(pybind11::dtype::of<T>())) {
      atomic_op = std::make_unique<AtomicRMWOp<T, Op>>(ptr, val, ret, mask,
                                                       numel, order);
    }
  }
};

template <RMWOp Op, typename... SupportedDTypes>
std::unique_ptr<AtomicOp>
makeAtomicRMWOp(pybind11::dtype dtype, const uint64_t *ptr, const void *val,
                void *ret, const bool *mask, size_t numel, int order) {
  // Iterate over all supported data types, make one that matches, and return
  std::unique_ptr<AtomicOp> atomic_op;
  OpCreator<Op> try_make_op{dtype, ptr,   val,   ret,
                            mask,  numel, order, atomic_op};

  (try_make_op.template create<SupportedDTypes>(), ...);
  if (!atomic_op) {
    throw std::invalid_argument("Unsupported data type");
  }
  // Make it a unique_ptr
  return atomic_op;
}

} // namespace

void init_triton_interpreter(py::module &&m) {
  using ret = py::return_value_policy;

  py::enum_<MemSemantic>(m, "MEM_SEMANTIC", py::module_local())
      .value("ACQUIRE_RELEASE", MemSemantic::ACQUIRE_RELEASE)
      .value("ACQUIRE", MemSemantic::ACQUIRE)
      .value("RELEASE", MemSemantic::RELEASE)
      .value("RELAXED", MemSemantic::RELAXED)
      .export_values();

  py::enum_<RMWOp>(m, "RMW_OP", py::module_local())
      .value("ADD", RMWOp::ADD)
      .value("FADD", RMWOp::FADD)
      .value("AND", RMWOp::AND)
      .value("OR", RMWOp::OR)
      .value("XOR", RMWOp::XOR)
      .value("XCHG", RMWOp::XCHG)
      .value("MAX", RMWOp::MAX)
      .value("MIN", RMWOp::MIN)
      .value("UMIN", RMWOp::UMIN)
      .value("UMAX", RMWOp::UMAX)
      .export_values();

  m.def("load",
        [](py::array_t<uint64_t> ptr, py::array_t<bool> mask, py::array other,
           py::dtype ret_dtype) -> py::array {
          int numel = ptr.size();
          auto shape =
              std::vector<ptrdiff_t>(ptr.shape(), ptr.shape() + ptr.ndim());
          py::array ret(ret_dtype, py::array::ShapeContainer{numel});
          py::array_t<uint64_t> reshaped_ptr = ptr.reshape({numel});
          py::array_t<bool> reshaped_mask = mask.reshape({numel});
          py::array reshaped_others = other.reshape({numel});
          for (size_t i = 0; i < ptr.size(); ++i) {
            if (reshaped_mask.at(i))
              memcpy(ret.mutable_data(i),
                     reinterpret_cast<void *>(reshaped_ptr.at(i)),
                     ret_dtype.itemsize());
            else
              memcpy(ret.mutable_data(i), reshaped_others.data(i),
                     ret_dtype.itemsize());
          }
          return ret.reshape(shape);
        });

  m.def("store",
        [](py::array_t<uint64_t> ptr, py::array value, py::array_t<bool> mask) {
          int numel = ptr.size();
          py::array_t<uint64_t> reshaped_ptr = ptr.reshape({numel});
          py::array_t<int8_t> reshaped_mask = mask.reshape({numel});
          py::array reshaped_value = value.reshape({numel});
          for (size_t i = 0; i < ptr.size(); ++i) {
            if (reshaped_mask.at(i)) {
              memcpy(reinterpret_cast<void *>(reshaped_ptr.mutable_at(i)),
                     reshaped_value.data(i), value.dtype().itemsize());
            }
          }
        });

  m.def("atomic_rmw",
        [](RMWOp rmw_op, py::array_t<uint64_t> ptr, py::array val,
           py::array_t<bool> mask, MemSemantic sem) -> py::array {
          int order = mem_semantic_map[sem];
          int numel = ptr.size();
          auto shape =
              std::vector<ptrdiff_t>(ptr.shape(), ptr.shape() + ptr.ndim());
          auto ret_dtype = val.dtype();
          py::array ret(ret_dtype, py::array::ShapeContainer{numel});
          py::array_t<uint64_t> reshaped_ptr = ptr.reshape({numel});
          py::array_t<bool> reshaped_mask = mask.reshape({numel});
          py::array reshaped_val = val.reshape({numel});
          auto *ptr_data = reshaped_ptr.data();
          auto *mask_data = reshaped_mask.data();
          auto *val_data = static_cast<const void *>(reshaped_val.data());
          auto *ret_data = static_cast<void *>(ret.mutable_data());

          std::unique_ptr<AtomicOp> atomic_op;

#define MAKE_ATOMIC_RMW_OP(OP_NAME, ...)                                       \
  case OP_NAME:                                                                \
    atomic_op = makeAtomicRMWOp<OP_NAME, __VA_ARGS__>(                         \
        ret_dtype, ptr_data, val_data, ret_data, mask_data, numel, order);     \
    break;

          switch (rmw_op) {
            MAKE_ATOMIC_RMW_OP(RMWOp::ADD, int32_t, uint32_t, int64_t, uint64_t)
            MAKE_ATOMIC_RMW_OP(RMWOp::FADD, float, double)
            MAKE_ATOMIC_RMW_OP(RMWOp::AND, int32_t, uint32_t, int64_t, uint64_t)
            MAKE_ATOMIC_RMW_OP(RMWOp::OR, int32_t, uint32_t, int64_t, uint64_t)
            MAKE_ATOMIC_RMW_OP(RMWOp::XOR, int32_t, uint32_t, int64_t, uint64_t)
            MAKE_ATOMIC_RMW_OP(RMWOp::MAX, int32_t, int64_t)
            MAKE_ATOMIC_RMW_OP(RMWOp::UMAX, uint32_t, uint64_t)
            MAKE_ATOMIC_RMW_OP(RMWOp::MIN, int32_t, int64_t)
            MAKE_ATOMIC_RMW_OP(RMWOp::UMIN, uint32_t, uint64_t)
            MAKE_ATOMIC_RMW_OP(RMWOp::XCHG, int32_t, uint32_t, int64_t,
                               uint64_t)
          default:
            throw std::invalid_argument("Unsupported RMW operation");
          }

#undef MAKE_ATOMIC_RMW_OP

          atomic_op->apply();
          return ret.reshape(shape);
        });

  m.def("atomic_cas",
        [](py::array_t<uint64_t> ptr, py::array &cmp, py::array &val,
           MemSemantic sem) -> py::array {
          int order = mem_semantic_map[sem];
          int numel = ptr.size();
          auto shape =
              std::vector<ptrdiff_t>(ptr.shape(), ptr.shape() + ptr.ndim());
          auto ret_dtype = cmp.dtype();
          py::array ret(ret_dtype, py::array::ShapeContainer{numel});
          py::array_t<uint64_t> reshaped_ptr = ptr.reshape({numel});
          py::array reshaped_cmp = cmp.reshape({numel});
          py::array reshaped_val = val.reshape({numel});
          auto itemsize = cmp.itemsize();
          memcpy(static_cast<void *>(ret.mutable_data()),
                 static_cast<const void *>(reshaped_cmp.data()),
                 itemsize * numel);
          AtomicCASOp(reshaped_ptr.data(), ret.mutable_data(),
                      static_cast<const void *>(reshaped_val.data()), itemsize,
                      numel, order)
              .apply();
          return ret.reshape(shape);
        });
}
