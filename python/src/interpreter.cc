#include <iostream>
#include <map>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

template <int BYTES>
bool atomic_compare_exchange(void *ptr, void *expected, void *desired,
                             int order) {
  if constexpr (BYTES == 1) {
    return __atomic_compare_exchange(
        static_cast<uint8_t *>(ptr), static_cast<uint8_t *>(expected),
        static_cast<uint8_t *>(desired), false, order, order);
  } else if constexpr (BYTES == 2) {
    return __atomic_compare_exchange(
        static_cast<uint16_t *>(ptr), static_cast<uint16_t *>(expected),
        static_cast<uint16_t *>(desired), false, order, order);
  } else if constexpr (BYTES == 4) {
    return __atomic_compare_exchange(
        static_cast<uint32_t *>(ptr), static_cast<uint32_t *>(expected),
        static_cast<uint32_t *>(desired), false, order, order);
  } else if constexpr (BYTES == 8) {
    return __atomic_compare_exchange(
        static_cast<uint64_t *>(ptr), static_cast<uint64_t *>(expected),
        static_cast<uint64_t *>(desired), false, order, order);
  } else {
    // The ‘__atomic’ builtins can be used with any integral scalar or pointer
    // type that is 1, 2, 4, or 8 bytes in length. 16-byte integral types are
    // also allowed if ‘__int128’ (see 128-bit Integers) is supported by the
    // architecture.
    // https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html
    throw std::invalid_argument("Invalid byte size");
  }
}

// Instantiate functions
template bool atomic_compare_exchange<1>(void *, void *, void *, int);
template bool atomic_compare_exchange<2>(void *, void *, void *, int);
template bool atomic_compare_exchange<4>(void *, void *, void *, int);
template bool atomic_compare_exchange<8>(void *, void *, void *, int);

typedef bool (*atomic_cas_func)(void *, void *, void *, int);

// Store function pointer in a map
std::map<int, atomic_cas_func> atomic_cas_func_map = {
    {1, atomic_compare_exchange<1>},
    {2, atomic_compare_exchange<2>},
    {4, atomic_compare_exchange<4>},
    {8, atomic_compare_exchange<8>},
};

} // namespace

enum class MemSemantic { ACQUIRE_RELEASE, ACQUIRE, RELEASE, RELAXED };

enum class RMWOp { ADD, FADD, AND, OR, XOR, XCHG, MAX, MIN, UMIN, UMAX };

void init_triton_interpreter(py::module &&m) {
  using ret = py::return_value_policy;

  py::enum_<MemSemantic>(m, "MEM_SEMANTIC", py::module_local())
      .value("ACQUIRE_RELEASE", MemSemantic::ACQUIRE_RELEASE)
      .value("ACQUIRE", MemSemantic::ACQUIRE)
      .value("RELEASE", MemSemantic::RELEASE)
      .value("RELAXED", MemSemantic::RELAXED)
      .export_values();

  m.def("load",
        [](py::array_t<uint64_t> ptrs, py::array_t<bool> masks, py::array other,
           py::dtype ret_dtype) -> py::array {
          int numel = ptrs.size();
          auto shape =
              std::vector<ptrdiff_t>(ptrs.shape(), ptrs.shape() + ptrs.ndim());
          py::array ret(ret_dtype, py::array::ShapeContainer{numel});
          py::array_t<uint64_t> reshaped_ptrs = ptrs.reshape({numel});
          py::array_t<bool> reshaped_masks = masks.reshape({numel});
          py::array reshaped_others = other.reshape({numel});
          for (size_t i = 0; i < ptrs.size(); ++i) {
            if (reshaped_masks.at(i))
              memcpy(ret.mutable_data(i),
                     reinterpret_cast<void *>(reshaped_ptrs.at(i)),
                     ret_dtype.itemsize());
            else
              memcpy(ret.mutable_data(i), reshaped_others.data(i),
                     ret_dtype.itemsize());
          }
          return ret.reshape(shape);
        });

  m.def("store", [](py::array_t<uint64_t> ptrs, py::array values,
                    py::array_t<bool> mask) {
    int numel = ptrs.size();
    py::array_t<uint64_t> reshaped_ptrs = ptrs.reshape({numel});
    py::array_t<int8_t> reshaped_masks = mask.reshape({numel});
    py::array reshaped_values = values.reshape({numel});
    for (size_t i = 0; i < ptrs.size(); ++i) {
      if (reshaped_masks.at(i)) {
        memcpy(reinterpret_cast<void *>(reshaped_ptrs.mutable_at(i)),
               reshaped_values.data(i), values.dtype().itemsize());
      }
    }
  });

  //m.def("atomic_rmw", [](RMWOp rmw_op, py::array_t<uint64_t> ptr, py::array val, py::array mask, MemSemantic sem) -> py:array {

  //});

  m.def("atomic_cas",
        [](py::array_t<uint64_t> ptrs, py::array &cmp, py::array &val,
          MemSemantic sem) -> py::array {
          // Use compiler builtin atomics instead of std::atomic which requires
          // each variable to be declared as atomic.
          // Currently work for clang and gcc.
          int order = 0;
          if (sem == MemSemantic::ACQUIRE_RELEASE) {
            order = __ATOMIC_ACQ_REL;
          } else if (sem == MemSemantic::ACQUIRE) {
            order = __ATOMIC_ACQUIRE;
          } else if (sem == MemSemantic::RELEASE) {
            order = __ATOMIC_RELEASE;
          } else if (sem == MemSemantic::RELAXED) {
            order = __ATOMIC_RELAXED;
          } else {
            throw std::invalid_argument("Invalid memory order");
          }
          int numel = ptrs.size();
          auto shape =
              std::vector<ptrdiff_t>(ptrs.shape(), ptrs.shape() + ptrs.ndim());
          auto ret_dtype = cmp.dtype();
          py::array ret(ret_dtype, py::array::ShapeContainer{numel});
          py::array_t<uint64_t> reshaped_ptrs = ptrs.reshape({numel});
          py::array reshaped_cmp = cmp.reshape({numel});
          py::array reshaped_val = val.reshape({numel});
          // Atomic operations perform bitwise comparison, so it's safe to
          // use number of bytes (itemsize) to determine the type of pointers
          auto itemsize = cmp.itemsize();
          memcpy(static_cast<void *>(ret.mutable_data()),
                 reinterpret_cast<void *>(reshaped_cmp.mutable_data()),
                 itemsize * numel);

          for (size_t i = 0; i < numel; ++i) {
            atomic_cas_func_map[itemsize](
                reinterpret_cast<void *>(reshaped_ptrs.mutable_at(i)),
                static_cast<void *>(reshaped_cmp.mutable_data(i)),
                static_cast<void *>(reshaped_val.mutable_data(i)), order);
          }
          return ret.reshape(shape);
        });
}
