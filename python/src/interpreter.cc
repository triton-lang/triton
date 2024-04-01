#include "interpreter.h"

namespace py = pybind11;

void init_triton_interpreter(py::module &&m) {
  using namespace triton::interpreter;
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

// clang-format off
    #define MAKE_ATOMIC_RMW_OP(OP_NAME, ...) \
      case OP_NAME: \
        atomic_op = makeAtomicRMWOp<OP_NAME, __VA_ARGS__>( \
            ret_dtype, ptr_data, val_data, ret_data, mask_data, numel, order); \
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
            MAKE_ATOMIC_RMW_OP(RMWOp::XCHG, int32_t, uint32_t, int64_t, uint64_t)
            default:
              throw std::invalid_argument("Unsupported RMW operation");
          }

    #undef MAKE_ATOMIC_RMW_OP
          // clang-format on

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
