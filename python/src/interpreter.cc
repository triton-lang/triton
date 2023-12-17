#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_triton_interpreter(py::module &&m) {
  using ret = py::return_value_policy;

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
}
