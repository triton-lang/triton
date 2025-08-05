#include <cstddef>
#include "pybind11/pybind11.h"
#include <climits>
#include <map>
#include <string>
#include <tuple>

namespace py = pybind11;
using DTypeKey = std::pair<PyObject*, bool>;

struct DTypeKeyCompare {
  bool operator()(const DTypeKey& lhs, const DTypeKey& rhs) const {
    if (lhs.first != rhs.first) {
      return lhs.first < rhs.first;
    }
    return lhs.second < rhs.second;
  }
};
using Dtype2Str = std::map<DTypeKey, py::object, DTypeKeyCompare>;


inline bool is_type_by_name(const py::object& obj, const std::string& name) {
  std::string class_str = py::str(obj.get_type().attr("__name__"));
  return class_str == name;
}

namespace {
  struct PythonGlobals {
    // python classes
    py::handle constexpr_cls;
    py::handle JITFunction;
    py::handle TensorDescriptor;
    py::handle GluonTensorDescriptor;
    py::handle canonicalize_dtype;
    py::handle canonicalize_ptr_dtype;

    Dtype2Str dtype2str;

    PythonGlobals(void) {
      auto m_lang = py::module::import("triton.language");
      constexpr_cls = py::object(m_lang.attr("constexpr")).release();
      auto m_jit = py::module::import("triton.runtime.jit");
      JITFunction = py::object(m_jit.attr("JITFunction")).release();
      auto m_desc = py::module::import("triton.tools.tensor_descriptor");
      TensorDescriptor = py::object(m_desc.attr("TensorDescriptor")).release();
      auto m_desc_gluon = py::module::import("triton.experimental.gluon.nvidia.hopper");
      GluonTensorDescriptor = py::object(m_desc_gluon.attr("TensorDescriptor")).release();
      auto m_canonicalize = py::module::import("triton._utils");
      canonicalize_dtype = py::object(m_canonicalize.attr("canonicalize_dtype")).release();
      canonicalize_ptr_dtype = py::object(m_canonicalize.attr("canonicalize_ptr_dtype")).release();
    }

    ~PythonGlobals() = default;
  };
}

std::pair<py::object, py::object> inline _specialize_int(
  py::object arg,
  bool specialize_value,
  bool align,
  PythonGlobals& globals
) {
  int64_t val = 0;
  bool overflow_i32 = false;
  bool overflow_i64 = false;

  try {
    val = arg.cast<int64_t>();
  } catch (const py::cast_error&) {
    overflow_i64 = true;
  }

  overflow_i32 = overflow_i64 || (val < INT32_MIN || val > INT32_MAX);

  py::object type_str;
  if (!overflow_i32) {
    type_str = py::str("i32");
  } else if (!overflow_i64) {
    type_str = py::str("i64");
  } else {
    type_str = py::str("u64");
  }

  py::object key = py::none();

  if (!specialize_value) {
    // key stays None
  } else if (align) {
    if (!overflow_i64) {
      key = (val % 16 == 0) ? py::str("D") : py::str("");
    } else {
      uint64_t val_u64 = static_cast<uint64_t>(val);
      key = (val_u64 % 16 == 0) ? py::str("D") : py::str("");
    }
  } else {
    key = py::str("");
  }

  return {type_str, key};
}

py::object inline _specialize_tensor_align(
  uint64_t data_ptr,
  bool specialize_value,
  bool align,
  PythonGlobals& globals
) {
  if (!specialize_value) {
    return py::none();
  }
  if (align && (data_ptr % 16 == 0)) {
    return py::str("D");
  } else {
    return py::str("");
  }
}

py::object inline _specialize_tensor_dtype(
  py::object arg,
  bool is_const,
  bool specialize_value,
  bool align,
  PythonGlobals& globals
) {
  py::object dtype = arg.attr("dtype");
  py::object dtype_canon = globals.canonicalize_ptr_dtype(dtype, is_const);
  return dtype_canon;
}

std::pair<py::object, py::object> inline _specialize_tensor(
  py::object arg,
  bool is_const,
  bool specialize_value,
  bool align,
  PythonGlobals& globals
) {
    DTypeKey dsk{arg.attr("dtype").ptr(), is_const};
    auto it = globals.dtype2str.find(dsk);
    py::object res;
    if (it != globals.dtype2str.end()) {
      res = it->second;
    } else {
      res = _specialize_tensor_dtype(arg, is_const, specialize_value, align, globals);
      globals.dtype2str[dsk] = res;
    }
    py::object key = _specialize_tensor_align(
      arg.attr("data_ptr")().cast<uint64_t>(),
      specialize_value,
      align,
      globals
    );
    return {res, key};
}

std::pair<py::object, py::object> inline _specialize_tensordesc(
  py::object arg,
  bool has_layout,
  PythonGlobals& globals
) {
  py::object base = arg.attr("base");
  if (!py::hasattr(base, "data_ptr")) {
    throw py::type_error("Expected TensorDescriptor.base to have 'data_ptr' attribute");
  }
  
  py::object inner = globals.canonicalize_dtype(base.attr("dtype"));
  std::string inner_str = py::str(inner);

  auto block_shape_vec = arg.attr("block_shape");
  std::string block_shape_str = py::str(py::list(block_shape_vec));

  std::string result = "tensordesc>";
  result += inner_str;
  result += block_shape_str;

  if (has_layout) {
    std::string layout_repr = py::str(py::repr(arg.attr("layout")));
    result += ",";
    result += layout_repr;
  }

  result += ">";

  return {py::str(result), py::none()};
}

std::pair<py::object, py::object> specialize_impl(
  py::object arg,
  bool is_const,
  bool specialize_value,
  bool align,
  PythonGlobals& globals
) {
  // if type(arg).__name__ == "Tensor"
  if (is_type_by_name(arg, "Tensor")) {
    return _specialize_tensor(arg, is_const, specialize_value, align, globals);
  }
  // if isinstance(arg, int)
  if (py::isinstance<py::int_>(arg)) {
    return _specialize_int(arg, specialize_value, align, globals);
  }
  // if arg is None
  if (arg.is_none()) {
    return {py::str("constexpr"), py::none()};
  }
  // if isinstance(arg, bool)
  if (py::isinstance<py::bool_>(arg)) {
    return {py::str("u1"), py::none()};
  }
  // if isinstance(arg, float)
  if (py::isinstance<py::float_>(arg)) {
    return {py::str("f32"), py::none()};
  }
  // if isinstance(arg, constexpr)
  if (py::isinstance(arg, globals.constexpr_cls)) {
    return {py::str("constexpr"), arg};
  }
  // if hasattr(arg, "data_ptr")
  if (py::hasattr(arg, "data_ptr")) {
    return _specialize_tensor(arg, is_const, specialize_value, align, globals);
  }
  // if isinstance(arg, JITFunction)
  if (py::isinstance(arg, globals.JITFunction)) {
    return {py::str("constexpr"), arg.attr("cache_key")};
  }
  // if hasattr(arg, "tma_desc_cpu_ptr")
  if (py::hasattr(arg, "tma_desc_cpu_ptr")) {
    return {py::str("nvTmaDesc"), py::none()};
  }
  // if isinstance(arg, tuple)
  if (py::isinstance<py::tuple>(arg)) {
    std::vector<py::object> tys, keys;

    auto seq = py::reinterpret_borrow<py::sequence>(arg);
    for (const auto& item : seq) {
      auto [ty, key] = specialize_impl(item, is_const, specialize_value, align, globals);
      tys.push_back(ty);
      keys.push_back(key);
    }

    py::object out_tys, out_keys;
    if (py::hasattr(arg, "_fields")) {
      py::handle tuple_type = arg.get_type();
      out_tys = tuple_type(tys);
      out_keys = tuple_type(keys);
    } else {
      py::tuple out_tys(tys.size()), out_keys(keys.size());
      for (size_t i = 0; i < tys.size(); ++i) {
        out_tys[i] = tys[i];
        out_keys[i] = keys[i];
      }
    }
    return {out_tys, out_keys};
  }
  // if isinstance(arg, TensorDescriptor)
  if (py::isinstance(arg, globals.TensorDescriptor)) {
    return _specialize_tensordesc(arg, false, globals);
  }
  // if isinstance(arg, GluonTensorDescriptor)
  if (py::isinstance(arg, globals.GluonTensorDescriptor)) {
    return _specialize_tensordesc(arg, true, globals);
  }
  throw py::type_error("Unsupported argument type for specialization: " + std::string(py::str(arg.get_type().attr("__name__"))));
}


PYBIND11_MODULE(__triton_specialize_all, m) { 
  static PythonGlobals globals;

  auto cleanup_lambda = []() {
    globals.dtype2str.clear();
  };

  m.def("specialize_impl", [](py::object arg, bool is_const, bool specialize_value, bool align) {
    return specialize_impl(arg, is_const, specialize_value, align, globals);
  }, "Specializes the given argument based on its type and properties.");

  auto atexit = py::module::import("atexit");
  atexit.attr("register")(py::cpp_function(cleanup_lambda));
}
