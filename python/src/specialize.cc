#include <Python.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <utility>

namespace {

namespace py = pybind11;

using DTypePtrKey = std::pair<Py_hash_t, bool>;
using DTypeKey = Py_hash_t;

struct DTypePtrKeyHash {
  std::size_t operator()(const DTypePtrKey &k) const {
    return std::hash<Py_hash_t>()(k.first) ^ (std::hash<bool>()(k.second) << 1);
  }
};

using DtypePtr2Str =
    std::unordered_map<DTypePtrKey, PyObject *, DTypePtrKeyHash>;
using Dtype2Str = std::unordered_map<DTypeKey, PyObject *>;

using TypeHandler = std::pair<py::object, py::object> (*)(PyObject *,
                                                          PyObject *, bool,
                                                          bool, bool);
using TypeHandlerCache = std::unordered_map<PyTypeObject *, TypeHandler>;

static std::pair<py::object, py::object>
specialize_arg(PyObject *backend, PyObject *arg, bool is_const,
               bool specialize_value, bool align);

static bool init_called = false;

static PyObject *constexpr_cls = nullptr;
static PyObject *jit_callable_cls = nullptr;
static PyObject *tensor_descriptor_cls = nullptr;
static PyObject *gluon_tensor_descriptor_cls = nullptr;
static PyObject *canonicalize_dtype_fn = nullptr;
static PyObject *canonicalize_ptr_dtype_fn = nullptr;
static PyObject *torch_tensor_cls = nullptr;

static PyObject *i32_str = nullptr;
static PyObject *i64_str = nullptr;
static PyObject *u64_str = nullptr;
static PyObject *fp32_str = nullptr;
static PyObject *u1_str = nullptr;
static PyObject *D_str = nullptr;
static PyObject *constexpr_str = nullptr;
static PyObject *empty_str = nullptr;
static PyObject *nvTmaDesc_str = nullptr;

static PyObject *base_attr = nullptr;
static PyObject *data_ptr_attr = nullptr;
static PyObject *dtype_attr = nullptr;
static PyObject *cache_key_attr = nullptr;
static PyObject *_fields_attr = nullptr;
static PyObject *block_shape_attr = nullptr;
static PyObject *layout_attr = nullptr;
static PyObject *has_native_tensor_spec_attr = nullptr;
static PyObject *get_tensor_spec_attr = nullptr;
static PyObject *align_kwarg = nullptr;

static DtypePtr2Str dtype_ptr2str;
static Dtype2Str dtype2str;
static TypeHandlerCache type_handler_cache;

// Wrappers to make steal and borrow slightly simpler. We use raw CPython API
// with py::object to handle decref, as using the pybind11 APIs adds exception
// handling overhead which is quite significant here.
py::object from_new_ref(py::handle val) {
  return py::reinterpret_steal<py::object>(val);
}
py::object from_borrowed_ref(py::handle val) {
  return py::reinterpret_borrow<py::object>(val);
}

PyObject *intern_from_string(const char *str) {
  PyObject *obj = PyUnicode_InternFromString(str);
  if (!obj)
    throw py::error_already_set();
  return obj;
}

PyObject *import_from(const char *module_name, const char *var_name) {
  py::object var = py::module_::import(module_name).attr(var_name);
  return var.release().ptr();
}

void init_interned_strings() {
  i32_str = intern_from_string("i32");
  i64_str = intern_from_string("i64");
  u64_str = intern_from_string("u64");
  fp32_str = intern_from_string("fp32");
  u1_str = intern_from_string("u1");
  D_str = intern_from_string("D");
  constexpr_str = intern_from_string("constexpr");
  empty_str = intern_from_string("");
  nvTmaDesc_str = intern_from_string("nvTmaDesc");

  base_attr = intern_from_string("base");
  data_ptr_attr = intern_from_string("data_ptr");
  dtype_attr = intern_from_string("dtype");
  cache_key_attr = intern_from_string("cache_key");
  _fields_attr = intern_from_string("_fields");
  block_shape_attr = intern_from_string("block_shape");
  layout_attr = intern_from_string("layout");
  has_native_tensor_spec_attr =
      intern_from_string("supports_native_tensor_specialization");
  get_tensor_spec_attr = intern_from_string("get_tensor_specialization");

  align_kwarg = py::make_tuple("align").release().ptr();
}

void init_type_handler_cache();

bool init_globals() noexcept try {
  // Import releavant symbols
  jit_callable_cls = import_from("triton.runtime.jit", "JITCallable");
  tensor_descriptor_cls =
      import_from("triton.tools.tensor_descriptor", "TensorDescriptor");
  gluon_tensor_descriptor_cls = import_from(
      "triton.experimental.gluon.nvidia.hopper", "TensorDescriptor");

  auto m_canonicalize = py::module_::import("triton._utils");
  canonicalize_dtype_fn = import_from("triton._utils", "canonicalize_dtype");
  canonicalize_ptr_dtype_fn =
      import_from("triton._utils", "canonicalize_ptr_dtype");
  constexpr_cls = import_from("triton.language", "constexpr");

  try {
    torch_tensor_cls = import_from("torch", "Tensor");
  } catch (py::error_already_set &e) {
  }

  init_interned_strings();
  init_type_handler_cache();

  init_called = true;
  return true;
} catch (py::error_already_set &e) {
  e.restore();
  return false;
}

std::pair<py::object, py::object> specialize_tensordesc(PyObject *arg,
                                                        bool has_layout) {
  auto base = from_new_ref(PyObject_GetAttr(arg, base_attr));
  if (!base)
    return {};

  auto dtype = from_new_ref(PyObject_GetAttr(base.ptr(), dtype_attr));
  if (!dtype)
    return {};

  PyObject *type_str;
  Py_hash_t dtype_hash = PyObject_Hash(dtype.ptr());
  if (dtype_hash == -1)
    return {};
  DTypeKey dsk{dtype_hash};
  auto it = dtype2str.find(dsk);
  if (it != dtype2str.end()) {
    type_str = it->second;
  } else {
    auto res = from_new_ref(PyObject_CallFunctionObjArgs(canonicalize_dtype_fn,
                                                         dtype.ptr(), nullptr));
    if (!res)
      return {};
    dtype2str[dsk] = res.ptr();
    type_str = res.release().ptr();
  }

  std::string desc_cstr;
  desc_cstr.reserve(128);
  desc_cstr = "tensordesc<";
  auto dtype_str = from_new_ref(PyObject_Str(type_str));
  if (!dtype_str)
    return {};

  const char *dtype_cstr = PyUnicode_AsUTF8(dtype_str.ptr());
  if (!dtype_cstr)
    return {};
  desc_cstr += dtype_cstr;

  auto block_shape_obj = from_new_ref(PyObject_GetAttr(arg, block_shape_attr));
  if (!block_shape_obj)
    return {};
  auto block_shape_list = from_new_ref(PySequence_List(block_shape_obj.ptr()));
  if (!block_shape_list)
    return {};
  auto block_shape_str = from_new_ref(PyObject_Str(block_shape_list.ptr()));
  if (!block_shape_str)
    return {};
  const char *block_shape_cstr = PyUnicode_AsUTF8(block_shape_str.ptr());
  if (!block_shape_cstr)
    return {};
  desc_cstr += block_shape_cstr;

  if (has_layout) {
    auto layout_obj = from_new_ref(PyObject_GetAttr(arg, layout_attr));
    if (!layout_obj)
      return {};
    auto layout_repr = from_new_ref(PyObject_Repr(layout_obj.ptr()));
    if (!layout_repr)
      return {};
    desc_cstr += ",";
    const char *layout_cstr = PyUnicode_AsUTF8(layout_repr.ptr());
    if (!layout_cstr)
      return {};
    desc_cstr += layout_cstr;
  }

  desc_cstr += ">";
  auto type_str_result = from_new_ref(PyUnicode_FromString(desc_cstr.c_str()));
  if (!type_str_result)
    return {};

  return {std::move(type_str_result), py::none()};
}

std::pair<py::object, py::object> handle_long_type(PyObject *backend,
                                                   PyObject *arg, bool is_const,
                                                   bool specialize_value,
                                                   bool align) {
  int overflow;
  long long val = PyLong_AsLongLongAndOverflow(arg, &overflow);
  if (PyErr_Occurred()) {
    return {};
  }

  if (specialize_value && (val == 1)) {
    return {from_borrowed_ref(constexpr_str), from_borrowed_ref(arg)};
  }

  py::handle type_str;
  py::handle key_obj;
  if (overflow == 0) {
    type_str = (val >= INT32_MIN && val <= INT32_MAX) ? i32_str : i64_str;
    if (specialize_value) {
      key_obj = (align && ((val & 15) == 0)) ? D_str : empty_str;
    }
  } else {
    unsigned long long val_64 = PyLong_AsUnsignedLongLong(arg);
    if (PyErr_Occurred()) {
      // this runs into an edge-case where the Python reference
      // returns i64 as type and alignment of the value despite
      // not being representable as such which at kernel launch later
      // will throw an OverflowError nevertheless, here we throw
      // OverflowError immediately
      PyErr_SetString(PyExc_OverflowError,
                      "integer to be specialized too large to represent");
      return {};
    }
    type_str = u64_str;
    if (specialize_value) {
      key_obj = (align && ((val_64 & 15) == 0)) ? D_str : empty_str;
    }
  }
  if (!key_obj) {
    return {from_borrowed_ref(type_str), py::none()};
  }
  return {from_borrowed_ref(type_str), from_borrowed_ref(key_obj)};
}

std::pair<py::object, py::object> handle_tensor(PyObject *backend,
                                                PyObject *arg, bool is_const,
                                                bool specialize_value,
                                                bool align) {
  // handle type_str specialization of a tensor
  auto dtype = from_new_ref(PyObject_GetAttr(arg, dtype_attr));
  if (!dtype)
    return {};

  Py_hash_t dtype_hash = PyObject_Hash(dtype.ptr());
  if (dtype_hash == -1)
    return {};

  DTypePtrKey dsk{dtype_hash, is_const};
  auto it = dtype_ptr2str.find(dsk);

  py::handle type_str;
  if (it != dtype_ptr2str.end()) {
    type_str = it->second;
  } else {
    auto canon_res =
        PyObject_CallFunctionObjArgs(canonicalize_ptr_dtype_fn, dtype.ptr(),
                                     is_const ? Py_True : Py_False, nullptr);
    if (!canon_res)
      return {};
    dtype_ptr2str[dsk] = canon_res;
    type_str = canon_res;
  }

  // handle alignment specialization of a tensor
  if (!specialize_value) {
    return {from_borrowed_ref(type_str), py::none()};
  }

  bool native_impl_available = false;
  auto native_spec_obj =
      from_new_ref(PyObject_GetAttr(backend, has_native_tensor_spec_attr));
  if (native_spec_obj) {
    native_impl_available = PyObject_IsTrue(native_spec_obj.ptr());
  } else {
    PyErr_Clear();
    // on error we fall back to native_impl_available = false gracefully
  }

  py::object key;
  if (native_impl_available) {
    auto data_ptr_result =
        from_new_ref(PyObject_CallMethodNoArgs(arg, data_ptr_attr));
    if (!data_ptr_result)
      return {};

    auto data_ptr = PyLong_AsUnsignedLongLong(data_ptr_result.ptr());
    if (PyErr_Occurred())
      return {};

    auto key_obj = (align && ((data_ptr & 15) == 0)) ? D_str : empty_str;
    key = from_borrowed_ref(key_obj);
  } else {
    PyObject *args[3] = {backend, arg, align ? Py_True : Py_False};
    PyObject *kwnames = align_kwarg;
    key = from_new_ref(
        PyObject_VectorcallMethod(get_tensor_spec_attr, args, 2, kwnames));
    if (!key)
      return {};
  }

  return {from_borrowed_ref(type_str), std::move(key)};
}

std::pair<py::object, py::object> handle_bool_type(PyObject *backend,
                                                   PyObject *arg, bool is_const,
                                                   bool specialize_value,
                                                   bool align) {
  return {from_borrowed_ref(u1_str), py::none()};
}

std::pair<py::object, py::object>
handle_float_type(PyObject *backend, PyObject *arg, bool is_const,
                  bool specialize_value, bool align) {
  return {from_borrowed_ref(fp32_str), py::none()};
}

std::pair<py::object, py::object>
handle_tensor_descriptor(PyObject *backend, PyObject *arg, bool is_const,
                         bool specialize_value, bool align) {
  return specialize_tensordesc(arg, false);
}

std::pair<py::object, py::object>
handle_gluon_tensor_descriptor(PyObject *backend, PyObject *arg, bool is_const,
                               bool specialize_value, bool align) {
  return specialize_tensordesc(arg, true);
}

std::pair<py::object, py::object>
handle_constexpr_type(PyObject *backend, PyObject *arg, bool is_const,
                      bool specialize_value, bool align) {
  return {from_borrowed_ref(constexpr_str), from_borrowed_ref(arg)};
}

std::pair<py::object, py::object>
handle_jit_callable(PyObject *backend, PyObject *arg, bool is_const,
                    bool specialize_value, bool align) {
  auto cache_key = from_new_ref(PyObject_GetAttr(arg, cache_key_attr));
  if (!cache_key)
    return {};
  return {from_borrowed_ref(constexpr_str), std::move(cache_key)};
}

std::pair<py::object, py::object> handle_tuple(PyObject *backend, PyObject *arg,
                                               bool is_const,
                                               bool specialize_value,
                                               bool align) {
  Py_ssize_t size = PyTuple_GET_SIZE(arg);
  if (size == 0) {
    // return tuple of empty tuples as in python reference
    return {from_borrowed_ref(arg), from_borrowed_ref(arg)};
  }

  bool is_namedtuple = PyObject_HasAttr(arg, _fields_attr);
  auto tuple_type = Py_TYPE(arg);

  // Create tuples directly instead of lists
  auto tys_tuple = from_new_ref(PyTuple_New(size));
  if (!tys_tuple)
    return {};

  auto keys_tuple = from_new_ref(PyTuple_New(size));
  if (!keys_tuple)
    return {};

  for (Py_ssize_t i = 0; i < size; ++i) {
    PyObject *item = PyTuple_GET_ITEM(arg, i); // Borrowed reference
    // python reference calls specialize recursively with default arguments set
    // currently this is is_const=False, specialize_value=True, align=True
    auto [type, key] = specialize_arg(backend, item, false, true, true);
    if (!type || !key)
      return {};
    // Steals reference
    PyTuple_SET_ITEM(tys_tuple.ptr(), i, type.release().ptr());
    PyTuple_SET_ITEM(keys_tuple.ptr(), i, key.release().ptr());
  }

  if (is_namedtuple) {
    tys_tuple = from_new_ref(
        PyObject_CallObject((PyObject *)tuple_type, tys_tuple.ptr()));
    if (!tys_tuple)
      return {};
    keys_tuple = from_new_ref(
        PyObject_CallObject((PyObject *)tuple_type, keys_tuple.ptr()));
    if (!keys_tuple)
      return {};
  }

  return {std::move(tys_tuple), std::move(keys_tuple)};
}

// initialize type handler which returns specialize impelemntations based on
// type(arg)
void init_type_handler_cache() {
  // Python Types (int, bool, float, tuple)
  type_handler_cache[&PyLong_Type] = handle_long_type;
  type_handler_cache[&PyBool_Type] = handle_bool_type;
  type_handler_cache[&PyFloat_Type] = handle_float_type;
  type_handler_cache[&PyTuple_Type] = handle_tuple;

  // torch.Tensor
  if (torch_tensor_cls && PyType_Check(torch_tensor_cls)) {
    type_handler_cache[(PyTypeObject *)torch_tensor_cls] = handle_tensor;
  }
  // TensorDescriptor
  if (tensor_descriptor_cls && PyType_Check(tensor_descriptor_cls)) {
    type_handler_cache[(PyTypeObject *)tensor_descriptor_cls] =
        handle_tensor_descriptor;
  }
  // GluonTensorDescriptor
  if (gluon_tensor_descriptor_cls &&
      PyType_Check(gluon_tensor_descriptor_cls)) {
    type_handler_cache[(PyTypeObject *)gluon_tensor_descriptor_cls] =
        handle_gluon_tensor_descriptor;
  }
  // constexpr
  if (constexpr_cls && PyType_Check(constexpr_cls)) {
    type_handler_cache[(PyTypeObject *)constexpr_cls] = handle_constexpr_type;
  }
  // JITCallable
  if (jit_callable_cls && PyType_Check(jit_callable_cls)) {
    type_handler_cache[(PyTypeObject *)jit_callable_cls] = handle_jit_callable;
  }
}

// specialization logic without passing of objects from Python (to be called in
// specialize_impl only)
std::pair<py::object, py::object> specialize_arg(PyObject *backend,
                                                 PyObject *arg, bool is_const,
                                                 bool specialize_value,
                                                 bool align) {
  // fast-path for default types
  PyTypeObject *arg_type = Py_TYPE(arg);
  auto it = type_handler_cache.find(arg_type);
  if (it != type_handler_cache.end()) {
    return it->second(backend, arg, is_const, specialize_value, align);
  }

  // separate handling of None
  if (Py_IsNone(arg)) {
    return {from_borrowed_ref(constexpr_str), py::none()};
  }

  // handling of sublcasses of tuples
  if (PyTuple_Check(arg)) {
    return handle_tuple(backend, arg, is_const, specialize_value, align);
  }

  // fallback paths checking full inheritance
  if (PyObject_IsInstance(arg, constexpr_cls)) {
    return handle_constexpr_type(backend, arg, is_const, specialize_value,
                                 align);
  }

  if (PyObject_IsInstance(arg, tensor_descriptor_cls)) {
    return handle_tensor_descriptor(backend, arg, is_const, specialize_value,
                                    align);
  }

  if (PyObject_IsInstance(arg, gluon_tensor_descriptor_cls)) {
    return handle_gluon_tensor_descriptor(backend, arg, is_const,
                                          specialize_value, align);
  }

  if (PyObject_IsInstance(arg, jit_callable_cls)) {
    return handle_jit_callable(backend, arg, is_const, specialize_value, align);
  }

  // fallback paths checking attributes directly
  if (PyObject_HasAttr(arg, data_ptr_attr)) {
    return handle_tensor(backend, arg, is_const, specialize_value, align);
  }

  return {};
}

// main entry-point from Python implementing specialization logic natively
PyObject *specialize_impl(PyObject *self, PyObject *const *args,
                          Py_ssize_t nargs) {
  if (!init_called) {
    if (!init_globals()) {
      return nullptr;
    }
  }

  if (nargs != 5) {
    PyErr_SetString(PyExc_TypeError,
                    "native_specialize_impl expected 5 arguments");
    return nullptr;
  }

  PyObject *backend = args[0];
  PyObject *arg = args[1];
  int is_const = PyObject_IsTrue(args[2]);
  int specialize_value = PyObject_IsTrue(args[3]);
  int align = PyObject_IsTrue(args[4]);

  if (is_const == -1 || specialize_value == -1 || align == -1) {
    PyErr_SetString(PyExc_TypeError, "native_specialize_impl expected boolean "
                                     "arguments for args2, args3, args4");
    return nullptr;
  }

  auto [type, key] =
      specialize_arg(backend, arg, is_const, specialize_value, align);

  // check if specialization failed
  if (!type || !key) {
    if (!PyErr_Occurred()) {
      PyErr_Format(PyExc_TypeError, "failed to specialize argument of type: %s",
                   Py_TYPE(arg)->tp_name);
    }
    return nullptr;
  }

  return PyTuple_Pack(2, type.ptr(), key.ptr());
}

static PyMethodDef module_methods[] = {
    {"native_specialize_impl", (PyCFunction)specialize_impl, METH_FASTCALL,
     nullptr},
    {nullptr, nullptr, 0, nullptr} // sentinel
};

} // anonymous namespace

void init_native_specialize(pybind11::module &m) {
  // add functions to module
  PyModule_AddFunctions(m.ptr(), module_methods);
}
