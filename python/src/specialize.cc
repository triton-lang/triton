#include <Python.h>
#include <climits>
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
using TensorDescCache = std::unordered_map<std::string, PyObject *>;

using TypeHandler = std::pair<py::object, py::object> (*)(PyObject *,
                                                          PyObject *, bool,
                                                          bool, bool);
using TypeHandlerCache = std::unordered_map<PyTypeObject *, TypeHandler>;

static std::pair<py::object, py::object>
specialize_arg(PyObject *backend, PyObject *arg, bool is_const,
               bool specialize_value, bool align);

static bool init_called = false;

static PyObject *constexpr_cls = nullptr;
static PyObject *jit_function_cls = nullptr;
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
static PyObject *tma_desc_cpu_ptr_attr = nullptr;
static PyObject *_fields_attr = nullptr;
static PyObject *block_shape_attr = nullptr;
static PyObject *layout_attr = nullptr;
static PyObject *has_native_tensor_spec_attr = nullptr;
static PyObject *get_tensor_spec_attr = nullptr;
static PyObject *align_kwarg = nullptr;

static DtypePtr2Str dtype_ptr2str;
static Dtype2Str dtype2str;
static TypeHandlerCache type_handler_cache;

static bool init_interned_strings() {
  PyObject *align_kwarg_str = nullptr;

  i32_str = PyUnicode_InternFromString("i32");
  if (!i32_str)
    goto cleanup;
  i64_str = PyUnicode_InternFromString("i64");
  if (!i64_str)
    goto cleanup;
  u64_str = PyUnicode_InternFromString("u64");
  if (!u64_str)
    goto cleanup;
  fp32_str = PyUnicode_InternFromString("fp32");
  if (!fp32_str)
    goto cleanup;
  u1_str = PyUnicode_InternFromString("u1");
  if (!u1_str)
    goto cleanup;
  D_str = PyUnicode_InternFromString("D");
  if (!D_str)
    goto cleanup;
  constexpr_str = PyUnicode_InternFromString("constexpr");
  if (!constexpr_str)
    goto cleanup;
  empty_str = PyUnicode_InternFromString("");
  if (!empty_str)
    goto cleanup;
  nvTmaDesc_str = PyUnicode_InternFromString("nvTmaDesc");
  if (!nvTmaDesc_str)
    goto cleanup;

  base_attr = PyUnicode_InternFromString("base");
  if (!base_attr)
    goto cleanup;
  data_ptr_attr = PyUnicode_InternFromString("data_ptr");
  if (!data_ptr_attr)
    goto cleanup;
  dtype_attr = PyUnicode_InternFromString("dtype");
  if (!dtype_attr)
    goto cleanup;
  cache_key_attr = PyUnicode_InternFromString("cache_key");
  if (!cache_key_attr)
    goto cleanup;
  tma_desc_cpu_ptr_attr = PyUnicode_InternFromString("tma_desc_cpu_ptr");
  if (!tma_desc_cpu_ptr_attr)
    goto cleanup;
  _fields_attr = PyUnicode_InternFromString("_fields");
  if (!_fields_attr)
    goto cleanup;
  block_shape_attr = PyUnicode_InternFromString("block_shape");
  if (!block_shape_attr)
    goto cleanup;
  layout_attr = PyUnicode_InternFromString("layout");
  if (!layout_attr)
    goto cleanup;
  has_native_tensor_spec_attr =
      PyUnicode_InternFromString("supports_native_tensor_specialization");
  if (!has_native_tensor_spec_attr)
    goto cleanup;
  get_tensor_spec_attr =
      PyUnicode_InternFromString("get_tensor_specialization");
  if (!get_tensor_spec_attr)
    goto cleanup;
  align_kwarg_str = PyUnicode_InternFromString("align");
  if (!align_kwarg_str)
    goto cleanup;
  align_kwarg = PyTuple_Pack(1, align_kwarg_str);
  Py_DECREF(align_kwarg_str);
  if (!align_kwarg)
    goto cleanup;

  return true;

cleanup:
  Py_CLEAR(align_kwarg_str);
  Py_CLEAR(align_kwarg);
  return false;
}

static void init_type_handler_cache();

static bool init_globals() {
  if (init_called) {
    return true;
  }

  PyObject *m_canonicalize = nullptr, *m_jit = nullptr, *m_desc = nullptr,
           *m_desc_gluon = nullptr, *m_constexpr = nullptr, *m_torch = nullptr;
  bool success = false;

  // import modules
  m_canonicalize = PyImport_ImportModule("triton._utils");
  if (!m_canonicalize)
    goto cleanup;

  m_jit = PyImport_ImportModule("triton.runtime.jit");
  if (!m_jit)
    goto cleanup;

  m_desc = PyImport_ImportModule("triton.tools.tensor_descriptor");
  if (!m_desc)
    goto cleanup;

  m_desc_gluon =
      PyImport_ImportModule("triton.experimental.gluon.nvidia.hopper");
  if (!m_desc_gluon)
    goto cleanup;

  m_constexpr = PyImport_ImportModule("triton.language");
  if (!m_constexpr)
    goto cleanup;

  m_torch = PyImport_ImportModule("torch");
  if (m_torch) {
    torch_tensor_cls = PyObject_GetAttrString(m_torch, "Tensor");
  } else {
    PyErr_Clear();
  }

  // get symbols from modules
  jit_function_cls = PyObject_GetAttrString(m_jit, "JITFunction");
  if (!jit_function_cls)
    goto cleanup;
  tensor_descriptor_cls = PyObject_GetAttrString(m_desc, "TensorDescriptor");
  if (!tensor_descriptor_cls)
    goto cleanup;
  gluon_tensor_descriptor_cls =
      PyObject_GetAttrString(m_desc_gluon, "TensorDescriptor");
  if (!gluon_tensor_descriptor_cls)
    goto cleanup;
  canonicalize_dtype_fn =
      PyObject_GetAttrString(m_canonicalize, "canonicalize_dtype");
  if (!canonicalize_dtype_fn)
    goto cleanup;
  canonicalize_ptr_dtype_fn =
      PyObject_GetAttrString(m_canonicalize, "canonicalize_ptr_dtype");
  if (!canonicalize_ptr_dtype_fn)
    goto cleanup;
  constexpr_cls = PyObject_GetAttrString(m_constexpr, "constexpr");
  if (!constexpr_cls)
    goto cleanup;

  // decref modules as we just needed them to import symbols above
  Py_CLEAR(m_jit);
  Py_CLEAR(m_desc);
  Py_CLEAR(m_desc_gluon);
  Py_CLEAR(m_canonicalize);
  Py_CLEAR(m_constexpr);
  Py_CLEAR(m_torch);

  success = init_interned_strings();
  if (!success) {
    goto cleanup;
  }
  init_type_handler_cache();

  init_called = true;

  return true;

cleanup:
  Py_CLEAR(m_canonicalize);
  Py_CLEAR(m_jit);
  Py_CLEAR(m_desc);
  Py_CLEAR(m_desc_gluon);
  Py_CLEAR(m_constexpr);
  Py_CLEAR(m_torch);
  return false;
}

static inline std::pair<py::object, py::object>
specialize_tensordesc(PyObject *arg, bool has_layout) {
  // no DECREF on these as they are cached internally or returned
  PyObject *res = nullptr, *type_str = nullptr, *type_str_result = nullptr;
  // normal temporary objects
  PyObject *base = nullptr, *dtype = nullptr, *dtype_str = nullptr;
  PyObject *block_shape_obj = nullptr, *block_shape_list = nullptr,
           *block_shape_str = nullptr;
  PyObject *layout_obj = nullptr, *layout_repr = nullptr;

  Py_hash_t dtype_hash = -1;
  DTypeKey dsk{-1};
  Dtype2Str::iterator it;
  TensorDescCache::iterator cache_it;

  std::string desc_cstr;
  desc_cstr.reserve(128);
  const char *dtype_cstr = nullptr;
  const char *block_shape_cstr = nullptr;
  const char *layout_cstr = nullptr;

  base = PyObject_GetAttr(arg, base_attr);
  if (!base)
    goto cleanup;

  dtype = PyObject_GetAttr(base, dtype_attr);
  if (!dtype)
    goto cleanup;

  dtype_hash = PyObject_Hash(dtype);
  if (dtype_hash == -1)
    goto cleanup;
  dsk = dtype_hash;
  it = dtype2str.find(dsk);
  if (it != dtype2str.end()) {
    type_str = it->second;
  } else {
    res = PyObject_CallFunctionObjArgs(canonicalize_dtype_fn, dtype, nullptr);
    if (!res) {
      goto cleanup;
    }
    dtype2str[dsk] = res;
    type_str = res;
  }

  desc_cstr = "tensordesc<";
  dtype_str = PyObject_Str(type_str);
  if (!dtype_str)
    goto cleanup;

  dtype_cstr = PyUnicode_AsUTF8(dtype_str);
  if (!dtype_cstr)
    goto cleanup;
  desc_cstr += dtype_cstr;

  block_shape_obj = PyObject_GetAttr(arg, block_shape_attr);
  if (!block_shape_obj)
    goto cleanup;
  block_shape_list = PySequence_List(block_shape_obj);
  if (!block_shape_list)
    goto cleanup;
  block_shape_str = PyObject_Str(block_shape_list);
  if (!block_shape_str)
    goto cleanup;
  block_shape_cstr = PyUnicode_AsUTF8(block_shape_str);
  if (!block_shape_cstr)
    goto cleanup;
  desc_cstr += block_shape_cstr;

  if (has_layout) {
    layout_obj = PyObject_GetAttr(arg, layout_attr);
    if (!layout_obj)
      goto cleanup;
    layout_repr = PyObject_Repr(layout_obj);
    if (!layout_repr)
      goto cleanup;
    desc_cstr += ",";
    layout_cstr = PyUnicode_AsUTF8(layout_repr);
    if (!layout_cstr)
      goto cleanup;
    desc_cstr += layout_cstr;
  }

  desc_cstr += ">";
  type_str_result = PyUnicode_FromString(desc_cstr.c_str());
  if (!type_str_result)
    goto cleanup;

  Py_DECREF(base);
  Py_DECREF(dtype);
  Py_DECREF(dtype_str);
  Py_DECREF(block_shape_obj);
  Py_DECREF(block_shape_list);
  Py_DECREF(block_shape_str);
  if (has_layout) {
    Py_DECREF(layout_obj);
    Py_DECREF(layout_repr);
  }

  return {py::reinterpret_steal<py::object>(type_str_result), py::none()};

cleanup:
  Py_XDECREF(base);
  Py_XDECREF(dtype);
  Py_XDECREF(dtype_str);
  Py_XDECREF(block_shape_obj);
  Py_XDECREF(block_shape_list);
  Py_XDECREF(block_shape_str);
  Py_XDECREF(layout_obj);
  Py_XDECREF(layout_repr);
  return {py::object(), py::object()};
}

static std::pair<py::object, py::object>
handle_long_type(PyObject *backend, PyObject *arg, bool is_const,
                 bool specialize_value, bool align) {
  PyObject *type_str;
  PyObject *key_obj = nullptr;
  py::object key = py::none();

  int overflow;
  long long val = PyLong_AsLongLongAndOverflow(arg, &overflow);
  if (PyErr_Occurred()) {
    return {py::object(), py::object()};
  }

  if (specialize_value && (val == 1)) {
    return {py::reinterpret_borrow<py::object>(constexpr_str),
            py::reinterpret_borrow<py::object>(arg)};
  }

  if (overflow == 0) {
    type_str = (val >= INT32_MIN && val <= INT32_MAX) ? i32_str : i64_str;
    if (specialize_value) {
      key_obj = (align && ((val & 15) == 0)) ? D_str : empty_str;
      key = py::reinterpret_borrow<py::object>(key_obj);
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
      return {py::object(), py::object()};
    } else {
      type_str = u64_str;
      if (specialize_value) {
        key_obj = (align && ((val_64 & 15) == 0)) ? D_str : empty_str;
        key = py::reinterpret_borrow<py::object>(key_obj);
      }
    }
  }

  return {py::reinterpret_borrow<py::object>(type_str), key};
}

static std::pair<py::object, py::object>
handle_tensor(PyObject *backend, PyObject *arg, bool is_const,
              bool specialize_value, bool align) {
  // no DECREF on these as they are used for objects we cache or return them
  PyObject *canon_res = nullptr, *type_str = nullptr, *key_res = nullptr;
  // normal temporary objects
  PyObject *dtype = nullptr, *native_spec_obj = nullptr;
  PyObject *data_ptr_result = nullptr;

  PyObject *key_obj = nullptr;
  py::object key;
  Py_hash_t dtype_hash = -1;
  DTypePtrKey dsk{0, false};
  DtypePtr2Str::iterator it;
  bool native_impl_available = false;
  uint64_t data_ptr = 0;

  // handle type_str specialization of a tensor
  dtype = PyObject_GetAttr(arg, dtype_attr);
  if (!dtype)
    goto cleanup;

  dtype_hash = PyObject_Hash(dtype);
  if (dtype_hash == -1)
    goto cleanup;

  dsk = {dtype_hash, is_const};
  it = dtype_ptr2str.find(dsk);

  if (it != dtype_ptr2str.end()) {
    type_str = it->second;
  } else {
    canon_res =
        PyObject_CallFunctionObjArgs(canonicalize_ptr_dtype_fn, dtype,
                                     is_const ? Py_True : Py_False, nullptr);
    if (!canon_res)
      goto cleanup;
    dtype_ptr2str[dsk] = canon_res;
    type_str = canon_res;
  }

  // handle alignment specialization of a tensor
  if (!specialize_value) {
    Py_DECREF(dtype);
    return {py::reinterpret_borrow<py::object>(type_str), py::none()};
  }

  native_spec_obj = PyObject_GetAttr(backend, has_native_tensor_spec_attr);
  if (native_spec_obj) {
    native_impl_available = PyObject_IsTrue(native_spec_obj);
  } else {
    PyErr_Clear();
    // on error we fall back to native_impl_available = false gracefully
  }

  if (native_impl_available) {
    data_ptr_result = PyObject_CallMethodNoArgs(arg, data_ptr_attr);
    if (!data_ptr_result)
      goto cleanup;

    data_ptr = PyLong_AsUnsignedLongLong(data_ptr_result);
    if (PyErr_Occurred())
      goto cleanup;

    key_obj = (align && ((data_ptr & 15) == 0)) ? D_str : empty_str;
    key = py::reinterpret_borrow<py::object>(key_obj);
  } else {
    PyObject *args[3] = {backend, arg, align ? Py_True : Py_False};
    PyObject *kwnames = align_kwarg;
    key_res = PyObject_VectorcallMethod(get_tensor_spec_attr, args, 2, kwnames);
    if (key_res) {
      key = py::reinterpret_steal<py::object>(key_res);
    } else {
      goto cleanup;
    }
  }

  Py_DECREF(dtype);
  Py_XDECREF(native_spec_obj);
  if (native_impl_available) {
    Py_DECREF(data_ptr_result);
  }
  return {py::reinterpret_borrow<py::object>(type_str), key};

cleanup:
  Py_XDECREF(dtype);
  Py_XDECREF(native_spec_obj);
  Py_XDECREF(data_ptr_result);
  return {py::object(), py::object()};
}

static std::pair<py::object, py::object>
handle_bool_type(PyObject *backend, PyObject *arg, bool is_const,
                 bool specialize_value, bool align) {
  return {py::reinterpret_borrow<py::object>(u1_str), py::none()};
}

static std::pair<py::object, py::object>
handle_float_type(PyObject *backend, PyObject *arg, bool is_const,
                  bool specialize_value, bool align) {
  return {py::reinterpret_borrow<py::object>(fp32_str), py::none()};
}

static std::pair<py::object, py::object>
handle_tensor_descriptor(PyObject *backend, PyObject *arg, bool is_const,
                         bool specialize_value, bool align) {
  return specialize_tensordesc(arg, false);
}

static std::pair<py::object, py::object>
handle_gluon_tensor_descriptor(PyObject *backend, PyObject *arg, bool is_const,
                               bool specialize_value, bool align) {
  return specialize_tensordesc(arg, true);
}

static std::pair<py::object, py::object>
handle_constexpr_type(PyObject *backend, PyObject *arg, bool is_const,
                      bool specialize_value, bool align) {
  return {py::reinterpret_borrow<py::object>(constexpr_str),
          py::reinterpret_borrow<py::object>(arg)};
}

static std::pair<py::object, py::object>
handle_jit_function(PyObject *backend, PyObject *arg, bool is_const,
                    bool specialize_value, bool align) {
  PyObject *cache_key = PyObject_GetAttr(arg, cache_key_attr);
  if (!cache_key) {
    return {py::object(), py::object()};
  }
  return {py::reinterpret_borrow<py::object>(constexpr_str),
          py::reinterpret_borrow<py::object>(cache_key)};
}

static std::pair<py::object, py::object>
handle_tuple(PyObject *backend, PyObject *arg, bool is_const,
             bool specialize_value, bool align) {
  Py_ssize_t size = PyTuple_GET_SIZE(arg);
  if (size == 0) {
    // return tuple of empty tuples as in python reference
    return {py::reinterpret_borrow<py::object>(arg),
            py::reinterpret_borrow<py::object>(arg)};
  }

  PyTypeObject *tuple_type = nullptr;
  PyObject *item = nullptr;
  PyObject *tys_tuple = nullptr, *keys_tuple = nullptr;
  // normal temporary objects for namedtuple case only
  PyObject *norm_tys_tuple = nullptr, *norm_keys_tuple = nullptr;
  std::vector<py::object> types(size);
  std::vector<py::object> keys(size);

  bool is_namedtuple = PyObject_HasAttr(arg, _fields_attr);
  tuple_type = Py_TYPE(arg);

  // Create tuples directly instead of lists
  tys_tuple = PyTuple_New(size);
  keys_tuple = PyTuple_New(size);

  if (!tys_tuple || !keys_tuple)
    goto cleanup;

  for (Py_ssize_t i = 0; i < size; ++i) {
    item = PyTuple_GET_ITEM(arg, i);
    // python reference calls specialize recursively with default arguments set
    // currently this is is_const=False, specialize_value=True, align=True
    auto [type, key] = specialize_arg(backend, item, false, true, true);
    if (PyErr_Occurred())
      goto cleanup;
    // check if specialization failed
    if (!type || !key)
      goto cleanup;
    types[i] = type;
    keys[i] = key;
  }

  for (Py_ssize_t i = 0; i < size; ++i) {
    PyTuple_SetItem(tys_tuple, i, types[i].inc_ref().ptr());
    PyTuple_SetItem(keys_tuple, i, keys[i].inc_ref().ptr());
  }

  keys.clear();
  types.clear();

  if (is_namedtuple) {
    norm_tys_tuple = tys_tuple;
    norm_keys_tuple = keys_tuple;

    tys_tuple = PyObject_CallObject((PyObject *)tuple_type, norm_tys_tuple);
    keys_tuple = PyObject_CallObject((PyObject *)tuple_type, norm_keys_tuple);
    if (!tys_tuple || !keys_tuple)
      goto cleanup;
    Py_DECREF(norm_tys_tuple);
    Py_DECREF(norm_keys_tuple);
  }

  return {py::reinterpret_steal<py::object>(tys_tuple),
          py::reinterpret_steal<py::object>(keys_tuple)};

cleanup:
  // cleanup potential left-overs
  for (auto &type : types) {
    Py_XDECREF(type.ptr());
  }
  types.clear();
  for (auto &key : keys) {
    Py_XDECREF(key.ptr());
  }
  keys.clear();
  Py_XDECREF(norm_tys_tuple);
  Py_XDECREF(norm_keys_tuple);
  Py_XDECREF(tys_tuple);
  Py_XDECREF(keys_tuple);
  return {py::object(), py::object()};
}

// initialize type handler which returns specialize impelemntations based on
// type(arg)
static void init_type_handler_cache() {
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
  // JITFunction
  if (jit_function_cls && PyType_Check(jit_function_cls)) {
    type_handler_cache[(PyTypeObject *)jit_function_cls] = handle_jit_function;
  }
}

// specialization logic without passing of objects from Python (to be called in
// specialize_impl only)
static std::pair<py::object, py::object>
specialize_arg(PyObject *backend, PyObject *arg, bool is_const,
               bool specialize_value, bool align) {
  // fast-path for default types
  PyTypeObject *arg_type = Py_TYPE(arg);
  auto it = type_handler_cache.find(arg_type);
  if (it != type_handler_cache.end()) {
    return it->second(backend, arg, is_const, specialize_value, align);
  }

  // separate handling of None
  if (Py_IsNone(arg)) {
    return {py::reinterpret_borrow<py::object>(constexpr_str), py::none()};
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

  if (PyObject_IsInstance(arg, jit_function_cls)) {
    return handle_jit_function(backend, arg, is_const, specialize_value, align);
  }

  // fallback paths checking attributes directly
  if (PyObject_HasAttr(arg, data_ptr_attr)) {
    return handle_tensor(backend, arg, is_const, specialize_value, align);
  }

  if (PyObject_HasAttr(arg, tma_desc_cpu_ptr_attr)) {
    return {py::reinterpret_borrow<py::object>(nvTmaDesc_str), py::none()};
  }

  return {py::object(), py::object()};
}

// main entry-point from Python implementing specialization logic natively
static PyObject *specialize_impl(PyObject *self, PyObject *const *args,
                                 Py_ssize_t nargs) {
  bool success = init_globals();
  if (!success) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(
          PyExc_RuntimeError,
          "failed to initialize __triton_specialize module objects");
    }
    return nullptr;
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
