#include <Python.h>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <pybind11/pybind11.h>

namespace {

namespace py = pybind11;

using DTypePtrKey = std::pair<Py_hash_t, bool>;
using DTypeKey = Py_hash_t;

struct DTypePtrKeyHash {
  std::size_t operator()(const DTypePtrKey& k) const {
    return std::hash<Py_hash_t>()(k.first) ^ (std::hash<bool>()(k.second) << 1);
  }
};

using DtypePtr2Str = std::unordered_map<DTypePtrKey, PyObject *, DTypePtrKeyHash>;
using Dtype2Str = std::unordered_map<DTypeKey, PyObject *>;
using TensorDescCache = std::unordered_map<std::string, PyObject *>;

using TypeHandler = std::pair<py::handle, py::handle>(*)(PyObject*, PyObject*, bool, bool, bool);
using TypeHandlerCache = std::unordered_map<PyTypeObject*, TypeHandler>;

std::pair<py::handle, py::handle> specialize_arg(PyObject *backend, PyObject *arg, bool is_const, bool specialize_value, bool align);

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

static DtypePtr2Str dtype_ptr2str;
static Dtype2Str dtype2str;
static TensorDescCache tensor_desc_cache;
static TypeHandlerCache type_handler_cache;

static bool lazy_init = false;

inline void xdecref_static() {}

template <typename T>
inline void xdecref_static(T& obj) {
  if (obj) {
    Py_XDECREF(obj);
    obj = nullptr;
  }
}

template<typename T, typename... Args>
inline void xdecref_static(T& obj, Args... rest) {
    if (obj) {
        Py_XDECREF(obj);
        obj = nullptr;
    }
    xdecref_static(rest...);
}

static void cleanup_lazy_str() {
  xdecref_static(jit_function_cls, tensor_descriptor_cls, gluon_tensor_descriptor_cls,
                 canonicalize_dtype_fn, canonicalize_ptr_dtype_fn, constexpr_cls, torch_tensor_cls);
}

static void cleanup_static_str() {
  xdecref_static(i32_str, i64_str, u64_str, fp32_str, u1_str, D_str,
                 constexpr_str, empty_str, nvTmaDesc_str);
  xdecref_static(data_ptr_attr, dtype_attr, cache_key_attr, tma_desc_cpu_ptr_attr, 
              _fields_attr, block_shape_attr, layout_attr, has_native_tensor_spec_attr, get_tensor_spec_attr);
}

static void init_type_handler_cache();

static void init_lazy() {
  if (!lazy_init) {
    PyObject *m_canonicalize = PyImport_ImportModule("triton._utils");
    if (!m_canonicalize) {
      return;
    }

    PyObject *m_jit = PyImport_ImportModule("triton.runtime.jit");
    if (!m_jit) {
      Py_XDECREF(m_canonicalize);
      return;
    }

    PyObject *m_desc = PyImport_ImportModule("triton.tools.tensor_descriptor");
    if (!m_desc) {
      Py_XDECREF(m_canonicalize);
      Py_XDECREF(m_jit);
      return;
    }

    PyObject *m_desc_gluon = PyImport_ImportModule("triton.experimental.gluon.nvidia.hopper");
    if (!m_desc_gluon) {
      Py_XDECREF(m_canonicalize);
      Py_XDECREF(m_jit);
      Py_XDECREF(m_desc);
      return;
    }

    PyObject *m_constexpr = PyImport_ImportModule("triton.language");
    if (!m_constexpr) {
      Py_XDECREF(m_canonicalize);
      Py_XDECREF(m_jit);
      Py_XDECREF(m_desc);
      Py_XDECREF(m_desc_gluon);
      return;
    }

    PyObject *m_torch = PyImport_ImportModule("torch");

    jit_function_cls = PyObject_GetAttrString(m_jit, "JITFunction");
    tensor_descriptor_cls = PyObject_GetAttrString(m_desc, "TensorDescriptor");
    gluon_tensor_descriptor_cls =
        PyObject_GetAttrString(m_desc_gluon, "TensorDescriptor");
    canonicalize_dtype_fn =
        PyObject_GetAttrString(m_canonicalize, "canonicalize_dtype");
    canonicalize_ptr_dtype_fn =
        PyObject_GetAttrString(m_canonicalize, "canonicalize_ptr_dtype");
    constexpr_cls = PyObject_GetAttrString(m_constexpr, "constexpr");
    if (m_torch) {
      torch_tensor_cls = PyObject_GetAttrString(m_torch, "Tensor");
    }

    // immediately decref modules as we just needed them to import symbols above
    Py_XDECREF(m_canonicalize);
    Py_XDECREF(m_jit);
    Py_XDECREF(m_desc);
    Py_XDECREF(m_desc_gluon);
    Py_XDECREF(m_constexpr);
    if (m_torch) {
      Py_XDECREF(m_torch);
    }

    if (!jit_function_cls || !tensor_descriptor_cls || !gluon_tensor_descriptor_cls || !canonicalize_dtype_fn || !canonicalize_ptr_dtype_fn || !constexpr_cls) {
      cleanup_lazy_str();
      return;
    }

    init_type_handler_cache();

    lazy_init = true;
  }
}

static void init_static_str() {
  i32_str = PyUnicode_InternFromString("i32");
  i64_str = PyUnicode_InternFromString("i64");
  u64_str = PyUnicode_InternFromString("u64");
  fp32_str = PyUnicode_InternFromString("fp32");
  u1_str = PyUnicode_InternFromString("u1");
  D_str = PyUnicode_InternFromString("D");
  constexpr_str = PyUnicode_InternFromString("constexpr");
  empty_str = PyUnicode_InternFromString("");
  nvTmaDesc_str = PyUnicode_InternFromString("nvTmaDesc");

  base_attr = PyUnicode_InternFromString("base");
  data_ptr_attr = PyUnicode_InternFromString("data_ptr");
  dtype_attr = PyUnicode_InternFromString("dtype");
  cache_key_attr = PyUnicode_InternFromString("cache_key");
  tma_desc_cpu_ptr_attr = PyUnicode_InternFromString("tma_desc_cpu_ptr");
  _fields_attr = PyUnicode_InternFromString("_fields");
  block_shape_attr = PyUnicode_InternFromString("block_shape");
  layout_attr = PyUnicode_InternFromString("layout");
  has_native_tensor_spec_attr = PyUnicode_InternFromString("supports_native_tensor_specialization");
  get_tensor_spec_attr = PyUnicode_InternFromString("get_tensor_specialization");
  
  if (!i32_str || !i64_str || !u64_str || !fp32_str || !u1_str || !D_str || 
      !constexpr_str || !empty_str || !nvTmaDesc_str || !data_ptr_attr || 
      !dtype_attr || !cache_key_attr || !tma_desc_cpu_ptr_attr || !base_attr ||
      !_fields_attr || !block_shape_attr || !layout_attr || !get_tensor_spec_attr || !has_native_tensor_spec_attr) {
    if (!PyErr_Occurred()){
      PyErr_SetString(PyExc_RuntimeError, "failed to initialize __triton_specialize module objects");
    }
    cleanup_static_str();
  }
}

static void module_cleanup() {
  for (auto& pair : dtype_ptr2str) {
    Py_XDECREF(pair.second);
  }
  for (auto& pair : dtype2str) {
    Py_XDECREF(pair.second);
  }
  for (auto& pair : tensor_desc_cache) {
    Py_XDECREF(pair.second);
  }
  dtype_ptr2str.clear();
  dtype2str.clear();
  tensor_desc_cache.clear();
  type_handler_cache.clear();

  cleanup_lazy_str();
  cleanup_static_str();
}

static inline std::pair<py::handle, py::handle>
specialize_int(PyObject *arg, bool specialize_value, bool align) {
  PyObject *type_str;
  PyObject *key = nullptr;

  int overflow;
  long long val = PyLong_AsLongLongAndOverflow(arg, &overflow);
  if (PyErr_Occurred()) {
    PyErr_Clear();
    return {py::handle(), py::handle()};
  }

  if (specialize_value && (val == 1)) {
    return {py::handle(constexpr_str), py::handle(arg)};
  }

  if (overflow == 0) {
    type_str = (val >= INT32_MIN && val <= INT32_MAX) ? i32_str : i64_str;
    if (specialize_value) {
      key = (align && ((val & 15) == 0)) ? D_str : empty_str;
    }
  } else {
    unsigned long long val_64 = PyLong_AsUnsignedLongLong(arg);
    type_str = u64_str;
    if (specialize_value) {
      key = (align && ((val_64 & 15) == 0)) ? D_str : empty_str;
    }
  }

  return {py::handle(type_str), py::handle(key)};
}

static inline py::handle
specialize_tensor_align(PyObject *tensor, bool specialize_value, bool align) {
  if (!specialize_value) {
    return py::handle();
  }

  PyObject *data_ptr_method = PyObject_GetAttr(tensor, data_ptr_attr);
  if (!data_ptr_method) {
    PyErr_Clear();
    return py::handle();
  }

  PyObject *data_ptr_result = PyObject_CallNoArgs(data_ptr_method);
  Py_DECREF(data_ptr_method);
  if (!data_ptr_result) {
    PyErr_Clear();
    return py::handle();
  }

  uint64_t data_ptr = PyLong_AsUnsignedLongLong(data_ptr_result);
  Py_DECREF(data_ptr_result);
  if (PyErr_Occurred()) {
    PyErr_Clear();
    return py::handle();
  }

  PyObject *result = (align && ((data_ptr & 15) == 0)) ? D_str : empty_str;

  return py::handle(result);
}

static inline std::pair<py::handle, py::handle>
specialize_tensor(PyObject *backend, PyObject *arg, bool is_const, bool specialize_value, bool align) {
  PyObject *dtype = PyObject_GetAttr(arg, dtype_attr);
  if (!dtype) {
    PyErr_Clear();
    return {py::handle(), py::handle()};
  }

  Py_hash_t dtype_hash = PyObject_Hash(dtype);
  if (dtype_hash == -1) {
    Py_DECREF(dtype);
    PyErr_Clear();
    return {py::handle(), py::handle()};
  }

  DTypePtrKey dsk{dtype_hash, is_const};
  auto it = dtype_ptr2str.find(dsk);

  PyObject *type_str;
  if (it != dtype_ptr2str.end()) {
    type_str = it->second;
  } else {
    PyObject *result = PyObject_CallFunctionObjArgs(
        canonicalize_ptr_dtype_fn, dtype, is_const ? Py_True : Py_False, nullptr);
    if (!result) {
      Py_DECREF(dtype);
      PyErr_Clear();
      return {py::handle(), py::handle()};
    }
    Py_INCREF(result);
    dtype_ptr2str[dsk] = result;
    type_str = result;
  }

  Py_DECREF(dtype);
  py::handle key;
  PyObject *native_spec_obj = PyObject_GetAttr(backend, has_native_tensor_spec_attr);
  bool native_impl_available = false;
  if (native_spec_obj) {
    native_impl_available = PyObject_IsTrue(native_spec_obj);
    Py_DECREF(native_spec_obj);
  } else {
    PyErr_Clear();
  }
  if (native_impl_available) {
    key = specialize_tensor_align(arg, specialize_value, align);
  } else {
    // call fallback
    PyObject* fallback_method = PyObject_GetAttr(backend, get_tensor_spec_attr);
    if (fallback_method) {
      PyObject *kwargs = PyDict_New();
      PyObject *args_tuple = PyTuple_Pack(1, arg);
      if (kwargs && args_tuple) {
        if (align) {
          PyDict_SetItemString(kwargs, "align", Py_True);
        }
        PyObject* res = PyObject_Call(fallback_method, args_tuple, kwargs);
        if (res) {
          key = py::handle(res);
        } else {
          PyErr_Clear();
          Py_DECREF(args_tuple);
          Py_DECREF(kwargs);
          return {py::handle(), py::handle()};
        }
        Py_DECREF(args_tuple);
        Py_DECREF(kwargs);
      } else {
        PyErr_Clear();
        Py_XDECREF(args_tuple);
        Py_XDECREF(kwargs);
      }
      Py_DECREF(fallback_method);
    } else {
      PyErr_Clear();
      return {py::handle(), py::handle()};
    }
  }

  return {py::handle(type_str), key};
}

static inline std::pair<py::handle, py::handle>
specialize_tensordesc(PyObject *arg, bool has_layout) {
  PyObject *base = PyObject_GetAttr(arg, base_attr);
  if (!base) {
    PyErr_Clear();
    return {py::handle(), py::handle()};
  }
  PyObject *dtype = PyObject_GetAttr(base, dtype_attr);
  if (!dtype) {
    Py_DECREF(base);
    PyErr_Clear();
    return {py::handle(), py::handle()};
  }
  Py_DECREF(base);

  Py_hash_t dtype_hash = PyObject_Hash(dtype);
  if (dtype_hash == -1) {
    Py_DECREF(dtype);
    PyErr_Clear();
    return {py::handle(), py::handle()};
  }
  DTypeKey dsk{dtype_hash};
  auto it = dtype2str.find(dsk);
  PyObject *type_str;
  if (it != dtype2str.end()) {
    type_str = it->second;
  } else {
    PyObject *res = PyObject_CallFunctionObjArgs(canonicalize_dtype_fn, dtype, nullptr);
    if (!res) {
      Py_DECREF(dtype);
      PyErr_Clear();
      return {py::handle(), py::handle()};
    }
    Py_INCREF(res);
    dtype2str[dsk] = res;
    type_str = res;
  }
  Py_DECREF(dtype);

  std::string cache_key;
  cache_key.reserve(128);
  cache_key = "tensordesc<";
  PyObject *dtype_str = PyObject_Str(type_str);
  if (!dtype_str) {
    PyErr_Clear();
    return {py::handle(), py::handle()};
  }
  const char *dtype_cstr = PyUnicode_AsUTF8(dtype_str);
  if (!dtype_cstr) {
    PyErr_Clear();
    Py_DECREF(dtype_str);
    return {py::handle(), py::handle()};
  }
  cache_key += dtype_cstr;
  Py_DECREF(dtype_str);

  PyObject *block_shape_obj = PyObject_GetAttr(arg, block_shape_attr);
  if (!block_shape_obj) {
    PyErr_Clear();
    return {py::handle(), py::handle()};
  }
  PyObject *block_shape_list = PySequence_List(block_shape_obj);
  Py_DECREF(block_shape_obj);
  if (!block_shape_list) {
    PyErr_Clear();
    return {py::handle(), py::handle()};
  }
  PyObject *block_shape_str = PyObject_Str(block_shape_list);
  Py_DECREF(block_shape_list);
  if (!block_shape_str) {
    PyErr_Clear();
    return {py::handle(), py::handle()};
  }
  const char* block_shape_cstr = PyUnicode_AsUTF8(block_shape_str);
  if (!block_shape_cstr) {
    Py_DECREF(block_shape_str);
    PyErr_Clear();
    return {py::handle(), py::handle()};
  }
  cache_key += block_shape_cstr;
  Py_DECREF(block_shape_str);

  if (has_layout) {
    PyObject *layout_obj = PyObject_GetAttr(arg, layout_attr);
    if (!layout_obj) {
      PyErr_Clear();
      return {py::handle(), py::handle()};
    }
    PyObject *layout_repr = PyObject_Repr(layout_obj);
    Py_DECREF(layout_obj);
    if (!layout_repr) {
      PyErr_Clear();
      return {py::handle(), py::handle()};
    }
    cache_key += ",";
    const char* layout_cstr = PyUnicode_AsUTF8(layout_repr);
    if (!layout_cstr) {
      PyErr_Clear();
      Py_DECREF(layout_repr);
      return {py::handle(), py::handle()};
    }
    cache_key += layout_cstr;
    Py_DECREF(layout_repr);
  }

  cache_key += ">";

  auto cache_it = tensor_desc_cache.find(cache_key);
  if (cache_it != tensor_desc_cache.end()) {
    return {py::handle(cache_it->second), py::handle()};
  }

  PyObject *type_str_result = PyUnicode_FromString(cache_key.c_str());
  if (!type_str_result) {
    PyErr_Clear();
    return {py::handle(), py::handle()};
  }

  tensor_desc_cache[cache_key] = type_str_result;
  return {py::handle(type_str_result), py::handle()};
}

static std::pair<py::handle, py::handle> handle_long_type(PyObject* backend, PyObject *arg, bool is_const, bool specialize_value, bool align) {
  return specialize_int(arg, specialize_value, align);
}

static std::pair<py::handle, py::handle> handle_tensor(PyObject* backend, PyObject *arg, bool is_const, bool specialize_value, bool align) {
  return specialize_tensor(backend, arg, is_const, specialize_value, align);
}

static std::pair<py::handle, py::handle> handle_bool_type(PyObject* backend, PyObject *arg, bool is_const, bool specialize_value, bool align) {
  return {py::handle(u1_str), py::handle()};
}

static std::pair<py::handle, py::handle> handle_float_type(PyObject* backend, PyObject *arg, bool is_const, bool specialize_value, bool align) {
  return {py::handle(fp32_str), py::handle()};
}

static std::pair<py::handle, py::handle> handle_tensor_descriptor(PyObject* backend, PyObject *arg, bool is_const, bool specialize_value, bool align) {
  return specialize_tensordesc(arg, false);
}

static std::pair<py::handle, py::handle> handle_gluon_tensor_descriptor(PyObject* backend, PyObject *arg, bool is_const, bool specialize_value, bool align) {
  return specialize_tensordesc(arg, true);
}

static std::pair<py::handle, py::handle> handle_constexpr_type(PyObject* backend, PyObject *arg, bool is_const, bool specialize_value, bool align) {
  return {py::handle(constexpr_str), py::handle(arg)};
}

static std::pair<py::handle, py::handle> handle_jit_function(PyObject* backend, PyObject *arg, bool is_const, bool specialize_value, bool align) {
  PyObject *cache_key = PyObject_GetAttr(arg, cache_key_attr);
  if (!cache_key) {
    PyErr_Clear();
    return {py::handle(constexpr_str), py::handle()};
  }
  py::handle key_handle(cache_key);
  Py_DECREF(cache_key);
  return {py::handle(constexpr_str), key_handle};
}

static std::pair<py::handle, py::handle> handle_tuple(PyObject* backend, PyObject *arg, bool is_const, bool specialize_value, bool align) {
  Py_ssize_t size = PyTuple_GET_SIZE(arg);
  if (size == 0) {
    // return tuple of empty tuples as in python reference
    return {py::handle(arg), py::handle(arg)};
  }

  bool is_namedtuple = PyObject_HasAttr(arg, _fields_attr);
  PyTypeObject *tuple_type = Py_TYPE(arg);

  PyObject *tys_list = PyList_New(size);
  PyObject *keys_list = PyList_New(size);

  if (!tys_list || !keys_list) {
    Py_XDECREF(tys_list);
    Py_XDECREF(keys_list);
    PyErr_Clear();
    return {py::handle(), py::handle()};
  }

  bool all_success = true;
  for (Py_ssize_t i = 0; i < size; ++i) {
    PyObject *item = PyTuple_GET_ITEM(arg, i);
    auto [type, key] = specialize_arg(backend, item, is_const, specialize_value, align);
    if (!type.ptr() && !key.ptr()) {
      all_success = false;
      break;
    }

    PyObject *type_obj = type.ptr() ? type.ptr() : Py_None;
    PyObject *key_obj = key.ptr() ? key.ptr() : Py_None;

    Py_INCREF(type_obj);
    Py_INCREF(key_obj);
    PyList_SET_ITEM(tys_list, i, type_obj);
    PyList_SET_ITEM(keys_list, i, key_obj);
  }

  if (!all_success) {
    Py_DECREF(tys_list);
    Py_DECREF(keys_list);
    return {py::handle(), py::handle()};
  }

  PyObject *tys_tuple = nullptr;
  PyObject *keys_tuple = nullptr;

  if (is_namedtuple) {
    PyObject *tys_args_tuple = PyList_AsTuple(tys_list);
    PyObject *keys_args_tuple = PyList_AsTuple(keys_list);
    
    if (tys_args_tuple && keys_args_tuple) {
      tys_tuple = PyObject_CallObject((PyObject*)tuple_type, tys_args_tuple);
      keys_tuple = PyObject_CallObject((PyObject*)tuple_type, keys_args_tuple);
    }
    
    Py_XDECREF(tys_args_tuple);
    Py_XDECREF(keys_args_tuple);
  } else {
    tys_tuple = PyList_AsTuple(tys_list);
    keys_tuple = PyList_AsTuple(keys_list);
  }

  Py_DECREF(tys_list);
  Py_DECREF(keys_list);

  if (!tys_tuple || !keys_tuple) {
    Py_XDECREF(tys_tuple);
    Py_XDECREF(keys_tuple);
    PyErr_Clear();
    return {py::handle(), py::handle()};
  }

  return {py::handle(tys_tuple), py::handle(keys_tuple)};
}

static void init_type_handler_cache() {
  type_handler_cache.clear();
  type_handler_cache[&PyLong_Type] = handle_long_type;
  type_handler_cache[&PyBool_Type] = handle_bool_type;
  type_handler_cache[&PyFloat_Type] = handle_float_type;
  type_handler_cache[&PyTuple_Type] = handle_tuple;

  if (torch_tensor_cls && PyType_Check(torch_tensor_cls)) {
    type_handler_cache[(PyTypeObject*)torch_tensor_cls] = handle_tensor;
  }
  if (tensor_descriptor_cls && PyType_Check(tensor_descriptor_cls)) {
    type_handler_cache[(PyTypeObject*)tensor_descriptor_cls] = handle_tensor_descriptor;
  }
  if (gluon_tensor_descriptor_cls && PyType_Check(gluon_tensor_descriptor_cls)) {
    type_handler_cache[(PyTypeObject*)gluon_tensor_descriptor_cls] = handle_gluon_tensor_descriptor;
  }
  if (constexpr_cls && PyType_Check(constexpr_cls)) {
    type_handler_cache[(PyTypeObject*)constexpr_cls] = handle_constexpr_type;
  }
  if (jit_function_cls && PyType_Check(jit_function_cls)) {
    type_handler_cache[(PyTypeObject*)jit_function_cls] = handle_jit_function;
  }
}


std::pair<py::handle, py::handle> specialize_arg(PyObject *backend,
                                                 PyObject *arg, 
                                                 bool is_const,
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
    return {py::handle(constexpr_str), py::handle()};
  }

  // handling of tuples
  if (PyTuple_Check(arg)) {
    return handle_tuple(backend, arg, is_const, specialize_value, align);
  }

  // fallback paths checking full inheritance
  if (PyObject_IsInstance(arg, constexpr_cls)) {
    return handle_constexpr_type(backend, arg, is_const, specialize_value, align);
  }

  if (PyObject_IsInstance(arg, tensor_descriptor_cls)) {
    return handle_tensor_descriptor(backend, arg, is_const, specialize_value, align);
  }

  if (PyObject_IsInstance(arg, gluon_tensor_descriptor_cls)) {
    return handle_gluon_tensor_descriptor(backend, arg, is_const, specialize_value, align);
  }

  if (PyObject_IsInstance(arg, jit_function_cls)) {
    return handle_jit_function(backend, arg, is_const, specialize_value, align);
  }

  // fallback paths checking attributes directly
  if (PyObject_HasAttr(arg, data_ptr_attr)) {
    return handle_tensor(backend, arg, is_const, specialize_value, align);
  }

  if (PyObject_HasAttr(arg, tma_desc_cpu_ptr_attr)) {
    return {py::handle(nvTmaDesc_str), py::handle()};
  }

  return {py::handle(), py::handle()};
}

static PyObject *specialize_impl(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
  init_lazy();

  if (nargs != 5) {
    PyErr_SetString(PyExc_TypeError, "expected 5 arguments to native specialize_impl");
    return nullptr;
  }

  PyObject* arg = args[1];
  PyObject* backend = args[0];
  int is_const = PyObject_IsTrue(args[2]);
  int specialize_value = PyObject_IsTrue(args[3]);
  int align = PyObject_IsTrue(args[4]);

  if (is_const == -1 || specialize_value == -1 || align == -1) {
    return nullptr;
  }

  auto [type, key] = specialize_arg(backend, arg, is_const, specialize_value, align);

  PyObject *type_obj = type ? type.ptr() : Py_None;
  PyObject *key_obj = key ? key.ptr() : Py_None;

  return PyTuple_Pack(2, type_obj, key_obj);
}

static PyMethodDef module_methods[] = {
    {"native_specialize_impl", (PyCFunction)(void(*)(void))specialize_impl, METH_FASTCALL, nullptr},
    {nullptr, nullptr, 0, nullptr}};

} // anonymous namespace

void init_native_specialize(pybind11::module &m) {
  init_static_str();

  PyObject *module_obj = m.ptr();

  for (int i = 0; module_methods[i].ml_name; ++i) {
    PyObject* func = PyCFunction_New(&module_methods[i], nullptr);
    if (!func) {
      throw py::error_already_set();
    }
    if (PyModule_AddObject(module_obj, module_methods[i].ml_name, func) < 0) {
      Py_DECREF(func);
      throw py::error_already_set();
    }
  }

  py::module atexit = py::module::import("atexit");
  atexit.attr("register")(py::cpp_function(module_cleanup));
}
